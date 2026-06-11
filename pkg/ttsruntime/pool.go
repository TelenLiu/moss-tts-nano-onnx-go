package ttsruntime

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/onnxconfig"
)

// WorkCore 表示一个独立的推理单元（子进程）。
type WorkCore struct {
	ID        int
	Worker    *WorkerProcess
	ActiveReqs atomic.Int64 // 当前活跃请求数
	mu        sync.Mutex    // 互斥锁，确保同一时间只有一个请求使用该推理单元
}

// Pool 管理多个推理单元（子进程），提供负载均衡访问。
type Pool struct {
	cores []*WorkCore
	mu    sync.RWMutex
	ready bool

	// 轮询计数器（用于最少连接负载均衡）
	nextCore atomic.Int64
}

// NewPool 创建推理单元池。
// 所有推理单元以子进程方式运行，每个子进程有独立的内存空间。
// workCores: 推理单元数量，coreCPUs: 每个单元的CPU线程数，coreMemMB: 每个单元的内存阈值MB。
func NewPool(modelDir string, workCores, coreCPUs, coreMemMB int, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*Pool, error) {
	if workCores < 1 {
		workCores = 1
	}
	if coreCPUs < 1 {
		coreCPUs = 1
	}

	pool := &Pool{
		cores: make([]*WorkCore, workCores),
	}

	log.Printf("[Pool] 初始化推理单元池: workCores=%d coreCPUs=%d coreMemMB=%d", workCores, coreCPUs, coreMemMB)

	for i := 0; i < workCores; i++ {
		log.Printf("[Pool] 启动推理子进程 #%d (threads=%d, memMB=%d)...", i, coreCPUs, coreMemMB)
		wp, err := NewWorkerProcess(i, modelDir, coreCPUs, coreMemMB, maxNewFrames, doSample, sampleMode, executionMode)
		if err != nil {
			// 清理已创建的子进程
			for j := 0; j < i; j++ {
				if pool.cores[j] != nil && pool.cores[j].Worker != nil {
					pool.cores[j].Worker.Close()
				}
			}
			return nil, fmt.Errorf("启动推理子进程 #%d 失败: %w", i, err)
		}
		pool.cores[i] = &WorkCore{
			ID:     i,
			Worker: wp,
		}
		log.Printf("[Pool] 推理子进程 #%d 启动完成", i)
	}

	pool.ready = true
	return pool, nil
}

// NewPoolFromConfig 从 onnx 配置创建推理单元池。
func NewPoolFromConfig(modelDir string, cfg *onnxconfig.Config, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*Pool, error) {
	return NewPool(modelDir, cfg.WorkCores, cfg.CoreCPUs, cfg.CoreMemMB, maxNewFrames, doSample, sampleMode, executionMode)
}

// Acquire 使用最少连接策略获取一个推理单元。
func (p *Pool) Acquire() *WorkCore {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.ready || len(p.cores) == 0 {
		return nil
	}

	// 单个单元直接返回
	if len(p.cores) == 1 {
		core := p.cores[0]
		core.mu.Lock()
		core.ActiveReqs.Add(1)
		return core
	}

	// 最少连接负载均衡
	bestIdx := 0
	minActive := p.cores[0].ActiveReqs.Load()
	for i := 1; i < len(p.cores); i++ {
		active := p.cores[i].ActiveReqs.Load()
		if active < minActive {
			minActive = active
			bestIdx = i
		}
	}

	core := p.cores[bestIdx]
	core.mu.Lock()
	core.ActiveReqs.Add(1)
	return core
}

// Release 释放推理单元。
func (p *Pool) Release(core *WorkCore) {
	if core != nil {
		core.ActiveReqs.Add(-1)
		core.mu.Unlock()
	}
}

// Close 关闭所有推理子进程。
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, core := range p.cores {
		if core != nil && core.Worker != nil {
			log.Printf("[Pool] 关闭推理子进程 #%d...", core.ID)
			core.Worker.Close()
		}
	}
	p.cores = nil
	p.ready = false
}

// IsReady 返回池是否就绪。
func (p *Pool) IsReady() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.ready
}

// WorkCoreCount 返回推理单元数量。
func (p *Pool) WorkCoreCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.cores)
}

// Status 返回每个推理单元的状态信息。
func (p *Pool) Status() []map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	status := make([]map[string]interface{}, len(p.cores))
	for i, core := range p.cores {
		status[i] = map[string]interface{}{
			"id":          core.ID,
			"active_reqs": core.ActiveReqs.Load(),
		}
	}
	return status
}

// SynthesizeWithContextEx 在最少连接的推理单元上执行合成。
func (p *Pool) SynthesizeWithContextEx(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (*SynthesisResult, error) {
	core := p.Acquire()
	if core == nil {
		return nil, fmt.Errorf("无可用推理单元")
	}
	defer p.Release(core)

	log.Printf("[Pool] 请求分配到推理子进程 #%d (active=%d)", core.ID, core.ActiveReqs.Load())
	return core.Worker.SynthesizeWithContextEx(ctx, text, voice, promptAudioPath, outputAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, streaming, maxNewFrames, voiceCloneMaxTextTokens, enableRobust, enableWeText, seed)
}

// SynthesizeStreamEx 在最少连接的推理单元上执行流式合成。
func (p *Pool) SynthesizeStreamEx(ctx context.Context, text string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (<-chan StreamChunk, error) {
	core := p.Acquire()
	if core == nil {
		return nil, fmt.Errorf("无可用推理单元")
	}

	log.Printf("[Pool] 流式请求分配到推理子进程 #%d (active=%d)", core.ID, core.ActiveReqs.Load())

	chunkChan, err := core.Worker.SynthesizeStreamEx(ctx, text, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, maxNewFrames, voiceCloneMaxTextTokens, enableRobust, enableWeText, seed)
	if err != nil {
		p.Release(core)
		return nil, err
	}

	// 包装 channel，在流结束后释放推理单元
	wrappedChan := make(chan StreamChunk, 64)
	go func() {
		defer close(wrappedChan)
		defer p.Release(core)
		for chunk := range chunkChan {
			wrappedChan <- chunk
		}
	}()

	return wrappedChan, nil
}

// ListBuiltinVoices 从第一个推理单元获取内置音色列表。
func (p *Pool) ListBuiltinVoices() []map[string]interface{} {
	core := p.Acquire()
	if core == nil {
		return nil
	}
	defer p.Release(core)
	return core.Worker.ListBuiltinVoices()
}

// EncodeText 使用第一个推理单元进行文本编码。
func (p *Pool) EncodeText(text string) []int {
	core := p.Acquire()
	if core == nil {
		return nil
	}
	defer p.Release(core)
	return core.Worker.EncodeText(text)
}

// CountTextTokens 使用第一个推理单元计算文本 token 数。
func (p *Pool) CountTextTokens(text string) int {
	core := p.Acquire()
	if core == nil {
		return 0
	}
	defer p.Release(core)
	return core.Worker.CountTextTokens(text)
}

// PreloadVoice 在所有推理单元上预加载音色。
func (p *Pool) PreloadVoice(preloadId, audioPath, voice string) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for _, core := range p.cores {
		if core.Worker != nil {
			if err := core.Worker.PreloadVoice(preloadId, audioPath, voice); err != nil {
				log.Printf("[Pool] 推理子进程 #%d 预加载音色失败: %v", core.ID, err)
			}
		}
	}
	return nil
}

// EnsureSeededRNG 确保推理单元的随机数生成器有合适的种子。
func (p *Pool) EnsureSeededRNG(seed *int64) {
	// 子进程架构下，RNG 在子进程内部管理
	// 通过 ping 确认子进程存活即可
	p.mu.RLock()
	defer p.mu.RUnlock()

	baseSeed := time.Now().UnixNano()
	if seed != nil {
		baseSeed = *seed
	}
	_ = baseSeed // 子进程内部自行管理 RNG

	for _, core := range p.cores {
		if core.Worker != nil {
			// 发送 ping 确认子进程存活
			core.Worker.Ping()
		}
	}
}
