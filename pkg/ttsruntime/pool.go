package ttsruntime

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/onnxconfig"
)

// WorkCore 表示一个独立的推理单元，包含自己的 OrtCpuRuntime 和 TtsRuntime。
type WorkCore struct {
	ID       int
	Runtime  *OnnxTtsRuntime
	ActiveReqs atomic.Int64 // 当前活跃请求数
	mu      sync.Mutex     // 互斥锁，确保同一时间只有一个请求使用该推理单元
}

// Pool 管理多个推理单元，提供负载均衡访问。
type Pool struct {
	cores []*WorkCore
	mu    sync.RWMutex
	ready bool

	// 轮询计数器（用于最少连接负载均衡）
	nextCore atomic.Int64
}

// NewPool 创建推理单元池。
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

	log.Printf("[Pool] 初始化推理单元池: workCores=%d coreCPUs=%d", workCores, coreCPUs)

	for i := 0; i < workCores; i++ {
		log.Printf("[Pool] 初始化推理单元 #%d (threads=%d)...", i, coreCPUs)
		rt, err := NewOnnxTtsRuntime(modelDir, coreCPUs, coreMemMB, maxNewFrames, doSample, sampleMode, executionMode)
		if err != nil {
			// 清理已创建的单元
			for j := 0; j < i; j++ {
				if pool.cores[j] != nil && pool.cores[j].Runtime != nil {
					pool.cores[j].Runtime.Close()
				}
			}
			return nil, fmt.Errorf("创建推理单元 #%d 失败: %w", i, err)
		}
		pool.cores[i] = &WorkCore{
			ID:      i,
			Runtime: rt,
		}
		log.Printf("[Pool] 推理单元 #%d 初始化完成", i)
	}

	pool.ready = true
	return pool, nil
}

// NewPoolFromConfig 从 onnx 配置创建推理单元池。
func NewPoolFromConfig(modelDir string, cfg *onnxconfig.Config, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*Pool, error) {
	return NewPool(modelDir, cfg.WorkCores, cfg.CoreCPUs, cfg.CoreMemMB, maxNewFrames, doSample, sampleMode, executionMode)
}

// Acquire 使用最少连接策略获取一个推理单元。
// 调用方在使用完毕后无需显式释放（活跃计数在 Synthesize 内部管理）。
func (p *Pool) Acquire() *WorkCore {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.ready || len(p.cores) == 0 {
		return nil
	}

	// 单个单元直接返回
	if len(p.cores) == 1 {
		core := p.cores[0]
		core.mu.Lock() // 获取互斥锁，确保串行访问
		core.ActiveReqs.Add(1)
		return core
	}

	// 最少连接负载均衡：找活跃请求最少的单元
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
	core.mu.Lock() // 获取互斥锁，确保串行访问
	core.ActiveReqs.Add(1)
	return core
}

// Release 释放推理单元（减少活跃请求计数并释放互斥锁）。
func (p *Pool) Release(core *WorkCore) {
	if core != nil {
		core.ActiveReqs.Add(-1)
		core.mu.Unlock() // 释放互斥锁，允许下一个请求使用
	}
}

// GetRuntime 获取最少连接的推理单元的 TTS Runtime。
// 这是便捷方法，适用于不需要精细控制活跃计数的场景。
func (p *Pool) GetRuntime() *OnnxTtsRuntime {
	core := p.Acquire()
	if core == nil {
		return nil
	}
	// 不减少计数，因为调用方会在 Synthesize 内部管理
	return core.Runtime
}

// Close 关闭所有推理单元。
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, core := range p.cores {
		if core != nil && core.Runtime != nil {
			log.Printf("[Pool] 关闭推理单元 #%d...", core.ID)
			core.Runtime.Close()
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

	log.Printf("[Pool] 请求分配到推理单元 #%d (active=%d)", core.ID, core.ActiveReqs.Load())
	return core.Runtime.SynthesizeWithContextEx(ctx, text, voice, promptAudioPath, outputAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, streaming, maxNewFrames, voiceCloneMaxTextTokens, enableRobust, enableWeText, seed)
}

// SynthesizeStreamEx 在最少连接的推理单元上执行流式合成。
func (p *Pool) SynthesizeStreamEx(ctx context.Context, text string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (<-chan StreamChunk, error) {
	core := p.Acquire()
	if core == nil {
		return nil, fmt.Errorf("无可用推理单元")
	}

	log.Printf("[Pool] 流式请求分配到推理单元 #%d (active=%d)", core.ID, core.ActiveReqs.Load())

	chunkChan, err := core.Runtime.SynthesizeStreamEx(ctx, text, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, maxNewFrames, voiceCloneMaxTextTokens, enableRobust, enableWeText, seed)
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
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.cores) > 0 && p.cores[0].Runtime != nil {
		return p.cores[0].Runtime.OrtRuntime.ListBuiltinVoices()
	}
	return nil
}

// EncodeText 使用第一个推理单元进行文本编码。
func (p *Pool) EncodeText(text string) []int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.cores) > 0 && p.cores[0].Runtime != nil {
		return p.cores[0].Runtime.EncodeText(text)
	}
	return nil
}

// CountTextTokens 使用第一个推理单元计算文本 token 数。
func (p *Pool) CountTextTokens(text string) int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.cores) > 0 && p.cores[0].Runtime != nil {
		return p.cores[0].Runtime.CountTextTokens(text)
	}
	return 0
}

// PreloadVoice 在所有推理单元上预加载音色。
func (p *Pool) PreloadVoice(preloadId, audioPath, voice string) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for _, core := range p.cores {
		if core.Runtime != nil && core.Runtime.PreloadCache != nil {
			if err := core.Runtime.PreloadCache.Preload(preloadId, audioPath, voice); err != nil {
				log.Printf("[Pool] 推理单元 #%d 预加载音色失败: %v", core.ID, err)
			}
		}
	}
	return nil
}

// EnsureSeededRNG 确保推理单元的随机数生成器有合适的种子。
// 在每个推理单元上设置独立的随机种子，避免多个单元产生相同结果。
func (p *Pool) EnsureSeededRNG(seed *int64) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	baseSeed := time.Now().UnixNano()
	if seed != nil {
		baseSeed = *seed
	}

	for i, core := range p.cores {
		if core.Runtime != nil && core.Runtime.OrtRuntime != nil {
			// 每个单元使用不同的种子（基于基础种子 + 单元ID偏移）
			coreSeed := baseSeed + int64(i)*12345
			core.Runtime.OrtRuntime.RNG = rand.New(rand.NewSource(coreSeed))
		}
	}
}
