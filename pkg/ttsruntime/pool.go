package ttsruntime

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/normalizer"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/onnxconfig"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
)

// CoreType 推理单元类型
type CoreType string

const (
	CoreTypeWork    CoreType = "work"    // 常驻核心
	CoreTypeReserve CoreType = "reserve" // 预留备用核心
)

// WorkCore 表示一个独立的推理单元（子进程）。
type WorkCore struct {
	Name       string   // 进程命名：workCore从1开始，reserveCore从r1开始
	Type       CoreType // 核心类型：work 或 reserve
	Worker     *WorkerProcess
	ActiveReqs atomic.Int64 // 当前活跃请求数
	mu         sync.Mutex   // 互斥锁，确保同一时间只有一个请求使用该推理单元

	// reserve 核心的闲置计时
	lastActiveTime atomic.Int64 // 上次活跃时间的 Unix 毫秒
}

// ReserveIdleTimeout 预留核心闲置超时时间
const ReserveIdleTimeout = 60 * time.Second

// pendingRequest 排队等候的请求
type pendingRequest struct {
	ctx  context.Context
	core chan *WorkCore // 请求者通过此 channel 接收分配到的核心
}

// Pool 管理多个推理单元（子进程），提供两级调度和负载均衡访问。
// 调度策略：
// 1. 请求优先分配给 workCores（常驻核心）
// 2. workCores 全忙时，启用 reserveCores（预留备用核心）
// 3. reserveCores 也全忙时，请求排队等候分配处理单元
type Pool struct {
	mu    sync.RWMutex
	ready bool

	// 常驻核心（常驻内存，不自动销毁）
	workCores []*WorkCore
	// 预留备用核心（闲置1分钟自动销毁）
	reserveCores []*WorkCore
	// 预留核心最大数量
	maxReserveCores int

	// 配置参数
	modelDir     string
	coreCPUs     int
	coreMemMB    int
	maxNewFrames *int
	doSample     *bool
	sampleMode   *string
	executionMode string

	// 排队等候的请求
	pendingMu    sync.Mutex
	pendingQueue []*pendingRequest

	// 预留核心自动销毁
	reserveNextSeq atomic.Int64 // 预留核心命名序号，从1开始
	stopCh         chan struct{}
}

// NewPool 创建推理单元池。
// 所有推理单元以子进程方式运行，每个子进程有独立的内存空间。
func NewPool(modelDir string, workCores, reserveWorkCores, coreCPUs, coreMemMB int, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*Pool, error) {
	if workCores < 1 {
		workCores = 1
	}
	if reserveWorkCores < 0 {
		reserveWorkCores = 0
	}
	if coreCPUs < 1 {
		coreCPUs = 1
	}

	pool := &Pool{
		workCores:       make([]*WorkCore, workCores),
		maxReserveCores: reserveWorkCores,
		modelDir:        modelDir,
		coreCPUs:        coreCPUs,
		coreMemMB:       coreMemMB,
		maxNewFrames:    maxNewFrames,
		doSample:        doSample,
		sampleMode:      sampleMode,
		executionMode:   executionMode,
		stopCh:          make(chan struct{}),
	}

	log.Printf("[Pool] 初始化推理单元池: workCores=%d reserveWorkCores=%d coreCPUs=%d coreMemMB=%d", workCores, reserveWorkCores, coreCPUs, coreMemMB)

	// 启动常驻核心，命名从1开始
	for i := 0; i < workCores; i++ {
		name := fmt.Sprintf("%d", i+1)
		log.Printf("[Pool] 启动常驻推理子进程 #%s (threads=%d, memMB=%d)...", name, coreCPUs, coreMemMB)
		wp, err := NewWorkerProcess(name, modelDir, coreCPUs, coreMemMB, maxNewFrames, doSample, sampleMode, executionMode)
		if err != nil {
			// 清理已创建的子进程
			for j := 0; j < i; j++ {
				if pool.workCores[j] != nil && pool.workCores[j].Worker != nil {
					pool.workCores[j].Worker.Close()
				}
			}
			return nil, fmt.Errorf("启动推理子进程 #%s 失败: %w", name, err)
		}
		pool.workCores[i] = &WorkCore{
			Name: name,
			Type: CoreTypeWork,
			Worker: wp,
		}
		log.Printf("[Pool] 常驻推理子进程 #%s 启动完成", name)
	}

	pool.ready = true

	// 启动预留核心闲置检测协程
	if reserveWorkCores > 0 {
		go pool.reserveIdleMonitor()
	}

	return pool, nil
}

// NewPoolFromConfig 从 onnx 配置创建推理单元池。
func NewPoolFromConfig(modelDir string, cfg *onnxconfig.Config, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*Pool, error) {
	return NewPool(modelDir, cfg.WorkCores, cfg.ReserveWorkCores, cfg.CoreCPUs, cfg.CoreMemMB, maxNewFrames, doSample, sampleMode, executionMode)
}

// Acquire 获取一个推理单元，按优先级：workCores -> reserveCores -> 启动新reserveCore -> 排队
func (p *Pool) Acquire(ctx context.Context) (*WorkCore, error) {
	// 第一优先级：workCores 中找空闲的
	if core := p.acquireFromWorkCores(); core != nil {
		return core, nil
	}

	// 第二优先级：已有的 reserveCores 中找空闲的
	if core := p.acquireFromReserveCores(); core != nil {
		return core, nil
	}

	// 第三优先级：启动新的 reserveCore
	if core := p.tryStartReserveCore(); core != nil {
		return core, nil
	}

	// 所有核心都忙，排队等候
	return p.enqueueWait(ctx)
}

// acquireFromWorkCores 从常驻核心中获取空闲单元
func (p *Pool) acquireFromWorkCores() *WorkCore {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.ready || len(p.workCores) == 0 {
		return nil
	}

	// 找最少活跃请求的核心
	bestIdx := -1
	minActive := int64(1) // 只选活跃请求为0的
	for i, core := range p.workCores {
		active := core.ActiveReqs.Load()
		if active < minActive {
			minActive = active
			bestIdx = i
			if active == 0 {
				break // 找到完全空闲的
			}
		}
	}

	if bestIdx == -1 {
		return nil // 所有常驻核心都忙
	}

	core := p.workCores[bestIdx]
	core.mu.Lock()
	core.ActiveReqs.Add(1)
	return core
}

// acquireFromReserveCores 从预留核心中获取空闲单元
func (p *Pool) acquireFromReserveCores() *WorkCore {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for _, core := range p.reserveCores {
		active := core.ActiveReqs.Load()
		if active == 0 && core.mu.TryLock() {
			core.ActiveReqs.Add(1)
			core.lastActiveTime.Store(time.Now().UnixMilli())
			return core
		}
	}
	return nil
}

// tryStartReserveCore 尝试启动一个新的预留核心
func (p *Pool) tryStartReserveCore() *WorkCore {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.reserveCores) >= p.maxReserveCores {
		return nil
	}

	seq := p.reserveNextSeq.Add(1)
	name := fmt.Sprintf("r%d", seq)

	log.Printf("[Pool] 启动预留推理子进程 #%s (threads=%d, memMB=%d)...", name, p.coreCPUs, p.coreMemMB)
	wp, err := NewWorkerProcess(name, p.modelDir, p.coreCPUs, p.coreMemMB, p.maxNewFrames, p.doSample, p.sampleMode, p.executionMode)
	if err != nil {
		log.Printf("[Pool] 启动预留推理子进程 #%s 失败: %v", name, err)
		return nil
	}

	core := &WorkCore{
		Name: name,
		Type: CoreTypeReserve,
		Worker: wp,
	}
	core.lastActiveTime.Store(time.Now().UnixMilli())
	core.mu.Lock()
	core.ActiveReqs.Add(1)
	p.reserveCores = append(p.reserveCores, core)

	log.Printf("[Pool] 预留推理子进程 #%s 启动完成", name)
	return core
}

// enqueueWait 排队等候分配处理单元
func (p *Pool) enqueueWait(ctx context.Context) (*WorkCore, error) {
	pending := &pendingRequest{
		ctx:  ctx,
		core: make(chan *WorkCore, 1),
	}

	p.pendingMu.Lock()
	p.pendingQueue = append(p.pendingQueue, pending)
	queueLen := len(p.pendingQueue)
	p.pendingMu.Unlock()

	log.Printf("[Pool] 所有核心忙，请求排队等候 (队列长度=%d)", queueLen)

	select {
	case core := <-pending.core:
		return core, nil
	case <-ctx.Done():
		// 从队列中移除
		p.pendingMu.Lock()
		for i, pr := range p.pendingQueue {
			if pr == pending {
				p.pendingQueue = append(p.pendingQueue[:i], p.pendingQueue[i+1:]...)
				break
			}
		}
		p.pendingMu.Unlock()
		return nil, ctx.Err()
	}
}

// dispatchPending 尝试将排队中的请求分配到空闲核心
func (p *Pool) dispatchPending() {
	p.pendingMu.Lock()
	defer p.pendingMu.Unlock()

	for len(p.pendingQueue) > 0 {
		// 尝试获取空闲核心
		core := p.acquireFromWorkCores()
		if core == nil {
			core = p.acquireFromReserveCores()
		}
		if core == nil {
			core = p.tryStartReserveCore()
		}
		if core == nil {
			return // 仍然没有空闲核心
		}

		// 将核心分配给排队的第一个请求
		pending := p.pendingQueue[0]
		p.pendingQueue = p.pendingQueue[1:]

		select {
		case pending.core <- core:
			log.Printf("[Pool] 排队请求已分配到推理子进程 #%s", core.Name)
		case <-pending.ctx.Done():
			// 请求已被取消，释放核心
			p.Release(core)
		}
	}
}

// Release 释放推理单元。
func (p *Pool) Release(core *WorkCore) {
	if core == nil {
		return
	}
	core.ActiveReqs.Add(-1)
	core.lastActiveTime.Store(time.Now().UnixMilli())
	core.mu.Unlock()

	// 释放后尝试分发排队请求
	go p.dispatchPending()
}

// reserveIdleMonitor 定期检查预留核心是否闲置超时，超时则销毁
func (p *Pool) reserveIdleMonitor() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			p.cleanupIdleReserveCores()
		case <-p.stopCh:
			return
		}
	}
}

// cleanupIdleReserveCores 清理闲置超时的预留核心
func (p *Pool) cleanupIdleReserveCores() {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := time.Now().UnixMilli()
	var remaining []*WorkCore

	for _, core := range p.reserveCores {
		active := core.ActiveReqs.Load()
		lastActive := core.lastActiveTime.Load()

		if active == 0 && now-lastActive > ReserveIdleTimeout.Milliseconds() {
			// 闲置超时，销毁进程
			log.Printf("[Pool] 预留推理子进程 #%s 闲置超时，销毁进程", core.Name)
			if core.Worker != nil {
				core.Worker.Close()
			}
		} else {
			remaining = append(remaining, core)
		}
	}

	p.reserveCores = remaining
}

// Close 关闭所有推理子进程。
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	close(p.stopCh)

	for _, core := range p.workCores {
		if core != nil && core.Worker != nil {
			log.Printf("[Pool] 关闭常驻推理子进程 #%s...", core.Name)
			core.Worker.Close()
		}
	}
	for _, core := range p.reserveCores {
		if core != nil && core.Worker != nil {
			log.Printf("[Pool] 关闭预留推理子进程 #%s...", core.Name)
			core.Worker.Close()
		}
	}
	p.workCores = nil
	p.reserveCores = nil
	p.ready = false

	// 通知所有排队请求
	p.pendingMu.Lock()
	for _, pr := range p.pendingQueue {
		close(pr.core)
	}
	p.pendingQueue = nil
	p.pendingMu.Unlock()
}

// IsReady 返回池是否就绪。
func (p *Pool) IsReady() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.ready
}

// WorkCoreCount 返回常驻推理单元数量。
func (p *Pool) WorkCoreCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.workCores)
}

// ReserveCoreCount 返回当前活跃的预留推理单元数量。
func (p *Pool) ReserveCoreCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.reserveCores)
}

// PendingCount 返回当前排队等候的请求数量。
func (p *Pool) PendingCount() int {
	p.pendingMu.Lock()
	defer p.pendingMu.Unlock()
	return len(p.pendingQueue)
}

// Status 返回每个推理单元的状态信息。
func (p *Pool) Status() []map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var status []map[string]interface{}

	for _, core := range p.workCores {
		status = append(status, map[string]interface{}{
			"name":        core.Name,
			"type":        string(core.Type),
			"active_reqs": core.ActiveReqs.Load(),
		})
	}
	for _, core := range p.reserveCores {
		status = append(status, map[string]interface{}{
			"name":        core.Name,
			"type":        string(core.Type),
			"active_reqs": core.ActiveReqs.Load(),
		})
	}
	return status
}

// AcquireForRead 获取一个核心用于只读操作（不经过排队，直接选最少连接的）
func (p *Pool) AcquireForRead() *WorkCore {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.ready || len(p.workCores) == 0 {
		return nil
	}

	// 优先从 workCores 中找最少连接的
	bestIdx := 0
	minActive := p.workCores[0].ActiveReqs.Load()
	for i := 1; i < len(p.workCores); i++ {
		active := p.workCores[i].ActiveReqs.Load()
		if active < minActive {
			minActive = active
			bestIdx = i
		}
	}

	core := p.workCores[bestIdx]
	core.mu.Lock()
	core.ActiveReqs.Add(1)
	return core
}

// SynthesizeWithContextEx 在可用的推理单元上执行合成。
// 文本预处理（包括 WeTextProcessing）在主进程完成，子进程只做推理，避免加载 WeTextProcessing 增加子进程内存。
func (p *Pool) SynthesizeWithContextEx(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int, overrides *ortruntime.GenerationOverrides) (*SynthesisResult, error) {
	core, err := p.Acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("获取推理单元失败: %w", err)
	}
	defer p.Release(core)

	// 主进程完成文本预处理，子进程不再需要加载 WeTextProcessing
	preparedText := normalizer.PrepareTTSText(text, enableRobust, enableWeText)
	log.Printf("[Pool] 文本预处理完成(主进程): 原始长度=%d 预处理后长度=%d (robust=%v wetext=%v)", len(text), len(preparedText), enableRobust, enableWeText)

	log.Printf("[Pool] 请求分配到推理子进程 #%s (type=%s, active=%d)", core.Name, core.Type, core.ActiveReqs.Load())
	return core.Worker.SynthesizeWithPreparedText(ctx, preparedText, voice, promptAudioPath, outputAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, streaming, maxNewFrames, voiceCloneMaxTextTokens, seed, overrides)
}

// SynthesizeStreamEx 在可用的推理单元上执行流式合成。
// 文本预处理（包括 WeTextProcessing）在主进程完成，子进程只做推理，避免加载 WeTextProcessing 增加子进程内存。
func (p *Pool) SynthesizeStreamEx(ctx context.Context, text string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int, overrides *ortruntime.GenerationOverrides) (<-chan StreamChunk, error) {
	core, err := p.Acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("获取推理单元失败: %w", err)
	}

	// 主进程完成文本预处理，子进程不再需要加载 WeTextProcessing
	preparedText := normalizer.PrepareTTSText(text, enableRobust, enableWeText)
	log.Printf("[Pool] 流式文本预处理完成(主进程): 原始长度=%d 预处理后长度=%d (robust=%v wetext=%v)", len(text), len(preparedText), enableRobust, enableWeText)

	log.Printf("[Pool] 流式请求分配到推理子进程 #%s (type=%s, active=%d)", core.Name, core.Type, core.ActiveReqs.Load())

	chunkChan, err := core.Worker.SynthesizeStreamWithPreparedText(ctx, preparedText, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, maxNewFrames, voiceCloneMaxTextTokens, seed, overrides)
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
	core := p.AcquireForRead()
	if core == nil {
		return nil
	}
	defer p.Release(core)
	return core.Worker.ListBuiltinVoices()
}

// EncodeText 使用第一个推理单元进行文本编码。
func (p *Pool) EncodeText(text string) []int {
	core := p.AcquireForRead()
	if core == nil {
		return nil
	}
	defer p.Release(core)
	return core.Worker.EncodeText(text)
}

// CountTextTokens 使用第一个推理单元计算文本 token 数。
func (p *Pool) CountTextTokens(text string) int {
	core := p.AcquireForRead()
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

	for _, core := range p.workCores {
		if core.Worker != nil {
			if err := core.Worker.PreloadVoice(preloadId, audioPath, voice); err != nil {
				log.Printf("[Pool] 推理子进程 #%s 预加载音色失败: %v", core.Name, err)
			}
		}
	}
	for _, core := range p.reserveCores {
		if core.Worker != nil {
			if err := core.Worker.PreloadVoice(preloadId, audioPath, voice); err != nil {
				log.Printf("[Pool] 预留推理子进程 #%s 预加载音色失败: %v", core.Name, err)
			}
		}
	}
	return nil
}

// EnsureSeededRNG 确保推理单元的随机数生成器有合适的种子。
func (p *Pool) EnsureSeededRNG(seed *int64) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	baseSeed := time.Now().UnixNano()
	if seed != nil {
		baseSeed = *seed
	}
	_ = baseSeed

	for _, core := range p.workCores {
		if core.Worker != nil {
			core.Worker.Ping()
		}
	}
	for _, core := range p.reserveCores {
		if core.Worker != nil {
			core.Worker.Ping()
		}
	}
}
