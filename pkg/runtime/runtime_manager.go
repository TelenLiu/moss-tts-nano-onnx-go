package runtime

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
)

var ErrNotInitialized = errors.New("runtime not initialized")

type InitProgress struct {
	Phase   string `json:"phase"`
	Message string `json:"message"`
	Percent int    `json:"percent"`
	Error   string `json:"error,omitempty"`
}

type ProgressCallback func(InitProgress)

type RuntimeManager struct {
	config       *deps.Config
	cpuThreads   int
	maxNewFrames int

	mu       sync.RWMutex
	ort      *ortruntime.OrtCpuRuntime
	tts      *ttsruntime.OnnxTtsRuntime
	ready    bool
	initErr  error
	progress []InitProgress
}

func NewRuntimeManager(cfg *deps.Config, cpuThreads, maxNewFrames int) *RuntimeManager {
	return &RuntimeManager{
		config:       cfg,
		cpuThreads:   cpuThreads,
		maxNewFrames: maxNewFrames,
		progress:     make([]InitProgress, 0),
	}
}

func (m *RuntimeManager) emitProgress(phase, message string, percent int, err error) {
	evt := InitProgress{
		Phase:   phase,
		Message: message,
		Percent: percent,
	}
	if err != nil {
		evt.Error = err.Error()
	}

	m.mu.Lock()
	m.progress = append(m.progress, evt)
	m.mu.Unlock()

	log.Printf("[%s] %s (%d%%)", phase, message, percent)
}

func (m *RuntimeManager) Initialize(ctx context.Context, progressCb ProgressCallback) error {
	m.mu.Lock()
	if m.ready {
		m.mu.Unlock()
		return nil
	}
	if m.initErr != nil {
		err := m.initErr
		m.mu.Unlock()
		return err
	}
	m.mu.Unlock()

	go m.doInitialize(ctx, progressCb)

	return nil
}

func (m *RuntimeManager) doInitialize(ctx context.Context, progressCb ProgressCallback) {
	m.mu.Lock()
	m.progress = make([]InitProgress, 0)
	m.mu.Unlock()

	report := func(phase, msg string, pct int, err error) {
		m.emitProgress(phase, msg, pct, err)
		if progressCb != nil {
			progressCb(InitProgress{Phase: phase, Message: msg, Percent: pct, Error: fmt.Sprintf("%v", err)})
		}
	}

	deps.SetDynlibPath(m.config.LibDir)

	report("check", "正在检测运行环境...", 5, nil)

	report("download", "正在检查 ONNX Runtime 本地依赖...", 10, nil)
	if err := deps.EnsureNativeLibs(m.config); err != nil {
		report("error", fmt.Sprintf("ONNX Runtime 依赖准备失败：%v", err), 0, err)
		m.mu.Lock()
		m.initErr = fmt.Errorf("ONNX Runtime 依赖准备失败：%w", err)
		m.mu.Unlock()
		return
	}
	report("download", "ONNX Runtime 依赖已就绪", 25, nil)

	report("download", "正在检查模型文件...", 30, nil)
	if err := deps.EnsureModels(m.config); err != nil {
		report("error", fmt.Sprintf("模型文件准备失败：%v", err), 0, err)
		m.mu.Lock()
		m.initErr = fmt.Errorf("模型文件准备失败：%w", err)
		m.mu.Unlock()
		return
	}
	report("download", "模型下载完成，正在加载...", 50, nil)

	report("load", "正在初始化 ONNX Runtime 环境...", 55, nil)
	if err := ortruntime.InitializeORT(m.config.LibDir); err != nil {
		report("error", fmt.Sprintf("ONNX Runtime 初始化失败：%v", err), 0, err)
		m.mu.Lock()
		m.initErr = fmt.Errorf("ONNX Runtime 初始化失败：%w", err)
		m.mu.Unlock()
		return
	}
	report("load", "ONNX Runtime 环境初始化成功", 70, nil)

	report("load", "正在加载 TTS 模型...", 75, nil)
	rt, err := ttsruntime.NewOnnxTtsRuntime(
		m.config.ModelDir, m.cpuThreads,
		&m.maxNewFrames, nil, nil, "hybrid", // 默认使用混合模式
	)
	if err != nil {
		report("error", fmt.Sprintf("TTS 运行时初始化失败：%v", err), 0, err)
		m.mu.Lock()
		m.initErr = fmt.Errorf("TTS 运行时初始化失败：%w", err)
		m.mu.Unlock()
		return
	}
	report("load", "TTS 模型加载完成", 95, nil)

	m.mu.Lock()
	m.ort = nil
	m.tts = rt
	m.ready = true
	m.mu.Unlock()

	report("ready", "系统就绪", 100, nil)
}

func (m *RuntimeManager) WaitReady(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			m.mu.RLock()
			ready := m.ready
			initErr := m.initErr
			m.mu.RUnlock()

			if initErr != nil {
				return initErr
			}
			if ready {
				return nil
			}
		}
	}
}

func (m *RuntimeManager) GetTTSRuntime() (*ttsruntime.OnnxTtsRuntime, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.ready {
		if m.initErr != nil {
			return nil, m.initErr
		}
		return nil, ErrNotInitialized
	}

	return m.tts, nil
}

func (m *RuntimeManager) IsReady() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.ready
}

func (m *RuntimeManager) GetProgress() []InitProgress {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]InitProgress, len(m.progress))
	copy(result, m.progress)
	return result
}

func (m *RuntimeManager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.tts != nil {
		m.tts.Close()
		m.tts = nil
	}
	if m.ort != nil {
		m.ort.Close()
		m.ort = nil
	}
	m.ready = false
}
