package stream

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
)

type MemoryStats struct {
	RSS          uint64 `json:"rss_bytes"`   // 进程实际物理内存（含C++库），用于阈值判断
	Alloc        uint64 `json:"alloc_bytes"` // Go 堆分配
	Sys          uint64 `json:"sys_bytes"`   // Go 从OS申请的总内存
	NumGC        uint32 `json:"num_gc"`
	PauseTotalNs uint64 `json:"pause_total_ns"`
	Timestamp    int64  `json:"timestamp"`
}

type MemoryMonitor struct {
	mu                sync.RWMutex
	lastStats         MemoryStats
	history           []MemoryStats
	maxHistory        int
	warningThreshold  uint64
	criticalThreshold uint64
}

func NewMemoryMonitor(warningMB, criticalMB uint64) *MemoryMonitor {
	return &MemoryMonitor{
		history:           make([]MemoryStats, 0, 64),
		maxHistory:        64,
		warningThreshold:  warningMB * 1024 * 1024,
		criticalThreshold: criticalMB * 1024 * 1024,
	}
}

// getProcessRSSBytes 获取当前进程的实际物理内存占用（RSS），单位字节
// runtime.MemStats.Alloc 只统计 Go 堆内存，不包括 ONNX Runtime 等 C++ 库的内存分配
func getProcessRSSBytes() uint64 {
	pid := os.Getpid()

	// Linux: 从 /proc/self/status 读取 VmRSS
	if data, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "VmRSS:") {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					if kb, err := strconv.ParseFloat(parts[1], 64); err == nil {
						return uint64(kb * 1024) // kB → bytes
					}
				}
			}
		}
	}

	// macOS / 其他 Unix: 使用 ps 命令
	cmd := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid))
	if output, err := cmd.Output(); err == nil {
		output = bytes.TrimSpace(output)
		if kb, err := strconv.ParseFloat(string(output), 64); err == nil {
			return uint64(kb * 1024) // kB → bytes
		}
	}

	// Windows: 使用 tasklist 命令
	cmd2 := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/FO", "CSV", "/NH")
	if output, err := cmd2.Output(); err == nil {
		fields := strings.Split(string(output), ",")
		if len(fields) >= 5 {
			memStr := strings.TrimSpace(strings.Trim(fields[len(fields)-1], `" K\r\n`))
			memStr = strings.ReplaceAll(memStr, ",", "")
			if kb, err := strconv.ParseFloat(memStr, 64); err == nil {
				return uint64(kb * 1024)
			}
		}
	}

	// 兜底：使用 Go 的 Sys 作为粗略估计
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Sys
}

func (m *MemoryMonitor) Sample() MemoryStats {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	rssBytes := getProcessRSSBytes()

	stats := MemoryStats{
		RSS:          rssBytes,
		Alloc:        mem.Alloc,
		Sys:          mem.Sys,
		NumGC:        mem.NumGC,
		PauseTotalNs: mem.PauseTotalNs,
		Timestamp:    time.Now().UnixNano(),
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.lastStats = stats
	m.history = append(m.history, stats)
	if len(m.history) > m.maxHistory {
		m.history = m.history[1:]
	}

	if stats.RSS > m.criticalThreshold {
		log.Warnf("[内存警告] 严重：RSS=%.1fMB Alloc=%.1fMB (阈值：%.1fMB)",
			float64(stats.RSS)/1024/1024, float64(stats.Alloc)/1024/1024,
			float64(m.criticalThreshold)/1024/1024)
	} else if stats.RSS > m.warningThreshold {
		log.Warnf("[内存警告] RSS=%.1fMB Alloc=%.1fMB (阈值：%.1fMB)",
			float64(stats.RSS)/1024/1024, float64(stats.Alloc)/1024/1024,
			float64(m.warningThreshold)/1024/1024)
	}

	return stats
}

func (m *MemoryMonitor) GetLastStats() MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.lastStats
}

func (m *MemoryMonitor) GetHistory() []MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]MemoryStats, len(m.history))
	copy(result, m.history)
	return result
}

func (m *MemoryMonitor) ForceGC() {
	log.Debugf("[内存管理] 触发强制 GC")
	runtime.GC()
	m.Sample()
}

type BackpressureController struct {
	mu               sync.RWMutex
	targetLatencyMs  float64
	currentLatencyMs float64
	bufferUsage      float64
	throttleFactor   float64
}

func NewBackpressureController(targetLatencyMs float64) *BackpressureController {
	return &BackpressureController{
		targetLatencyMs: targetLatencyMs,
		throttleFactor:  1.0,
	}
}

func (b *BackpressureController) Update(bufferSize, bufferCapacity int, processingLatencyMs float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.bufferUsage = float64(bufferSize) / float64(bufferCapacity)
	b.currentLatencyMs = processingLatencyMs

	if b.bufferUsage > 0.9 {
		b.throttleFactor = 0.5
		log.Debugf("[背压] 高负载：buffer 使用率 %.0f%%，降低生成速度", b.bufferUsage*100)
	} else if b.bufferUsage > 0.7 {
		b.throttleFactor = 0.75
	} else if b.bufferUsage < 0.3 {
		b.throttleFactor = 1.2
		log.Debugf("[背压] 低负载：buffer 使用率 %.0f%%，提高生成速度", b.bufferUsage*100)
	} else {
		b.throttleFactor = 1.0
	}

	if b.currentLatencyMs > b.targetLatencyMs*2 {
		b.throttleFactor *= 0.8
		log.Debugf("[背压] 高延迟：%.1fms (目标：%.1fms)", b.currentLatencyMs, b.targetLatencyMs)
	}
}

func (b *BackpressureController) GetThrottleFactor() float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.throttleFactor
}

func (b *BackpressureController) ShouldPause() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.throttleFactor < 0.6
}

type StreamStats struct {
	FramesGenerated int64   `json:"frames_generated"`
	FramesDecoded   int64   `json:"frames_decoded"`
	FramesSent      int64   `json:"frames_sent"`
	CurrentBuffer   int     `json:"current_buffer_size"`
	DecodeBudget    int     `json:"decode_budget"`
	LeadSeconds     float64 `json:"lead_seconds"`
	IsThrottled     bool    `json:"is_throttled"`
}

type StreamController struct {
	mu             sync.RWMutex
	stats          StreamStats
	startTime      time.Time
	firstAudioTime time.Time
	emittedSamples int64
	sampleRate     int
	memoryMonitor  *MemoryMonitor
	backpressure   *BackpressureController
}

func NewStreamController(sampleRate int) *StreamController {
	return &StreamController{
		startTime:     time.Now(),
		sampleRate:    sampleRate,
		memoryMonitor: NewMemoryMonitor(500, 800),
		backpressure:  NewBackpressureController(200),
	}
}

func (c *StreamController) RecordFrameGenerated() {
	c.mu.Lock()
	c.stats.FramesGenerated++
	c.mu.Unlock()
}

func (c *StreamController) RecordFrameDecoded(samples int) {
	c.mu.Lock()
	c.stats.FramesDecoded++
	c.emittedSamples += int64(samples)
	if c.stats.FramesDecoded == 1 {
		c.firstAudioTime = time.Now()
	}
	c.mu.Unlock()
}

func (c *StreamController) RecordFrameSent() {
	c.mu.Lock()
	c.stats.FramesSent++
	c.mu.Unlock()
}

func (c *StreamController) UpdateBufferStatus(bufferSize, bufferCapacity int) {
	c.mu.Lock()
	c.stats.CurrentBuffer = bufferSize
	c.mu.Unlock()

	c.backpressure.Update(bufferSize, bufferCapacity, c.GetLatencyMs())
}

func (c *StreamController) GetLeadSeconds() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.firstAudioTime.IsZero() {
		return 0
	}

	elapsedSeconds := time.Since(c.firstAudioTime).Seconds()
	emittedSeconds := float64(c.emittedSamples) / float64(c.sampleRate)
	return emittedSeconds - elapsedSeconds
}

func (c *StreamController) GetLatencyMs() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.firstAudioTime.IsZero() {
		return 0
	}
	return time.Since(c.firstAudioTime).Seconds() * 1000
}

func (c *StreamController) CalculateDecodeBudget() int {
	leadSeconds := c.GetLeadSeconds()

	c.mu.Lock()
	c.stats.LeadSeconds = leadSeconds
	c.mu.Unlock()

	var budget int
	if leadSeconds < 0.20 {
		budget = 1
	} else if leadSeconds < 0.55 {
		budget = 2
	} else if leadSeconds < 1.10 {
		budget = 4
	} else {
		budget = 8
	}

	throttle := c.backpressure.GetThrottleFactor()
	if throttle < 1.0 {
		budget = int(float64(budget) * throttle)
		if budget < 1 {
			budget = 1
		}
	}

	c.mu.Lock()
	c.stats.DecodeBudget = budget
	c.mu.Unlock()

	return budget
}

func (c *StreamController) GetStats() StreamStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := c.stats
	stats.IsThrottled = c.backpressure.GetThrottleFactor() < 1.0
	return stats
}

func (c *StreamController) GetMemoryStats() MemoryStats {
	return c.memoryMonitor.Sample()
}

func (c *StreamController) ShouldPause() bool {
	return c.backpressure.ShouldPause()
}
