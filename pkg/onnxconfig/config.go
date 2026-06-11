package onnxconfig

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// Config 表示 conf/onnx.json 的内容。
type Config struct {
	WorkCores int `json:"workCores"` // 最大同时启动的推理单元数（默认1，至少1）
	CoreCPUs  int `json:"coreCPUs"`  // 每个推理单元最大允许的CPU内核数（默认4，至少1）
	CoreMemMB int `json:"coreMemMB"` // 每个推理单元的内存阈值MB（默认800,单元内存峰值控制在3.2GB），超过时触发Session重置
}

// DefaultConfigPath 返回默认配置文件路径。
func DefaultConfigPath() string {
	cwd, _ := os.Getwd()
	return filepath.Join(cwd, "conf", "onnx.json")
}

// DefaultConfig 返回默认配置。
func DefaultConfig() *Config {
	return &Config{
		WorkCores: 1,
		CoreCPUs:  4,
		CoreMemMB: 800,
	}
}

// Load 从指定路径加载配置。文件不存在时返回默认配置。
func Load(path string) (*Config, error) {
	if path == "" {
		path = DefaultConfigPath()
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			cfg := DefaultConfig()
			cfg.CoreCPUs = defaultCoreCPUs()
			return cfg, nil
		}
		return nil, fmt.Errorf("读取 onnx 配置失败: %w", err)
	}

	cfg := DefaultConfig()
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("解析 onnx 配置失败: %w", err)
	}

	cfg = normalize(cfg)
	return cfg, nil
}

// normalize 规范化配置，确保值在合法范围内。
func normalize(cfg *Config) *Config {
	if cfg.WorkCores < 1 {
		cfg.WorkCores = 1
	}
	if cfg.CoreCPUs < 1 {
		cfg.CoreCPUs = 1
	}
	if cfg.CoreMemMB < 100 {
		cfg.CoreMemMB = 800
	}
	// 约束：workCores * coreCPUs 不超过 CPU 核心数
	totalCores := runtime.NumCPU()
	if cfg.WorkCores*cfg.CoreCPUs > totalCores {
		// 优先保证 workCores，缩减 coreCPUs
		cfg.CoreCPUs = totalCores / cfg.WorkCores
		if cfg.CoreCPUs < 1 {
			cfg.CoreCPUs = 1
		}
	}
	return cfg
}

// defaultCoreCPUs 返回默认的每个推理单元CPU数。
func defaultCoreCPUs() int {
	n := runtime.NumCPU()
	if n >= 4 {
		return 4
	}
	return max(1, n)
}
