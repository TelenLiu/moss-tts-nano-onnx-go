// Package appconfig 负责加载 conf/app.json 中的业务配置。
//
// 当前支持的字段：
//   - logLevel:  日志级别（由 pkg/log 自行读取，本包不处理）
//   - mp3Volume: MP3 输出默认音量倍数（默认 1.0）
package appconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// Config 表示 conf/app.json 的业务配置部分。
type Config struct {
	LogLevel  string  `json:"logLevel"`
	MP3Volume float64 `json:"mp3Volume"`
}

var (
	once     sync.Once
	cfgCache *Config
)

// DefaultConfigPath 返回默认配置文件路径。
func DefaultConfigPath() string {
	cwd, _ := os.Getwd()
	return filepath.Join(cwd, "conf", "app.json")
}

// DefaultConfig 返回默认配置（mp3Volume 默认 1.0）。
func DefaultConfig() *Config {
	return &Config{
		MP3Volume: 1.0,
	}
}

// Load 从指定路径加载配置。path 为空时使用 DefaultConfigPath。
// 文件不存在或解析失败时返回默认配置。
func Load(path string) (*Config, error) {
	if path == "" {
		path = DefaultConfigPath()
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return DefaultConfig(), nil
		}
		return nil, err
	}

	cfg := DefaultConfig()
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	// 规范化：mp3Volume <= 0 视为未设置，回退到默认 1.0
	if cfg.MP3Volume <= 0 {
		cfg.MP3Volume = 1.0
	}
	return cfg, nil
}

// LoadOnce 以单例方式加载配置（只读取一次文件）。
func LoadOnce() *Config {
	once.Do(func() {
		c, err := Load("")
		if err != nil || c == nil {
			c = DefaultConfig()
		}
		cfgCache = c
	})
	return cfgCache
}

// MP3Volume 返回默认 MP3 音量倍数。
func MP3Volume() float64 {
	return LoadOnce().MP3Volume
}
