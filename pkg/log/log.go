// Package log 提供基于 logrus 的统一日志分级输出。
//
// 使用方式：
//
//	import "github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
//
//	log.Infof("服务启动于 %s", addr)
//	log.Debugf("详细调试信息: %v", data)
//	log.Warnf("警告: %v", err)
//	log.Errorf("失败: %v", err)
//
// 日志级别优先级：环境变量 LOG_LEVEL > conf/app.json logLevel > 默认 info
// 可选值: debug, info, warn, error
package log

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
)

var std = logrus.New()

func init() {
	// 默认格式：文本格式，包含时间、级别、消息
	std.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
		ForceColors:     false,
		DisableQuote:    true,
	})

	// 默认输出到 stderr
	std.SetOutput(os.Stderr)

	// 确定日志级别：环境变量 > app.json > 默认 info
	levelStr := resolveLogLevel()
	switch levelStr {
	case "debug":
		std.SetLevel(logrus.DebugLevel)
	case "info":
		std.SetLevel(logrus.InfoLevel)
	case "warn":
		std.SetLevel(logrus.WarnLevel)
	case "error":
		std.SetLevel(logrus.ErrorLevel)
	default:
		std.SetLevel(logrus.InfoLevel)
	}
}

// resolveLogLevel 按优先级确定日志级别：环境变量 > conf/app.json > 默认 info
func resolveLogLevel() string {
	// 1. 环境变量
	if env := strings.ToLower(os.Getenv("LOG_LEVEL")); env != "" {
		return env
	}

	// 2. conf/app.json
	if level := readAppLogLevel(); level != "" {
		return level
	}

	// 3. 默认
	return "info"
}

// readAppLogLevel 从 conf/app.json 读取 logLevel 字段
func readAppLogLevel() string {
	cwd, _ := os.Getwd()
	paths := []string{
		filepath.Join(cwd, "conf", "app.json"),
	}
	// 也尝试从可执行文件目录查找
	if exe, err := os.Executable(); err == nil {
		paths = append(paths, filepath.Join(filepath.Dir(exe), "conf", "app.json"))
	}

	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var cfg struct {
			LogLevel string `json:"logLevel"`
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			continue
		}
		if cfg.LogLevel != "" {
			return strings.ToLower(cfg.LogLevel)
		}
	}
	return ""
}

// Debugf 输出 debug 级别日志
func Debugf(format string, args ...interface{}) {
	std.Debugf(format, args...)
}

// Infof 输出 info 级别日志
func Infof(format string, args ...interface{}) {
	std.Infof(format, args...)
}

// Infoln 输出 info 级别日志（带换行）
func Infoln(args ...interface{}) {
	std.Infoln(args...)
}

// Warnf 输出 warn 级别日志
func Warnf(format string, args ...interface{}) {
	std.Warnf(format, args...)
}

// Errorf 输出 error 级别日志
func Errorf(format string, args ...interface{}) {
	std.Errorf(format, args...)
}

// Debug 输出 debug 级别日志
func Debug(args ...interface{}) {
	std.Debug(args...)
}

// Info 输出 info 级别日志
func Info(args ...interface{}) {
	std.Info(args...)
}

// Warn 输出 warn 级别日志
func Warn(args ...interface{}) {
	std.Warn(args...)
}

// Error 输出 error 级别日志
func Error(args ...interface{}) {
	std.Error(args...)
}

// Fatalf 输出 fatal 级别日志并退出
func Fatalf(format string, args ...interface{}) {
	std.Fatalf(format, args...)
}

// Fatal 输出 fatal 级别日志并退出
func Fatal(args ...interface{}) {
	std.Fatal(args...)
}

// Printf 兼容标准 log.Printf，按 info 级别输出
func Printf(format string, args ...interface{}) {
	std.Infof(format, args...)
}

// Print 兼容标准 log.Print，按 info 级别输出
func Print(args ...interface{}) {
	std.Info(args...)
}

// Println 兼容标准 log.Println，按 info 级别输出
func Println(args ...interface{}) {
	std.Infoln(args...)
}

// SetLevel 动态设置日志级别
func SetLevel(level string) error {
	lvl, err := logrus.ParseLevel(level)
	if err != nil {
		return fmt.Errorf("无效的日志级别 %q: %w", level, err)
	}
	std.SetLevel(lvl)
	return nil
}

// SetOutput 设置日志输出目标
func SetOutput(w *os.File) {
	std.SetOutput(w)
}
