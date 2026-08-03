//go:build !darwin && !linux && !freebsd && !netbsd && !openbsd && !dragonfly && !solaris && !aix

package ttsruntime

import "os"

// tryLinkWorkerExe 在其余平台（如 Windows）上不尝试链接/克隆，直接由调用方回退到文件复制。
func tryLinkWorkerExe(exePath, target string) bool {
	// 目标已存在则删除（可能是旧版本残留）
	os.Remove(target)
	return false
}
