//go:build linux || freebsd || netbsd || openbsd || dragonfly || solaris || aix

package ttsruntime

import (
	"os"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
)

// tryLinkWorkerExe 在 Linux 等平台上使用硬链接（os.Link）创建 worker 可执行文件。
// Linux 的 ps/htop 基于 argv[0] 显示进程名，不会出现 macOS 上基于 inode 的名称漂移；
// 硬链接共享 inode，不占用额外磁盘空间。
// 返回 true 表示已创建好目标文件。
func tryLinkWorkerExe(exePath, target string) bool {
	// 目标已存在且与源文件为同一 inode（即指向当前二进制），直接复用
	if fi2, err := os.Lstat(target); err == nil {
		if fi2.Mode().IsRegular() {
			if fi1, err1 := os.Stat(exePath); err1 == nil && os.SameFile(fi1, fi2) {
				log.Debugf("[WorkerProcess] 复用已有的 worker 硬链接: %s", target)
				return true
			}
		}
		// 文件存在但不匹配（含旧的符号链接），删除重建
		os.Remove(target)
	}

	err := os.Link(exePath, target)
	if err == nil {
		log.Printf("[WorkerProcess] 创建 worker 硬链接: %s -> %s", target, exePath)
		return true
	}
	log.Debugf("[WorkerProcess] 硬链接失败（可能跨文件系统），回退文件复制: %v", err)
	return false
}
