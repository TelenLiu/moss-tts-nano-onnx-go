//go:build darwin

package ttsruntime

import (
	"os"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
	"golang.org/x/sys/unix"
)

// tryLinkWorkerExe 在 macOS 上优先使用 APFS 写时复制克隆（unix.Clonefile）创建 worker 可执行文件。
// 克隆文件拥有独立的 inode（活动监视器显示独立进程名，不会与主进程混淆），
// 且数据块与源文件共享物理存储（零额外磁盘占用，仅写时复制）。
// 注意：这里绝不回退到硬链接——硬链接在 macOS 上会导致主进程名被随机显示为某个 worker 名。
// 返回 true 表示已创建好目标文件。
func tryLinkWorkerExe(exePath, target string) bool {
	// 克隆副本无法可靠校验是否与当前二进制一致，直接删除旧文件重建（clonefile 为 O(1) 元数据操作）
	os.Remove(target)

	if err := unix.Clonefile(exePath, target, 0); err != nil {
		log.Debugf("[WorkerProcess] clonefile 失败（可能非 APFS/跨卷），回退文件复制: %v", err)
		return false
	}
	log.Printf("[WorkerProcess] 创建 worker APFS 克隆: %s", target)
	return true
}
