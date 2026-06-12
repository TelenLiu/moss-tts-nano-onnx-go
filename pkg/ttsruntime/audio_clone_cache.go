package ttsruntime

import (
	"crypto/sha256"
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// AudioCloneCache 基于文件的音频克隆编码缓存
// 使用音频数据的 SHA-256 hash 作为 key，gob 序列化存储到 cache/audio_clone_gob/ 目录
// 支持跨进程共享，主进程定时清理过期文件
type AudioCloneCache struct {
	cacheDir string
	mu       sync.Mutex
}

// audioCloneGobEntry gob 序列化的缓存条目
type audioCloneGobEntry struct {
	CreatedAt time.Time
	Codes     [][]int
}

// NewAudioCloneCache 创建音频克隆缓存
func NewAudioCloneCache(cacheDir string) *AudioCloneCache {
	return &AudioCloneCache{cacheDir: cacheDir}
}

// HashAudioFile 计算音频文件的 SHA-256 hash
func (c *AudioCloneCache) HashAudioFile(audioPath string) (string, error) {
	data, err := os.ReadFile(audioPath)
	if err != nil {
		return "", fmt.Errorf("读取音频文件失败: %w", err)
	}
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:]), nil
}

// HashAudioData 计算音频二进制数据的 SHA-256 hash
func (c *AudioCloneCache) HashAudioData(data []byte) string {
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:])
}

// Get 从缓存中读取编码结果，不存在返回 nil
func (c *AudioCloneCache) Get(hashKey string) [][]int {
	c.mu.Lock()
	defer c.mu.Unlock()

	filePath := c.gobPath(hashKey)
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil
	}

	var entry audioCloneGobEntry
	if err := gobDecode(data, &entry); err != nil {
		log.Printf("[AudioCloneCache] 解码缓存文件失败 %s: %v，删除损坏文件", filePath, err)
		os.Remove(filePath)
		return nil
	}

	log.Printf("[AudioCloneCache] 命中文件缓存: %s (frames=%d, created=%s)", hashKey[:12], len(entry.Codes), entry.CreatedAt.Format("2006-01-02 15:04:05"))
	return entry.Codes
}

// Put 将编码结果写入缓存
func (c *AudioCloneCache) Put(hashKey string, codes [][]int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := os.MkdirAll(c.cacheDir, 0755); err != nil {
		return fmt.Errorf("创建缓存目录失败: %w", err)
	}

	entry := audioCloneGobEntry{
		CreatedAt: time.Now(),
		Codes:     codes,
	}

	data, err := gobEncode(&entry)
	if err != nil {
		return fmt.Errorf("编码缓存数据失败: %w", err)
	}

	filePath := c.gobPath(hashKey)
	tmpPath := filePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("写入缓存文件失败: %w", err)
	}
	// 原子重命名，避免并发读写损坏
	if err := os.Rename(tmpPath, filePath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("重命名缓存文件失败: %w", err)
	}

	log.Printf("[AudioCloneCache] 写入文件缓存: %s (frames=%d)", hashKey[:12], len(codes))
	return nil
}

// CleanExpired 清理创建时间超过 expHours 小时的缓存文件
func (c *AudioCloneCache) CleanExpired(expHours int) {
	if expHours <= 0 {
		expHours = 24
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	entries, err := os.ReadDir(c.cacheDir)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[AudioCloneCache] 读取缓存目录失败: %v", err)
		}
		return
	}

	threshold := time.Now().Add(-time.Duration(expHours) * time.Hour)
	cleaned := 0

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".gob" {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		// 使用文件修改时间作为判断依据（写入时就是创建时间）
		if info.ModTime().Before(threshold) {
			filePath := filepath.Join(c.cacheDir, entry.Name())
			if err := os.Remove(filePath); err != nil {
				log.Printf("[AudioCloneCache] 删除过期缓存失败 %s: %v", entry.Name(), err)
			} else {
				cleaned++
			}
		}
	}

	if cleaned > 0 {
		log.Printf("[AudioCloneCache] 清理过期缓存: 删除 %d 个文件 (阈值=%d小时)", cleaned, expHours)
	}
}

// StartCleanupLoop 启动定时清理 goroutine，每隔 interval 执行一次清理
func (c *AudioCloneCache) StartCleanupLoop(expHours int, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// 启动时先清理一次
		c.CleanExpired(expHours)

		for range ticker.C {
			c.CleanExpired(expHours)
		}
	}()
	log.Printf("[AudioCloneCache] 启动定时清理: 间隔=%v 过期阈值=%d小时", interval, expHours)
}

func (c *AudioCloneCache) gobPath(hashKey string) string {
	return filepath.Join(c.cacheDir, hashKey+".gob")
}

func gobEncode(v interface{}) ([]byte, error) {
	var buf []byte
	enc := gob.NewEncoder((*sliceWriter)(&buf))
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return buf, nil
}

func gobDecode(data []byte, v interface{}) error {
	dec := gob.NewDecoder((*sliceReader)(&data))
	return dec.Decode(v)
}

// sliceWriter / sliceReader 用于 gob 编解码的辅助类型
type sliceWriter []byte

func (w *sliceWriter) Write(p []byte) (int, error) {
	*w = append(*w, p...)
	return len(p), nil
}

type sliceReader []byte

func (r *sliceReader) Read(p []byte) (int, error) {
	n := copy(p, *r)
	*r = (*r)[n:]
	if n == 0 {
		return 0, fmt.Errorf("EOF")
	}
	return n, nil
}
