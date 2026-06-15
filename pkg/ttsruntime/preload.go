package ttsruntime

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// PreloadData 预加载的音频数据
type PreloadData struct {
	ID             string    // 预加载ID
	AudioCodes     [][]int   // 音频编码数据
	Text           string    // 示例文本
	AudioPath      string    // 音频文件路径
	LoadTime       time.Time // 加载时间
	LastAccessTime time.Time // 最后访问时间
}

// PreloadCache 预加载缓存管理器
type PreloadCache struct {
	mu            sync.RWMutex
	cache         map[string]*PreloadData // 内存缓存
	maxInMemory   int                     // 内存中最多保留的数量
	cacheDir      string                  // 缓存目录
	rt            *OnnxTtsRuntime         // TTS运行时
	accessOrder   []string                // 访问顺序（用于LRU淘汰）
}

// NewPreloadCache 创建预加载缓存管理器
func NewPreloadCache(cacheDir string, maxInMemory int, rt *OnnxTtsRuntime) *PreloadCache {
	if maxInMemory <= 0 {
		maxInMemory = 2 // 默认保留2个
	}
	pc := &PreloadCache{
		cache:       make(map[string]*PreloadData),
		maxInMemory: maxInMemory,
		cacheDir:    cacheDir,
		rt:          rt,
		accessOrder: make([]string, 0),
	}
	// 创建缓存目录
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		log.Printf("[PreloadCache] 创建缓存目录失败: %v", err)
	}
	return pc
}

// Get 获取预加载数据
func (pc *PreloadCache) Get(id string) (*PreloadData, error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// 检查内存缓存
	if data, ok := pc.cache[id]; ok {
		data.LastAccessTime = time.Now()
		pc.updateAccessOrder(id)
		log.Printf("[PreloadCache] 从内存缓存获取: %s (frames=%d)", id, len(data.AudioCodes))
		return data, nil
	}

	// 检查磁盘缓存
	cacheFile := pc.getCacheFilePath(id)
	if _, err := os.Stat(cacheFile); err == nil {
		data, err := pc.loadFromDisk(id)
		if err != nil {
			log.Printf("[PreloadCache] 从磁盘加载失败: %v", err)
			return nil, err
		}
		// 加入内存缓存
		pc.addToMemoryCache(id, data)
		log.Printf("[PreloadCache] 从磁盘缓存加载: %s (frames=%d)", id, len(data.AudioCodes))
		return data, nil
	}

	return nil, fmt.Errorf("preload data not found: %s", id)
}

// Preload 预加载音频数据
func (pc *PreloadCache) Preload(id string, audioPath string, text string) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// 检查是否已缓存
	if _, ok := pc.cache[id]; ok {
		log.Printf("[PreloadCache] 已缓存: %s", id)
		return nil
	}

	// 检查磁盘缓存
	cacheFile := pc.getCacheFilePath(id)
	if _, err := os.Stat(cacheFile); err == nil {
		log.Printf("[PreloadCache] 磁盘缓存已存在: %s", id)
		return nil
	}

	// 编码音频
	log.Printf("[PreloadCache] 预加载音频: %s -> %s", id, audioPath)
	audioCodes := pc.rt.EncodeReferenceAudioWithOptions(audioPath, true)
	if audioCodes == nil {
		return fmt.Errorf("音频编码失败: %s", audioPath)
	}

	// 创建预加载数据
	data := &PreloadData{
		ID:             id,
		AudioCodes:     audioCodes,
		Text:           text,
		AudioPath:      audioPath,
		LoadTime:       time.Now(),
		LastAccessTime: time.Now(),
	}

	// 保存到磁盘
	if err := pc.saveToDisk(id, data); err != nil {
		log.Printf("[PreloadCache] 保存到磁盘失败: %v", err)
	}

	// 加入内存缓存
	pc.addToMemoryCache(id, data)

	log.Printf("[PreloadCache] 预加载完成: %s (frames=%d)", id, len(audioCodes))
	return nil
}

// addToMemoryCache 加入内存缓存（带LRU淘汰）
func (pc *PreloadCache) addToMemoryCache(id string, data *PreloadData) {
	// 如果已存在，更新数据
	if _, ok := pc.cache[id]; ok {
		pc.cache[id] = data
		pc.updateAccessOrder(id)
		return
	}

	// 如果超过最大数量，淘汰最旧的
	if len(pc.cache) >= pc.maxInMemory {
		// 淘汰最旧的（第一个）
		if len(pc.accessOrder) > 0 {
			evictID := pc.accessOrder[0]
			log.Printf("[PreloadCache] LRU淘汰: %s", evictID)
			delete(pc.cache, evictID)
			pc.accessOrder = pc.accessOrder[1:]
		}
	}

	// 加入缓存
	pc.cache[id] = data
	pc.accessOrder = append(pc.accessOrder, id)
}

// updateAccessOrder 更新访问顺序（LRU）
func (pc *PreloadCache) updateAccessOrder(id string) {
	// 移除旧的记录
	for i, accessID := range pc.accessOrder {
		if accessID == id {
			pc.accessOrder = append(pc.accessOrder[:i], pc.accessOrder[i+1:]...)
			break
		}
	}
	// 添加到末尾（最新访问）
	pc.accessOrder = append(pc.accessOrder, id)
}

// saveToDisk 保存到磁盘缓存
func (pc *PreloadCache) saveToDisk(id string, data *PreloadData) error {
	cacheFile := pc.getCacheFilePath(id)
	file, err := os.Create(cacheFile)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(data)
}

// loadFromDisk 从磁盘缓存加载
func (pc *PreloadCache) loadFromDisk(id string) (*PreloadData, error) {
	cacheFile := pc.getCacheFilePath(id)
	file, err := os.Open(cacheFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data PreloadData
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, err
	}

	data.LastAccessTime = time.Now()
	return &data, nil
}

// getCacheFilePath 获取缓存文件路径
func (pc *PreloadCache) getCacheFilePath(id string) string {
	return filepath.Join(pc.cacheDir, fmt.Sprintf("%s.gob", id))
}

// Clear 清空缓存
func (pc *PreloadCache) Clear() {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	pc.cache = make(map[string]*PreloadData)
	pc.accessOrder = make([]string, 0)

	// 清空磁盘缓存
	if err := os.RemoveAll(pc.cacheDir); err != nil {
		log.Printf("[PreloadCache] 清空磁盘缓存失败: %v", err)
	}
	if err := os.MkdirAll(pc.cacheDir, 0755); err != nil {
		log.Printf("[PreloadCache] 重新创建缓存目录失败: %v", err)
	}

	log.Printf("[PreloadCache] 缓存已清空")
}

// Stats 获取缓存统计信息
func (pc *PreloadCache) Stats() map[string]interface{} {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	return map[string]interface{}{
		"memory_count":   len(pc.cache),
		"max_in_memory":  pc.maxInMemory,
		"access_order":   pc.accessOrder,
		"cache_dir":      pc.cacheDir,
	}
}