package ttsruntime

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
)

// preloadCacheVersion 预加载缓存版本号。
// 修改参考音频预处理/编码逻辑（如采样精度、时长截断、编码帧数上限等）后递增此版本号，
// 使旧的磁盘 gob 缓存自动失效，避免复用旧编码结果导致克隆异常。
const preloadCacheVersion = 4

// PreloadData 预加载的音频数据
type PreloadData struct {
	Version        int       // 缓存版本号，不匹配时视为失效
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
		log.Warnf("[PreloadCache] 创建缓存目录失败: %v", err)
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
		log.Debugf("[PreloadCache] 从内存缓存获取: %s (frames=%d)", id, len(data.AudioCodes))
		return data, nil
	}

	// 检查磁盘缓存
	cacheFile := pc.getCacheFilePath(id)
	if _, err := os.Stat(cacheFile); err == nil {
		data, err := pc.loadFromDisk(id)
		if err != nil || data == nil || data.Version != preloadCacheVersion {
			if err == nil && data != nil {
				log.Infof("[PreloadCache] 磁盘缓存版本不匹配，忽略: %s (got=%d want=%d)", id, data.Version, preloadCacheVersion)
			}
			// 旧版本缓存不可用，删除并视为未命中
			os.Remove(cacheFile)
			return nil, fmt.Errorf("preload data version mismatch: %s", id)
		}
		// 加入内存缓存
		pc.addToMemoryCache(id, data)
		log.Debugf("[PreloadCache] 从磁盘缓存加载: %s (frames=%d)", id, len(data.AudioCodes))
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
		log.Debugf("[PreloadCache] 已缓存: %s", id)
		return nil
	}

	// 检查磁盘缓存
	cacheFile := pc.getCacheFilePath(id)
	if _, err := os.Stat(cacheFile); err == nil {
		if existing, err := pc.loadFromDisk(id); err == nil && existing != nil && existing.Version == preloadCacheVersion {
			log.Debugf("[PreloadCache] 磁盘缓存已存在: %s", id)
			return nil
		}
		// 版本不匹配或加载失败：删除旧缓存，重新编码
		log.Infof("[PreloadCache] 磁盘缓存版本不匹配，重新预加载: %s", id)
		os.Remove(cacheFile)
	}

	// 编码音频
	log.Debugf("[PreloadCache] 预加载音频: %s -> %s", id, audioPath)
	audioCodes := pc.rt.EncodeReferenceAudioWithOptions(audioPath, true)
	if audioCodes == nil {
		return fmt.Errorf("音频编码失败: %s", audioPath)
	}

	// 创建预加载数据
	data := &PreloadData{
		Version:        preloadCacheVersion,
		ID:             id,
		AudioCodes:     audioCodes,
		Text:           text,
		AudioPath:      audioPath,
		LoadTime:       time.Now(),
		LastAccessTime: time.Now(),
	}

	// 保存到磁盘
	if err := pc.saveToDisk(id, data); err != nil {
		log.Warnf("[PreloadCache] 保存到磁盘失败: %v", err)
	}

	// 加入内存缓存
	pc.addToMemoryCache(id, data)

	log.Debugf("[PreloadCache] 预加载完成: %s (frames=%d)", id, len(audioCodes))
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
			log.Debugf("[PreloadCache] LRU淘汰: %s", evictID)
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
		log.Warnf("[PreloadCache] 清空磁盘缓存失败: %v", err)
	}
	if err := os.MkdirAll(pc.cacheDir, 0755); err != nil {
		log.Warnf("[PreloadCache] 重新创建缓存目录失败: %v", err)
	}

	log.Debugf("[PreloadCache] 缓存已清空")
}

// Remove 移除指定 id 的缓存条目（内存 + 磁盘）
func (pc *PreloadCache) Remove(id string) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// 从内存缓存移除
	if _, ok := pc.cache[id]; ok {
		delete(pc.cache, id)
		for i, accessID := range pc.accessOrder {
			if accessID == id {
				pc.accessOrder = append(pc.accessOrder[:i], pc.accessOrder[i+1:]...)
				break
			}
		}
	}

	// 从磁盘缓存移除
	cacheFile := pc.getCacheFilePath(id)
	if err := os.Remove(cacheFile); err != nil && !os.IsNotExist(err) {
		log.Warnf("[PreloadCache] 移除磁盘缓存失败: %s: %v", id, err)
	}

	log.Debugf("[PreloadCache] 已移除缓存条目: %s", id)
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