package audio

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
	"unsafe"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
)

// MP3EncodeConfig MP3 编码配置
type MP3EncodeConfig struct {
	SampleRate int     // 目标采样率，默认 44100
	VBRQuality float64 // VBR 质量 (0-9, 0最高 9最低)，默认 7 (较低质量节省空间)
}

// DefaultMP3EncodeConfig 默认 MP3 编码配置
var DefaultMP3EncodeConfig = MP3EncodeConfig{
	SampleRate: 44100,
	VBRQuality: 7,
}

// mp3EncoderCache 缓存探测到的可用 MP3 编码器名称
var (
	mp3EncoderCache     string
	mp3EncoderCacheOnce sync.Once
	ffmpegPathCache     string
	ffmpegPathOnce      sync.Once
)

// GetFFmpegPath 获取可用的 ffmpeg 路径：优先本地 lib/ffmpeg，然后系统 PATH
func GetFFmpegPath() string {
	ffmpegPathOnce.Do(func() {
		// 1. 优先使用本地 lib/ffmpeg 下的 ffmpeg（含 libmp3lame）
		// 尝试从可执行文件目录和工作目录查找
		exePath, _ := os.Executable()
		exeDir := filepath.Dir(exePath)
		cwd, _ := os.Getwd()

		searchDirs := []string{
			filepath.Join(cwd, "lib"),
			filepath.Join(exeDir, "lib"),
		}
		for _, libDir := range searchDirs {
			localFFmpeg := deps.FindLocalFFmpeg(libDir)
			if localFFmpeg != "" {
				ffmpegPathCache = localFFmpeg
				log.Printf("[Audio] 使用本地 FFmpeg: %s", localFFmpeg)
				return
			}
		}
		// 2. 使用系统 PATH 中的 ffmpeg
		if path, err := exec.LookPath("ffmpeg"); err == nil {
			ffmpegPathCache = path
			log.Printf("[Audio] 使用系统 FFmpeg: %s", path)
			return
		}
		ffmpegPathCache = ""
		log.Printf("[Audio] 未找到 FFmpeg，MP3 编码不可用")
	})
	return ffmpegPathCache
}

// detectMP3Encoder 探测 ffmpeg 可用的 MP3 编码器
func detectMP3Encoder(ffmpegBin string) string {
	mp3EncoderCacheOnce.Do(func() {
		if ffmpegBin == "" {
			mp3EncoderCache = ""
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		cmd := exec.CommandContext(ctx, ffmpegBin, "-hide_banner", "-encoders")
		var out bytes.Buffer
		cmd.Stdout = &out
		cmd.Run()
		cancel()

		output := out.Bytes()
		// 依次尝试 libmp3lame, mp3
		for _, enc := range []string{"libmp3lame", "mp3"} {
			if bytes.Contains(output, []byte(enc)) {
				mp3EncoderCache = enc
				log.Printf("[Audio] 检测到 FFmpeg MP3 编码器: %s", enc)
				return
			}
		}
		mp3EncoderCache = ""
		log.Printf("[Audio] FFmpeg 未找到 MP3 编码器")
	})
	return mp3EncoderCache
}

// EncodeMP3 将 float32 波形编码为 MP3 格式，使用 ffmpeg 子进程
// 直接 pipe float32 PCM 到 ffmpeg，跳过中间 WAV 编码
// 如果 ffmpeg 不可用或无 MP3 编码器，回退到 WAV 格式
func EncodeMP3(waveform []float32, channels, sampleRate int, cfg MP3EncodeConfig) ([]byte, error) {
	ffmpegBin := GetFFmpegPath()
	encoder := detectMP3Encoder(ffmpegBin)
	if ffmpegBin == "" || encoder == "" {
		log.Printf("[EncodeMP3] FFmpeg 或 MP3 编码器不可用，回退到 WAV 格式")
		return EncodeWAV(waveform, channels, sampleRate)
	}

	// 直接 pipe float32 PCM 到 ffmpeg，无需中间 WAV 编码
	args := []string{
		"-y",
		"-v", "fatal",
		"-f", "f32le",
		"-ac", fmt.Sprintf("%d", channels),
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-i", "pipe:0",
		"-c:a", encoder,
		"-ar", fmt.Sprintf("%d", cfg.SampleRate),
		"-q:a", fmt.Sprintf("%.0f", cfg.VBRQuality),
		"-f", "mp3",
		"pipe:1",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, ffmpegBin, args...)

	// 将 float32 波形直接作为 PCM bytes 传给 ffmpeg stdin
	pcmData := unsafe.Slice((*byte)(unsafe.Pointer(&waveform[0])), len(waveform)*4)
	cmd.Stdin = bytes.NewReader(pcmData)

	var outBuf bytes.Buffer
	var stderrBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &stderrBuf

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg MP3 编码失败: %w (stderr: %s)", err, stderrBuf.String())
	}

	return outBuf.Bytes(), nil
}

// WriteMP3 将 float32 波形编码为 MP3 文件
func WriteMP3(path string, waveform []float32, channels, sampleRate int, cfg MP3EncodeConfig) error {
	data, err := EncodeMP3(waveform, channels, sampleRate, cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// IsFFmpegAvailable 检查 ffmpeg 是否可用
func IsFFmpegAvailable() bool {
	_, err := exec.LookPath("ffmpeg")
	return err == nil
}

func WriteWAV(path string, waveform []float32, channels, sampleRate int) error {
	data, err := EncodeWAV(waveform, channels, sampleRate)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func EncodeWAV(waveform []float32, channels, sampleRate int) ([]byte, error) {
	numSamples := len(waveform) / channels
	if numSamples == 0 {
		numSamples = len(waveform)
		channels = 1
	}
	dataSize := numSamples * channels * 2

	// normalizeVolume: 防止削波，保证 WAV 格式合规
	// 仅在 maxAbs > 1.0 时归一化，避免不必要的内存分配
	maxAbs := float32(0)
	for _, s := range waveform {
		if s < 0 {
			s = -s
		}
		if s > maxAbs {
			maxAbs = s
		}
	}
	normFactor := float32(1.0)
	if maxAbs > 1.0 {
		normFactor = float32(0.95) / maxAbs
	}

	buf := make([]byte, 44+dataSize)
	// RIFF header
	copy(buf[0:4], []byte("RIFF"))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(36+dataSize))
	copy(buf[8:12], []byte("WAVE"))
	copy(buf[12:16], []byte("fmt "))
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], 1)
	binary.LittleEndian.PutUint16(buf[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:32], uint32(sampleRate*channels*2))
	binary.LittleEndian.PutUint16(buf[32:34], uint16(channels*2))
	binary.LittleEndian.PutUint16(buf[34:36], uint16(16))
	copy(buf[36:40], []byte("data"))
	binary.LittleEndian.PutUint32(buf[40:44], uint32(dataSize))

	// PCM 数据编码，直接操作 []byte 避免反射开销
	for i, s := range waveform {
		s *= normFactor
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		val := int16(s * 32767)
		binary.LittleEndian.PutUint16(buf[44+i*2:], uint16(val))
	}

	return buf, nil
}

func normalizeVolume(waveform []float32) []float32 {
	if len(waveform) == 0 {
		return waveform
	}
	var maxAbs float32 = 0
	for _, s := range waveform {
		abs := float32(math.Abs(float64(s)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs < 1e-6 {
		return waveform
	}
	result := make([]float32, len(waveform))
	normFactor := float32(0.95) / maxAbs
	for i, s := range waveform {
		result[i] = s * normFactor
	}
	return result
}

func ReadWAV(path string) ([]float32, int, int, error) {
	return ReadWAVMaxDuration(path, 30)
}

func ReadWAVMaxDuration(path string, maxDurationSecs int) ([]float32, int, int, error) {
	if GetFFmpegPath() != "" {
		return readWithFFmpeg(path)
	}
	return readRIFFWAV(path)
}

func readWithFFmpeg(path string) ([]float32, int, int, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("无法获取绝对路径: %w", err)
	}

	resolvedPath := absPath
	if info, err := os.Stat(absPath); err == nil && info.Mode()&os.ModeSymlink != 0 {
		if rl, err := filepath.EvalSymlinks(absPath); err == nil {
			resolvedPath = rl
		}
	}

	tmpDir := filepath.Dir(resolvedPath)
	tmpFile := filepath.Join(tmpDir, fmt.Sprintf(".tmp_convert_%d.wav", time.Now().UnixNano()))
	defer os.Remove(tmpFile)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, GetFFmpegPath(), "-y",
		"-v", "fatal",
		"-i", resolvedPath,
		"-t", "30",
		"-f", "wav",
		"-acodec", "pcm_s16le",
		tmpFile,
	)

	if err := cmd.Run(); err != nil {
		return nil, 0, 0, fmt.Errorf("ffmpeg转码失败: %w", err)
	}

	return readRIFFWAV(tmpFile)
}

func readRIFFWAV(path string) ([]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	header := make([]byte, 12)
	if _, err := f.Read(header); err != nil {
		return nil, 0, 0, err
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, 0, 0, fmt.Errorf("ffmpeg输出不是有效的WAV")
	}

	var channels, sampleRate, dataSize int
	for {
		chunkHeader := make([]byte, 8)
		_, err := f.Read(chunkHeader)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("读取chunk头失败: %w", err)
		}
		chunkID := string(chunkHeader[0:4])
		chunkSize := int(binary.LittleEndian.Uint32(chunkHeader[4:8]))

		if chunkID == "fmt " {
			fmtData := make([]byte, 16)
			if _, err := f.Read(fmtData); err != nil {
				return nil, 0, 0, err
			}
			channels = int(binary.LittleEndian.Uint16(fmtData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(fmtData[4:8]))
			remaining := chunkSize - 16
			if remaining > 0 {
				f.Seek(int64(remaining), 1)
			}
		} else if chunkID == "data" {
			dataSize = chunkSize
			break
		} else {
			pad := chunkSize % 2
			f.Seek(int64(chunkSize+pad), 1)
		}
	}

	data := make([]byte, dataSize)
	if _, err := f.Read(data); err != nil {
		return nil, 0, 0, err
	}

	numSamples := dataSize / 2
	waveform := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		pcm16 := int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
		waveform[i] = float32(pcm16) / 32768.0
	}

	return waveform, channels, sampleRate, nil
}

func Resample(waveform []float32, origRate, targetRate, channels int) []float32 {
	if origRate == targetRate {
		return waveform
	}
	ratio := float64(targetRate) / float64(origRate)
	totalSamples := len(waveform)
	origFrames := totalSamples / channels
	targetFrames := int(float64(origFrames) * ratio)
	result := make([]float32, targetFrames*channels)
	for i := 0; i < targetFrames; i++ {
		srcFrame := float64(i) / ratio
		srcFrame0 := int(srcFrame)
		if srcFrame0 >= origFrames-1 {
			srcFrame0 = origFrames - 2
		}
		if srcFrame0 < 0 {
			srcFrame0 = 0
		}
		srcFrame1 := srcFrame0 + 1
		t := srcFrame - float64(srcFrame0)
		for ch := 0; ch < channels; ch++ {
			v0 := waveform[srcFrame0*channels+ch]
			v1 := waveform[srcFrame1*channels+ch]
			result[i*channels+ch] = float32((1-t)*float64(v0) + t*float64(v1))
		}
	}
	return result
}

func MergeAudioChannels(channelArrays [][]float32) []float32 {
	if len(channelArrays) == 0 {
		return nil
	}
	if len(channelArrays) == 1 {
		return channelArrays[0]
	}
	minLen := len(channelArrays[0])
	for _, ch := range channelArrays[1:] {
		if len(ch) < minLen {
			minLen = len(ch)
		}
	}
	// 合并为交错格式(interleaved): [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
	result := make([]float32, minLen*len(channelArrays))
	for chIdx, ch := range channelArrays {
		for i := 0; i < minLen; i++ {
			result[i*len(channelArrays)+chIdx] = ch[i]
		}
	}
	return result
}

func ConcatWaveforms(waveforms [][]float32) []float32 {
	if len(waveforms) == 0 {
		return nil
	}
	var totalLen int
	for _, w := range waveforms {
		totalLen += len(w)
	}
	result := make([]float32, 0, totalLen)
	for _, w := range waveforms {
		result = append(result, w...)
	}
	return result
}

func MakeSilence(durationSamples, channels int) []float32 {
	return make([]float32, durationSamples*channels)
}

// MaxReferenceAudioDurationSec 参考音频最大时长（秒），超过此时长将被截断以避免 OOM
const MaxReferenceAudioDurationSec = 15

func LoadReferenceAudio(path string, targetSampleRate, targetChannels int) ([]float32, int, int, error) {
	waveform, channels, sampleRate, err := ReadWAV(path)
	if err != nil {
		return nil, 0, 0, err
	}

	if sampleRate != targetSampleRate {
		waveform = Resample(waveform, sampleRate, targetSampleRate, channels)
		sampleRate = targetSampleRate
	}

	// 截断过长的参考音频，避免编码后帧数过多导致 OOM
	maxSamples := MaxReferenceAudioDurationSec * sampleRate * channels
	if len(waveform) > maxSamples {
		log.Printf("[LoadReferenceAudio] 参考音频过长(%d样本, 约%.1f秒)，截断至%d秒",
			len(waveform), float64(len(waveform))/float64(sampleRate*channels), float64(MaxReferenceAudioDurationSec))
		waveform = waveform[:maxSamples]
	}

	if channels == targetChannels {
		return waveform, channels, sampleRate, nil
	}

	if channels == 1 && targetChannels > 1 {
		result := make([]float32, len(waveform)*targetChannels)
		for i := 0; i < len(waveform); i++ {
			for ch := 0; ch < targetChannels; ch++ {
				result[i*targetChannels+ch] = waveform[i]
			}
		}
		return result, targetChannels, sampleRate, nil
	}

	if channels > 1 && targetChannels == 1 {
		frames := len(waveform) / channels
		result := make([]float32, frames)
		for i := 0; i < frames; i++ {
			var sum float32
			for ch := 0; ch < channels; ch++ {
				sum += waveform[i*channels+ch]
			}
			result[i] = sum / float32(channels)
		}
		return result, 1, sampleRate, nil
	}

	return nil, 0, 0, fmt.Errorf("不支持的音频通道转换: %d -> %d", channels, targetChannels)
}
