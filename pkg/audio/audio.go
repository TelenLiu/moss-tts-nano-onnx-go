package audio

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
	"unsafe"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
	resample "github.com/gunter-q12/resample"
)

// MP3EncodeConfig MP3 编码配置
type MP3EncodeConfig struct {
	SampleRate int     // 目标采样率，默认 44100
	VBRQuality float64 // VBR 质量 (0-9, 0最高 9最低)，默认 7 (较低质量节省空间)
	Volume     float64 // 音量倍数 (1.0=原始音量，>1放大，<1减小)，默认 1.0。<=0 或未设置时按 1.0 处理
}

// DefaultMP3EncodeConfig 默认 MP3 编码配置
var DefaultMP3EncodeConfig = MP3EncodeConfig{
	SampleRate: 44100,
	VBRQuality: 7,
	Volume:     1.0,
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
				log.Debugf("[Audio] 使用本地 FFmpeg: %s", localFFmpeg)
				return
			}
		}
		// 2. 使用系统 PATH 中的 ffmpeg
		if path, err := exec.LookPath("ffmpeg"); err == nil {
			ffmpegPathCache = path
			log.Debugf("[Audio] 使用系统 FFmpeg: %s", path)
			return
		}
		ffmpegPathCache = ""
		log.Debugf("[Audio] 未找到 FFmpeg，MP3 编码不可用")
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
				log.Debugf("[Audio] 检测到 FFmpeg MP3 编码器: %s", enc)
				return
			}
		}
		mp3EncoderCache = ""
		log.Debugf("[Audio] FFmpeg 未找到 MP3 编码器")
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
		log.Debugf("[EncodeMP3] FFmpeg 或 MP3 编码器不可用，回退到 WAV 格式")
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
	}
	// 音量调节：仅当 Volume 有效且不等于 1.0 时添加 volume 滤镜
	// ffmpeg volume 滤镜会做软限幅，不会硬削波爆音
	vol := cfg.Volume
	if vol <= 0 {
		vol = 1.0
	}
	if vol != 1.0 {
		args = append(args, "-af", fmt.Sprintf("volume=%.6g", vol))
	}
	args = append(args,
		"-c:a", encoder,
		"-ar", fmt.Sprintf("%d", cfg.SampleRate),
		"-q:a", fmt.Sprintf("%.0f", cfg.VBRQuality),
		"-f", "mp3",
		"pipe:1",
	)

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
	// 与 Python 一致：使用硬削波（clip）+ 四舍五入，不做整体归一化
	for i, s := range waveform {
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		val := int16(math.Round(float64(s) * 32767.0))
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

// loadWithFFmpegResampled 使用 ffmpeg 加载音频并完成重采样和通道转换
// 优先使用 soxr 重采样器（质量接近 torchaudio），不可用时回退到 ffmpeg 默认 swr 重采样器
func loadWithFFmpegResampled(path string, targetSampleRate, targetChannels int) ([]float32, int, int, error) {
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

	ffmpegPath := GetFFmpegPath()
	if ffmpegPath == "" {
		return nil, 0, 0, fmt.Errorf("ffmpeg 不可用")
	}

	tmpDir := filepath.Dir(resolvedPath)
	tmpFile := filepath.Join(tmpDir, fmt.Sprintf(".tmp_resample_%d.wav", time.Now().UnixNano()))
	defer os.Remove(tmpFile)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 尝试使用 ffmpeg 的 soxr 重采样器（质量接近 torchaudio）
	// 注意：soxr 引擎需要 ffmpeg 编译时启用 libsoxr 支持，并非所有 ffmpeg 都可用
	soxrCmd := exec.CommandContext(ctx, ffmpegPath, "-y",
		"-v", "fatal",
		"-i", resolvedPath,
		"-t", fmt.Sprintf("%d", MaxReferenceAudioDurationSec),
		"-af", "aresample=resampler=soxr:precision=28",
		"-ar", fmt.Sprintf("%d", targetSampleRate),
		"-ac", fmt.Sprintf("%d", targetChannels),
		"-f", "wav",
		"-acodec", "pcm_s16le",
		tmpFile,
	)

	if err := soxrCmd.Run(); err == nil {
		waveform, channels, sampleRate, err := readRIFFWAV(tmpFile)
		if err == nil {
			// 不裁剪ffmpeg滤波器延迟静音：裁剪会导致codec_encode编码结果与Python不一致
			return waveform, channels, sampleRate, nil
		}
	}

	// soxr 不可用，回退到 ffmpeg 默认的 swr 重采样器
	// swr 质量仍远好于 Go 端的线性插值
	os.Remove(tmpFile)
	swrCmd := exec.CommandContext(ctx, ffmpegPath, "-y",
		"-v", "fatal",
		"-i", resolvedPath,
		"-t", fmt.Sprintf("%d", MaxReferenceAudioDurationSec),
		"-ar", fmt.Sprintf("%d", targetSampleRate),
		"-ac", fmt.Sprintf("%d", targetChannels),
		"-f", "wav",
		"-acodec", "pcm_s16le",
		tmpFile,
	)

	if err := swrCmd.Run(); err != nil {
		return nil, 0, 0, fmt.Errorf("ffmpeg重采样失败: %w", err)
	}

	waveform, channels, sampleRate, err := readRIFFWAV(tmpFile)
	if err != nil {
		return nil, 0, 0, err
	}
	// 不裁剪ffmpeg滤波器延迟静音：裁剪会导致codec_encode编码结果与Python不一致
	return waveform, channels, sampleRate, nil
}

// trimLeadingSilence 裁剪波形开头的静音段
// ffmpeg 重采样会在开头插入滤波器延迟（约100-200ms静音），需要裁剪掉
// 否则 codec_encode 会将静音段编码为静音帧，干扰音色克隆
func trimLeadingSilence(waveform []float32, channels int) []float32 {
	if channels <= 0 {
		return waveform
	}
	// 查找第一个非静音帧的位置
	// 使用帧级别检测（每帧 = channels 个样本）
	silenceThreshold := float32(1e-4) // 非常低的阈值，只裁剪真正的静音
	firstNonSilentFrame := 0
	numFrames := len(waveform) / channels
	for i := 0; i < numFrames; i++ {
		frameMax := float32(0)
		for ch := 0; ch < channels; ch++ {
			absVal := waveform[i*channels+ch]
			if absVal < 0 {
				absVal = -absVal
			}
			if absVal > frameMax {
				frameMax = absVal
			}
		}
		if frameMax > silenceThreshold {
			firstNonSilentFrame = i
			break
		}
		if i == numFrames-1 {
			// 全部静音，返回原始波形
			return waveform
		}
	}

	if firstNonSilentFrame == 0 {
		return waveform
	}

	// 保留一点前置静音（约5ms），避免裁剪掉语音起始的过渡段
	// 假设采样率约48000，5ms ≈ 240个样本 ≈ 240/channels 个帧
	padFrames := 240 / channels
	if padFrames < 1 {
		padFrames = 1
	}
	startFrame := firstNonSilentFrame - padFrames
	if startFrame < 0 {
		startFrame = 0
	}

	trimmed := waveform[startFrame*channels:]
	if len(trimmed) < len(waveform)/2 {
		// 裁剪掉超过一半，可能有问题，返回原始波形
		return waveform
	}
	return trimmed
}

// MaxReferenceAudioDurationSec 参考音频最大时长（秒），超过此时长将被截断以避免 OOM
const MaxReferenceAudioDurationSec = 15

// loadWithHighQualityResample 使用 Go 内置高质量重采样加载参考音频
// 基于 bandlimited interpolation（Kaiser 窗 sinc 插值），与 Python torchaudio.functional.resample 算法一致
// 这确保 codec_encode 的输入波形与 Python 端一致，避免音色偏差
func loadWithHighQualityResample(path string, targetSampleRate, targetChannels int) ([]float32, int, int, error) {
	// 1. 先用 ffmpeg 解码为原始 PCM（不做重采样）
	ffmpegPath := GetFFmpegPath()
	if ffmpegPath == "" {
		return nil, 0, 0, fmt.Errorf("ffmpeg 不可用")
	}

	tmpDir := filepath.Dir(path)
	tmpFile := filepath.Join(tmpDir, fmt.Sprintf(".tmp_pcm_%d.wav", time.Now().UnixNano()))
	defer os.Remove(tmpFile)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 用 ffmpeg 解码为 WAV（保持原始采样率，仅做格式转换和通道转换）
	cmd := exec.CommandContext(ctx, ffmpegPath, "-y",
		"-v", "fatal",
		"-i", path,
		"-t", fmt.Sprintf("%d", MaxReferenceAudioDurationSec),
		"-ac", fmt.Sprintf("%d", targetChannels),
		"-f", "wav",
		"-acodec", "pcm_s16le",
		tmpFile,
	)
	if err := cmd.Run(); err != nil {
		return nil, 0, 0, fmt.Errorf("ffmpeg 解码失败: %w", err)
	}

	// 2. 读取 WAV 文件获取原始采样率和波形
	waveform, channels, sampleRate, err := readRIFFWAV(tmpFile)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("读取 WAV 失败: %w", err)
	}

	// 3. 如果采样率已经是目标采样率，直接返回
	if sampleRate == targetSampleRate {
		return waveform, channels, sampleRate, nil
	}

	// 4. 使用 bandlimited interpolation 进行高质量重采样
	// gunter-q12/resample 库基于与 torchaudio 相同的 sinc 插值算法
	// 使用 KaiserBest 滤波器获得最高质量
	resampled, err := resampleWaveform(waveform, channels, sampleRate, targetSampleRate)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("重采样失败: %w", err)
	}

	return resampled, channels, targetSampleRate, nil
}

// resampleWaveform 使用 bandlimited interpolation 重采样
func resampleWaveform(waveform []float32, channels, inRate, outRate int) ([]float32, error) {
	if inRate == outRate {
		return waveform, nil
	}

	// 将交错格式的 float32 样本转换为 bytes
	inputBytes := make([]byte, len(waveform)*4)
	for i, v := range waveform {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(inputBytes[i*4:], bits)
	}

	// 创建输出缓冲区
	var outBuf bytes.Buffer

	// 创建重采样器：输出写入 outBuf
	r, err := resample.New(&outBuf, resample.FormatFloat32, inRate, outRate, channels, resample.WithKaiserBestFilter())
	if err != nil {
		return nil, fmt.Errorf("创建重采样器失败: %w", err)
	}

	// 执行重采样
	n, err := r.Write(inputBytes)
	if err != nil {
		return nil, fmt.Errorf("重采样写入失败: %w", err)
	}
	if n != len(inputBytes) {
		log.Warnf("[resampleWaveform] 警告: 写入 %d 字节，期望 %d 字节", n, len(inputBytes))
	}

	// 使用 ReadFrom 刷新重采样器内部缓冲区
	// 传入空的 reader 以确保所有数据都被处理
	if _, err := r.ReadFrom(bytes.NewReader([]byte{})); err != nil {
		log.Warnf("[resampleWaveform] 刷新重采样器失败: %v", err)
	}

	// 从 outBuf 读取重采样后的 bytes 并转换为 float32
	outputBytes := outBuf.Bytes()
	if len(outputBytes)%4 != 0 {
		return nil, fmt.Errorf("重采样输出字节数不是4的倍数: %d", len(outputBytes))
	}

	result := make([]float32, len(outputBytes)/4)
	for i := range result {
		bits := binary.LittleEndian.Uint32(outputBytes[i*4:])
		result[i] = math.Float32frombits(bits)
	}

	return result, nil
}

func LoadReferenceAudio(path string, targetSampleRate, targetChannels int) ([]float32, int, int, error) {
	// 优先使用 Go 内置的高质量重采样（基于 bandlimited interpolation，与 torchaudio 算法一致）
	// 这比 ffmpeg 的 swr 重采样器质量更高，能确保 codec_encode 结果与 Python 端一致
	waveform, channels, sampleRate, err := loadWithHighQualityResample(path, targetSampleRate, targetChannels)
	if err == nil {
		return waveform, channels, sampleRate, nil
	}

	// 高质量重采样失败，回退到 ffmpeg
	log.Debugf("[LoadReferenceAudio] 高质量重采样失败(%v)，回退到 ffmpeg", err)
	waveform, channels, sampleRate, err = loadWithFFmpegResampled(path, targetSampleRate, targetChannels)
	if err == nil {
		// 截断过长的参考音频
		maxSamples := MaxReferenceAudioDurationSec * targetSampleRate * targetChannels
		if len(waveform) > maxSamples {
			log.Debugf("[LoadReferenceAudio] 参考音频过长(%d样本, 约%.1f秒)，截断至%d秒",
				len(waveform), float64(len(waveform))/float64(targetSampleRate*targetChannels), float64(MaxReferenceAudioDurationSec))
			waveform = waveform[:maxSamples]
		}
		return waveform, targetChannels, targetSampleRate, nil
	}

	// ffmpeg 不可用时回退到原有逻辑
	log.Debugf("[LoadReferenceAudio] ffmpeg重采样失败(%v)，回退到内置重采样", err)
	waveform, channels, sampleRate, err = ReadWAV(path)
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
		log.Debugf("[LoadReferenceAudio] 参考音频过长(%d样本, 约%.1f秒)，截断至%d秒",
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
