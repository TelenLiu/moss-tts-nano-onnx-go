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
	"time"
)

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

	var buf bytes.Buffer
	writeLE := func(data interface{}) {
		binary.Write(&buf, binary.LittleEndian, data)
	}

	writeLE([]byte("RIFF"))
	writeLE(uint32(36 + dataSize))
	writeLE([]byte("WAVE"))
	writeLE([]byte("fmt "))
	writeLE(uint32(16))
	writeLE(uint16(1))
	writeLE(uint16(channels))
	writeLE(uint32(sampleRate))
	writeLE(uint32(sampleRate * channels * 2))
	writeLE(uint16(channels * 2))
	writeLE(uint16(16))
	writeLE([]byte("data"))
	writeLE(uint32(dataSize))

	samples := waveform
	if channels > 1 && len(waveform)%channels != 0 {
		samples = waveform[:numSamples*channels]
	}

	normSamples := normalizeVolume(samples)

	pcmBuf := make([]byte, len(normSamples)*2)
	for i, s := range normSamples {
		clamped := math.Max(-1.0, math.Min(1.0, float64(s)))
		pcm16 := int16(math.Round(clamped * 32767.0))
		pcmBuf[i*2] = byte(pcm16)
		pcmBuf[i*2+1] = byte(pcm16 >> 8)
	}
	buf.Write(pcmBuf)
	return buf.Bytes(), nil
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
	if _, err := exec.LookPath("ffmpeg"); err == nil {
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

	cmd := exec.CommandContext(ctx, "ffmpeg", "-y",
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

func LoadReferenceAudio(path string, targetSampleRate, targetChannels int) ([]float32, int, int, error) {
	waveform, channels, sampleRate, err := ReadWAV(path)
	if err != nil {
		return nil, 0, 0, err
	}

	if sampleRate != targetSampleRate {
		waveform = Resample(waveform, sampleRate, targetSampleRate, channels)
		sampleRate = targetSampleRate
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
