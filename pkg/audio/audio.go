package audio

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

func WriteWAV(path string, waveform []float32, channels, sampleRate int) error {
	numSamples := len(waveform) / channels
	if numSamples == 0 {
		numSamples = len(waveform)
		channels = 1
	}
	dataSize := numSamples * channels * 2

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	writeLE := func(data interface{}) {
		binary.Write(f, binary.LittleEndian, data)
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

	buf := make([]byte, 0, len(normSamples)*2)
	for _, s := range normSamples {
		clamped := math.Max(-1.0, math.Min(1.0, float64(s)))
		pcm16 := int16(math.Round(clamped * 32767.0))
		buf = append(buf, byte(pcm16), byte(pcm16>>8))
	}
	_, err = f.Write(buf)
	return err
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
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	header := make([]byte, 44)
	if _, err := f.Read(header); err != nil {
		return nil, 0, 0, err
	}

	channels := int(binary.LittleEndian.Uint16(header[22:24]))
	sampleRate := int(binary.LittleEndian.Uint32(header[24:28]))
	dataSize := int(binary.LittleEndian.Uint32(header[40:44]))

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
