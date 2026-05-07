package config

import (
	"fmt"
	"runtime"
)

const (
	MinCpuThreads       = 1
	MaxCpuThreads       = 64
	MinMaxNewFrames     = 50
	MaxMaxNewFrames     = 2000
	MinVoiceCloneTokens = 10
	MaxVoiceCloneTokens = 500
	DefaultCpuThreads   = -1
	DefaultMaxNewFrames = 375
	DefaultVoiceTokens  = 75
)

type TtsConfig struct {
	CpuThreads       int
	MaxNewFrames     int
	VoiceCloneTokens int
	TextTemperature  float64
	TextTopP         float64
	TextTopK         int
	AudioTemperature float64
	AudioTopP        float64
	AudioTopK        int
	AudioRepPenalty  float64
	Seed             int
}

func DefaultTtsConfig() *TtsConfig {
	return &TtsConfig{
		CpuThreads:       DefaultCpuThreads,
		MaxNewFrames:     DefaultMaxNewFrames,
		VoiceCloneTokens: DefaultVoiceTokens,
		TextTemperature:  1.0,
		TextTopP:         1.0,
		TextTopK:         50,
		AudioTemperature: 0.8,
		AudioTopP:        0.95,
		AudioTopK:        25,
		AudioRepPenalty:  1.2,
		Seed:             -1,
	}
}

func (c *TtsConfig) Validate() error {
	if c.CpuThreads != DefaultCpuThreads {
		if c.CpuThreads < MinCpuThreads {
			return fmt.Errorf("cpu-threads 必须 >= %d", MinCpuThreads)
		}
		if c.CpuThreads > MaxCpuThreads {
			return fmt.Errorf("cpu-threads 必须 <= %d", MaxCpuThreads)
		}
		numCPU := runtime.NumCPU()
		if c.CpuThreads > numCPU {
			fmt.Printf("警告：cpu-threads (%d) 超过 CPU 核心数 (%d)，可能影响性能\n", c.CpuThreads, numCPU)
		}
	}

	if c.MaxNewFrames < MinMaxNewFrames {
		return fmt.Errorf("max-new-frames 必须 >= %d", MinMaxNewFrames)
	}
	if c.MaxNewFrames > MaxMaxNewFrames {
		return fmt.Errorf("max-new-frames 必须 <= %d", MaxMaxNewFrames)
	}

	if c.VoiceCloneTokens < MinVoiceCloneTokens {
		return fmt.Errorf("voice-clone-max-text-tokens 必须 >= %d", MinVoiceCloneTokens)
	}
	if c.VoiceCloneTokens > MaxVoiceCloneTokens {
		return fmt.Errorf("voice-clone-max-text-tokens 必须 <= %d", MaxVoiceCloneTokens)
	}

	if c.TextTemperature <= 0 {
		return fmt.Errorf("text-temperature 必须 > 0")
	}
	if c.TextTopP < 0 || c.TextTopP > 1 {
		return fmt.Errorf("text-top-p 必须在 0-1 之间")
	}
	if c.TextTopK < 1 {
		return fmt.Errorf("text-top-k 必须 >= 1")
	}

	if c.AudioTemperature <= 0 {
		return fmt.Errorf("audio-temperature 必须 > 0")
	}
	if c.AudioTopP < 0 || c.AudioTopP > 1 {
		return fmt.Errorf("audio-top-p 必须在 0-1 之间")
	}
	if c.AudioTopK < 1 {
		return fmt.Errorf("audio-top-k 必须 >= 1")
	}

	if c.AudioRepPenalty <= 0 {
		return fmt.Errorf("audio-repetition-penalty 必须 > 0")
	}

	return nil
}

func (c *TtsConfig) GetEffectiveCpuThreads() int {
	if c.CpuThreads == DefaultCpuThreads {
		n := runtime.NumCPU()
		if n > 1 {
			return n - 1
		}
		return 1
	}
	return c.CpuThreads
}
