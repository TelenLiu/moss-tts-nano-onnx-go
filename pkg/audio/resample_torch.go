package audio

import (
	"math"
)

// 本文件实现与 torchaudio.functional.resample 完全一致的重采样算法
// （sinc_interp_hann 方法），确保需要重采样的参考音频在 Go 端得到与 Python
// 端 bit-exact 的波形，从而保证 codec_encode 码和 EOS 判定一致。
//
// 对应 Python 默认参数：
//   lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann"

const (
	torchLowpassFilterWidth = 6
	torchRolloff            = 0.99
)

// torchSincResample 对交错格式的多声道波形执行 torchaudio 兼容重采样。
// waveform: 交错 float32（[s0ch0,s0ch1,...,s1ch0,...]）
// channels: 声道数
// 返回重采样后的交错 float32。
func torchSincResample(waveform []float32, channels, origFreq, newFreq int) []float32 {
	if origFreq == newFreq {
		return waveform
	}
	g := gcdInt(origFreq, newFreq)
	origFreq /= g
	newFreq /= g

	// 每个声道的输入/输出样本数。
	inFrames := len(waveform) / channels
	targetLength := int(math.Ceil(float64(newFreq*inFrames) / float64(origFreq)))

	// 构建核：kernels[phase][tap]，phase ∈ [0,newFreq)。
	baseFreq := float64(minInt(origFreq, newFreq)) * torchRolloff
	width := int(math.Ceil(float64(torchLowpassFilterWidth) * float64(origFreq) / baseFreq))
	scale := baseFreq / float64(origFreq)

	// idx 长度 = 2*width + origFreq
	idxLen := 2*width + origFreq
	kernels := make([][]float64, newFreq)
	for phase := 0; phase < newFreq; phase++ {
		kernels[phase] = make([]float64, idxLen)
		// t_phase = (0 - phase)/newFreq + idx/origFreq, 其中 idx 从 -width 到 width+origFreq-1
		tBase := -float64(phase) / float64(newFreq)
		for k := 0; k < idxLen; k++ {
			idxVal := float64(k-width) / float64(origFreq)
			t := (tBase + idxVal) * baseFreq
			if t > torchLowpassFilterWidth {
				t = torchLowpassFilterWidth
			}
			if t < -torchLowpassFilterWidth {
				t = -torchLowpassFilterWidth
			}
			// hann 窗：cos(t*pi/W/2)^2
			window := math.Cos(t * math.Pi / torchLowpassFilterWidth / 2.0)
			window *= window
			tPi := t * math.Pi
			var sincVal float64
			if tPi == 0 {
				sincVal = 1.0
			} else {
				sincVal = math.Sin(tPi) / tPi
			}
			kernels[phase][k] = sincVal * window * scale
		}
	}

	// 对每个声道做 conv1d：pad (width, width+origFreq)，stride=origFreq。
	output := make([]float32, targetLength*channels)
	for ch := 0; ch < channels; ch++ {
		in := make([]float64, inFrames)
		for i := 0; i < inFrames; i++ {
			in[i] = float64(waveform[i*channels+ch])
		}
		// 左侧补 width 个零，右侧补 width+origFreq 个零。
		padded := make([]float64, inFrames+2*width+origFreq)
		copy(padded[width:], in)

		for outIdx := 0; outIdx < targetLength; outIdx++ {
			// torch conv1d 输出位置 n 对应卷积起点 n*origFreq。
			// resampled 形状 (1, newFreq, L_out)，transpose 后：
			//   resampled_flat[outIdx] = conv_out[outIdx % newFreq, outIdx / newFreq]
			phase := outIdx % newFreq
			convCol := outIdx / newFreq
			start := convCol * origFreq
			var acc float64
			ker := kernels[phase]
			for k := 0; k < idxLen; k++ {
				acc += padded[start+k] * ker[k]
			}
			output[outIdx*channels+ch] = float32(acc)
		}
	}
	return output
}

func gcdInt(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
