package ttsruntime

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/audio"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/normalizer"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	DefaultInterChunkPauseShortSec = 0.40
	DefaultInterChunkPauseLongSec  = 0.24
	// ChunkContextFrames 每个 chunk 从前一个 chunk 继承的音频帧数，用于提供声学上下文
	// 减少chunk衔接处的音色突变和杂音
	ChunkContextFrames = 8
)

var SentenceEndPunctuation = map[rune]bool{'.': true, '!': true, '?': true, '。': true, '！': true, '？': true, '；': true, ';': true}
var ClauseSplitPunctuation = map[rune]bool{',': true, '，': true, '、': true, '；': true, ';': true, '：': true, ':': true}
var ClosingPunctuation = map[rune]bool{'"': true, '\'': true, '\u201d': true, '\u2019': true, ')': true, ']': true, '}': true, '）': true, '】': true, '》': true, '」': true, '』': true}

type SynthesisResult struct {
	AudioPath    string
	AudioData    []byte
	SampleRate   int
	AudioSamples int
	Waveform     []float32
	Channels     int
	TextChunks   []string
	SampleMode   string
	DoSample     bool
	Streaming    bool
	ElapsedSec   float64
	SeedUsed     int64
}

type StreamChunk struct {
	Waveform   []float32
	SampleRate int
	Channels   int
	ChunkIndex int
	IsPause    bool
	SeedUsed   int64
}

type OnnxTtsRuntime struct {
	OrtRuntime          *ortruntime.OrtCpuRuntime
	SPModel             *tokenizer.Processor
	PreloadCache        *PreloadCache
	AudioCloneCache     *AudioCloneCache                // 音频克隆编码的文件缓存（hash key + gob，跨进程共享）
	MemoryThresholdMB   int                             // 长文本多 chunk 推理时，单 chunk 处理后的内存上限MB，超过此阈值触发 ForceResetSessions
	GenerationOverrides *ortruntime.GenerationOverrides // 运行时采样参数覆盖
}

func NewOnnxTtsRuntime(modelDir string, threadCount int, coreMemMB int, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*OnnxTtsRuntime, error) {
	rt, err := ortruntime.NewOrtCpuRuntime(modelDir, threadCount, maxNewFrames, doSample, sampleMode, executionMode)
	if err != nil {
		return nil, fmt.Errorf("创建 OrtCpuRuntime 失败: %w", err)
	}
	if err := rt.CreateSessions(); err != nil {
		return nil, fmt.Errorf("创建 ONNX sessions 失败: %w", err)
	}
	manifest := rt.Manifest
	modelFiles := manifest["model_files"].(map[string]interface{})
	tokenizerRelPath := "tokenizer.model"
	if tp, ok := modelFiles["tokenizer_model"]; ok {
		tokenizerRelPath = fmt.Sprintf("%v", tp)
	}
	tokenizerPath := rt.ResolveManifestRelativePath(tokenizerRelPath)
	sp, err := tokenizer.NewProcessor(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("加载 SentencePiece 分词器失败: %w", err)
	}

	// 创建preload缓存
	if coreMemMB <= 0 {
		coreMemMB = 800
	}
	ttsRuntime := &OnnxTtsRuntime{
		OrtRuntime:        rt,
		SPModel:           sp,
		MemoryThresholdMB: coreMemMB,
	}
	// 使用 cache/audio_clone_gob 作为音频克隆编码缓存目录（跨进程共享）
	audioCloneCacheDir := filepath.Join(filepath.Dir(modelDir), "lib", "cache", "audio_clone_gob")
	ttsRuntime.AudioCloneCache = NewAudioCloneCache(audioCloneCacheDir)
	// 使用 lib/cache/assets_preload 作为缓存目录
	cacheDir := filepath.Join(filepath.Dir(modelDir), "lib", "cache", "assets_preload")
	ttsRuntime.PreloadCache = NewPreloadCache(cacheDir, 2, ttsRuntime)

	return ttsRuntime, nil
}

func (t *OnnxTtsRuntime) EncodeText(text string) []int {
	return t.SPModel.EncodeAsIDs(text)
}

func (t *OnnxTtsRuntime) CountTextTokens(text string) int {
	return t.SPModel.CountTokens(text)
}

func (t *OnnxTtsRuntime) buildGenerationOverrides(sampleMode string, doSample bool) *ortruntime.GenerationOverrides {
	o := t.GenerationOverrides
	if o == nil {
		o = &ortruntime.GenerationOverrides{}
	}
	o.SampleMode = &sampleMode
	o.DoSample = &doSample
	return o
}

func (t *OnnxTtsRuntime) PrepareSynthesisText(text string, enableNormalize bool) string {
	if enableNormalize {
		return normalizer.PrepareTTSText(text, true, true)
	}
	return text
}

func (t *OnnxTtsRuntime) PrepareSynthesisTextEx(text string, enableRobust bool, enableWeText bool) string {
	return normalizer.PrepareTTSText(text, enableRobust, enableWeText)
}

func (t *OnnxTtsRuntime) PrepareSynthesisTextWithVoice(text string, enableRobust bool, enableWeText bool, voice string) string {
	return normalizer.PrepareTTSTextWithVoice(text, enableRobust, enableWeText, voice)
}

func (t *OnnxTtsRuntime) SplitVoiceCloneText(text string, maxTokens int) []string {
	normalizedText := strings.TrimSpace(text)
	if normalizedText == "" {
		return nil
	}
	safeMaxTokens := max(1, maxTokens)
	preparedText := prepareTextForSentenceChunking(normalizedText)
	sentenceCandidates := splitTextByPunctuation(preparedText, SentenceEndPunctuation)
	if len(sentenceCandidates) == 0 {
		sentenceCandidates = []string{strings.TrimSpace(preparedText)}
	}
	log.Printf("[SplitVoiceCloneText] 原始句子数: %d, maxTokens=%d", len(sentenceCandidates), safeMaxTokens)
	for i, s := range sentenceCandidates {
		log.Printf("[SplitVoiceCloneText]   句子 %d: %q (tokens=%d)", i+1, s, t.CountTextTokens(s))
	}
	type slice struct {
		tokenCount int
		text       string
	}
	var sentenceSlices []slice
	for _, sentenceText := range sentenceCandidates {
		normalizedSentence := strings.TrimSpace(sentenceText)
		if normalizedSentence == "" {
			continue
		}
		sentenceTokenCount := t.CountTextTokens(normalizedSentence)
		if sentenceTokenCount <= safeMaxTokens {
			sentenceSlices = append(sentenceSlices, slice{sentenceTokenCount, normalizedSentence})
			continue
		}
		clauseCandidates := splitTextByPunctuation(normalizedSentence, ClauseSplitPunctuation)
		if len(clauseCandidates) <= 1 {
			clauseCandidates = []string{normalizedSentence}
		}
		for _, clauseText := range clauseCandidates {
			normalizedClause := strings.TrimSpace(clauseText)
			if normalizedClause == "" {
				continue
			}
			clauseTokenCount := t.CountTextTokens(normalizedClause)
			if clauseTokenCount <= safeMaxTokens {
				sentenceSlices = append(sentenceSlices, slice{clauseTokenCount, normalizedClause})
				continue
			}
			for _, piece := range t.splitTextByTokenBudget(normalizedClause, safeMaxTokens) {
				normalizedPiece := strings.TrimSpace(piece)
				if normalizedPiece != "" {
					sentenceSlices = append(sentenceSlices, slice{t.CountTextTokens(normalizedPiece), normalizedPiece})
				}
			}
		}
	}
	var chunks []string
	currentChunk := ""
	currentChunkTokenCount := 0
	for _, s := range sentenceSlices {
		if currentChunk == "" {
			currentChunk = s.text
			currentChunkTokenCount = s.tokenCount
			continue
		}
		if currentChunkTokenCount+s.tokenCount > safeMaxTokens {
			chunks = append(chunks, strings.TrimSpace(currentChunk))
			currentChunk = s.text
			currentChunkTokenCount = s.tokenCount
		} else {
			currentChunk = joinSentenceParts(currentChunk, s.text)
			currentChunkTokenCount = t.CountTextTokens(currentChunk)
		}
	}
	if currentChunk != "" {
		chunks = append(chunks, strings.TrimSpace(currentChunk))
	}
	log.Printf("[SplitVoiceCloneText] 分块结果: %d 块", len(chunks))
	for i, chunk := range chunks {
		log.Printf("[SplitVoiceCloneText]   chunk[%d]: %q (tokens=%d)", i, chunk, t.CountTextTokens(chunk))
	}
	if len(chunks) > 1 {
		return chunks
	}
	return []string{normalizedText}
}

func (t *OnnxTtsRuntime) splitTextByTokenBudget(text string, maxTokens int) []string {
	remainingText := strings.TrimSpace(text)
	if remainingText == "" {
		return nil
	}
	var pieces []string
	preferredBoundaryChars := make(map[rune]bool)
	for k := range ClauseSplitPunctuation {
		preferredBoundaryChars[k] = true
	}
	for k := range SentenceEndPunctuation {
		preferredBoundaryChars[k] = true
	}
	preferredBoundaryChars[' '] = true
	for remainingText != "" {
		if t.CountTextTokens(remainingText) <= maxTokens {
			pieces = append(pieces, remainingText)
			break
		}
		low, high := 1, len(remainingText)
		bestPrefixLength := 1
		for low <= high {
			middle := (low + high) / 2
			candidate := strings.TrimSpace(remainingText[:middle])
			if candidate == "" {
				low = middle + 1
				continue
			}
			if t.CountTextTokens(candidate) <= maxTokens {
				bestPrefixLength = middle
				low = middle + 1
			} else {
				high = middle - 1
			}
		}
		cutIndex := bestPrefixLength
		prefix := remainingText[:bestPrefixLength]
		preferredIndex := -1
		scanMin := len(prefix) - 25
		if scanMin < -1 {
			scanMin = -1
		}
		for scanIndex := len(prefix) - 1; scanIndex > scanMin; scanIndex-- {
			if preferredBoundaryChars[rune(prefix[scanIndex])] {
				preferredIndex = scanIndex + 1
				break
			}
		}
		if preferredIndex > 0 {
			cutIndex = preferredIndex
		}
		piece := strings.TrimSpace(remainingText[:cutIndex])
		if piece == "" {
			piece = strings.TrimSpace(remainingText[:bestPrefixLength])
			cutIndex = bestPrefixLength
		}
		pieces = append(pieces, piece)
		remainingText = strings.TrimSpace(remainingText[cutIndex:])
	}
	return pieces
}

func (t *OnnxTtsRuntime) ResolvePromptAudioCodes(voice string, promptAudioPath string) [][]int {
	return t.ResolvePromptAudioCodesWithPreload(voice, promptAudioPath, "", "")
}

func (t *OnnxTtsRuntime) ResolvePromptAudioCodesWithPreload(voice string, promptAudioPath string, preloadId string, preloadAudioPath string) [][]int {
	log.Printf("[ResolvePromptAudioCodes] voice=%q promptAudioPath=%q preloadId=%q preloadAudioPath=%q", voice, promptAudioPath, preloadId, preloadAudioPath)

	var result [][]int

	// 优先使用preloadId
	if preloadId != "" && t.PreloadCache != nil {
		log.Printf("[ResolvePromptAudioCodes] 尝试使用preload缓存: %s", preloadId)
		data, err := t.PreloadCache.Get(preloadId)
		if err != nil {
			log.Printf("[ResolvePromptAudioCodes] preload缓存获取失败: %v，尝试预加载", err)
			// 尝试预加载
			if preloadAudioPath != "" {
				if err := t.PreloadCache.Preload(preloadId, preloadAudioPath, ""); err != nil {
					log.Printf("[ResolvePromptAudioCodes] 预加载失败: %v", err)
				} else {
					// 再次尝试获取
					data, err = t.PreloadCache.Get(preloadId)
					if err == nil {
						log.Printf("[ResolvePromptAudioCodes] 使用preload缓存: %s (frames=%d)", preloadId, len(data.AudioCodes))
						result = data.AudioCodes
					}
				}
			}
		} else {
			log.Printf("[ResolvePromptAudioCodes] 使用preload缓存: %s (frames=%d)", preloadId, len(data.AudioCodes))
			result = data.AudioCodes
		}
	} else if preloadId != "" && t.PreloadCache == nil {
		log.Printf("[ResolvePromptAudioCodes] 警告: PreloadCache未初始化，无法使用preloadId")
	}

	if result == nil && promptAudioPath != "" {
		log.Printf("[ResolvePromptAudioCodes] 使用上传的参考音频: %s", promptAudioPath)
		codes := t.EncodeReferenceAudio(promptAudioPath)
		if codes != nil {
			log.Printf("[ResolvePromptAudioCodes] 参考音频编码成功: %d 帧", len(codes))
		} else {
			log.Printf("[ResolvePromptAudioCodes] 参考音频编码失败，将回退到内置音色")
		}
		if codes != nil {
			result = codes
		}
	}

	if result == nil {
		resolvedVoice := voice
		if resolvedVoice == "" {
			voices := t.OrtRuntime.ListBuiltinVoices()
			if len(voices) > 0 {
				resolvedVoice = fmt.Sprintf("%v", voices[0]["voice"])
			}
		}
		log.Printf("[ResolvePromptAudioCodes] 使用内置音色: %s", resolvedVoice)
		for _, v := range t.OrtRuntime.ListBuiltinVoices() {
			if fmt.Sprintf("%v", v["voice"]) == resolvedVoice {
				codes, ok := v["prompt_audio_codes"].([]interface{})
				if !ok {
					log.Printf("[ResolvePromptAudioCodes] 警告: 音色 %s 的prompt_audio_codes格式不正确，跳过", resolvedVoice)
					continue
				}
				parsed := make([][]int, len(codes))
				allOk := true
				for i, codeRow := range codes {
					row, ok := codeRow.([]interface{})
					if !ok {
						log.Printf("[ResolvePromptAudioCodes] 警告: 音色 %s 的第%d行格式不正确", resolvedVoice, i)
						allOk = false
						break
					}
					parsed[i] = make([]int, len(row))
					for j, val := range row {
						parsed[i][j] = int(ortruntime.ToFloat64(val))
					}
				}
				if allOk {
					log.Printf("[ResolvePromptAudioCodes] 内置音色 %s 加载成功: %d 帧", resolvedVoice, len(parsed))
					result = parsed
					break
				}
				log.Printf("[ResolvePromptAudioCodes] 警告: 音色 %s 加载失败，继续查找", resolvedVoice)
			}
		}
		if result == nil {
			log.Printf("[ResolvePromptAudioCodes] 警告: 未找到内置音色 %s，使用第一个内置音色作为默认", resolvedVoice)
			voices := t.OrtRuntime.ListBuiltinVoices()
			if len(voices) > 0 {
				for _, v := range voices {
					codes, ok := v["prompt_audio_codes"].([]interface{})
					if !ok {
						continue
					}
					parsed := make([][]int, len(codes))
					allOk := true
					for i, codeRow := range codes {
						row, ok := codeRow.([]interface{})
						if !ok {
							allOk = false
							break
						}
						parsed[i] = make([]int, len(row))
						for j, val := range row {
							parsed[i][j] = int(ortruntime.ToFloat64(val))
						}
					}
					if allOk {
						log.Printf("[ResolvePromptAudioCodes] 使用第一个可用音色: %s (%d 帧)", fmt.Sprintf("%v", v["voice"]), len(parsed))
						result = parsed
						break
					}
				}
			}
		}
	}

	// 统一截断保护：限制 audio codes 帧数，避免 prefill 输入序列过长导致 OOM
	const maxPromptAudioFrames = 300
	if len(result) > maxPromptAudioFrames {
		log.Printf("[ResolvePromptAudioCodes] 音频编码帧数过多(%d帧)，截断至%d帧", len(result), maxPromptAudioFrames)
		result = result[:maxPromptAudioFrames]
	}

	if result == nil {
		log.Printf("[ResolvePromptAudioCodes] 错误: 没有可用的内置音色，返回空")
	}
	return result
}

func (t *OnnxTtsRuntime) EncodeReferenceAudio(audioPath string) [][]int {
	return t.EncodeReferenceAudioWithOptions(audioPath, false)
}

// EncodeReferenceAudioWithOptions 编码参考音频
// skipAudioCloneCacheWrite=true 时跳过写入 AudioCloneCache（用于 PreloadCache，因其自身已有永久存储，避免重复写入）
func (t *OnnxTtsRuntime) EncodeReferenceAudioWithOptions(audioPath string, skipAudioCloneCacheWrite bool) [][]int {
	// 计算音频文件 hash，用于文件缓存 key
	hashKey, err := t.AudioCloneCache.HashAudioFile(audioPath)
	if err != nil {
		log.Printf("[EncodeReferenceAudio] 计算音频hash失败: %v, 跳过缓存", err)
	} else {
		// 先查文件缓存，命中则直接返回，跳过阶段1
		if cached := t.AudioCloneCache.Get(hashKey); cached != nil {
			return cached
		}
	}

	log.Printf("[EncodeReferenceAudio] 开始编码参考音频: %s", audioPath)

	// ========== 阶段1：音频编码 ==========
	// 先销毁阶段2的推理 session，确保只有 CodecEncode 在内存中，避免峰值叠加
	log.Printf("[EncodeReferenceAudio] [阶段1] 销毁推理 sessions，切换到音频编码阶段...")
	t.OrtRuntime.DestroyAllSessions()
	runtime.GC()
	debug.FreeOSMemory()

	codecConfig := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
	targetSampleRate := int(ortruntime.ToFloat64(codecConfig["sample_rate"]))
	targetChannels := int(ortruntime.ToFloat64(codecConfig["channels"]))
	numQuantizers := int(ortruntime.ToFloat64(codecConfig["num_quantizers"]))
	log.Printf("[EncodeReferenceAudio] codec配置: sampleRate=%d channels=%d numQuantizers=%d", targetSampleRate, targetChannels, numQuantizers)

	waveform, channels, sampleRate, err := audio.LoadReferenceAudio(audioPath, targetSampleRate, targetChannels)
	if err != nil {
		log.Printf("[EncodeReferenceAudio] 加载参考音频失败: %v, 使用内置音色", err)
		// 加载失败，直接进入阶段2
		t.OrtRuntime.CreateSessions()
		return nil
	}
	log.Printf("[EncodeReferenceAudio] 加载完成: 原始采样率=%d 通道数=%d 总样本数=%d", sampleRate, channels, len(waveform))

	waveformLength := len(waveform) / channels

	// 直接构建平面格式数据: [ch0_s0, ch0_s1, ..., ch1_s0, ch1_s1, ...]
	planarWaveform := make([]float32, len(waveform))
	for ch := 0; ch < channels; ch++ {
		for i := 0; i < waveformLength; i++ {
			planarWaveform[ch*waveformLength+i] = waveform[i*channels+ch]
		}
	}

	flatWaveform := planarWaveform
	wfDims := []int64{1, int64(channels), int64(waveformLength)}
	waveformTensor, _ := ort.NewTensor(wfDims, flatWaveform)
	inputLengths := []int32{int32(waveformLength)}
	inputLengthsTensor, _ := ort.NewTensor([]int64{1}, inputLengths)

	inputs := []ort.Value{waveformTensor, inputLengthsTensor}
	outputs := make([]ort.Value, 2)

	log.Printf("[EncodeReferenceAudio] [阶段1] 创建 CodecEncode session 并编码...")

	// 懒加载 CodecEncode session（此时推理 session 已销毁，不会内存叠加）
	if err := t.OrtRuntime.EnsureCodecEncodeSession(); err != nil {
		log.Printf("[EncodeReferenceAudio] CodecEncode session 创建失败: %v, 使用内置音色", err)
		t.OrtRuntime.CreateSessions()
		return nil
	}

	err = t.OrtRuntime.RunSession(t.OrtRuntime.Onnx.CodecEncode.Session, inputs, outputs)
	waveformTensor.Destroy()
	inputLengthsTensor.Destroy()

	if err != nil {
		log.Printf("[EncodeReferenceAudio] codec_encode 失败: %v, 使用内置音色", err)
		t.OrtRuntime.DestroyCodecEncodeSession()
		runtime.GC()
		debug.FreeOSMemory()
		t.OrtRuntime.CreateSessions()
		return nil
	}

	audioCodesData := getInt32Data(outputs[0])
	audioCodesShape := outputs[0].GetShape()
	audioCodeLengthsData := getInt32Data(outputs[1])
	log.Printf("[EncodeReferenceAudio] codec_encode 成功: audioCodes形状=%v audioCodeLengths=%v 数据长度=%d",
		audioCodesShape, audioCodeLengthsData, len(audioCodesData))
	// 调试：打印前32个原始数据
	if len(audioCodesData) >= 32 {
		log.Printf("[EncodeReferenceAudio] audioCodesData[:32]=%v", audioCodesData[:32])
	}

	for _, v := range outputs {
		if v != nil {
			v.Destroy()
		}
	}

	codeLength := int(audioCodeLengthsData[0])
	if len(audioCodesShape) >= 3 {
		codeLength = minInt(codeLength, int(audioCodesShape[1]))
	}

	// 截断过长的音频编码帧，避免 prefill 输入序列过长导致 OOM
	maxPromptAudioFrames := 300 // 约 24 秒（帧率 12.5/秒）
	if codeLength > maxPromptAudioFrames {
		log.Printf("[EncodeReferenceAudio] 音频编码帧数过多(%d帧)，截断至%d帧", codeLength, maxPromptAudioFrames)
		codeLength = maxPromptAudioFrames
	}

	log.Printf("[EncodeReferenceAudio] 解析codeLength=%d", codeLength)

	promptAudioCodes := make([][]int, codeLength)
	for frameIndex := 0; frameIndex < codeLength; frameIndex++ {
		promptAudioCodes[frameIndex] = make([]int, numQuantizers)
		for q := 0; q < numQuantizers; q++ {
			offset := frameIndex*numQuantizers + q
			if offset < len(audioCodesData) {
				promptAudioCodes[frameIndex][q] = int(audioCodesData[offset])
			}
		}
	}

	if codeLength > 0 {
		log.Printf("[EncodeReferenceAudio] 参考音频编码完成: %d 帧, 第一帧codes=%v, 最后一帧codes=%v",
			codeLength, promptAudioCodes[0], promptAudioCodes[codeLength-1])
	} else {
		log.Printf("[EncodeReferenceAudio] 参考音频编码完成: 0 帧")
	}

	// 缓存编码结果到文件，后续相同音频无需再调用 CodecEncode
	// skipAudioCloneCacheWrite 时跳过写入（PreloadCache 自身已有永久存储，避免重复）
	if hashKey != "" && !skipAudioCloneCacheWrite {
		if err := t.AudioCloneCache.Put(hashKey, promptAudioCodes); err != nil {
			log.Printf("[EncodeReferenceAudio] 写入缓存失败: %v", err)
		}
	}

	// ========== 阶段1结束 → 阶段2准备 ==========
	// 销毁 CodecEncode session，释放阶段1内存，然后重建推理 session
	log.Printf("[EncodeReferenceAudio] [阶段1→阶段2] 销毁 CodecEncode，重建推理 sessions...")
	t.OrtRuntime.DestroyCodecEncodeSession()
	runtime.GC()
	debug.FreeOSMemory()

	// 重建推理 sessions（阶段2）
	if err := t.OrtRuntime.CreateSessions(); err != nil {
		log.Printf("[EncodeReferenceAudio] 重建推理 sessions 失败: %v", err)
	}

	return promptAudioCodes
}

func (t *OnnxTtsRuntime) Synthesize(text string, voice string, promptAudioPath string, outputAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (*SynthesisResult, error) {
	return t.SynthesizeWithContext(context.Background(), text, voice, promptAudioPath, outputAudioPath, sampleMode, doSample, streaming, maxNewFrames, voiceCloneMaxTextTokens, enableNormalize, seed)
}

func (t *OnnxTtsRuntime) SynthesizeEx(text string, voice string, promptAudioPath string, outputAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (*SynthesisResult, error) {
	return t.SynthesizeWithContextEx(context.Background(), text, voice, promptAudioPath, outputAudioPath, "", "", sampleMode, doSample, streaming, maxNewFrames, voiceCloneMaxTextTokens, enableRobust, enableWeText, seed)
}

func (t *OnnxTtsRuntime) SynthesizeWithContextEx(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (*SynthesisResult, error) {
	startTime := time.Now()

	t.OrtRuntime.AcquireSession()
	defer t.OrtRuntime.ReleaseSession()
	// 推理结束后（无论正常完成还是中断），执行内存清理
	interrupted := false
	defer func() {
		runtime.GC()
		debug.FreeOSMemory()
		if interrupted {
			// 中断时强制重置 Session，释放 ONNX 内存池
			t.OrtRuntime.ForceResetSessions()
			log.Printf("[Synthesize] 推理被中断，Session 已强制重置，ONNX 内存池已释放")
		}
	}()

	t.OrtRuntime.CheckAndReleaseIdleSessions()

	log.Printf("[Synthesize] ========== 阶段1：音频编码 ==========")
	log.Printf("[Synthesize] 开始合成：text=%q voice=%q promptAudioPath=%q preloadId=%q sampleMode=%s doSample=%v maxNewFrames=%d", text, voice, promptAudioPath, preloadId, sampleMode, doSample, maxNewFrames)

	var rngSeed int64 = 1234
	if seed != nil {
		rngSeed = int64(*seed)
		log.Printf("[Synthesize] 使用随机种子: %d", *seed)
	} else {
		rngSeed = time.Now().UnixNano()
		log.Printf("[Synthesize] 使用随机种子(基于时间): %d", rngSeed)
	}
	t.OrtRuntime.RNG = rand.New(rand.NewSource(rngSeed))

	// 构建采样参数覆盖
	overrides := t.buildGenerationOverrides(sampleMode, doSample)

	preparedText := t.PrepareSynthesisTextWithVoice(text, enableRobust, enableWeText, voice)
	log.Printf("[Synthesize] 文本预处理完成: 原始长度=%d 预处理后长度=%d (robust=%v wetext=%v)", len(text), len(preparedText), enableRobust, enableWeText)
	promptAudioCodes := t.ResolvePromptAudioCodesWithPreload(voice, promptAudioPath, preloadId, preloadAudioPath)
	if promptAudioCodes == nil {
		log.Printf("[Synthesize] 警告: promptAudioCodes 为 nil，将使用空列表")
		promptAudioCodes = [][]int{}
	} else {
		// 截断过长的参考音频帧，减少 prefill 输入长度
		promptAudioCodes = truncatePromptAudioCodes(promptAudioCodes, defaultMaxPromptAudioFrames)
		log.Printf("[Synthesize] promptAudioCodes: %d 帧", len(promptAudioCodes))
	}

	// ========== 阶段2：TTS 推理 ==========
	// 此时阶段1的 CodecEncode session 已销毁，只有推理 session 在内存中
	log.Printf("[Synthesize] ========== 阶段2：TTS 推理 ==========")
	textChunks := t.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens)
	log.Printf("[Synthesize] 文本分块: %d 块", len(textChunks))
	for i, chunk := range textChunks {
		log.Printf("[Synthesize]   chunk[%d]: %q (tokens=%d)", i, chunk, t.CountTextTokens(chunk))
	}
	codecMeta := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
	sampleRate := int(ortruntime.ToFloat64(codecMeta["sample_rate"]))
	channels := int(ortruntime.ToFloat64(codecMeta["channels"]))
	log.Printf("[Synthesize] codec配置: sampleRate=%d channels=%d", sampleRate, channels)

	// 确保推理 sessions 存在（可能被 ForceResetSessions 销毁后尚未重建）
	if t.OrtRuntime.Onnx.Inference == nil || t.OrtRuntime.Onnx.Inference.CodecDecodeStep == nil {
		log.Printf("[Synthesize] 推理 sessions 不存在，重新创建...")
		if err := t.OrtRuntime.CreateSessions(); err != nil {
			return nil, fmt.Errorf("重建推理 sessions 失败: %w", err)
		}
	}

	// 使用流式 codec 解码，避免一次性分配大 tensor 导致内存峰值过高
	// 注意：长文本多 chunk 时，chunk 间会 ForceResetSessions 重建 session，
	// 所以不能用 defer 重置 streamingCodecSession，需手动管理
	streamingCodecSession := ortruntime.NewCodecStreamingDecodeSession(t.OrtRuntime.CodecMeta, t.OrtRuntime.Onnx.Inference.CodecDecodeStep.Session, t.OrtRuntime)

	var allWaveforms [][]float32
	var prevChunkTailFrames [][]int // 前一个 chunk 的最后几帧，作为下一个 chunk 的声学上下文
	for chunkIndex, chunkText := range textChunks {
		select {
		case <-ctx.Done():
			interrupted = true
			if streamingCodecSession != nil {
				streamingCodecSession.Reset()
			}
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		default:
		}
		log.Printf("[Synthesize] 处理 chunk %d/%d...", chunkIndex+1, len(textChunks))
		logMemoryStats(fmt.Sprintf("chunk %d/%d 开始", chunkIndex+1, len(textChunks)))

		// 每个 chunk 重置 RNG 到相同种子，确保各 chunk 的随机值序列一致，减少音色差异
		t.OrtRuntime.RNG = rand.New(rand.NewSource(rngSeed))

		textTokenIDs := t.EncodeText(chunkText)
		log.Printf("[Synthesize]   文本编码完成: %d tokens", len(textTokenIDs))

		// 将前一个 chunk 的尾部帧作为声学上下文追加到参考音频后面
		chunkPromptCodes := promptAudioCodes
		if chunkIndex > 0 && len(prevChunkTailFrames) > 0 {
			chunkPromptCodes = make([][]int, len(promptAudioCodes), len(promptAudioCodes)+len(prevChunkTailFrames))
			copy(chunkPromptCodes, promptAudioCodes)
			chunkPromptCodes = append(chunkPromptCodes, prevChunkTailFrames...)
			log.Printf("[Synthesize]   追加 %d 帧上下文帧到参考音频", len(prevChunkTailFrames))
		}

		requestRows := t.OrtRuntime.BuildVoiceCloneRequestRows(chunkPromptCodes, textTokenIDs)
		log.Printf("[Synthesize]   请求行构建完成：%d 行", len(requestRows["inputIds"]))

		// 根据文本 token 数估算最大帧数，避免无效 decode 步骤
		effectiveMaxFrames := estimateMaxNewFrames(len(textTokenIDs), maxNewFrames)
		generatedFrames := t.OrtRuntime.GenerateAudioFramesWithContextAndOverrides(ctx, requestRows, effectiveMaxFrames, overrides)
		if ctx.Err() != nil {
			interrupted = true
			if streamingCodecSession != nil {
				streamingCodecSession.Reset()
			}
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		}
		log.Printf("[Synthesize]   音频帧生成完成: %d 帧 (maxNewFrames=%d effective=%d)", len(generatedFrames), maxNewFrames, effectiveMaxFrames)

		// 保存当前 chunk 的最后几帧，作为下一个 chunk 的声学上下文
		if len(generatedFrames) >= ChunkContextFrames {
			prevChunkTailFrames = make([][]int, ChunkContextFrames)
			copy(prevChunkTailFrames, generatedFrames[len(generatedFrames)-ChunkContextFrames:])
		} else if len(generatedFrames) > 0 {
			prevChunkTailFrames = make([][]int, len(generatedFrames))
			copy(prevChunkTailFrames, generatedFrames)
		}

		// 使用流式 codec 解码替代一次性全量解码，降低内存峰值
		var channelArrays [][]float32
		var audioLength int
		if streamingCodecSession != nil {
			streamingCodecSession.Reset()
			channelArrays, audioLength = decodeFramesStreaming(streamingCodecSession, generatedFrames)
		} else {
			channelArrays, audioLength = t.OrtRuntime.DecodeFullAudioWithContext(ctx, generatedFrames)
		}
		if ctx.Err() != nil {
			interrupted = true
			if streamingCodecSession != nil {
				streamingCodecSession.Reset()
			}
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		}
		log.Printf("[Synthesize]   音频解码完成: channels=%d samples=%d", len(channelArrays), audioLength)
		if len(channelArrays) > 0 {
			merged := audio.MergeAudioChannels(channelArrays)
			allWaveforms = append(allWaveforms, merged)
		}
		if chunkIndex < len(textChunks)-1 {
			pauseSeconds := estimateInterChunkPauseSeconds(chunkText)
			pauseSamples := int(math.Round(float64(sampleRate) * pauseSeconds))
			if pauseSamples > 0 {
				allWaveforms = append(allWaveforms, audio.MakeSilence(pauseSamples, channels))
			}
		}

		logMemoryStats(fmt.Sprintf("chunk %d/%d 处理完成", chunkIndex+1, len(textChunks)))

		// 长文本多 chunk 间：先做轻量清理，内存超阈值时才 ForceResetSessions
		// 避免每个 chunk 间都重建 Session 的开销（约1-2秒/次）
		if chunkIndex < len(textChunks)-1 && len(textChunks) > 1 {
			runtime.GC()
			debug.FreeOSMemory()
			streamingCodecSession, _ = t.resetSessionsIfOverMemory(streamingCodecSession,
				fmt.Sprintf("chunk %d/%d", chunkIndex+1, len(textChunks)))
		}
	}
	// 最终清理 codec streaming session
	if streamingCodecSession != nil {
		streamingCodecSession.Reset()
	}
	waveform := audio.ConcatWaveforms(allWaveforms)

	// AudioData 延迟编码：子进程通过 attachment 传 Waveform bytes，AudioData 由调用方按需编码
	var resolvedOutputPath string
	if outputAudioPath != "" {
		resolvedOutputPath = outputAudioPath
		if err := audio.WriteWAV(resolvedOutputPath, waveform, channels, sampleRate); err != nil {
			return nil, fmt.Errorf("写入 WAV 文件失败: %w", err)
		}
	}

	// 释放中间数据，帮助 GC 回收大块内存
	allWaveforms = nil

	elapsed := time.Since(startTime).Seconds()
	audioSamples := len(waveform) / channels
	return &SynthesisResult{
		AudioPath:    resolvedOutputPath,
		SampleRate:   sampleRate,
		AudioSamples: audioSamples,
		Waveform:     waveform,
		Channels:     channels,
		TextChunks:   textChunks,
		SampleMode:   sampleMode,
		DoSample:     doSample,
		Streaming:    streaming,
		ElapsedSec:   elapsed,
		SeedUsed:     rngSeed,
	}, nil
}

func (t *OnnxTtsRuntime) SynthesizeWithContext(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (*SynthesisResult, error) {
	startTime := time.Now()

	t.OrtRuntime.AcquireSession()
	defer t.OrtRuntime.ReleaseSession()
	// 推理结束后（无论正常完成还是中断），执行内存清理
	interrupted := false
	defer func() {
		runtime.GC()
		debug.FreeOSMemory()
		if interrupted {
			// 中断时强制重置 Session，释放 ONNX 内存池
			t.OrtRuntime.ForceResetSessions()
			log.Printf("[Synthesize] 推理被中断，Session 已强制重置，ONNX 内存池已释放")
		}
	}()

	// 检查并释放空闲超时的 Session
	t.OrtRuntime.CheckAndReleaseIdleSessions()

	log.Printf("[Synthesize] 开始合成：text=%q voice=%q promptAudioPath=%q sampleMode=%s doSample=%v maxNewFrames=%d", text, voice, promptAudioPath, sampleMode, doSample, maxNewFrames)

	// 保存初始随机种子，只在开始时设置一次，确保RNG状态在分块之间连续
	var rngSeed int64 = 1234 // 默认种子
	if seed != nil {
		rngSeed = int64(*seed)
		log.Printf("[Synthesize] 使用随机种子: %d", *seed)
	} else {
		// 如果没有指定种子，使用当前时间作为种子，确保每次合成都是独立的
		rngSeed = time.Now().UnixNano()
		log.Printf("[Synthesize] 使用随机种子(基于时间): %d", rngSeed)
	}
	t.OrtRuntime.RNG = rand.New(rand.NewSource(rngSeed))

	// 构建采样参数覆盖
	overrides := t.buildGenerationOverrides(sampleMode, doSample)

	preparedText := t.PrepareSynthesisText(text, enableNormalize)
	log.Printf("[Synthesize] 文本预处理完成: 原始长度=%d 预处理后长度=%d", len(text), len(preparedText))
	promptAudioCodes := t.ResolvePromptAudioCodes(voice, promptAudioPath)
	if promptAudioCodes == nil {
		log.Printf("[Synthesize] 警告: promptAudioCodes 为 nil，将使用空列表")
		promptAudioCodes = [][]int{}
	} else {
		// 截断过长的参考音频帧，减少 prefill 输入长度
		promptAudioCodes = truncatePromptAudioCodes(promptAudioCodes, defaultMaxPromptAudioFrames)
		log.Printf("[Synthesize] promptAudioCodes: %d 帧", len(promptAudioCodes))
	}
	textChunks := t.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens)
	log.Printf("[Synthesize] 文本分块: %d 块", len(textChunks))
	for i, chunk := range textChunks {
		log.Printf("[Synthesize]   chunk[%d]: %q (tokens=%d)", i, chunk, t.CountTextTokens(chunk))
	}
	codecMeta := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
	sampleRate := int(ortruntime.ToFloat64(codecMeta["sample_rate"]))
	channels := int(ortruntime.ToFloat64(codecMeta["channels"]))
	log.Printf("[Synthesize] codec配置: sampleRate=%d channels=%d", sampleRate, channels)

	// 确保推理 sessions 存在（可能被 ForceResetSessions 销毁后尚未重建）
	if t.OrtRuntime.Onnx.Inference == nil || t.OrtRuntime.Onnx.Inference.CodecDecodeStep == nil {
		log.Printf("[Synthesize] 推理 sessions 不存在，重新创建...")
		if err := t.OrtRuntime.CreateSessions(); err != nil {
			return nil, fmt.Errorf("重建推理 sessions 失败: %w", err)
		}
	}

	// 使用流式 codec 解码，避免一次性分配大 tensor 导致内存峰值过高
	// 注意：长文本多 chunk 时，chunk 间会 ForceResetSessions 重建 session，
	// 所以不能用 defer 重置 streamingCodecSession，需手动管理
	streamingCodecSession := ortruntime.NewCodecStreamingDecodeSession(t.OrtRuntime.CodecMeta, t.OrtRuntime.Onnx.Inference.CodecDecodeStep.Session, t.OrtRuntime)

	var allWaveforms [][]float32
	var prevChunkTailFrames [][]int // 前一个 chunk 的最后几帧，作为下一个 chunk 的声学上下文
	for chunkIndex, chunkText := range textChunks {
		select {
		case <-ctx.Done():
			interrupted = true
			if streamingCodecSession != nil {
				streamingCodecSession.Reset()
			}
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		default:
		}
		log.Printf("[Synthesize] 处理 chunk %d/%d...", chunkIndex+1, len(textChunks))
		logMemoryStats(fmt.Sprintf("chunk %d/%d 开始", chunkIndex+1, len(textChunks)))

		// 每个 chunk 重置 RNG 到相同种子，确保各 chunk 的随机值序列一致，减少音色差异
		t.OrtRuntime.RNG = rand.New(rand.NewSource(rngSeed))

		textTokenIDs := t.EncodeText(chunkText)
		log.Printf("[Synthesize]   文本编码完成: %d tokens", len(textTokenIDs))

		// 将前一个 chunk 的尾部帧作为声学上下文追加到参考音频后面
		chunkPromptCodes := promptAudioCodes
		if chunkIndex > 0 && len(prevChunkTailFrames) > 0 {
			chunkPromptCodes = make([][]int, len(promptAudioCodes), len(promptAudioCodes)+len(prevChunkTailFrames))
			copy(chunkPromptCodes, promptAudioCodes)
			chunkPromptCodes = append(chunkPromptCodes, prevChunkTailFrames...)
			log.Printf("[Synthesize]   追加 %d 帧上下文帧到参考音频", len(prevChunkTailFrames))
		}

		requestRows := t.OrtRuntime.BuildVoiceCloneRequestRows(chunkPromptCodes, textTokenIDs)
		log.Printf("[Synthesize]   请求行构建完成：%d 行", len(requestRows["inputIds"]))

		// 根据文本 token 数估算最大帧数，避免无效 decode 步骤
		effectiveMaxFrames := estimateMaxNewFrames(len(textTokenIDs), maxNewFrames)
		generatedFrames := t.OrtRuntime.GenerateAudioFramesWithContextAndOverrides(ctx, requestRows, effectiveMaxFrames, overrides)
		if ctx.Err() != nil {
			interrupted = true
			if streamingCodecSession != nil {
				streamingCodecSession.Reset()
			}
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		}
		log.Printf("[Synthesize]   音频帧生成完成: %d 帧 (maxNewFrames=%d effective=%d)", len(generatedFrames), maxNewFrames, effectiveMaxFrames)

		// 保存当前 chunk 的最后几帧，作为下一个 chunk 的声学上下文
		if len(generatedFrames) >= ChunkContextFrames {
			prevChunkTailFrames = make([][]int, ChunkContextFrames)
			copy(prevChunkTailFrames, generatedFrames[len(generatedFrames)-ChunkContextFrames:])
		} else if len(generatedFrames) > 0 {
			prevChunkTailFrames = make([][]int, len(generatedFrames))
			copy(prevChunkTailFrames, generatedFrames)
		}

		// 使用流式 codec 解码替代一次性全量解码，降低内存峰值
		var channelArrays [][]float32
		var audioLength int
		if streamingCodecSession != nil {
			streamingCodecSession.Reset()
			channelArrays, audioLength = decodeFramesStreaming(streamingCodecSession, generatedFrames)
		} else {
			channelArrays, audioLength = t.OrtRuntime.DecodeFullAudioWithContext(ctx, generatedFrames)
		}
		if ctx.Err() != nil {
			interrupted = true
			if streamingCodecSession != nil {
				streamingCodecSession.Reset()
			}
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		}
		log.Printf("[Synthesize]   音频解码完成: channels=%d samples=%d", len(channelArrays), audioLength)
		if len(channelArrays) > 0 {
			merged := audio.MergeAudioChannels(channelArrays)
			allWaveforms = append(allWaveforms, merged)
		}
		if chunkIndex < len(textChunks)-1 {
			pauseSeconds := estimateInterChunkPauseSeconds(chunkText)
			pauseSamples := int(math.Round(float64(sampleRate) * pauseSeconds))
			if pauseSamples > 0 {
				allWaveforms = append(allWaveforms, audio.MakeSilence(pauseSamples, channels))
			}
		}

		logMemoryStats(fmt.Sprintf("chunk %d/%d 处理完成", chunkIndex+1, len(textChunks)))

		// 长文本多 chunk 间：先做轻量清理，内存超阈值时才 ForceResetSessions
		// 避免每个 chunk 间都重建 Session 的开销（约1-2秒/次）
		if chunkIndex < len(textChunks)-1 && len(textChunks) > 1 {
			runtime.GC()
			debug.FreeOSMemory()
			streamingCodecSession, _ = t.resetSessionsIfOverMemory(streamingCodecSession,
				fmt.Sprintf("chunk %d/%d", chunkIndex+1, len(textChunks)))
		}
	}
	// 最终清理 codec streaming session
	if streamingCodecSession != nil {
		streamingCodecSession.Reset()
	}
	waveform := audio.ConcatWaveforms(allWaveforms)

	// AudioData 延迟编码：子进程通过 attachment 传 Waveform bytes，AudioData 由调用方按需编码
	var resolvedOutputPath string
	if outputAudioPath != "" {
		resolvedOutputPath = outputAudioPath
		if err := audio.WriteWAV(resolvedOutputPath, waveform, channels, sampleRate); err != nil {
			return nil, fmt.Errorf("写入 WAV 文件失败: %w", err)
		}
	}

	// 释放中间数据，帮助 GC 回收大块内存
	allWaveforms = nil

	elapsed := time.Since(startTime).Seconds()
	audioSamples := len(waveform) / channels
	return &SynthesisResult{
		AudioPath:    resolvedOutputPath,
		SampleRate:   sampleRate,
		AudioSamples: audioSamples,
		Waveform:     waveform,
		Channels:     channels,
		TextChunks:   textChunks,
		SampleMode:   sampleMode,
		DoSample:     doSample,
		Streaming:    streaming,
		ElapsedSec:   elapsed,
		SeedUsed:     rngSeed,
	}, nil
}

func (t *OnnxTtsRuntime) SynthesizeStreamEx(ctx context.Context, text string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (<-chan StreamChunk, error) {
	// 增大channel buffer以避免阻塞音频生成
	chunkChan := make(chan StreamChunk, 64)

	go func() {
		defer close(chunkChan)

		t.OrtRuntime.AcquireSession()
		defer t.OrtRuntime.ReleaseSession()

		// 推理结束后（无论正常完成还是中断），执行内存清理
		interrupted := false
		defer func() {
			runtime.GC()
			debug.FreeOSMemory()
			if interrupted {
				// 中断时强制重置 Session，释放 ONNX 内存池
				t.OrtRuntime.ForceResetSessions()
				log.Printf("[SynthesizeStream] 推理被中断，Session 已强制重置，ONNX 内存池已释放")
			}
		}()

		if seed != nil {
			t.OrtRuntime.RNG = rand.New(rand.NewSource(int64(*seed)))
		}

		// 记录实际使用的种子
		var rngSeedUsed int64
		if seed != nil {
			rngSeedUsed = int64(*seed)
		} else {
			rngSeedUsed = time.Now().UnixNano()
			t.OrtRuntime.RNG = rand.New(rand.NewSource(rngSeedUsed))
		}

		// 构建采样参数覆盖
		overrides := t.buildGenerationOverrides(sampleMode, doSample)

		preparedText := t.PrepareSynthesisTextWithVoice(text, enableRobust, enableWeText, voice)
		promptAudioCodes := t.ResolvePromptAudioCodesWithPreload(voice, promptAudioPath, preloadId, preloadAudioPath)
		if promptAudioCodes == nil {
			promptAudioCodes = [][]int{}
		} else {
			promptAudioCodes = truncatePromptAudioCodes(promptAudioCodes, defaultMaxPromptAudioFrames)
		}
		textChunks := t.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens)

		codecMeta := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
		sampleRate := int(ortruntime.ToFloat64(codecMeta["sample_rate"]))
		channels := int(ortruntime.ToFloat64(codecMeta["channels"]))

		// 确保推理 sessions 存在（可能被 ForceResetSessions 销毁后尚未重建）
		if t.OrtRuntime.Onnx.Inference == nil || t.OrtRuntime.Onnx.Inference.CodecDecodeStep == nil {
			log.Printf("[SynthesizeStream] 推理 sessions 不存在，重新创建...")
			if err := t.OrtRuntime.CreateSessions(); err != nil {
				log.Printf("[SynthesizeStream] 重建推理 sessions 失败: %v", err)
				close(chunkChan)
				return
			}
		}

		// 注意：长文本多 chunk 时，chunk 间会 ForceResetSessions 重建 session，
		// 所以不能用 defer 重置 streamingSession，需手动管理
		streamingSession := ortruntime.NewCodecStreamingDecodeSession(t.OrtRuntime.CodecMeta, t.OrtRuntime.Onnx.Inference.CodecDecodeStep.Session, t.OrtRuntime)
		if streamingSession == nil {
			log.Printf("[SynthesizeStream] 错误: 无法创建流式解码会话")
			return
		}

		var prevChunkTailFrames [][]int // 前一个 chunk 的最后几帧，作为下一个 chunk 的声学上下文
		for chunkIndex, chunkText := range textChunks {
			select {
			case <-ctx.Done():
				interrupted = true
				if streamingSession != nil {
					streamingSession.Reset()
				}
				return
			default:
			}

			// 每个 chunk 重置 RNG 到相同种子，确保各 chunk 的随机值序列一致，减少音色差异
			t.OrtRuntime.RNG = rand.New(rand.NewSource(rngSeedUsed))

			textTokenIDs := t.EncodeText(chunkText)

			// 将前一个 chunk 的尾部帧作为声学上下文追加到参考音频后面
			chunkPromptCodes := promptAudioCodes
			if chunkIndex > 0 && len(prevChunkTailFrames) > 0 {
				chunkPromptCodes = make([][]int, len(promptAudioCodes), len(promptAudioCodes)+len(prevChunkTailFrames))
				copy(chunkPromptCodes, promptAudioCodes)
				chunkPromptCodes = append(chunkPromptCodes, prevChunkTailFrames...)
				log.Printf("[SynthesizeStream]   追加 %d 帧上下文帧到参考音频", len(prevChunkTailFrames))
			}

			requestRows := t.OrtRuntime.BuildVoiceCloneRequestRows(chunkPromptCodes, textTokenIDs)
			log.Printf("[SynthesizeStream] 处理 chunk %d/%d: maxNewFrames=%d", chunkIndex+1, len(textChunks), maxNewFrames)
			logMemoryStats(fmt.Sprintf("stream chunk %d/%d 开始", chunkIndex+1, len(textChunks)))

			pendingDecodeFrames := make([][]int, 0)
			emittedSamplesTotal := 0
			firstAudioEmittedAt := float64(0)
			hasEmittedAudio := false

			streamingSession.Reset()

			decodePending := func(force bool) {
				pendingCount := len(pendingDecodeFrames)
				if pendingCount <= 0 {
					return
				}
				decodeBudget := resolveStreamDecodeFrameBudget(emittedSamplesTotal, sampleRate, firstAudioEmittedAt, hasEmittedAudio)
				if !force && pendingCount < max(1, decodeBudget) {
					return
				}
				frameBudget := pendingCount
				if !force {
					frameBudget = minInt(pendingCount, max(1, decodeBudget))
				}
				frameChunk := pendingDecodeFrames[:frameBudget]
				pendingDecodeFrames = pendingDecodeFrames[frameBudget:]

				channelArrays, audioLength := streamingSession.RunFrames(frameChunk)
				if audioLength <= 0 {
					return
				}
				if !hasEmittedAudio {
					hasEmittedAudio = true
					firstAudioEmittedAt = float64(time.Now().UnixNano()) / 1e9
				}
				emittedSamplesTotal += audioLength

				merged := audio.MergeAudioChannels(channelArrays)
				isFirstChunk := chunkIndex == 0 && !hasEmittedAudio
				select {
				case <-ctx.Done():
					return
				case chunkChan <- StreamChunk{
					Waveform:   merged,
					SampleRate: sampleRate,
					Channels:   channels,
					ChunkIndex: chunkIndex,
					IsPause:    false,
					SeedUsed: func() int64 {
						if isFirstChunk {
							return rngSeedUsed
						}
						return 0
					}(),
				}:
				}
			}

			onFrame := func(generatedFrames [][]int, _ int, frame []int) {
				pendingDecodeFrames = append(pendingDecodeFrames, frame)
				decodePending(false)
				// 保存当前 chunk 的最后几帧，作为下一个 chunk 的声学上下文
				if len(generatedFrames) >= ChunkContextFrames {
					prevChunkTailFrames = make([][]int, ChunkContextFrames)
					copy(prevChunkTailFrames, generatedFrames[len(generatedFrames)-ChunkContextFrames:])
				} else if len(generatedFrames) > 0 {
					prevChunkTailFrames = make([][]int, len(generatedFrames))
					copy(prevChunkTailFrames, generatedFrames)
				}
			}

			// 根据文本 token 数估算最大帧数，避免无效 decode 步骤
			effectiveMaxFrames := estimateMaxNewFrames(len(textTokenIDs), maxNewFrames)
			_ = t.OrtRuntime.GenerateAudioFramesWithCallbackAndOverrides(ctx, requestRows, effectiveMaxFrames, onFrame, overrides)
			// 如果推理被中断，标记并退出
			select {
			case <-ctx.Done():
				interrupted = true
				return
			default:
			}
			decodePending(true)
			streamingSession.Reset()

			logMemoryStats(fmt.Sprintf("stream chunk %d/%d 处理完成", chunkIndex+1, len(textChunks)))

			if chunkIndex < len(textChunks)-1 {
				// 长文本多 chunk 间：先做轻量清理，内存超阈值时才 ForceResetSessions
				runtime.GC()
				debug.FreeOSMemory()
				streamingSession, _ = t.resetSessionsIfOverMemory(streamingSession,
					fmt.Sprintf("stream chunk %d/%d", chunkIndex+1, len(textChunks)))

				pauseSeconds := estimateInterChunkPauseSeconds(chunkText)
				pauseSamples := int(math.Round(float64(sampleRate) * pauseSeconds))
				if pauseSamples > 0 {
					pauseWaveform := audio.MakeSilence(pauseSamples, channels)
					select {
					case <-ctx.Done():
						interrupted = true
						return
					case chunkChan <- StreamChunk{
						Waveform:   pauseWaveform,
						SampleRate: sampleRate,
						Channels:   channels,
						ChunkIndex: chunkIndex,
						IsPause:    true,
					}:
					}
				}
			}
		}
		// 最终清理 streaming session
		if streamingSession != nil {
			streamingSession.Reset()
		}
	}()

	return chunkChan, nil
}

func (t *OnnxTtsRuntime) SynthesizeStream(ctx context.Context, text string, voice string, promptAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (<-chan StreamChunk, error) {
	return t.SynthesizeStreamEx(ctx, text, voice, promptAudioPath, "", "", sampleMode, doSample, maxNewFrames, voiceCloneMaxTextTokens, true, true, seed)
}

func resolveStreamDecodeFrameBudget(emittedSamplesTotal, sampleRate int, firstAudioEmittedAt float64, hasEmittedAudio bool) int {
	if !hasEmittedAudio || sampleRate <= 0 {
		// 首次解码时立即返回较大的批次，尽快发出首个音频
		return 4
	}
	elapsedSeconds := math.Max(0.0, float64(time.Now().UnixNano())/1e9-firstAudioEmittedAt)
	emittedSeconds := float64(emittedSamplesTotal) / float64(sampleRate)
	leadSeconds := emittedSeconds - elapsedSeconds
	// 优化：增加批处理大小，加快流式返回速度
	if leadSeconds < 0.10 {
		return 4
	}
	if leadSeconds < 0.30 {
		return 8
	}
	if leadSeconds < 0.80 {
		return 16
	}
	return 32
}

func (t *OnnxTtsRuntime) Close() {
	t.OrtRuntime.Close()
}

func prepareTextForSentenceChunking(text string) string {
	normalizedText := strings.TrimSpace(text)
	if normalizedText == "" {
		return ""
	}
	normalizedText = strings.ReplaceAll(normalizedText, "\r", " ")
	normalizedText = strings.ReplaceAll(normalizedText, "\n", " ")
	for strings.Contains(normalizedText, "  ") {
		normalizedText = strings.ReplaceAll(normalizedText, "  ", " ")
	}
	if normalizer.ContainsCJK(normalizedText) {
		lastRune := []rune(normalizedText)
		if len(lastRune) > 0 && !SentenceEndPunctuation[lastRune[len(lastRune)-1]] {
			normalizedText += "。"
		}
		return normalizedText
	}
	runes := []rune(normalizedText)
	if len(runes) > 0 && runes[0] >= 'a' && runes[0] <= 'z' {
		runes[0] = runes[0] - 'a' + 'A'
		normalizedText = string(runes)
	}
	if len(runes) > 0 {
		lastCh := runes[len(runes)-1]
		if (lastCh >= 'a' && lastCh <= 'z') || (lastCh >= 'A' && lastCh <= 'Z') || (lastCh >= '0' && lastCh <= '9') {
			normalizedText += "."
		}
	}
	words := strings.Fields(normalizedText)
	if len(words) < 5 {
		normalizedText = "        " + normalizedText
	}
	return normalizedText
}

func splitTextByPunctuation(text string, punctuation map[rune]bool) []string {
	var sentences []string
	var currentChars []rune
	runes := []rune(text)
	index := 0
	for index < len(runes) {
		ch := runes[index]
		currentChars = append(currentChars, ch)
		if punctuation[ch] {
			lookahead := index + 1
			for lookahead < len(runes) && ClosingPunctuation[runes[lookahead]] {
				currentChars = append(currentChars, runes[lookahead])
				lookahead++
			}
			sentence := strings.TrimSpace(string(currentChars))
			if sentence != "" {
				sentences = append(sentences, sentence)
			}
			currentChars = currentChars[:0]
			for lookahead < len(runes) && (runes[lookahead] == ' ' || runes[lookahead] == '\t') {
				lookahead++
			}
			index = lookahead
			continue
		}
		index++
	}
	tail := strings.TrimSpace(string(currentChars))
	if tail != "" {
		sentences = append(sentences, tail)
	}
	return sentences
}

func joinSentenceParts(left, right string) string {
	if left == "" {
		return right
	}
	if right == "" {
		return left
	}
	if normalizer.ContainsCJK(left) || normalizer.ContainsCJK(right) {
		return left + right
	}
	return left + " " + right
}

func estimateInterChunkPauseSeconds(textChunk string) float64 {
	wordCount := len(strings.Fields(textChunk))
	if wordCount <= 4 {
		return DefaultInterChunkPauseShortSec
	}
	return DefaultInterChunkPauseLongSec
}

// decodeFramesStreaming 使用流式 codec 解码，分批处理帧以降低内存峰值
func decodeFramesStreaming(session *ortruntime.CodecStreamingDecodeSession, frames [][]int) ([][]float32, int) {
	if len(frames) == 0 {
		return nil, 0
	}
	// 与 Python 实现一致：使用 8 帧一批进行流式解码
	// 批次过大会导致流式解码的注意力缓存不一致，影响音质
	const batchSize = 8

	// 先用第一批获取通道数
	firstBatch := frames[:min(batchSize, len(frames))]
	firstChannels, firstLength := session.RunFrames(firstBatch)
	if firstLength <= 0 || len(firstChannels) == 0 {
		return nil, 0
	}
	numChannels := len(firstChannels)

	// 按通道收集所有波形
	perChannelWaveforms := make([][][]float32, numChannels) // perChannelWaveforms[ch] = list of waveforms for channel ch
	for ch, chData := range firstChannels {
		perChannelWaveforms[ch] = [][]float32{chData}
	}
	totalSamples := firstLength

	// 处理剩余批次
	for i := batchSize; i < len(frames); i += batchSize {
		end := i + batchSize
		if end > len(frames) {
			end = len(frames)
		}
		batch := frames[i:end]
		channelArrays, audioLength := session.RunFrames(batch)
		if audioLength > 0 && len(channelArrays) > 0 {
			for ch := 0; ch < numChannels && ch < len(channelArrays); ch++ {
				perChannelWaveforms[ch] = append(perChannelWaveforms[ch], channelArrays[ch])
			}
			totalSamples += audioLength
		}
	}

	// 合并每个通道的波形
	result := make([][]float32, numChannels)
	for ch := 0; ch < numChannels; ch++ {
		result[ch] = audio.ConcatWaveforms(perChannelWaveforms[ch])
	}
	return result, totalSamples
}

// estimateMaxNewFrames 根据文本 token 数估算合理的最大生成帧数
// 经验值：每个 token 约生成 3-5 帧，下限 50 帧，上限为用户指定的 maxNewFrames
func estimateMaxNewFrames(tokenCount int, maxNewFrames int) int {
	estimated := tokenCount * 5
	if estimated < 50 {
		estimated = 50
	}
	if estimated > maxNewFrames {
		estimated = maxNewFrames
	}
	return estimated
}

// getProcessRSSMB 获取当前进程的实际物理内存占用（RSS），单位 MB
// runtime.MemStats.Alloc 只统计 Go 堆内存，不包括 ONNX Runtime 等 C++ 库的内存分配
// 因此必须使用操作系统级 RSS 来做内存阈值判断
func getProcessRSSMB() float64 {
	pid := os.Getpid()

	// Linux: 从 /proc/self/status 读取 VmRSS
	if data, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "VmRSS:") {
				// VmRSS:    1234567 kB
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					if kb, err := strconv.ParseFloat(parts[1], 64); err == nil {
						return kb / 1024 // kB → MB
					}
				}
			}
		}
	}

	// macOS / 其他 Unix: 使用 ps 命令
	cmd := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid))
	if output, err := cmd.Output(); err == nil {
		output = bytes.TrimSpace(output)
		if kb, err := strconv.ParseFloat(string(output), 64); err == nil {
			return kb / 1024 // kB → MB
		}
	}

	// Windows: 使用 tasklist 命令
	cmd2 := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/FO", "CSV", "/NH")
	if output, err := cmd2.Output(); err == nil {
		// 输出格式: "moss-tts-nano-onnx-go.exe","12345","Services","0","1,623,456 K"
		fields := strings.Split(string(output), ",")
		if len(fields) >= 5 {
			memStr := strings.TrimSpace(strings.Trim(fields[len(fields)-1], `" K\r\n`))
			memStr = strings.ReplaceAll(memStr, ",", "")
			if kb, err := strconv.ParseFloat(memStr, 64); err == nil {
				return kb / 1024
			}
		}
	}

	// 兜底：使用 Go 的 Sys 作为粗略估计
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Sys) / 1024 / 1024
}

// logMemoryStats 记录当前进程内存使用情况，帮助诊断内存增长
// 返回当前 RSS 字节数，供内存阈值判断使用
func logMemoryStats(label string) uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	rssMB := getProcessRSSMB()
	log.Printf("[Memory] %s: RSS=%.1fMB Alloc=%.1fMB Sys=%.1fMB HeapAlloc=%.1fMB",
		label, rssMB, float64(m.Alloc)/1024/1024, float64(m.Sys)/1024/1024,
		float64(m.HeapAlloc)/1024/1024)
	// 返回 RSS 字节数
	return uint64(rssMB * 1024 * 1024)
}

// resetSessionsIfOverMemory 检查当前内存（使用进程RSS），超过阈值时执行 ForceResetSessions
// 返回是否执行了重置，以及重建后的 codec streaming session
func (t *OnnxTtsRuntime) resetSessionsIfOverMemory(streamingCodecSession *ortruntime.CodecStreamingDecodeSession, chunkLabel string) (*ortruntime.CodecStreamingDecodeSession, bool) {
	rssMB := getProcessRSSMB()
	threshold := t.MemoryThresholdMB
	if threshold <= 0 {
		threshold = 800
	}
	if rssMB <= float64(threshold) {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		log.Printf("[Memory] %s: RSS=%.1fMB (阈值=%dMB), Alloc=%.1fMB, 跳过Session重置",
			chunkLabel, rssMB, threshold, float64(m.Alloc)/1024/1024)
		return streamingCodecSession, false
	}
	log.Printf("[Memory] %s: RSS=%.1fMB > 阈值%dMB, 执行ForceResetSessions", chunkLabel, rssMB, threshold)
	// 1. 重置 codec streaming 状态
	if streamingCodecSession != nil {
		streamingCodecSession.Reset()
	}
	// 2. Go 侧内存释放
	runtime.GC()
	debug.FreeOSMemory()
	// 3. 销毁并重建 ONNX Session，释放 C++ 内存池
	if err := t.OrtRuntime.ForceResetSessions(); err != nil {
		log.Printf("[Memory] 警告: ForceResetSessions失败: %v", err)
		return streamingCodecSession, false
	}
	// 4. 用新 Session 重建 codec streaming session
	newSession := ortruntime.NewCodecStreamingDecodeSession(t.OrtRuntime.CodecMeta, t.OrtRuntime.Onnx.Inference.CodecDecodeStep.Session, t.OrtRuntime)
	logMemoryStats(chunkLabel + " ForceResetSessions后")
	return newSession, true
}

// truncatePromptAudioCodes 截断过长的参考音频帧，只保留前 maxFrames 帧
// 与 Python 端对齐：Python 不做主动截断，仅在 ResolvePromptAudioCodes 中有 300 帧 OOM 保护
// 保留此函数作为安全上限，但默认值从 20 提高到 300（与 ResolvePromptAudioCodes 一致）
const defaultMaxPromptAudioFrames = 300

func truncatePromptAudioCodes(codes [][]int, maxFrames int) [][]int {
	if maxFrames <= 0 {
		maxFrames = defaultMaxPromptAudioFrames
	}
	if len(codes) <= maxFrames {
		return codes
	}
	log.Printf("[TruncatePromptAudio] 参考音频从 %d 帧截断为 %d 帧", len(codes), maxFrames)
	return codes[:maxFrames]
}

func ToFloat64(v interface{}) float64 {
	return ortruntime.ToFloat64(v)
}

func getInt32Data(v ort.Value) []int32 {
	if v == nil {
		return nil
	}
	if t, ok := v.(*ort.Tensor[int32]); ok {
		return t.GetData()
	}
	if c, ok := v.(*ort.CustomDataTensor); ok {
		rawData := c.GetData()
		result := make([]int32, len(rawData)/4)
		for i := range result {
			result[i] = int32(binary.LittleEndian.Uint32(rawData[i*4:]))
		}
		return result
	}
	return nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var _ = json.Unmarshal
