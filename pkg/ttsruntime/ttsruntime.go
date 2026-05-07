package ttsruntime

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"

	"runtime"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/audio"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/normalizer"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	DefaultInterChunkPauseShortSec = 0.15
	DefaultInterChunkPauseLongSec  = 0.10
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
}

type StreamChunk struct {
	Waveform   []float32
	SampleRate int
	Channels   int
	ChunkIndex int
	IsPause    bool
}

type OnnxTtsRuntime struct {
	OrtRuntime *ortruntime.OrtCpuRuntime
	SPModel    *tokenizer.Processor
}

func NewOnnxTtsRuntime(modelDir string, threadCount int, maxNewFrames *int, doSample *bool, sampleMode *string) (*OnnxTtsRuntime, error) {
	rt, err := ortruntime.NewOrtCpuRuntime(modelDir, threadCount, maxNewFrames, doSample, sampleMode)
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
	return &OnnxTtsRuntime{
		OrtRuntime: rt,
		SPModel:    sp,
	}, nil
}

func (t *OnnxTtsRuntime) EncodeText(text string) []int {
	return t.SPModel.EncodeAsIDs(text)
}

func (t *OnnxTtsRuntime) CountTextTokens(text string) int {
	return t.SPModel.CountTokens(text)
}

func (t *OnnxTtsRuntime) PrepareSynthesisText(text string, enableNormalize bool) string {
	if enableNormalize {
		return normalizer.NormalizeTTSText(text)
	}
	return text
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
	log.Printf("[ResolvePromptAudioCodes] voice=%q promptAudioPath=%q", voice, promptAudioPath)
	if promptAudioPath != "" {
		log.Printf("[ResolvePromptAudioCodes] 使用上传的参考音频: %s", promptAudioPath)
		codes := t.EncodeReferenceAudio(promptAudioPath)
		if codes != nil {
			log.Printf("[ResolvePromptAudioCodes] 参考音频编码成功: %d 帧", len(codes))
		} else {
			log.Printf("[ResolvePromptAudioCodes] 参考音频编码失败，将回退到内置音色")
		}
		if codes != nil {
			return codes
		}
	}
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
			result := make([][]int, len(codes))
			allOk := true
			for i, codeRow := range codes {
				row, ok := codeRow.([]interface{})
				if !ok {
					log.Printf("[ResolvePromptAudioCodes] 警告: 音色 %s 的第%d行格式不正确", resolvedVoice, i)
					allOk = false
					break
				}
				result[i] = make([]int, len(row))
				for j, val := range row {
					result[i][j] = int(ortruntime.ToFloat64(val))
				}
			}
			if allOk {
				log.Printf("[ResolvePromptAudioCodes] 内置音色 %s 加载成功: %d 帧", resolvedVoice, len(result))
				return result
			}
			log.Printf("[ResolvePromptAudioCodes] 警告: 音色 %s 加载失败，继续查找", resolvedVoice)
		}
	}
	log.Printf("[ResolvePromptAudioCodes] 警告: 未找到内置音色 %s，使用第一个内置音色作为默认", resolvedVoice)
	voices := t.OrtRuntime.ListBuiltinVoices()
	if len(voices) > 0 {
		for _, v := range voices {
			codes, ok := v["prompt_audio_codes"].([]interface{})
			if !ok {
				continue
			}
			result := make([][]int, len(codes))
			allOk := true
			for i, codeRow := range codes {
				row, ok := codeRow.([]interface{})
				if !ok {
					allOk = false
					break
				}
				result[i] = make([]int, len(row))
				for j, val := range row {
					result[i][j] = int(ortruntime.ToFloat64(val))
				}
			}
			if allOk {
				log.Printf("[ResolvePromptAudioCodes] 使用第一个可用音色: %s (%d 帧)", fmt.Sprintf("%v", v["voice"]), len(result))
				return result
			}
		}
	}
	log.Printf("[ResolvePromptAudioCodes] 错误: 没有可用的内置音色，返回空")
	return nil
}

func (t *OnnxTtsRuntime) EncodeReferenceAudio(audioPath string) [][]int {
	log.Printf("[EncodeReferenceAudio] 开始编码参考音频: %s", audioPath)

	codecConfig := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
	targetSampleRate := int(ortruntime.ToFloat64(codecConfig["sample_rate"]))
	targetChannels := int(ortruntime.ToFloat64(codecConfig["channels"]))
	numQuantizers := int(ortruntime.ToFloat64(codecConfig["num_quantizers"]))
	log.Printf("[EncodeReferenceAudio] codec配置: sampleRate=%d channels=%d numQuantizers=%d", targetSampleRate, targetChannels, numQuantizers)

	waveform, channels, sampleRate, err := audio.LoadReferenceAudio(audioPath, targetSampleRate, targetChannels)
	if err != nil {
		log.Printf("[EncodeReferenceAudio] 加载参考音频失败: %v, 使用内置音色", err)
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

	log.Printf("[EncodeReferenceAudio] 调用 codec_encode...")
	err = t.OrtRuntime.Onnx.CodecEncode.Session.Run(inputs, outputs)
	waveformTensor.Destroy()
	inputLengthsTensor.Destroy()

	if err != nil {
		log.Printf("[EncodeReferenceAudio] codec_encode 失败: %v, 使用内置音色", err)
		return nil
	}

	audioCodesData := getInt32Data(outputs[0])
	audioCodesShape := outputs[0].GetShape()
	audioCodeLengthsData := getInt32Data(outputs[1])
	log.Printf("[EncodeReferenceAudio] codec_encode 成功: audioCodes形状=%v audioCodeLengths=%v 数据长度=%d",
		audioCodesShape, audioCodeLengthsData, len(audioCodesData))

	for _, v := range outputs {
		if v != nil {
			v.Destroy()
		}
	}

	codeLength := int(audioCodeLengthsData[0])
	if len(audioCodesShape) >= 3 {
		codeLength = minInt(codeLength, int(audioCodesShape[1]))
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
	return promptAudioCodes
}

func (t *OnnxTtsRuntime) Synthesize(text string, voice string, promptAudioPath string, outputAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (*SynthesisResult, error) {
	return t.SynthesizeWithContext(context.Background(), text, voice, promptAudioPath, outputAudioPath, sampleMode, doSample, streaming, maxNewFrames, voiceCloneMaxTextTokens, enableNormalize, seed)
}

func (t *OnnxTtsRuntime) SynthesizeWithContext(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (*SynthesisResult, error) {
	startTime := time.Now()

	// 检查并释放空闲超时的 Session
	t.OrtRuntime.CheckAndReleaseIdleSessions()

	log.Printf("[Synthesize] 开始合成：text=%q voice=%q promptAudioPath=%q sampleMode=%s doSample=%v maxNewFrames=%d", text, voice, promptAudioPath, sampleMode, doSample, maxNewFrames)
	if seed != nil {
		t.OrtRuntime.RNG = rand.New(rand.NewSource(int64(*seed)))
		log.Printf("[Synthesize] 使用随机种子: %d", *seed)
	}
	preparedText := t.PrepareSynthesisText(text, enableNormalize)
	log.Printf("[Synthesize] 文本预处理完成: 原始长度=%d 预处理后长度=%d", len(text), len(preparedText))
	promptAudioCodes := t.ResolvePromptAudioCodes(voice, promptAudioPath)
	if promptAudioCodes == nil {
		log.Printf("[Synthesize] 警告: promptAudioCodes 为 nil，将使用空列表")
		promptAudioCodes = [][]int{}
	} else {
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
	var allWaveforms [][]float32
	for chunkIndex, chunkText := range textChunks {
		select {
		case <-ctx.Done():
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		default:
		}
		log.Printf("[Synthesize] 处理 chunk %d/%d...", chunkIndex+1, len(textChunks))
		textTokenIDs := t.EncodeText(chunkText)
		log.Printf("[Synthesize]   文本编码完成: %d tokens", len(textTokenIDs))
		requestRows := t.OrtRuntime.BuildVoiceCloneRequestRows(promptAudioCodes, textTokenIDs)
		log.Printf("[Synthesize]   请求行构建完成: %d 行", len(requestRows["inputIds"]))
		generatedFrames := t.OrtRuntime.GenerateAudioFramesWithContext(ctx, requestRows)
		if ctx.Err() != nil {
			log.Printf("[Synthesize] 合成被取消")
			return nil, ctx.Err()
		}
		log.Printf("[Synthesize]   音频帧生成完成: %d 帧", len(generatedFrames))
		channelArrays, audioLength := t.OrtRuntime.DecodeFullAudioWithContext(ctx, generatedFrames)
		if ctx.Err() != nil {
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
	}
	waveform := audio.ConcatWaveforms(allWaveforms)

	var audioData []byte
	var resolvedOutputPath string
	if outputAudioPath != "" {
		resolvedOutputPath = outputAudioPath
		if err := audio.WriteWAV(resolvedOutputPath, waveform, channels, sampleRate); err != nil {
			return nil, fmt.Errorf("写入 WAV 文件失败: %w", err)
		}
	}
	audioData, err := audio.EncodeWAV(waveform, channels, sampleRate)
	if err != nil {
		return nil, fmt.Errorf("编码 WAV 失败: %w", err)
	}

	// 强制 GC，释放内存
	runtime.GC()

	// 销毁所有 Session，强制 ONNX Runtime 释放内存池
	// 这是防止内存持续增长的关键步骤
	t.OrtRuntime.ResetSessions()
	log.Printf("[Synthesize] Session 已重置，内存已释放")

	elapsed := time.Since(startTime).Seconds()
	audioSamples := len(waveform) / channels
	return &SynthesisResult{
		AudioPath:    resolvedOutputPath,
		AudioData:    audioData,
		SampleRate:   sampleRate,
		AudioSamples: audioSamples,
		Waveform:     waveform,
		Channels:     channels,
		TextChunks:   textChunks,
		SampleMode:   sampleMode,
		DoSample:     doSample,
		Streaming:    streaming,
		ElapsedSec:   elapsed,
	}, nil
}

func (t *OnnxTtsRuntime) SynthesizeStream(ctx context.Context, text string, voice string, promptAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (<-chan StreamChunk, error) {
	chunkChan := make(chan StreamChunk, 16)

	go func() {
		defer close(chunkChan)

		if seed != nil {
			t.OrtRuntime.RNG = rand.New(rand.NewSource(int64(*seed)))
		}

		preparedText := t.PrepareSynthesisText(text, enableNormalize)
		promptAudioCodes := t.ResolvePromptAudioCodes(voice, promptAudioPath)
		if promptAudioCodes == nil {
			promptAudioCodes = [][]int{}
		}
		textChunks := t.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens)

		codecMeta := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
		sampleRate := int(ortruntime.ToFloat64(codecMeta["sample_rate"]))
		channels := int(ortruntime.ToFloat64(codecMeta["channels"]))

		streamingSession := ortruntime.NewCodecStreamingDecodeSession(t.OrtRuntime.CodecMeta, t.OrtRuntime.Onnx.CodecDecodeStep.Session)
		if streamingSession == nil {
			log.Printf("[SynthesizeStream] 错误: 无法创建流式解码会话")
			return
		}
		defer streamingSession.Reset()

		for chunkIndex, chunkText := range textChunks {
			select {
			case <-ctx.Done():
				return
			default:
			}

			textTokenIDs := t.EncodeText(chunkText)
			requestRows := t.OrtRuntime.BuildVoiceCloneRequestRows(promptAudioCodes, textTokenIDs)

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
				select {
				case <-ctx.Done():
					return
				case chunkChan <- StreamChunk{
					Waveform:   merged,
					SampleRate: sampleRate,
					Channels:   channels,
					ChunkIndex: chunkIndex,
					IsPause:    false,
				}:
				}
			}

			onFrame := func(_ [][]int, _ int, frame []int) {
				pendingDecodeFrames = append(pendingDecodeFrames, frame)
				decodePending(false)
			}

			_ = t.OrtRuntime.GenerateAudioFramesWithCallback(ctx, requestRows, onFrame)
			decodePending(true)
			streamingSession.Reset()

			if chunkIndex < len(textChunks)-1 {
				pauseSeconds := estimateInterChunkPauseSeconds(chunkText)
				pauseSamples := int(math.Round(float64(sampleRate) * pauseSeconds))
				if pauseSamples > 0 {
					pauseWaveform := audio.MakeSilence(pauseSamples, channels)
					select {
					case <-ctx.Done():
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
	}()

	return chunkChan, nil
}

func resolveStreamDecodeFrameBudget(emittedSamplesTotal, sampleRate int, firstAudioEmittedAt float64, hasEmittedAudio bool) int {
	if !hasEmittedAudio || sampleRate <= 0 {
		return 1
	}
	elapsedSeconds := math.Max(0.0, float64(time.Now().UnixNano())/1e9-firstAudioEmittedAt)
	emittedSeconds := float64(emittedSamplesTotal) / float64(sampleRate)
	leadSeconds := emittedSeconds - elapsedSeconds
	if leadSeconds < 0.20 {
		return 1
	}
	if leadSeconds < 0.55 {
		return 2
	}
	if leadSeconds < 1.10 {
		return 4
	}
	return 8
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
