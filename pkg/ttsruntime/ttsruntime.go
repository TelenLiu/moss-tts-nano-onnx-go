package ttsruntime

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/audio"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/normalizer"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/tokenizer"
)

const (
	DefaultInterChunkPauseShortSec = 0.40
	DefaultInterChunkPauseLongSec  = 0.24
)

var SentenceEndPunctuation = map[rune]bool{'.': true, '!': true, '?': true, '。': true, '！': true, '？': true, '；': true, ';': true}
var ClauseSplitPunctuation = map[rune]bool{',': true, '，': true, '、': true, '；': true, ';': true, '：': true, ':': true}
var ClosingPunctuation = map[rune]bool{'"': true, '\'': true, '\u201d': true, '\u2019': true, ')': true, ']': true, '}': true, '）': true, '】': true, '》': true, '」': true, '』': true}

type SynthesisResult struct {
	AudioPath    string
	SampleRate   int
	Waveform     []float32
	Channels     int
	TextChunks   []string
	SampleMode   string
	DoSample     bool
	Streaming    bool
	ElapsedSec   float64
}

type OnnxTtsRuntime struct {
	OrtRuntime  *ortruntime.OrtCpuRuntime
	SPModel     *tokenizer.Processor
	OutputDir   string
}

func NewOnnxTtsRuntime(modelDir string, threadCount int, maxNewFrames *int, doSample *bool, sampleMode *string, outputDir string) (*OnnxTtsRuntime, error) {
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
	absOutputDir, _ := filepath.Abs(outputDir)
	os.MkdirAll(absOutputDir, 0755)
	return &OnnxTtsRuntime{
		OrtRuntime: rt,
		SPModel:    sp,
		OutputDir:  absOutputDir,
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
	if promptAudioPath != "" {
		return t.EncodeReferenceAudio(promptAudioPath)
	}
	resolvedVoice := voice
	if resolvedVoice == "" {
		voices := t.OrtRuntime.ListBuiltinVoices()
		if len(voices) > 0 {
			resolvedVoice = fmt.Sprintf("%v", voices[0]["voice"])
		}
	}
	for _, v := range t.OrtRuntime.ListBuiltinVoices() {
		if fmt.Sprintf("%v", v["voice"]) == resolvedVoice {
			codes, ok := v["prompt_audio_codes"].([]interface{})
			if !ok {
				continue
			}
			result := make([][]int, len(codes))
			for i, codeRow := range codes {
				row, ok := codeRow.([]interface{})
				if !ok {
					continue
				}
				result[i] = make([]int, len(row))
				for j, val := range row {
					result[i][j] = int(ortruntime.ToFloat64(val))
				}
			}
			return result
		}
	}
	log.Printf("警告: 未找到内置音色 %s，使用默认", resolvedVoice)
	voices := t.OrtRuntime.ListBuiltinVoices()
	if len(voices) > 0 {
		codes, _ := voices[0]["prompt_audio_codes"].([]interface{})
		result := make([][]int, len(codes))
		for i, codeRow := range codes {
			row, _ := codeRow.([]interface{})
			result[i] = make([]int, len(row))
			for j, val := range row {
				result[i][j] = int(ortruntime.ToFloat64(val))
			}
		}
		return result
	}
	return nil
}

func (t *OnnxTtsRuntime) EncodeReferenceAudio(audioPath string) [][]int {
	log.Printf("编码参考音频: %s (stub - 需要 ONNX codec_encode session)", audioPath)
	return [][]int{{0, 0, 0, 0}}
}

func (t *OnnxTtsRuntime) Synthesize(text string, voice string, promptAudioPath string, outputAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableNormalize bool, seed *int) (*SynthesisResult, error) {
	startTime := time.Now()
	if seed != nil {
		t.OrtRuntime.RNG = rand.New(rand.NewSource(int64(*seed)))
	}
	preparedText := t.PrepareSynthesisText(text, enableNormalize)
	promptAudioCodes := t.ResolvePromptAudioCodes(voice, promptAudioPath)
	textChunks := t.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens)
	codecMeta := t.OrtRuntime.CodecMeta["codec_config"].(map[string]interface{})
	sampleRate := int(ortruntime.ToFloat64(codecMeta["sample_rate"]))
	channels := int(ortruntime.ToFloat64(codecMeta["channels"]))
	var allWaveforms [][]float32
	for chunkIndex, chunkText := range textChunks {
		textTokenIDs := t.EncodeText(chunkText)
		requestRows := t.OrtRuntime.BuildVoiceCloneRequestRows(promptAudioCodes, textTokenIDs)
		generatedFrames := t.OrtRuntime.GenerateAudioFrames(requestRows)
		channelArrays, audioLength := t.OrtRuntime.DecodeFullAudio(generatedFrames)
		log.Printf("  chunk %d/%d: text=%q frames=%d audio_samples=%d", chunkIndex+1, len(textChunks), chunkText, len(generatedFrames), audioLength)
		if len(channelArrays) > 0 {
			for _, ch := range channelArrays {
				allWaveforms = append(allWaveforms, ch)
			}
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
	resolvedOutputPath := outputAudioPath
	if resolvedOutputPath == "" {
		resolvedOutputPath = filepath.Join(t.OutputDir, "infer_onnx_output.wav")
	}
	if err := audio.WriteWAV(resolvedOutputPath, waveform, channels, sampleRate); err != nil {
		return nil, fmt.Errorf("写入 WAV 文件失败: %w", err)
	}
	elapsed := time.Since(startTime).Seconds()
	return &SynthesisResult{
		AudioPath:  resolvedOutputPath,
		SampleRate: sampleRate,
		Waveform:   waveform,
		Channels:   channels,
		TextChunks: textChunks,
		SampleMode: sampleMode,
		DoSample:   doSample,
		Streaming:  streaming,
		ElapsedSec: elapsed,
	}, nil
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

var _ = json.Unmarshal
