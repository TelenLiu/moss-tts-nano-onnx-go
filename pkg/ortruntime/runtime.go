package ortruntime

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/sampler"
	ort "github.com/yalue/onnxruntime_go"
)

var ManifestCandidateRelativePaths = []string{
	"browser_poc_manifest.json",
	"MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
}

var ModelDirAliasMap = map[string]string{
	"MOSS-TTS-Nano-ONNX-CPU":             "MOSS-TTS-Nano-100M-ONNX",
	"MOSS-Audio-Tokenizer-Nano-ONNX-CPU": "MOSS-Audio-Tokenizer-Nano-ONNX",
}

// TimedSession 带时间戳的 Session 包装
type TimedSession struct {
	Session  *ort.DynamicAdvancedSession
	LastUsed time.Time
}

type OnnxSessions struct {
	Prefill                *TimedSession
	Decode                 *TimedSession
	LocalDecoder           *TimedSession
	LocalCachedStep        *TimedSession
	LocalFixedSampledFrame *TimedSession
	CodecEncode            *TimedSession
	CodecDecode            *TimedSession
	CodecDecodeStep        *TimedSession
}

type CodecStreamingDecodeSession struct {
	CodecMeta        map[string]interface{}
	Session          *ort.DynamicAdvancedSession
	TransformerSpecs []map[string]interface{}
	AttentionSpecs   []map[string]interface{}
	StateFeeds       map[string]ort.Value
}

func NewCodecStreamingDecodeSession(codecMeta map[string]interface{}, session *ort.DynamicAdvancedSession) *CodecStreamingDecodeSession {
	s := &CodecStreamingDecodeSession{
		CodecMeta:  codecMeta,
		Session:    session,
		StateFeeds: make(map[string]ort.Value),
	}
	if streamingDecode, ok := codecMeta["streaming_decode"].(map[string]interface{}); ok {
		if toffsets, ok := streamingDecode["transformer_offsets"].([]interface{}); ok {
			for _, spec := range toffsets {
				if m, ok := spec.(map[string]interface{}); ok {
					s.TransformerSpecs = append(s.TransformerSpecs, m)
				}
			}
		}
		if acaches, ok := streamingDecode["attention_caches"].([]interface{}); ok {
			for _, spec := range acaches {
				if m, ok := spec.(map[string]interface{}); ok {
					s.AttentionSpecs = append(s.AttentionSpecs, m)
				}
			}
		}
	}
	s.Reset()
	return s
}

func (s *CodecStreamingDecodeSession) Reset() {
	for _, v := range s.StateFeeds {
		if v != nil {
			v.Destroy()
		}
	}
	s.StateFeeds = make(map[string]ort.Value)
	for _, spec := range s.TransformerSpecs {
		shape := toInt64Slice(spec["shape"])
		data := make([]int32, totalDimFromShape(shape))
		t, _ := ort.NewTensor(shape, data)
		s.StateFeeds[fmt.Sprintf("%v", spec["input_name"])] = t
	}
	for _, spec := range s.AttentionSpecs {
		offsetShape := toInt64Slice(spec["offset_shape"])
		cacheShape := toInt64Slice(spec["cache_shape"])
		positionsShape := toInt64Slice(spec["positions_shape"])
		offsetData := make([]int32, totalDimFromShape(offsetShape))
		cacheData := make([]float32, totalDimFromShape(cacheShape))
		positionsData := make([]int32, totalDimFromShape(positionsShape))
		for i := range positionsData {
			positionsData[i] = -1
		}
		offsetT, _ := ort.NewTensor(offsetShape, offsetData)
		cacheKeysT, _ := ort.NewTensor(cacheShape, cacheData)
		cacheValuesT, _ := ort.NewTensor(cacheShape, cacheData)
		positionsT, _ := ort.NewTensor(positionsShape, positionsData)
		s.StateFeeds[fmt.Sprintf("%v", spec["offset_input_name"])] = offsetT
		s.StateFeeds[fmt.Sprintf("%v", spec["cached_keys_input_name"])] = cacheKeysT
		s.StateFeeds[fmt.Sprintf("%v", spec["cached_values_input_name"])] = cacheValuesT
		s.StateFeeds[fmt.Sprintf("%v", spec["cached_positions_input_name"])] = positionsT
	}
}

func (s *CodecStreamingDecodeSession) RunFrames(frameRows [][]int) ([][]float32, int) {
	if len(frameRows) == 0 || s.Session == nil {
		return nil, 0
	}
	codecConfig := s.CodecMeta["codec_config"].(map[string]interface{})
	numQuantizers := int(toFloat64(codecConfig["num_quantizers"]))
	frameCount := len(frameRows)

	audioCodes := make([]int32, frameCount*numQuantizers)
	for fi, frame := range frameRows {
		for ci := 0; ci < numQuantizers; ci++ {
			val := int32(0)
			if ci < len(frame) {
				val = int32(frame[ci])
			}
			audioCodes[fi*numQuantizers+ci] = val
		}
	}

	audioCodesShape := []int64{1, int64(frameCount), int64(numQuantizers)}
	audioCodesTensor, _ := ort.NewTensor(audioCodesShape, audioCodes)
	audioCodeLengths := []int32{int32(frameCount)}
	audioCodeLengthsTensor, _ := ort.NewTensor([]int64{1}, audioCodeLengths)

	inputs := []ort.Value{audioCodesTensor, audioCodeLengthsTensor}
	inputNames := []string{"audio_codes", "audio_code_lengths"}

	for _, spec := range s.TransformerSpecs {
		name := fmt.Sprintf("%v", spec["input_name"])
		if v, ok := s.StateFeeds[name]; ok {
			inputs = append(inputs, v)
			inputNames = append(inputNames, name)
		}
	}
	for _, spec := range s.AttentionSpecs {
		for _, key := range []string{"offset_input_name", "cached_keys_input_name", "cached_values_input_name", "cached_positions_input_name"} {
			name := fmt.Sprintf("%v", spec[key])
			if v, ok := s.StateFeeds[name]; ok {
				inputs = append(inputs, v)
				inputNames = append(inputNames, name)
			}
		}
	}

	outputs := make([]ort.Value, len(inputNames))
	err := s.Session.Run(inputs, outputs)

	audioCodesTensor.Destroy()
	audioCodeLengthsTensor.Destroy()

	if err != nil {
		log.Printf("  codec_decode_step 失败: %v", err)
		for _, v := range outputs {
			if v != nil {
				v.Destroy()
			}
		}
		return nil, 0
	}

	outputNames := []string{"audio", "audio_lengths"}
	for _, spec := range s.TransformerSpecs {
		outputNames = append(outputNames, fmt.Sprintf("%v", spec["output_name"]))
	}
	for _, spec := range s.AttentionSpecs {
		for _, key := range []string{"offset_output_name", "cached_keys_output_name", "cached_values_output_name", "cached_positions_output_name"} {
			outputNames = append(outputNames, fmt.Sprintf("%v", spec[key]))
		}
	}

	namedOutputs := make(map[string]ort.Value)
	for i, name := range outputNames {
		if i < len(outputs) {
			namedOutputs[name] = outputs[i]
		}
	}

	for _, spec := range s.TransformerSpecs {
		inName := fmt.Sprintf("%v", spec["input_name"])
		outName := fmt.Sprintf("%v", spec["output_name"])
		if oldV, ok := s.StateFeeds[inName]; ok && oldV != nil {
			oldV.Destroy()
		}
		if newV, ok := namedOutputs[outName]; ok && newV != nil {
			s.StateFeeds[inName] = newV
		}
	}
	for _, spec := range s.AttentionSpecs {
		pairs := []struct{ inKey, outKey string }{
			{"offset_input_name", "offset_output_name"},
			{"cached_keys_input_name", "cached_keys_output_name"},
			{"cached_values_input_name", "cached_values_output_name"},
			{"cached_positions_input_name", "cached_positions_output_name"},
		}
		for _, pair := range pairs {
			inName := fmt.Sprintf("%v", spec[pair.inKey])
			outName := fmt.Sprintf("%v", spec[pair.outKey])
			if oldV, ok := s.StateFeeds[inName]; ok && oldV != nil {
				oldV.Destroy()
			}
			if newV, ok := namedOutputs[outName]; ok && newV != nil {
				s.StateFeeds[inName] = newV
			}
		}
	}

	audioLengthData := getInt32Data(namedOutputs["audio_lengths"])
	audioLength := int32(0)
	if len(audioLengthData) > 0 {
		audioLength = audioLengthData[0]
	}

	audioData := getFloat32Data(namedOutputs["audio"])
	audioShape := namedOutputs["audio"].GetShape()

	numChannels := 1
	maxSamplesPerChannel := int(audioLength)
	if len(audioShape) >= 3 {
		numChannels = int(audioShape[1])
		maxSamplesPerChannel = int(audioShape[2])
	} else if len(audioShape) >= 2 {
		numChannels = int(audioShape[1])
	}
	if numChannels <= 0 {
		numChannels = 1
	}

	actualSamplesPerChannel := int(audioLength)
	if actualSamplesPerChannel > maxSamplesPerChannel {
		actualSamplesPerChannel = maxSamplesPerChannel
	}

	channels := make([][]float32, numChannels)
	for ch := 0; ch < numChannels; ch++ {
		startOff := ch * maxSamplesPerChannel
		endOff := startOff + actualSamplesPerChannel
		if endOff > len(audioData) {
			endOff = len(audioData)
		}
		if startOff >= len(audioData) {
			channels[ch] = make([]float32, 0)
			continue
		}
		channels[ch] = make([]float32, endOff-startOff)
		copy(channels[ch], audioData[startOff:endOff])
	}

	if audioV, ok := namedOutputs["audio"]; ok && audioV != nil {
		audioV.Destroy()
	}
	if alV, ok := namedOutputs["audio_lengths"]; ok && alV != nil {
		alV.Destroy()
	}

	return channels, actualSamplesPerChannel
}

type OrtCpuRuntime struct {
	ModelDir           string
	ThreadCount        int
	ManifestPath       string
	ManifestDir        string
	Manifest           map[string]interface{}
	TTSMetaPath        string
	CodecMetaPath      string
	TTSMeta            map[string]interface{}
	CodecMeta          map[string]interface{}
	RNG                *rand.Rand
	Onnx               *OnnxSessions
	SessionIdleTimeout time.Duration // Session 空闲超时（默认 10 秒）
	SessionMutex       sync.Mutex    // 保护 sessions 的并发访问
}

func NewOrtCpuRuntime(modelDir string, threadCount int, maxNewFrames *int, doSample *bool, sampleMode *string) (*OrtCpuRuntime, error) {
	rt := &OrtCpuRuntime{
		ModelDir:           modelDir,
		ThreadCount:        max(1, threadCount),
		RNG:                rand.New(rand.NewSource(time.Now().UnixNano())),
		SessionIdleTimeout: 10 * time.Second, // 默认 10 秒超时
	}
	manifestPath, err := rt.resolveManifestPath(modelDir)
	if err != nil {
		return nil, err
	}
	rt.ManifestPath = manifestPath
	rt.ManifestDir = filepath.Dir(manifestPath)
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("读取 manifest 失败: %w", err)
	}
	if err := json.Unmarshal(manifestData, &rt.Manifest); err != nil {
		return nil, fmt.Errorf("解析 manifest JSON 失败: %w", err)
	}
	gd := rt.Manifest["generation_defaults"].(map[string]interface{})
	if maxNewFrames != nil {
		gd["max_new_frames"] = float64(*maxNewFrames)
	}
	if doSample != nil {
		gd["do_sample"] = *doSample
	}
	rawSampleMode := ""
	if sampleMode != nil {
		rawSampleMode = *sampleMode
	} else if sm, ok := gd["sample_mode"]; ok {
		rawSampleMode = fmt.Sprintf("%v", sm)
	}
	rawDoSample := false
	if ds, ok := gd["do_sample"]; ok {
		rawDoSample = toBool(ds)
	}
	normalized := sampler.NormalizeSampleMode(rawSampleMode, rawDoSample)
	gd["sample_mode"] = normalized
	gd["do_sample"] = normalized != sampler.SampleModeGreedy
	modelFiles := rt.Manifest["model_files"].(map[string]interface{})
	rt.TTSMetaPath = rt.ResolveManifestRelativePath(fmt.Sprintf("%v", modelFiles["tts_meta"]))
	rt.CodecMetaPath = rt.ResolveManifestRelativePath(fmt.Sprintf("%v", modelFiles["codec_meta"]))
	ttsMetaData, err := os.ReadFile(rt.TTSMetaPath)
	if err != nil {
		return nil, fmt.Errorf("读取 TTS meta 失败: %w", err)
	}
	if err := json.Unmarshal(ttsMetaData, &rt.TTSMeta); err != nil {
		return nil, fmt.Errorf("解析 TTS meta JSON 失败: %w", err)
	}
	codecMetaData, err := os.ReadFile(rt.CodecMetaPath)
	if err != nil {
		return nil, fmt.Errorf("读取 Codec meta 失败: %w", err)
	}
	if err := json.Unmarshal(codecMetaData, &rt.CodecMeta); err != nil {
		return nil, fmt.Errorf("解析 Codec meta JSON 失败: %w", err)
	}
	return rt, nil
}

func (rt *OrtCpuRuntime) CreateSessions() error {
	ttsDir := filepath.Dir(rt.TTSMetaPath)
	codecDir := filepath.Dir(rt.CodecMetaPath)
	ttsFiles := rt.TTSMeta["files"].(map[string]interface{})
	codecFiles := rt.CodecMeta["files"].(map[string]interface{})
	onnxInfo := rt.TTSMeta["onnx"].(map[string]interface{})
	sessions := &OnnxSessions{}

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("创建 SessionOptions 失败: %w", err)
	}
	defer sessionOptions.Destroy()
	if err := sessionOptions.SetGraphOptimizationLevel(ort.GraphOptimizationLevel(99)); err != nil {
		log.Printf("  警告: 设置图优化级别失败: %v", err)
	}
	if err := sessionOptions.SetIntraOpNumThreads(rt.ThreadCount); err != nil {
		log.Printf("  警告: 设置 IntraOp 线程数失败: %v", err)
	}
	if err := sessionOptions.SetInterOpNumThreads(1); err != nil {
		log.Printf("  警告: 设置 InterOp 线程数失败: %v", err)
	}

	load := func(name, dir string, filename interface{}, inputNames, outputNames []string) error {
		if filename == nil {
			return nil
		}
		onnxPath := filepath.Join(dir, fmt.Sprintf("%v", filename))
		log.Printf("  加载 ONNX session: %s (inputs=%d outputs=%d threads=%d)", name, len(inputNames), len(outputNames), rt.ThreadCount)
		sess, err := ort.NewDynamicAdvancedSession(onnxPath, inputNames, outputNames, sessionOptions)
		if err != nil {
			return fmt.Errorf("创建 ONNX session 失败 (%s): %w", name, err)
		}
		switch name {
		case "prefill":
			sessions.Prefill = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "decode":
			sessions.Decode = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "local_decoder":
			sessions.LocalDecoder = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "local_cached_step":
			sessions.LocalCachedStep = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "local_fixed_sampled_frame":
			sessions.LocalFixedSampledFrame = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "codec_encode":
			sessions.CodecEncode = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "codec_decode":
			sessions.CodecDecode = &TimedSession{Session: sess, LastUsed: time.Now()}
		case "codec_decode_step":
			sessions.CodecDecodeStep = &TimedSession{Session: sess, LastUsed: time.Now()}
		}
		return nil
	}

	prefillOutputs := toStrSlice(onnxInfo["prefill_output_names"])
	if err := load("prefill", ttsDir, ttsFiles["prefill"], []string{"input_ids", "attention_mask"}, prefillOutputs); err != nil {
		return err
	}

	decodeInputs := toStrSlice(onnxInfo["decode_input_names"])
	decodeOutputs := toStrSlice(onnxInfo["decode_output_names"])
	if err := load("decode", ttsDir, ttsFiles["decode_step"], decodeInputs, decodeOutputs); err != nil {
		return err
	}

	load("local_decoder", ttsDir, ttsFiles["local_decoder"], nil, nil)

	if f, ok := ttsFiles["local_cached_step"]; ok && f != nil {
		lcInputs := toStrSlice(onnxInfo["local_cached_input_names"])
		lcOutputs := toStrSlice(onnxInfo["local_cached_output_names"])
		load("local_cached_step", ttsDir, f, lcInputs, lcOutputs)
	}
	if f, ok := ttsFiles["local_fixed_sampled_frame"]; ok && f != nil {
		lfInputs := toStrSlice(onnxInfo["local_fixed_sampled_frame_input_names"])
		lfOutputs := toStrSlice(onnxInfo["local_fixed_sampled_frame_output_names"])
		load("local_fixed_sampled_frame", ttsDir, f, lfInputs, lfOutputs)
	}

	if err := load("codec_encode", codecDir, codecFiles["encode"], []string{"waveform", "input_lengths"}, []string{"audio_codes", "audio_code_lengths"}); err != nil {
		return err
	}
	if err := load("codec_decode", codecDir, codecFiles["decode_full"], []string{"audio_codes", "audio_code_lengths"}, []string{"audio", "audio_lengths"}); err != nil {
		return err
	}
	if f, ok := codecFiles["decode_step"]; ok && f != nil {
		codecOnnxInfo := rt.CodecMeta["onnx"].(map[string]interface{})
		dsInputs := toStrSlice(codecOnnxInfo["decode_step_input_names"])
		dsOutputs := toStrSlice(codecOnnxInfo["decode_step_output_names"])
		load("codec_decode_step", codecDir, f, dsInputs, dsOutputs)
	}
	rt.Onnx = sessions
	return nil
}

func InitializeORT(libDir string) error {
	ortDir := filepath.Join(libDir, "onnxruntime")
	libPath := findOrtLibForInit(ortDir)
	if libPath == "" {
		return fmt.Errorf("未找到 ONNX Runtime 动态库 (搜索目录: %s)", ortDir)
	}
	ensureOrtSymlink(ortDir, libPath)
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("初始化 ONNX Runtime 环境失败: %w", err)
	}
	log.Printf("ONNX Runtime 环境初始化成功 (lib=%s)", libPath)
	return nil
}

func (rt *OrtCpuRuntime) ListBuiltinVoices() []map[string]interface{} {
	voices := rt.Manifest["builtin_voices"].([]interface{})
	result := make([]map[string]interface{}, len(voices))
	for i, v := range voices {
		result[i] = v.(map[string]interface{})
	}
	return result
}

func (rt *OrtCpuRuntime) ListTextSamples() []map[string]interface{} {
	samples := rt.Manifest["text_samples"].([]interface{})
	result := make([]map[string]interface{}, len(samples))
	for i, v := range samples {
		result[i] = v.(map[string]interface{})
	}
	return result
}

func (rt *OrtCpuRuntime) BuildTextRows(tokenIDs []int) [][]int32 {
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	nvq := int(toFloat64(ttsConfig["n_vq"]))
	rowWidth := nvq + 1
	audioPad := int32(toFloat64(ttsConfig["audio_pad_token_id"]))
	rows := make([][]int32, len(tokenIDs))
	for i, tid := range tokenIDs {
		row := make([]int32, rowWidth)
		for j := range row {
			row[j] = audioPad
		}
		row[0] = int32(tid)
		rows[i] = row
	}
	return rows
}

func (rt *OrtCpuRuntime) BuildAudioPrefixRows(promptAudioCodes [][]int, slotTokenID *int) [][]int32 {
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	nvq := int(toFloat64(ttsConfig["n_vq"]))
	rowWidth := nvq + 1
	audioPad := int32(toFloat64(ttsConfig["audio_pad_token_id"]))
	resolvedSlot := int32(toFloat64(ttsConfig["audio_user_slot_token_id"]))
	if slotTokenID != nil {
		resolvedSlot = int32(*slotTokenID)
	}
	rows := make([][]int32, len(promptAudioCodes))
	for i, codeRow := range promptAudioCodes {
		row := make([]int32, rowWidth)
		for j := range row {
			row[j] = audioPad
		}
		row[0] = resolvedSlot
		for j := 0; j < len(codeRow) && j < nvq; j++ {
			row[j+1] = int32(codeRow[j])
		}
		rows[i] = row
	}
	return rows
}

func (rt *OrtCpuRuntime) BuildVoiceCloneRequestRows(promptAudioCodes [][]int, textTokenIDs []int) map[string][][]int32 {
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	promptTemplates := rt.Manifest["prompt_templates"].(map[string]interface{})
	prefixIDs := toIntSlice(promptTemplates["user_prompt_prefix_token_ids"])
	prefixIDs = append(prefixIDs, int(toFloat64(ttsConfig["audio_start_token_id"])))
	suffixIDs := []int{int(toFloat64(ttsConfig["audio_end_token_id"]))}
	suffixIDs = append(suffixIDs, toIntSlice(promptTemplates["user_prompt_after_reference_token_ids"])...)
	suffixIDs = append(suffixIDs, textTokenIDs...)
	suffixIDs = append(suffixIDs, toIntSlice(promptTemplates["assistant_prompt_prefix_token_ids"])...)
	suffixIDs = append(suffixIDs, int(toFloat64(ttsConfig["audio_start_token_id"])))
	var allRows [][]int32
	prefixRows := rt.BuildTextRows(prefixIDs)
	audioRows := rt.BuildAudioPrefixRows(promptAudioCodes, nil)
	suffixRows := rt.BuildTextRows(suffixIDs)
	allRows = append(allRows, prefixRows...)
	allRows = append(allRows, audioRows...)
	allRows = append(allRows, suffixRows...)
	log.Printf("[BuildVoiceCloneRequestRows] prefixRows=%d audioRows=%d suffixRows=%d total=%d textTokens=%d audioFrames=%d",
		len(prefixRows), len(audioRows), len(suffixRows), len(allRows), len(textTokenIDs), len(promptAudioCodes))
	mask := make([][]int32, 1)
	mask[0] = make([]int32, len(allRows))
	for i := range mask[0] {
		mask[0][i] = 1
	}
	return map[string][][]int32{"inputIds": allRows, "attentionMask": mask}
}

func (rt *OrtCpuRuntime) GenerateAudioFrames(requestRows map[string][][]int32) [][]int {
	return rt.GenerateAudioFramesWithContext(context.Background(), requestRows)
}

func (rt *OrtCpuRuntime) GenerateAudioFramesWithContext(ctx context.Context, requestRows map[string][][]int32) [][]int {
	return rt.GenerateAudioFramesWithCallback(ctx, requestRows, 0, nil)
}

type FrameCallback func(generatedFrames [][]int, stepIndex int, frame []int)

func (rt *OrtCpuRuntime) GenerateAudioFramesWithCallback(ctx context.Context, requestRows map[string][][]int32, maxNewFrames int, onFrame FrameCallback) [][]int {
	rt.SessionMutex.Lock()
	defer rt.SessionMutex.Unlock()

	gd := rt.Manifest["generation_defaults"].(map[string]interface{})
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	onnxInfo := rt.TTSMeta["onnx"].(map[string]interface{})
	nvq := int(toFloat64(ttsConfig["n_vq"]))
	// 使用传入的 maxNewFrames，如果为 0 则使用默认值
	if maxNewFrames <= 0 {
		maxNewFrames = int(toFloat64(gd["max_new_frames"]))
	}
	rowWidth := nvq + 1
	audioCodebookSizes := toIntSlice(ttsConfig["audio_codebook_sizes"])
	audioCodebookSize := 1024
	if len(audioCodebookSizes) > 0 {
		audioCodebookSize = audioCodebookSizes[0]
	}
	sampleMode := fmt.Sprintf("%v", gd["sample_mode"])

	select {
	case <-ctx.Done():
		return nil
	default:
	}

	inputIDs3D := [][][]int32{requestRows["inputIds"]}
	flatIDs, idDims := flatten3dInt32(inputIDs3D)
	inputIDsTensor, _ := ort.NewTensor(idDims, flatIDs)
	flatMask, maskDims := flatten2dInt32(requestRows["attentionMask"])
	attentionMaskTensor, _ := ort.NewTensor(maskDims, flatMask)

	prefillInputs := []ort.Value{inputIDsTensor, attentionMaskTensor}
	prefillOutputNames := toStrSlice(onnxInfo["prefill_output_names"])
	prefillOutputs := make([]ort.Value, len(prefillOutputNames))

	log.Printf("  运行 prefill (输入长度=%d)...", len(requestRows["inputIds"]))
	if err := rt.Onnx.Prefill.Session.Run(prefillInputs, prefillOutputs); err != nil {
		log.Printf("  prefill 失败：%v", err)
		inputIDsTensor.Destroy()
		attentionMaskTensor.Destroy()
		return nil
	}
	inputIDsTensor.Destroy()
	attentionMaskTensor.Destroy()
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.Prefill)

	namedPrefillOutputs := make(map[string]ort.Value)
	for i, name := range prefillOutputNames {
		namedPrefillOutputs[name] = prefillOutputs[i]
	}

	globalHidden := extractLastHidden(namedPrefillOutputs["global_hidden"])

	pastValidLength := int32(0)
	for _, v := range requestRows["attentionMask"][0] {
		pastValidLength += v
	}

	pastByName := make(map[string]ort.Value)
	for _, name := range prefillOutputNames[1:] {
		pastName := strings.Replace(name, "present_", "past_", 1)
		pastByName[pastName] = namedPrefillOutputs[name]
	}

	var generatedFrames [][]int
	prevTokensByChannel := make([][]int, nvq)
	prevTokenSetsByChannel := make([]map[int]bool, nvq)
	for i := 0; i < nvq; i++ {
		prevTokenSetsByChannel[i] = make(map[int]bool)
	}

	decodeInputNames := toStrSlice(onnxInfo["decode_input_names"])
	decodeOutputNames := toStrSlice(onnxInfo["decode_output_names"])

	localPast := make(map[string][]float32)
	localPVL := 0

	for stepIndex := 0; stepIndex < maxNewFrames; stepIndex++ {
		select {
		case <-ctx.Done():
			log.Printf("  生成被取消")
			for _, v := range pastByName {
				v.Destroy()
			}
			for _, v := range prefillOutputs {
				v.Destroy()
			}
			return nil
		default:
		}

		var frame []int
		shouldContinue := true

		if rt.Onnx.LocalFixedSampledFrame != nil && sampleMode == sampler.SampleModeFixed {
			shouldContinue, frame = rt.runLocalFixedSampledFrame(
				globalHidden, prevTokenSetsByChannel, nvq, audioCodebookSize)
		} else if rt.Onnx.LocalCachedStep != nil {
			shouldContinue, frame, localPast, localPVL = rt.runLocalCachedStepFull(
				globalHidden, prevTokensByChannel, prevTokenSetsByChannel, nvq, audioCodebookSize, gd, localPast, localPVL)
		} else {
			log.Printf("  step %d: 无可用的 local 模型", stepIndex)
			break
		}

		// 按照 Python 源码的顺序：先添加帧，再检查停止信号
		for ci, token := range frame {
			prevTokensByChannel[ci] = append(prevTokensByChannel[ci], token)
			prevTokenSetsByChannel[ci][token] = true
		}
		generatedFrames = append(generatedFrames, frame)

		if !shouldContinue {
			log.Printf("  step %d: 停止信号", stepIndex)
			break
		}

		if onFrame != nil {
			onFrame(generatedFrames, stepIndex, frame)
		}

		audioPad := int32(toFloat64(ttsConfig["audio_pad_token_id"]))
		assistSlot := int32(toFloat64(ttsConfig["audio_assistant_slot_token_id"]))
		nextRow := make([]int32, rowWidth)
		for j := range nextRow {
			nextRow[j] = audioPad
		}
		nextRow[0] = assistSlot
		for idx, token := range frame {
			if idx+1 < rowWidth {
				nextRow[idx+1] = int32(token)
			}
		}
		nextRow3D := [][][]int32{{nextRow}}
		flatNextRow, nextRowDims := flatten3dInt32(nextRow3D)
		nextRowTensor, _ := ort.NewTensor(nextRowDims, flatNextRow)
		pvlData := []int32{pastValidLength}
		pvlTensor, _ := ort.NewTensor([]int64{1}, pvlData)

		decodeInputs := make([]ort.Value, len(decodeInputNames))
		decodeInputs[0] = nextRowTensor
		decodeInputs[1] = pvlTensor
		for i := 2; i < len(decodeInputNames); i++ {
			if v, ok := pastByName[decodeInputNames[i]]; ok {
				decodeInputs[i] = v
			}
		}

		decodeOutputs := make([]ort.Value, len(decodeOutputNames))
		if err := rt.Onnx.Decode.Session.Run(decodeInputs, decodeOutputs); err != nil {
			log.Printf("  decode_step 失败：%v", err)
			nextRowTensor.Destroy()
			pvlTensor.Destroy()
			break
		}
		nextRowTensor.Destroy()
		pvlTensor.Destroy()
		// 更新 Session 使用时间
		rt.UpdateSessionTime(rt.Onnx.Decode)

		namedDecodeOutputs := make(map[string]ort.Value)
		for i, name := range decodeOutputNames {
			namedDecodeOutputs[name] = decodeOutputs[i]
		}

		globalHidden = extractLastHidden(namedDecodeOutputs["global_hidden"])
		pastValidLength++

		// 销毁旧的 past 值和 decodeOutputs
		for _, oldPast := range pastByName {
			oldPast.Destroy()
		}
		// 注意：namedDecodeOutputs 中的值会被转移到 newPastByName，所以这里不需要销毁

		// 创建新的 pastByName，只保存需要传递到下一轮的 past 值
		newPastByName := make(map[string]ort.Value)
		for _, name := range decodeOutputNames[1:] {
			pastName := strings.Replace(name, "present_", "past_", 1)
			if v, ok := namedDecodeOutputs[name]; ok && v != nil {
				newPastByName[pastName] = v
			}
		}
		pastByName = newPastByName

		if (stepIndex+1)%10 == 0 {
			log.Printf("  已生成 %d/%d 帧", stepIndex+1, maxNewFrames)
		}
	}

	for _, v := range pastByName {
		v.Destroy()
	}
	for _, v := range prefillOutputs {
		v.Destroy()
	}

	log.Printf("  生成完成: %d 帧", len(generatedFrames))
	return generatedFrames
}

func flattenRowToInner(row []int32) []int32 {
	return row
}

func extractLastHidden(v ort.Value) []float32 {
	if v == nil {
		return nil
	}
	shape := v.GetShape()
	data := getFloat32Data(v)
	if data == nil || len(data) == 0 {
		return nil
	}
	ndim := len(shape)
	if ndim == 2 {
		result := make([]float32, len(data))
		copy(result, data)
		return result
	}
	if ndim != 3 || shape[0] != 1 {
		result := make([]float32, len(data))
		copy(result, data)
		return result
	}
	lastDim := shape[2]
	totalSize := int64(1)
	for _, d := range shape {
		totalSize *= d
	}
	startOff := int(totalSize - lastDim)
	if startOff < 0 {
		startOff = 0
	}
	result := make([]float32, lastDim)
	copy(result, data[startOff:])
	return result
}

func (rt *OrtCpuRuntime) runLocalFixedSampledFrame(globalHidden []float32, prevTokenSets []map[int]bool, nvq, audioCodebookSize int) (bool, []int) {
	ghShape := []int64{1, int64(len(globalHidden))}
	ghTensor, _ := ort.NewTensor(ghShape, globalHidden)

	repetitionMask := make([]int32, nvq*audioCodebookSize)
	for ci, tokenSet := range prevTokenSets {
		for tid := range tokenSet {
			if tid >= 0 && tid < audioCodebookSize {
				repetitionMask[ci*audioCodebookSize+tid] = 1
			}
		}
	}
	repMaskShape := []int64{1, int64(nvq), int64(audioCodebookSize)}
	repMaskTensor, _ := ort.NewTensor(repMaskShape, repetitionMask)

	assistantRU := []float32{clampRand(float32(rt.RNG.Float32()))}
	assRUTensor, _ := ort.NewTensor([]int64{1}, assistantRU)

	audioRU := make([]float32, nvq)
	for i := range audioRU {
		audioRU[i] = clampRand(float32(rt.RNG.Float32()))
	}
	audioRUTensor, _ := ort.NewTensor([]int64{1, int64(nvq)}, audioRU)

	inputs := []ort.Value{ghTensor, repMaskTensor, assRUTensor, audioRUTensor}
	outputs := make([]ort.Value, 2)

	err := rt.Onnx.LocalFixedSampledFrame.Session.Run(inputs, outputs)
	ghTensor.Destroy()
	repMaskTensor.Destroy()
	assRUTensor.Destroy()
	audioRUTensor.Destroy()

	if err != nil {
		log.Printf("  local_fixed_sampled_frame 失败: %v", err)
		return false, nil
	}
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.LocalFixedSampledFrame)

	shouldContinueData := getInt32Data(outputs[0])
	shouldContinue := int32(0)
	if len(shouldContinueData) > 0 {
		shouldContinue = shouldContinueData[0]
	}

	frameData := getInt32Data(outputs[1])
	frame := make([]int, nvq)
	for i := 0; i < nvq && i < len(frameData); i++ {
		frame[i] = int(frameData[i])
	}

	outputs[0].Destroy()
	outputs[1].Destroy()
	return shouldContinue != 0, frame
}

func (rt *OrtCpuRuntime) runLocalCachedStepFull(globalHidden []float32, prevTokens [][]int, prevTokenSets []map[int]bool, nvq, audioCodebookSize int, gd map[string]interface{}, localPast map[string][]float32, localPVL int) (bool, []int, map[string][]float32, int) {
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	onnxInfo := rt.TTSMeta["onnx"].(map[string]interface{})
	audioAssistSlotID := int(toFloat64(ttsConfig["audio_assistant_slot_token_id"]))
	audioEndTokenID := int(toFloat64(ttsConfig["audio_end_token_id"]))
	inputNames := toStrSlice(onnxInfo["local_cached_input_names"])
	outputNames := toStrSlice(onnxInfo["local_cached_output_names"])

	doSample := true
	if ds, ok := gd["do_sample"]; ok {
		doSample = toBool(ds)
	}
	temperature := 1.0
	if t, ok := gd["temperature"]; ok {
		temperature = toFloat64(t)
	}
	topK := 50
	if k, ok := gd["top_k"]; ok {
		topK = int(toFloat64(k))
	}
	topP := 1.0
	if p, ok := gd["top_p"]; ok {
		topP = toFloat64(p)
	}
	repetitionPenalty := float32(1.0)
	if rp, ok := gd["repetition_penalty"]; ok {
		repetitionPenalty = float32(toFloat64(rp))
	}

	ghShape := []int64{1, int64(len(globalHidden))}

	textLogits, _, _, localPastData := rt.runCachedStep(globalHidden, ghShape, 0, 0, 0, 0, localPVL, localPast, inputNames, outputNames)
	localPVL++

	if textLogits == nil {
		return false, nil, localPastData, localPVL
	}

	nextTextToken := sampler.SampleAssistantTextToken(textLogits, audioAssistSlotID, audioEndTokenID, doSample, temperature, topK, topP, rt.RNG)
	if nextTextToken != audioAssistSlotID {
		return false, nil, localPastData, localPVL
	}

	_, audioLogits, audioLogitsShape, localPastData := rt.runCachedStep(globalHidden, ghShape, nextTextToken, 0, 0, 1, localPVL, localPastData, inputNames, outputNames)
	localPVL++

	if audioLogits == nil {
		return false, nil, localPastData, localPVL
	}

	perChannel := int(audioLogitsShape[len(audioLogitsShape)-1])
	firstChannelLogits := make([]float32, perChannel)
	copy(firstChannelLogits, audioLogits[:minInt(perChannel, len(audioLogits))])

	sampledToken := sampler.SampleAudioToken(firstChannelLogits, prevTokens[0], prevTokenSets[0], doSample, temperature, topK, topP, repetitionPenalty, rt.RNG)
	frame := []int{sampledToken}
	previousToken := sampledToken

	for ci := 1; ci < nvq; ci++ {
		_, audioLogits2, audioLogitsShape2, localPastData2 := rt.runCachedStep(globalHidden, ghShape, 0, previousToken, ci-1, 2, localPVL, localPastData, inputNames, outputNames)
		localPVL++

		if audioLogits2 == nil {
			break
		}

		localPastData = localPastData2
		perChannel2 := int(audioLogitsShape2[len(audioLogitsShape2)-1])
		startOff := ci * perChannel2
		endOff := minInt(startOff+perChannel2, len(audioLogits2))
		channelLogits := make([]float32, perChannel2)
		if startOff < len(audioLogits2) {
			copy(channelLogits, audioLogits2[startOff:endOff])
		}

		sampledToken2 := sampler.SampleAudioToken(channelLogits, prevTokens[ci], prevTokenSets[ci], doSample, temperature, topK, topP, repetitionPenalty, rt.RNG)
		frame = append(frame, sampledToken2)
		previousToken = sampledToken2
	}

	return true, frame, localPastData, localPVL
}

func (rt *OrtCpuRuntime) runCachedStep(globalHidden []float32, ghShape []int64, textTokenID, audioTokenID, channelIndex, stepType, pastValidLength int, localPast map[string][]float32, inputNames, outputNames []string) ([]float32, []float32, []int64, map[string][]float32) {
	ghTensor, _ := ort.NewTensor(ghShape, globalHidden)
	ttTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(textTokenID)})
	atTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(audioTokenID)})
	ciTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(channelIndex)})
	stTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(stepType)})
	pvlTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(pastValidLength)})

	inputs := make([]ort.Value, len(inputNames))
	inputs[0] = ghTensor
	inputs[1] = ttTensor
	inputs[2] = atTensor
	inputs[3] = ciTensor
	inputs[4] = stTensor
	inputs[5] = pvlTensor
	localHeads := int64(8)
	localHeadDim := int64(64)
	if mc, ok := rt.TTSMeta["model_config"].(map[string]interface{}); ok {
		if lh, exists := mc["local_heads"]; exists {
			localHeads = int64(toFloat64(lh))
		}
		if lhd, exists := mc["local_head_dim"]; exists {
			localHeadDim = int64(toFloat64(lhd))
		}
	}
	for i := 6; i < len(inputNames); i++ {
		pastData, ok := localPast[inputNames[i]]
		if !ok {
			pastData = make([]float32, 0)
		}
		pastShape := []int64{1, 0, localHeads, localHeadDim}
		t, _ := ort.NewTensor(pastShape, pastData)
		inputs[i] = t
	}

	outputs := make([]ort.Value, len(outputNames))
	err := rt.Onnx.LocalCachedStep.Session.Run(inputs, outputs)

	ghTensor.Destroy()
	ttTensor.Destroy()
	atTensor.Destroy()
	ciTensor.Destroy()
	stTensor.Destroy()
	pvlTensor.Destroy()
	for i := 6; i < len(inputs); i++ {
		if inputs[i] != nil {
			inputs[i].Destroy()
		}
	}

	if err != nil {
		log.Printf("  local_cached_step 失败：%v", err)
		return nil, nil, nil, nil
	}
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.LocalCachedStep)

	namedOut := make(map[string]ort.Value)
	for i, name := range outputNames {
		namedOut[name] = outputs[i]
	}

	var textLogits []float32
	if tl, ok := namedOut["text_logits"]; ok && tl != nil {
		textLogits = getFloat32Data(tl)
	}

	var audioLogits []float32
	var audioLogitsShape []int64
	if al, ok := namedOut["audio_logits"]; ok && al != nil {
		audioLogits = getFloat32Data(al)
		audioLogitsShape = al.GetShape()
	}

	nextLocalPast := make(map[string][]float32)
	for i := 2; i < len(outputNames); i++ {
		pastName := strings.Replace(outputNames[i], "local_present_", "local_past_", 1)
		if v, ok := namedOut[outputNames[i]]; ok && v != nil {
			data := getFloat32Data(v)
			copied := make([]float32, len(data))
			copy(copied, data)
			nextLocalPast[pastName] = copied
		}
	}

	for _, v := range outputs {
		if v != nil {
			v.Destroy()
		}
	}

	return textLogits, audioLogits, audioLogitsShape, nextLocalPast
}

func (rt *OrtCpuRuntime) DecodeFullAudio(generatedFrames [][]int) ([][]float32, int) {
	return rt.DecodeFullAudioWithContext(context.Background(), generatedFrames)
}

func (rt *OrtCpuRuntime) DecodeFullAudioWithContext(ctx context.Context, generatedFrames [][]int) ([][]float32, int) {
	if len(generatedFrames) == 0 || rt.Onnx.CodecDecode == nil {
		return nil, 0
	}

	select {
	case <-ctx.Done():
		return nil, 0
	default:
	}

	codecConfig := rt.CodecMeta["codec_config"].(map[string]interface{})
	numQuantizers := int(toFloat64(codecConfig["num_quantizers"]))
	frameCount := len(generatedFrames)

	audioCodesFlat := make([]int32, frameCount*numQuantizers)
	for fi, frame := range generatedFrames {
		for ci := 0; ci < numQuantizers; ci++ {
			val := int32(0)
			if ci < len(frame) {
				val = int32(frame[ci])
			}
			audioCodesFlat[fi*numQuantizers+ci] = val
		}
	}

	audioCodesShape := []int64{1, int64(frameCount), int64(numQuantizers)}
	audioCodesTensor, _ := ort.NewTensor(audioCodesShape, audioCodesFlat)
	audioCodeLengths := []int32{int32(frameCount)}
	audioCodeLengthsTensor, _ := ort.NewTensor([]int64{1}, audioCodeLengths)

	inputs := []ort.Value{audioCodesTensor, audioCodeLengthsTensor}
	outputs := make([]ort.Value, 2)

	err := rt.Onnx.CodecDecode.Session.Run(inputs, outputs)
	audioCodesTensor.Destroy()
	audioCodeLengthsTensor.Destroy()

	if err != nil {
		log.Printf("  codec_decode_full 失败：%v", err)
		return nil, 0
	}
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.CodecDecode)

	audioLengthData := getInt32Data(outputs[1])
	audioLength := int32(0)
	if len(audioLengthData) > 0 {
		audioLength = audioLengthData[0]
	}

	audioData := getFloat32Data(outputs[0])
	audioShape := outputs[0].GetShape()

	for _, v := range outputs {
		if v != nil {
			v.Destroy()
		}
	}

	if len(audioData) == 0 || audioLength == 0 {
		return nil, 0
	}

	numChannels := 1
	maxSamplesPerChannel := int(audioLength)
	if len(audioShape) >= 3 {
		numChannels = int(audioShape[1])
		maxSamplesPerChannel = int(audioShape[2])
	} else if len(audioShape) >= 1 {
		numChannels = 1
	}
	if numChannels <= 0 {
		numChannels = 1
	}

	channels := make([][]float32, numChannels)
	for ch := 0; ch < numChannels; ch++ {
		startOff := ch * maxSamplesPerChannel
		endOff := startOff + int(audioLength)
		if endOff > len(audioData) {
			endOff = len(audioData)
		}
		if startOff >= len(audioData) {
			channels[ch] = make([]float32, 0)
			continue
		}
		channels[ch] = make([]float32, endOff-startOff)
		copy(channels[ch], audioData[startOff:endOff])
	}

	log.Printf("  解码音频: frames=%d channels=%d samples=%d audioShape=%v", frameCount, numChannels, audioLength, audioShape)
	return channels, int(audioLength)
}

func (rt *OrtCpuRuntime) Close() {
	if rt.Onnx != nil {
		s := rt.Onnx
		if s.Prefill != nil {
			s.Prefill.Session.Destroy()
			s.Prefill = nil
		}
		if s.Decode != nil {
			s.Decode.Session.Destroy()
			s.Decode = nil
		}
		if s.LocalDecoder != nil {
			s.LocalDecoder.Session.Destroy()
			s.LocalDecoder = nil
		}
		if s.LocalCachedStep != nil {
			s.LocalCachedStep.Session.Destroy()
			s.LocalCachedStep = nil
		}
		if s.LocalFixedSampledFrame != nil {
			s.LocalFixedSampledFrame.Session.Destroy()
			s.LocalFixedSampledFrame = nil
		}
		if s.CodecEncode != nil {
			s.CodecEncode.Session.Destroy()
			s.CodecEncode = nil
		}
		if s.CodecDecode != nil {
			s.CodecDecode.Session.Destroy()
			s.CodecDecode = nil
		}
		if s.CodecDecodeStep != nil {
			s.CodecDecodeStep.Session.Destroy()
			s.CodecDecodeStep = nil
		}
	}
}

// CheckAndReleaseIdleSessions 检查并释放所有空闲超时的 Session（导出方法）
func (rt *OrtCpuRuntime) CheckAndReleaseIdleSessions() {
	if rt.SessionIdleTimeout <= 0 {
		return // 禁用超时释放
	}

	now := time.Now()
	count := 0
	log.Printf("[ORT] 开始检查空闲 Session (超时阈值：%v)...", rt.SessionIdleTimeout)

	// 检查每个 Session
	count += rt.checkAndReleaseOne("prefill", rt.Onnx.Prefill, now)
	count += rt.checkAndReleaseOne("decode", rt.Onnx.Decode, now)
	count += rt.checkAndReleaseOne("local_decoder", rt.Onnx.LocalDecoder, now)
	count += rt.checkAndReleaseOne("local_cached_step", rt.Onnx.LocalCachedStep, now)
	count += rt.checkAndReleaseOne("local_fixed_sampled_frame", rt.Onnx.LocalFixedSampledFrame, now)
	count += rt.checkAndReleaseOne("codec_encode", rt.Onnx.CodecEncode, now)
	count += rt.checkAndReleaseOne("codec_decode", rt.Onnx.CodecDecode, now)
	count += rt.checkAndReleaseOne("codec_decode_step", rt.Onnx.CodecDecodeStep, now)

	if count > 0 {
		log.Printf("[ORT] 发现 %d 个空闲 Session (超时：%v)，执行重置...", count, rt.SessionIdleTimeout)
		// 调用 ResetSessions 来销毁并重新创建所有 Session
		if err := rt.ResetSessions(); err != nil {
			log.Printf("[ORT] 重置 Session 失败：%v", err)
		} else {
			log.Printf("[ORT] 重置完成，已释放 %d 个空闲 Session", count)
		}
	}
}

// checkAndReleaseOne 检查并释放单个 Session（内部方法）
func (rt *OrtCpuRuntime) checkAndReleaseOne(name string, timed *TimedSession, now time.Time) int {
	if timed != nil && timed.Session != nil {
		idleTime := now.Sub(timed.LastUsed)
		log.Printf("[ORT]   Session '%s': 空闲 %v (阈值：%v)", name, idleTime, rt.SessionIdleTimeout)
		if idleTime > rt.SessionIdleTimeout {
			log.Printf("[ORT] Session '%s' 空闲超时 (%v > %v)，标记需要重置", name, idleTime, rt.SessionIdleTimeout)
			return 1
		}
	} else {
		log.Printf("[ORT]   Session '%s': 未加载", name)
	}
	return 0
}

// UpdateSessionTime 更新 Session 使用时间（在成功使用后调用）
func (rt *OrtCpuRuntime) UpdateSessionTime(timed *TimedSession) {
	if timed != nil {
		timed.LastUsed = time.Now()
	}
}

// ResetSessions 销毁并重新创建所有 ONNX Session
func (rt *OrtCpuRuntime) ResetSessions() error {
	log.Printf("[ORT] 销毁旧 sessions...")

	// 获取互斥锁以防止与正在运行的请求发生竞态条件
	rt.SessionMutex.Lock()
	defer rt.SessionMutex.Unlock()

	// 销毁所有 Session
	s := rt.Onnx
	if s == nil {
		return nil
	}

	if s.Prefill != nil {
		s.Prefill.Session.Destroy()
		s.Prefill = nil
	}
	if s.Decode != nil {
		s.Decode.Session.Destroy()
		s.Decode = nil
	}
	if s.LocalDecoder != nil {
		s.LocalDecoder.Session.Destroy()
		s.LocalDecoder = nil
	}
	if s.LocalCachedStep != nil {
		s.LocalCachedStep.Session.Destroy()
		s.LocalCachedStep = nil
	}
	if s.LocalFixedSampledFrame != nil {
		s.LocalFixedSampledFrame.Session.Destroy()
		s.LocalFixedSampledFrame = nil
	}
	if s.CodecEncode != nil {
		s.CodecEncode.Session.Destroy()
		s.CodecEncode = nil
	}
	if s.CodecDecode != nil {
		s.CodecDecode.Session.Destroy()
		s.CodecDecode = nil
	}
	if s.CodecDecodeStep != nil {
		s.CodecDecodeStep.Session.Destroy()
		s.CodecDecodeStep = nil
	}

	// 重新创建
	log.Printf("[ORT] 重新创建 sessions...")
	return rt.CreateSessions()
}

func (rt *OrtCpuRuntime) resolveManifestPath(modelDir string) (string, error) {
	var triedPaths []string
	for _, relPath := range ManifestCandidateRelativePaths {
		candidate := filepath.Join(modelDir, relPath)
		absCandidate, _ := filepath.Abs(candidate)
		triedPaths = append(triedPaths, absCandidate)
		if _, err := os.Stat(absCandidate); err == nil {
			return absCandidate, nil
		}
	}
	return "", fmt.Errorf("browser_poc_manifest.json 未找到: %s", strings.Join(triedPaths, ", "))
}

func (rt *OrtCpuRuntime) ResolveManifestRelativePath(relativePath string) string {
	resolved := filepath.Join(rt.ManifestDir, relativePath)
	if _, err := os.Stat(resolved); err == nil {
		return resolved
	}
	relText := strings.ReplaceAll(relativePath, "\\", "/")
	for legacyName, canonicalName := range ModelDirAliasMap {
		if strings.Contains("/"+relText+"/", "/"+legacyName+"/") {
			rewritten := strings.ReplaceAll(relText, legacyName, canonicalName)
			rewrittenPath := filepath.Join(rt.ManifestDir, rewritten)
			if _, err := os.Stat(rewrittenPath); err == nil {
				return rewrittenPath
			}
		}
	}
	return resolved
}

func findOrtLibForInit(ortDir string) string {
	libDir := filepath.Join(ortDir, "lib")
	directNames := []string{"libonnxruntime.dylib", "libonnxruntime.so", "onnxruntime.dll"}
	for _, dir := range []string{ortDir, libDir} {
		for _, name := range directNames {
			if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
				return filepath.Join(dir, name)
			}
		}
	}
	for _, pattern := range []string{
		filepath.Join(libDir, "libonnxruntime*.dylib"),
		filepath.Join(libDir, "libonnxruntime.so*"),
	} {
		matches, err := filepath.Glob(pattern)
		if err == nil && len(matches) > 0 {
			for _, m := range matches {
				if !strings.Contains(m, ".dSYM") {
					return m
				}
			}
		}
	}
	return ""
}

func ensureOrtSymlink(ortDir, actualLibPath string) {
	var symlinkName string
	switch {
	case strings.HasSuffix(actualLibPath, ".dylib"):
		symlinkName = "libonnxruntime.dylib"
	case strings.HasSuffix(actualLibPath, ".so") || strings.Contains(actualLibPath, ".so."):
		symlinkName = "libonnxruntime.so"
	default:
		return
	}
	symlinkPath := filepath.Join(ortDir, symlinkName)
	if _, err := os.Stat(symlinkPath); err == nil {
		return
	}
	os.Symlink(actualLibPath, symlinkPath)
}

func getFloat32Data(v ort.Value) []float32 {
	if v == nil {
		return nil
	}
	if t, ok := v.(*ort.Tensor[float32]); ok {
		return t.GetData()
	}
	if c, ok := v.(*ort.CustomDataTensor); ok {
		rawData := c.GetData()
		result := make([]float32, len(rawData)/4)
		for i := range result {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(rawData[i*4:]))
		}
		return result
	}
	return nil
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

func clampRand(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 0.99999994 {
		return 0.99999994
	}
	return v
}

func flatten3dInt32(nested [][][]int32) ([]int32, []int64) {
	dim0 := int64(len(nested))
	dim1 := int64(0)
	dim2 := int64(0)
	if dim0 > 0 {
		dim1 = int64(len(nested[0]))
		if dim1 > 0 {
			dim2 = int64(len(nested[0][0]))
		}
	}
	data := make([]int32, dim0*dim1*dim2)
	offset := 0
	for i := range nested {
		for j := range nested[i] {
			for k := range nested[i][j] {
				data[offset] = nested[i][j][k]
				offset++
			}
		}
	}
	return data, []int64{dim0, dim1, dim2}
}

func flatten2dInt32(nested [][]int32) ([]int32, []int64) {
	dim0 := int64(len(nested))
	dim1 := int64(0)
	if dim0 > 0 {
		dim1 = int64(len(nested[0]))
	}
	data := make([]int32, dim0*dim1)
	offset := 0
	for i := range nested {
		for j := range nested[i] {
			data[offset] = nested[i][j]
			offset++
		}
	}
	return data, []int64{dim0, dim1}
}

func totalDimFromShape(shape []int64) int64 {
	var total int64 = 1
	for _, d := range shape {
		total *= d
	}
	return total
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func toBool(v interface{}) bool {
	switch val := v.(type) {
	case bool:
		return val
	case float64:
		return val != 0
	case int:
		return val != 0
	case string:
		return val == "true" || val == "1"
	default:
		return false
	}
}

func ToFloat64(v interface{}) float64 { return toFloat64(v) }

func toFloat64(v interface{}) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case int:
		return float64(val)
	case int64:
		return float64(val)
	case json.Number:
		f, _ := val.Float64()
		return f
	default:
		var result float64
		fmt.Sscanf(fmt.Sprintf("%v", v), "%f", &result)
		return result
	}
}

func toIntSlice(v interface{}) []int {
	switch val := v.(type) {
	case []interface{}:
		result := make([]int, len(val))
		for i, item := range val {
			result[i] = int(toFloat64(item))
		}
		return result
	case []int:
		return val
	default:
		return nil
	}
}

func toStrSlice(v interface{}) []string {
	switch val := v.(type) {
	case []interface{}:
		result := make([]string, len(val))
		for i, item := range val {
			result[i] = fmt.Sprintf("%v", item)
		}
		return result
	case []string:
		return val
	default:
		return nil
	}
}

func toInt64Slice(v interface{}) []int64 {
	switch val := v.(type) {
	case []interface{}:
		result := make([]int64, len(val))
		for i, item := range val {
			result[i] = int64(toFloat64(item))
		}
		return result
	case []int64:
		return val
	case []int:
		result := make([]int64, len(val))
		for i, item := range val {
			result[i] = int64(item)
		}
		return result
	case []float64:
		result := make([]int64, len(val))
		for i, item := range val {
			result[i] = int64(item)
		}
		return result
	default:
		return nil
	}
}
