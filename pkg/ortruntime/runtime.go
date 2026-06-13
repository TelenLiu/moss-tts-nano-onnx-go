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
	"sync/atomic"
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

// InferenceSessions 阶段2（TTS推理）使用的 session 集合
// 包含 prefill、decode、local 模型和 codec_decode 系列
// 这些 session 在推理阶段同时存在于内存中
type InferenceSessions struct {
	Prefill                *TimedSession
	Decode                 *TimedSession
	LocalDecoder           *TimedSession
	LocalGreedyFrame       *TimedSession
	LocalCachedStep        *TimedSession
	LocalFixedSampledFrame *TimedSession
	CodecDecode            *TimedSession
	CodecDecodeStep        *TimedSession
}

// OnnxSessions 所有 ONNX session 的容器
// 按"阶段"管理：阶段1（音频编码）只有 CodecEncode，阶段2（TTS推理）使用 Inference
// 两个阶段的 session 不会同时存在于内存中，避免内存峰值叠加
type OnnxSessions struct {
	Inference   *InferenceSessions // 阶段2：TTS 推理 session 集合
	CodecEncode *TimedSession      // 阶段1：音频编码 session（懒加载，用完即销毁）
}

type CodecStreamingDecodeSession struct {
	CodecMeta        map[string]interface{}
	Session          *ort.DynamicAdvancedSession
	Runtime          *OrtCpuRuntime // 用于并发安全的 Session.Run 调用
	TransformerSpecs []map[string]interface{}
	AttentionSpecs   []map[string]interface{}
	StateFeeds       map[string]ort.Value
}

func NewCodecStreamingDecodeSession(codecMeta map[string]interface{}, session *ort.DynamicAdvancedSession, runtime *OrtCpuRuntime) *CodecStreamingDecodeSession {
	s := &CodecStreamingDecodeSession{
		CodecMeta:  codecMeta,
		Session:    session,
		Runtime:    runtime,
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
	err := s.Runtime.RunSession(s.Session, inputs, outputs)

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
	ExecutionMode      string // "hybrid", "cpu", "gpu"
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
	SessionRunMutex    sync.Mutex    // 保护所有 Session.Run 调用（ONNX Session 非线程安全）
	ActiveRequests     atomic.Int64  // 当前活跃请求数，用于防止在请求期间销毁 session

	// local_cached_step 缓存的元数据，避免每帧17次重复查找
	cachedStepMeta cachedStepMetadata
}

// cachedStepMetadata 缓存 local_cached_step 推理所需的元数据
type cachedStepMetadata struct {
	localHeads  int64
	localHeadDim int64
	inputNames  []string
	outputNames []string
	initialized bool
}

// AcquireSession 标记有活跃请求正在使用 session，防止 session 被销毁。
// 每个合成请求开始时调用，结束时调用 ReleaseSession。
func (rt *OrtCpuRuntime) AcquireSession() {
	rt.ActiveRequests.Add(1)
}

// ReleaseSession 标记活跃请求结束，允许 session 被销毁。
func (rt *OrtCpuRuntime) ReleaseSession() {
	rt.ActiveRequests.Add(-1)
}

// RunSession 安全地执行 ONNX Session.Run，通过互斥锁保证并发安全。
// ONNX Runtime 的 Session 不是线程安全的，多个 goroutine 同时调用同一个 Session.Run 会导致 SIGSEGV。
func (rt *OrtCpuRuntime) RunSession(session *ort.DynamicAdvancedSession, inputs []ort.Value, outputs []ort.Value) error {
	rt.SessionRunMutex.Lock()
	err := session.Run(inputs, outputs)
	rt.SessionRunMutex.Unlock()
	return err
}

func NewOrtCpuRuntime(modelDir string, threadCount int, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*OrtCpuRuntime, error) {
	rt := &OrtCpuRuntime{
		ModelDir:           modelDir,
		ThreadCount:        max(1, threadCount),
		ExecutionMode:      executionMode,
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
	infSessions := &InferenceSessions{}

	// 尝试创建session，如果GPU加载失败则回退到CPU
	tryCreateSessions := func(useGPU bool) error {
		sessionOptions, err := ort.NewSessionOptions()
		if err != nil {
			return fmt.Errorf("创建 SessionOptions 失败: %w", err)
		}
		defer sessionOptions.Destroy()
		// 使用 EXTENDED 级别替代 ALL(99)，减少内存预分配开销，推理速度差异极小
		if err := sessionOptions.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableExtended); err != nil {
			log.Printf("  警告: 设置图优化级别失败: %v", err)
		}
		// 禁用内存模式优化，减少 ONNX Runtime 内存池预留空间
		if err := sessionOptions.AddSessionConfigEntry("session.enable_mem_pattern", "0"); err != nil {
			log.Printf("  警告: 设置 enable_mem_pattern 失败: %v", err)
		}
		// 启用内存复用，允许释放的 tensor 内存被后续分配复用
		if err := sessionOptions.AddSessionConfigEntry("session.enable_mem_reuse", "1"); err != nil {
			log.Printf("  警告: 设置 enable_mem_reuse 失败: %v", err)
		}
		// 禁用 arena 分配器，避免 ONNX Runtime 使用 jemalloc 等内存池导致内存不释放
		// arena 分配器会缓存已释放的内存块供后续复用，但不会归还给操作系统
		if err := sessionOptions.AddSessionConfigEntry("session.use_arena", "0"); err != nil {
			log.Printf("  警告: 设置 use_arena 失败: %v", err)
		}

		// 根据是否使用GPU配置SessionOptions
		if useGPU {
			gpuSuccess := false

			// 尝试CUDA
			cudaOptions, err := ort.NewCUDAProviderOptions()
			if err == nil {
				defer cudaOptions.Destroy()
				cudaOptions.Update(map[string]string{
					"device_id": "0",
				})
				if err := sessionOptions.AppendExecutionProviderCUDA(cudaOptions); err == nil {
					log.Printf("  使用CUDA执行提供程序")
					gpuSuccess = true
				}
			}

			// 如果CUDA失败，尝试CoreML
			if !gpuSuccess {
				// 检查模型是否有外部数据文件，CoreML不支持外部数据
				hasExternalData := hasExternalDataFiles(ttsDir, ttsFiles, codecDir, codecFiles)
				if hasExternalData {
					log.Printf("  模型使用外部数据文件，CoreML不支持此格式，跳过CoreML")
				} else {
					coremlFlags := uint32(0)
					if err := sessionOptions.AppendExecutionProviderCoreML(coremlFlags); err == nil {
						log.Printf("  使用CoreML执行提供程序 (Apple M1/M2)")
						gpuSuccess = true
					}
				}
			}

			if gpuSuccess {
				threads := rt.ThreadCount
				if threads > 1 {
					threads = threads - 1
				}
				if err := sessionOptions.SetIntraOpNumThreads(threads); err != nil {
					log.Printf("  警告: 设置 IntraOp 线程数失败: %v", err)
				}
			} else {
				log.Printf("  GPU不可用，使用CPU模式")
				if err := sessionOptions.SetIntraOpNumThreads(rt.ThreadCount); err != nil {
					log.Printf("  警告: 设置 IntraOp 线程数失败: %v", err)
				}
			}
		} else {
			log.Printf("  使用CPU执行提供程序")
			if err := sessionOptions.SetIntraOpNumThreads(rt.ThreadCount); err != nil {
				log.Printf("  警告: 设置 IntraOp 线程数失败: %v", err)
			}
		}

		if err := sessionOptions.SetInterOpNumThreads(1); err != nil {
			log.Printf("  警告: 设置 InterOp 线程数失败: %v", err)
		}

		// 加载模型
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
				infSessions.Prefill = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "decode":
				infSessions.Decode = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "local_decoder":
				infSessions.LocalDecoder = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "local_greedy_frame":
				infSessions.LocalGreedyFrame = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "local_cached_step":
				infSessions.LocalCachedStep = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "local_fixed_sampled_frame":
				infSessions.LocalFixedSampledFrame = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "codec_encode":
				// codec_encode 属于阶段1，不在阶段2创建
				log.Printf("  跳过 codec_encode session（属于阶段1，懒加载按需创建）")
				sess.Destroy()
			case "codec_decode":
				infSessions.CodecDecode = &TimedSession{Session: sess, LastUsed: time.Now()}
			case "codec_decode_step":
				infSessions.CodecDecodeStep = &TimedSession{Session: sess, LastUsed: time.Now()}
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

		if f, ok := ttsFiles["local_greedy_frame"]; ok && f != nil {
			lgInputs := []string{"global_hidden", "repetition_seen_mask", "repetition_penalty"}
			lgOutputs := []string{"should_continue", "frame_token_ids"}
			load("local_greedy_frame", ttsDir, f, lgInputs, lgOutputs)
		}

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

		// codec_encode 属于阶段1（音频编码），不在阶段2创建
		// 首次调用 EncodeReferenceAudio 时通过 EnsureCodecEncodeSession 按需创建
		log.Printf("  跳过 codec_encode session（属于阶段1，懒加载按需创建）")

		if err := load("codec_decode", codecDir, codecFiles["decode_full"], []string{"audio_codes", "audio_code_lengths"}, []string{"audio", "audio_lengths"}); err != nil {
			return err
		}
		if f, ok := codecFiles["decode_step"]; ok && f != nil {
			codecOnnxInfo := rt.CodecMeta["onnx"].(map[string]interface{})
			dsInputs := toStrSlice(codecOnnxInfo["decode_step_input_names"])
			dsOutputs := toStrSlice(codecOnnxInfo["decode_step_output_names"])
			load("codec_decode_step", codecDir, f, dsInputs, dsOutputs)
		}

		return nil
	}

	// 根据推理模式决定是否尝试GPU
	useGPU := rt.ExecutionMode == "hybrid" || rt.ExecutionMode == "gpu"

	// 如果是GPU模式或混合模式，先尝试GPU
	if useGPU {
		log.Printf("  尝试使用GPU加载模型...")
		err := tryCreateSessions(true)
		if err != nil {
			log.Printf("  GPU加载失败: %v", err)
			// 如果是混合模式，回退到CPU
			if rt.ExecutionMode == "hybrid" {
				log.Printf("  回退到CPU模式...")
				rt.ExecutionMode = "cpu"
				err = tryCreateSessions(false)
				if err != nil {
					return err
				}
			} else {
				// 如果是仅GPU模式，返回错误
				return fmt.Errorf("GPU模式加载失败，无法回退到CPU: %w", err)
			}
		}
	} else {
		// 仅CPU模式
		err := tryCreateSessions(false)
		if err != nil {
			return err
		}
	}

	rt.Onnx = &OnnxSessions{Inference: infSessions}
	return nil
}

// hasExternalDataFiles 检查指定目录下是否有外部数据文件（如 .data 文件）
// CoreML不支持外部数据文件格式，所以有外部数据文件时不能使用CoreML
func hasExternalDataFiles(ttsDir string, ttsFiles map[string]interface{}, codecDir string, codecFiles map[string]interface{}) bool {
	checkDir := func(dir string, filenames []string) bool {
		for _, fn := range filenames {
			if fn == "" {
				continue
			}
			// 移除可能的".onnx"后缀
			baseName := strings.TrimSuffix(fn, ".onnx")
			// 检查常见的外部数据文件名
			candidates := []string{
				baseName + ".data",
				baseName + "_data",
				baseName + ".onnx.data",
			}
			for _, candidate := range candidates {
				dataPath := filepath.Join(dir, candidate)
				if _, err := os.Stat(dataPath); err == nil {
					log.Printf("  [hasExternalDataFiles] 发现外部数据文件: %s", dataPath)
					return true
				}
			}
		}
		return false
	}

	// 收集TTS模型文件名
	ttsFilenames := []string{}
	for _, v := range ttsFiles {
		if v != nil {
			ttsFilenames = append(ttsFilenames, fmt.Sprintf("%v", v))
		}
	}
	if checkDir(ttsDir, ttsFilenames) {
		return true
	}

	// 收集Codec模型文件名
	codecFilenames := []string{}
	for _, v := range codecFiles {
		if v != nil {
			codecFilenames = append(codecFilenames, fmt.Sprintf("%v", v))
		}
	}
	if checkDir(codecDir, codecFilenames) {
		return true
	}

	return false
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
	return rt.GenerateAudioFramesWithContext(context.Background(), requestRows, 0)
}

func (rt *OrtCpuRuntime) GenerateAudioFramesWithContext(ctx context.Context, requestRows map[string][][]int32, maxNewFrames int) [][]int {
	return rt.GenerateAudioFramesWithCallbackAndOverrides(ctx, requestRows, maxNewFrames, nil, nil)
}

func (rt *OrtCpuRuntime) GenerateAudioFramesWithContextAndOverrides(ctx context.Context, requestRows map[string][][]int32, maxNewFrames int, overrides *GenerationOverrides) [][]int {
	return rt.GenerateAudioFramesWithCallbackAndOverrides(ctx, requestRows, maxNewFrames, nil, overrides)
}

type FrameCallback func(generatedFrames [][]int, stepIndex int, frame []int)

// GenerationOverrides 运行时采样参数覆盖，优先级高于 manifest 中的 generation_defaults
type GenerationOverrides struct {
	DoSample              *bool
	SampleMode            *string
	TextTemperature       *float64
	TextTopK              *int
	TextTopP              *float64
	AudioTemperature      *float64
	AudioTopK             *int
	AudioTopP             *float64
	AudioRepetitionPenalty *float64
}

func (rt *OrtCpuRuntime) GenerateAudioFramesWithCallback(ctx context.Context, requestRows map[string][][]int32, maxNewFrames int, onFrame FrameCallback) [][]int {
	return rt.GenerateAudioFramesWithCallbackAndOverrides(ctx, requestRows, maxNewFrames, onFrame, nil)
}

func (rt *OrtCpuRuntime) GenerateAudioFramesWithCallbackAndOverrides(ctx context.Context, requestRows map[string][][]int32, maxNewFrames int, onFrame FrameCallback, overrides *GenerationOverrides) [][]int {
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

	// 应用运行时参数覆盖
	if overrides != nil {
		if overrides.SampleMode != nil {
			sampleMode = *overrides.SampleMode
		}
		if overrides.DoSample != nil {
			gd["do_sample"] = *overrides.DoSample
		}
		if overrides.TextTemperature != nil {
			gd["text_temperature"] = *overrides.TextTemperature
		}
		if overrides.TextTopK != nil {
			gd["text_top_k"] = *overrides.TextTopK
		}
		if overrides.TextTopP != nil {
			gd["text_top_p"] = *overrides.TextTopP
		}
		if overrides.AudioTemperature != nil {
			gd["audio_temperature"] = *overrides.AudioTemperature
		}
		if overrides.AudioTopK != nil {
			gd["audio_top_k"] = *overrides.AudioTopK
		}
		if overrides.AudioTopP != nil {
			gd["audio_top_p"] = *overrides.AudioTopP
		}
		if overrides.AudioRepetitionPenalty != nil {
			gd["audio_repetition_penalty"] = *overrides.AudioRepetitionPenalty
		}
	}

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
	if err := rt.RunSession(rt.Onnx.Inference.Prefill.Session, prefillInputs, prefillOutputs); err != nil {
		log.Printf("  prefill 失败：%v", err)
		inputIDsTensor.Destroy()
		attentionMaskTensor.Destroy()
		return nil
	}
	inputIDsTensor.Destroy()
	attentionMaskTensor.Destroy()
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.Inference.Prefill)

	namedPrefillOutputs := make(map[string]ort.Value)
	for i, name := range prefillOutputNames {
		namedPrefillOutputs[name] = prefillOutputs[i]
	}

	globalHidden := extractLastHidden(namedPrefillOutputs["global_hidden"])

	// 立即销毁 global_hidden tensor，数据已拷贝到 globalHidden 切片
	if gh, ok := namedPrefillOutputs["global_hidden"]; ok && gh != nil {
		gh.Destroy()
		namedPrefillOutputs["global_hidden"] = nil
		// 同步清理 prefillOutputs 中的引用
		for i, name := range prefillOutputNames {
			if name == "global_hidden" && i < len(prefillOutputs) {
				prefillOutputs[i] = nil
			}
		}
	}

	pastValidLength := int32(0)
	for _, v := range requestRows["attentionMask"][0] {
		pastValidLength += v
	}

	pastByName := make(map[string]ort.Value)
	for _, name := range prefillOutputNames[1:] {
		pastName := strings.Replace(name, "present_", "past_", 1)
		if v, ok := namedPrefillOutputs[name]; ok && v != nil {
			pastByName[pastName] = v
		}
	}

	var generatedFrames [][]int
	prevTokensByChannel := make([][]int, nvq)
	prevTokenSetsByChannel := make([]map[int]bool, nvq)
	for i := 0; i < nvq; i++ {
		prevTokenSetsByChannel[i] = make(map[int]bool)
	}

	decodeInputNames := toStrSlice(onnxInfo["decode_input_names"])
	decodeOutputNames := toStrSlice(onnxInfo["decode_output_names"])

	// 获取 audio_repetition_penalty 默认值（用于 local_greedy_frame）
	audioRepetitionPenalty := float32(1.0)
	if rp, ok := gd["audio_repetition_penalty"]; ok {
		audioRepetitionPenalty = float32(toFloat64(rp))
	}
	doSample := true
	if ds, ok := gd["do_sample"]; ok {
		doSample = toBool(ds)
	}

	for stepIndex := 0; stepIndex < maxNewFrames; stepIndex++ {
		select {
		case <-ctx.Done():
			log.Printf("  生成被取消")
			// 先销毁 pastByName，然后将 prefillOutputs 中对应的值设为 nil 避免 double-free
			for _, v := range pastByName {
				if v != nil {
					v.Destroy()
				}
			}
			// pastByName 中的值和 prefillOutputs 共享同一 ort.Value 对象
			// 需要将 prefillOutputs 中已销毁的设为 nil
			destroyedSet := make(map[ort.Value]bool)
			for _, v := range pastByName {
				if v != nil {
					destroyedSet[v] = true
				}
			}
			for i, v := range prefillOutputs {
				if v != nil && destroyedSet[v] {
					prefillOutputs[i] = nil
				}
			}
			for _, v := range prefillOutputs {
				if v != nil {
					v.Destroy()
				}
			}
			return nil
		default:
		}

		var frame []int
		shouldContinue := true

		if rt.Onnx.Inference.LocalGreedyFrame != nil && !doSample {
			shouldContinue, frame = rt.runLocalGreedyFrame(
				globalHidden, prevTokenSetsByChannel, nvq, audioCodebookSize, audioRepetitionPenalty)
		} else if rt.Onnx.Inference.LocalFixedSampledFrame != nil && (sampleMode == sampler.SampleModeFixed || (!doSample && rt.Onnx.Inference.LocalGreedyFrame == nil)) {
			// greedy 模式且无 local_greedy_frame 时，也使用 local_fixed_sampled_frame
			// 因为 argmax 采样会产生静音帧，回退到与 fixed 模式相同的概率采样
			shouldContinue, frame = rt.runLocalFixedSampledFrame(
				globalHidden, prevTokenSetsByChannel, nvq, audioCodebookSize)
		} else if rt.Onnx.Inference.LocalCachedStep != nil {
			// local_past 每帧重置，与 Python 行为一致
			localPast := make(map[string][]float32)
			localPVL := 0
			shouldContinue, frame, _, _ = rt.runLocalCachedStepFull(
				globalHidden, prevTokensByChannel, prevTokenSetsByChannel, nvq, audioCodebookSize, gd, localPast, localPVL)
		} else if rt.Onnx.Inference.LocalDecoder != nil {
			// local_decoder 兜底路径：无 KV cache 的原始解码器
			shouldContinue, frame = rt.runLocalDecoderFull(
				globalHidden, prevTokensByChannel, prevTokenSetsByChannel, nvq, audioCodebookSize, gd)
		} else {
			log.Printf("  step %d: 无可用的 local 模型", stepIndex)
			break
		}

		// 按照 Python 源码的顺序：先检查停止信号，如果停止则不添加帧
		if !shouldContinue {
			log.Printf("  step %d: 停止信号 (frame=nil=%v)", stepIndex, frame == nil)
			break
		}

		// 添加帧到结果中
		for ci, token := range frame {
			prevTokensByChannel[ci] = append(prevTokensByChannel[ci], token)
			prevTokenSetsByChannel[ci][token] = true
		}
		generatedFrames = append(generatedFrames, frame)

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
		if err := rt.RunSession(rt.Onnx.Inference.Decode.Session, decodeInputs, decodeOutputs); err != nil {
			log.Printf("  decode_step 失败：%v", err)
			nextRowTensor.Destroy()
			pvlTensor.Destroy()
			break
		}
		nextRowTensor.Destroy()
		pvlTensor.Destroy()
		// 更新 Session 使用时间
		rt.UpdateSessionTime(rt.Onnx.Inference.Decode)

		namedDecodeOutputs := make(map[string]ort.Value)
		for i, name := range decodeOutputNames {
			namedDecodeOutputs[name] = decodeOutputs[i]
		}

		globalHidden = extractLastHidden(namedDecodeOutputs["global_hidden"])

		// 立即销毁 global_hidden tensor，数据已拷贝到 globalHidden 切片
		if gh, ok := namedDecodeOutputs["global_hidden"]; ok && gh != nil {
			gh.Destroy()
			namedDecodeOutputs["global_hidden"] = nil
		}

		pastValidLength++

		// 销毁旧的 past 值
		for _, oldPast := range pastByName {
			oldPast.Destroy()
		}

		// 销毁 decodeOutputs 中非 past 的输出（如 logits 等），避免内存累积
		for _, name := range decodeOutputNames {
			if name == "global_hidden" {
				continue // 已销毁
			}
			isPast := strings.HasPrefix(name, "present_")
			if !isPast {
				if v, ok := namedDecodeOutputs[name]; ok && v != nil {
					v.Destroy()
					namedDecodeOutputs[name] = nil
				}
			}
		}

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
		if v != nil {
			v.Destroy()
		}
	}
	// pastByName 中的值和 prefillOutputs 共享同一 ort.Value 对象
	// 需要将 prefillOutputs 中已销毁的设为 nil 避免 double-free
	destroyedSet := make(map[ort.Value]bool)
	for _, v := range pastByName {
		if v != nil {
			destroyedSet[v] = true
		}
	}
	for i, v := range prefillOutputs {
		if v != nil && destroyedSet[v] {
			prefillOutputs[i] = nil
		}
	}
	for _, v := range prefillOutputs {
		if v != nil {
			v.Destroy()
		}
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

func (rt *OrtCpuRuntime) runLocalDecoder(globalHidden []float32, textTokenID int, framePrefix []int, nvq int) ([]float32, []float32) {
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	audioPad := int(toFloat64(ttsConfig["audio_pad_token_id"]))

	ghShape := []int64{1, int64(len(globalHidden))}
	ghTensor, _ := ort.NewTensor(ghShape, globalHidden)
	ttTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(textTokenID)})

	paddedPrefix := make([]int32, nvq-1)
	for i := range paddedPrefix {
		paddedPrefix[i] = int32(audioPad)
	}
	for i := 0; i < len(framePrefix) && i < nvq-1; i++ {
		paddedPrefix[i] = int32(framePrefix[i])
	}
	apShape := []int64{1, int64(nvq - 1)}
	apTensor, _ := ort.NewTensor(apShape, paddedPrefix)

	inputs := []ort.Value{ghTensor, ttTensor, apTensor}
	outputs := make([]ort.Value, 2)

	err := rt.RunSession(rt.Onnx.Inference.LocalDecoder.Session, inputs, outputs)
	ghTensor.Destroy()
	ttTensor.Destroy()
	apTensor.Destroy()

	if err != nil {
		log.Printf("  local_decoder 失败: %v", err)
		return nil, nil
	}
	rt.UpdateSessionTime(rt.Onnx.Inference.LocalDecoder)

	textLogits := getFloat32Data(outputs[0])
	audioLogits := getFloat32Data(outputs[1])

	outputs[0].Destroy()
	outputs[1].Destroy()
	return textLogits, audioLogits
}

func (rt *OrtCpuRuntime) runLocalDecoderFull(globalHidden []float32, prevTokens [][]int, prevTokenSets []map[int]bool, nvq, audioCodebookSize int, gd map[string]interface{}) (bool, []int) {
	ttsConfig := rt.Manifest["tts_config"].(map[string]interface{})
	audioAssistSlotID := int(toFloat64(ttsConfig["audio_assistant_slot_token_id"]))

	doSample := true
	if ds, ok := gd["do_sample"]; ok {
		doSample = toBool(ds)
	}
	textTemperature := 1.0
	if t, ok := gd["text_temperature"]; ok {
		textTemperature = toFloat64(t)
	}
	textTopK := 50
	if k, ok := gd["text_top_k"]; ok {
		textTopK = int(toFloat64(k))
	}
	textTopP := 1.0
	if p, ok := gd["text_top_p"]; ok {
		textTopP = toFloat64(p)
	}
	audioTemperature := 0.8
	if t, ok := gd["audio_temperature"]; ok {
		audioTemperature = toFloat64(t)
	}
	audioTopK := 25
	if k, ok := gd["audio_top_k"]; ok {
		audioTopK = int(toFloat64(k))
	}
	audioTopP := 0.95
	if p, ok := gd["audio_top_p"]; ok {
		audioTopP = toFloat64(p)
	}
	audioRepetitionPenalty := float32(1.0)
	if rp, ok := gd["audio_repetition_penalty"]; ok {
		audioRepetitionPenalty = float32(toFloat64(rp))
	}

	localTextLogits, _ := rt.runLocalDecoder(globalHidden, 0, nil, nvq)
	if localTextLogits == nil {
		return false, nil
	}

	nextTextToken := sampler.SampleAssistantTextToken(localTextLogits, audioAssistSlotID, int(toFloat64(ttsConfig["audio_end_token_id"])), doSample, textTemperature, textTopK, textTopP, rt.RNG)
	if nextTextToken != audioAssistSlotID {
		return false, nil
	}

	var frame []int
	for ci := 0; ci < nvq; ci++ {
		_, audioLogits := rt.runLocalDecoder(globalHidden, nextTextToken, frame, nvq)
		if audioLogits == nil {
			break
		}
		perChannel := audioCodebookSize
		startOff := ci * perChannel
		endOff := minInt(startOff+perChannel, len(audioLogits))
		channelLogits := make([]float32, perChannel)
		if startOff < len(audioLogits) {
			copy(channelLogits, audioLogits[startOff:endOff])
		}
		sampledToken := sampler.SampleAudioToken(channelLogits, prevTokens[ci], prevTokenSets[ci], doSample, audioTemperature, audioTopK, audioTopP, audioRepetitionPenalty, rt.RNG)
		frame = append(frame, sampledToken)
		prevTokens[ci] = append(prevTokens[ci], sampledToken)
		prevTokenSets[ci][sampledToken] = true
	}
	return true, frame
}

func (rt *OrtCpuRuntime) runLocalGreedyFrame(globalHidden []float32, prevTokenSets []map[int]bool, nvq, audioCodebookSize int, repetitionPenalty float32) (bool, []int) {
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

	repPenaltyTensor, _ := ort.NewTensor([]int64{1}, []float32{repetitionPenalty})

	inputs := []ort.Value{ghTensor, repMaskTensor, repPenaltyTensor}
	outputs := make([]ort.Value, 2)

	err := rt.RunSession(rt.Onnx.Inference.LocalGreedyFrame.Session, inputs, outputs)
	ghTensor.Destroy()
	repMaskTensor.Destroy()
	repPenaltyTensor.Destroy()

	if err != nil {
		log.Printf("  local_greedy_frame 失败: %v", err)
		return false, nil
	}
	rt.UpdateSessionTime(rt.Onnx.Inference.LocalGreedyFrame)

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

	err := rt.RunSession(rt.Onnx.Inference.LocalFixedSampledFrame.Session, inputs, outputs)
	ghTensor.Destroy()
	repMaskTensor.Destroy()
	assRUTensor.Destroy()
	audioRUTensor.Destroy()

	if err != nil {
		log.Printf("  local_fixed_sampled_frame 失败: %v", err)
		return false, nil
	}
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.Inference.LocalFixedSampledFrame)

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
	audioAssistSlotID := int(toFloat64(ttsConfig["audio_assistant_slot_token_id"]))
	audioEndTokenID := int(toFloat64(ttsConfig["audio_end_token_id"]))

	// 使用缓存的元数据，避免每帧17次重复查找
	meta := &rt.cachedStepMeta
	if !meta.initialized {
		onnxInfo := rt.TTSMeta["onnx"].(map[string]interface{})
		meta.inputNames = toStrSlice(onnxInfo["local_cached_input_names"])
		meta.outputNames = toStrSlice(onnxInfo["local_cached_output_names"])
		meta.localHeads = int64(8)
		meta.localHeadDim = int64(64)
		if mc, ok := rt.TTSMeta["model_config"].(map[string]interface{}); ok {
			if lh, exists := mc["local_heads"]; exists {
				meta.localHeads = int64(toFloat64(lh))
			}
			if lhd, exists := mc["local_head_dim"]; exists {
				meta.localHeadDim = int64(toFloat64(lhd))
			}
		}
		meta.initialized = true
	}

	doSample := true
	if ds, ok := gd["do_sample"]; ok {
		doSample = toBool(ds)
	}
	textTemperature := 1.0
	if t, ok := gd["text_temperature"]; ok {
		textTemperature = toFloat64(t)
	} else if t, ok := gd["temperature"]; ok {
		textTemperature = toFloat64(t)
	}
	textTopK := 50
	if k, ok := gd["text_top_k"]; ok {
		textTopK = int(toFloat64(k))
	} else if k, ok := gd["top_k"]; ok {
		textTopK = int(toFloat64(k))
	}
	textTopP := 1.0
	if p, ok := gd["text_top_p"]; ok {
		textTopP = toFloat64(p)
	} else if p, ok := gd["top_p"]; ok {
		textTopP = toFloat64(p)
	}
	audioTemperature := 0.8
	if t, ok := gd["audio_temperature"]; ok {
		audioTemperature = toFloat64(t)
	}
	audioTopK := 25
	if k, ok := gd["audio_top_k"]; ok {
		audioTopK = int(toFloat64(k))
	}
	audioTopP := 0.95
	if p, ok := gd["audio_top_p"]; ok {
		audioTopP = toFloat64(p)
	}
	audioRepetitionPenalty := float32(1.0)
	if rp, ok := gd["audio_repetition_penalty"]; ok {
		audioRepetitionPenalty = float32(toFloat64(rp))
	}

	// 使用优化的 runCachedStepOpt，直接传递 ort.Value KV cache 避免数据拷贝
	ghShape := []int64{1, int64(len(globalHidden))}
	// 创建 globalHidden tensor 一次，所有17步复用
	ghTensor, _ := ort.NewTensor(ghShape, globalHidden)
	defer ghTensor.Destroy()

	// 将 map[string][]float32 转换为 map[string]ort.Value
	localPastValues := make(map[string]ort.Value)
	for k, v := range localPast {
		if len(v) == 0 {
			pastShape := []int64{1, 0, meta.localHeads, meta.localHeadDim}
			t, _ := ort.NewTensor(pastShape, make([]float32, 0))
			localPastValues[k] = t
		} else {
			seqLen := int64(len(v)) / (meta.localHeads * meta.localHeadDim)
			pastShape := []int64{1, seqLen, meta.localHeads, meta.localHeadDim}
			t, _ := ort.NewTensor(pastShape, v)
			localPastValues[k] = t
		}
	}

	textLogits, _, _, localPastValuesNew := rt.runCachedStepOpt(ghTensor, 0, 0, 0, 0, localPVL, localPastValues, meta)
	localPVL++

	// 销毁旧的 past values
	for _, v := range localPastValues {
		if v != nil {
			v.Destroy()
		}
	}

	if textLogits == nil {
		// 销毁新创建的 past values
		for _, v := range localPastValuesNew {
			if v != nil {
				v.Destroy()
			}
		}
		return false, nil, localPast, localPVL
	}

	nextTextToken := sampler.SampleAssistantTextToken(textLogits, audioAssistSlotID, audioEndTokenID, doSample, textTemperature, textTopK, textTopP, rt.RNG)
	if nextTextToken != audioAssistSlotID {
		// 销毁新创建的 past values
		for _, v := range localPastValuesNew {
			if v != nil {
				v.Destroy()
			}
		}
		return false, nil, localPast, localPVL
	}

	_, audioLogits, audioLogitsShape, localPastValuesNew2 := rt.runCachedStepOpt(ghTensor, nextTextToken, 0, 0, 1, localPVL, localPastValuesNew, meta)
	localPVL++

	// 销毁上一轮的 past values
	for _, v := range localPastValuesNew {
		if v != nil {
			v.Destroy()
		}
	}

	if audioLogits == nil {
		for _, v := range localPastValuesNew2 {
			if v != nil {
				v.Destroy()
			}
		}
		return false, nil, localPast, localPVL
	}

	perChannel := int(audioLogitsShape[len(audioLogitsShape)-1])
	firstChannelLogits := make([]float32, perChannel)
	copy(firstChannelLogits, audioLogits[:minInt(perChannel, len(audioLogits))])

	sampledToken := sampler.SampleAudioToken(firstChannelLogits, prevTokens[0], prevTokenSets[0], doSample, audioTemperature, audioTopK, audioTopP, audioRepetitionPenalty, rt.RNG)
	frame := []int{sampledToken}
	previousToken := sampledToken

	currentPastValues := localPastValuesNew2
	for ci := 1; ci < nvq; ci++ {
		_, audioLogits2, audioLogitsShape2, nextPastValues := rt.runCachedStepOpt(ghTensor, 0, previousToken, ci-1, 2, localPVL, currentPastValues, meta)
		localPVL++

		if audioLogits2 == nil {
			// 销毁剩余的 past values
			for _, v := range currentPastValues {
				if v != nil {
					v.Destroy()
				}
			}
			break
		}

		// 销毁上一轮的 past values（currentPastValues 中的值已被 runCachedStepOpt 内部使用完毕）
		// 注意：runCachedStepOpt 不销毁输入的 past values，需要我们手动销毁
		for _, v := range currentPastValues {
			if v != nil {
				v.Destroy()
			}
		}
		currentPastValues = nextPastValues

		perChannel2 := int(audioLogitsShape2[len(audioLogitsShape2)-1])
		startOff := ci * perChannel2
		endOff := minInt(startOff+perChannel2, len(audioLogits2))
		channelLogits := make([]float32, perChannel2)
		if startOff < len(audioLogits2) {
			copy(channelLogits, audioLogits2[startOff:endOff])
		}

		sampledToken2 := sampler.SampleAudioToken(channelLogits, prevTokens[ci], prevTokenSets[ci], doSample, audioTemperature, audioTopK, audioTopP, audioRepetitionPenalty, rt.RNG)
		frame = append(frame, sampledToken2)
		previousToken = sampledToken2
	}

	// 销毁最后一轮的 past values
	for _, v := range currentPastValues {
		if v != nil {
			v.Destroy()
		}
	}

	return true, frame, localPast, localPVL
}

// runCachedStepOpt 是 runCachedStep 的优化版本，直接使用 ort.Value 传递 KV cache，避免数据拷贝
func (rt *OrtCpuRuntime) runCachedStepOpt(ghTensor ort.Value, textTokenID, audioTokenID, channelIndex, stepType, pastValidLength int, localPastValues map[string]ort.Value, meta *cachedStepMetadata) ([]float32, []float32, []int64, map[string]ort.Value) {
	ttTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(textTokenID)})
	atTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(audioTokenID)})
	ciTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(channelIndex)})
	stTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(stepType)})
	pvlTensor, _ := ort.NewTensor([]int64{1}, []int32{int32(pastValidLength)})

	inputs := make([]ort.Value, len(meta.inputNames))
	inputs[0] = ghTensor
	inputs[1] = ttTensor
	inputs[2] = atTensor
	inputs[3] = ciTensor
	inputs[4] = stTensor
	inputs[5] = pvlTensor
	for i := 6; i < len(meta.inputNames); i++ {
		if v, ok := localPastValues[meta.inputNames[i]]; ok && v != nil {
			inputs[i] = v
		} else {
			pastShape := []int64{1, 0, meta.localHeads, meta.localHeadDim}
			t, _ := ort.NewTensor(pastShape, make([]float32, 0))
			inputs[i] = t
		}
	}

	outputs := make([]ort.Value, len(meta.outputNames))
	err := rt.RunSession(rt.Onnx.Inference.LocalCachedStep.Session, inputs, outputs)

	ttTensor.Destroy()
	atTensor.Destroy()
	ciTensor.Destroy()
	stTensor.Destroy()
	pvlTensor.Destroy()
	// 销毁临时创建的空 past tensor（不是 localPastValues 中的）
	for i := 6; i < len(meta.inputNames); i++ {
		if v, ok := localPastValues[meta.inputNames[i]]; !ok || v == nil {
			if inputs[i] != nil {
				inputs[i].Destroy()
			}
		}
	}

	if err != nil {
		log.Printf("  local_cached_step 失败：%v", err)
		// 销毁 outputs
		for _, v := range outputs {
			if v != nil {
				v.Destroy()
			}
		}
		return nil, nil, nil, nil
	}
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.Inference.LocalCachedStep)

	var textLogits []float32
	var audioLogits []float32
	var audioLogitsShape []int64

	for i, name := range meta.outputNames {
		v := outputs[i]
		if v == nil {
			continue
		}
		if name == "text_logits" {
			textLogits = getFloat32Data(v)
			v.Destroy()
			outputs[i] = nil
		} else if name == "audio_logits" {
			audioLogits = getFloat32Data(v)
			audioLogitsShape = v.GetShape()
			v.Destroy()
			outputs[i] = nil
		}
	}

	// 保存 present_ 输出为下一轮的 past 输入，直接传递 ort.Value 避免数据拷贝
	nextLocalPastValues := make(map[string]ort.Value)
	for i, name := range meta.outputNames {
		if strings.HasPrefix(name, "local_present_") {
			pastName := strings.Replace(name, "local_present_", "local_past_", 1)
			if outputs[i] != nil {
				nextLocalPastValues[pastName] = outputs[i]
				outputs[i] = nil // 防止下面被销毁
			}
		}
	}

	// 销毁剩余未处理的 outputs
	for _, v := range outputs {
		if v != nil {
			v.Destroy()
		}
	}

	return textLogits, audioLogits, audioLogitsShape, nextLocalPastValues
}

func (rt *OrtCpuRuntime) DecodeFullAudio(generatedFrames [][]int) ([][]float32, int) {
	return rt.DecodeFullAudioWithContext(context.Background(), generatedFrames)
}

func (rt *OrtCpuRuntime) DecodeFullAudioWithContext(ctx context.Context, generatedFrames [][]int) ([][]float32, int) {
	if len(generatedFrames) == 0 || rt.Onnx.Inference.CodecDecode == nil {
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

	err := rt.RunSession(rt.Onnx.Inference.CodecDecode.Session, inputs, outputs)
	audioCodesTensor.Destroy()
	audioCodeLengthsTensor.Destroy()

	if err != nil {
		log.Printf("  codec_decode_full 失败：%v", err)
		return nil, 0
	}
	// 更新 Session 使用时间
	rt.UpdateSessionTime(rt.Onnx.Inference.CodecDecode)

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

	log.Printf("  解码音频: frames=%d channels=%d samples=%d audioShape=%v audioDataLen=%d", frameCount, numChannels, audioLength, audioShape, len(audioData))
	return channels, int(audioLength)
}

func (rt *OrtCpuRuntime) Close() {
	if rt.Onnx != nil {
		rt.destroyInferenceSessions()
		rt.destroyCodecEncodeSession()
	}
}

// destroyInferenceSessions 销毁阶段2（TTS推理）的所有 session
// 调用方需持有 SessionMutex
func (rt *OrtCpuRuntime) destroyInferenceSessions() {
	s := rt.Onnx
	if s == nil || s.Inference == nil {
		return
	}
	inf := s.Inference
	if inf.Prefill != nil {
		inf.Prefill.Session.Destroy()
		inf.Prefill = nil
	}
	if inf.Decode != nil {
		inf.Decode.Session.Destroy()
		inf.Decode = nil
	}
	if inf.LocalDecoder != nil {
		inf.LocalDecoder.Session.Destroy()
		inf.LocalDecoder = nil
	}
	if inf.LocalGreedyFrame != nil {
		inf.LocalGreedyFrame.Session.Destroy()
		inf.LocalGreedyFrame = nil
	}
	if inf.LocalCachedStep != nil {
		inf.LocalCachedStep.Session.Destroy()
		inf.LocalCachedStep = nil
	}
	if inf.LocalFixedSampledFrame != nil {
		inf.LocalFixedSampledFrame.Session.Destroy()
		inf.LocalFixedSampledFrame = nil
	}
	if inf.CodecDecode != nil {
		inf.CodecDecode.Session.Destroy()
		inf.CodecDecode = nil
	}
	if inf.CodecDecodeStep != nil {
		inf.CodecDecodeStep.Session.Destroy()
		inf.CodecDecodeStep = nil
	}
	s.Inference = nil
}

// destroyCodecEncodeSession 销毁阶段1（音频编码）的 CodecEncode session
// 调用方需持有 SessionMutex
func (rt *OrtCpuRuntime) destroyCodecEncodeSession() {
	s := rt.Onnx
	if s == nil || s.CodecEncode == nil {
		return
	}
	s.CodecEncode.Session.Destroy()
	s.CodecEncode = nil
}

// CheckAndReleaseIdleSessions 检查并释放所有空闲超时的 Session（导出方法）
// 当有活跃请求时跳过，避免销毁正在使用的 session
func (rt *OrtCpuRuntime) CheckAndReleaseIdleSessions() {
	if rt.SessionIdleTimeout <= 0 {
		return // 禁用超时释放
	}

	// 有活跃请求时不执行重置，避免销毁正在使用的 session
	if rt.ActiveRequests.Load() > 0 {
		log.Printf("[ORT] 跳过空闲 Session 检查：当前有 %d 个活跃请求", rt.ActiveRequests.Load())
		return
	}

	now := time.Now()
	count := 0
	log.Printf("[ORT] 开始检查空闲 Session (超时阈值：%v)...", rt.SessionIdleTimeout)

	// 检查每个 Session
	count += rt.checkAndReleaseOne("prefill", rt.Onnx.Inference.Prefill, now)
	count += rt.checkAndReleaseOne("decode", rt.Onnx.Inference.Decode, now)
	count += rt.checkAndReleaseOne("local_decoder", rt.Onnx.Inference.LocalDecoder, now)
	count += rt.checkAndReleaseOne("local_greedy_frame", rt.Onnx.Inference.LocalGreedyFrame, now)
	count += rt.checkAndReleaseOne("local_cached_step", rt.Onnx.Inference.LocalCachedStep, now)
	count += rt.checkAndReleaseOne("local_fixed_sampled_frame", rt.Onnx.Inference.LocalFixedSampledFrame, now)
	count += rt.checkAndReleaseOne("codec_encode", rt.Onnx.CodecEncode, now)
	count += rt.checkAndReleaseOne("codec_decode", rt.Onnx.Inference.CodecDecode, now)
	count += rt.checkAndReleaseOne("codec_decode_step", rt.Onnx.Inference.CodecDecodeStep, now)

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
// 当有活跃请求时跳过，避免销毁正在使用的 session
func (rt *OrtCpuRuntime) ResetSessions() error {
	return rt.resetSessionsInternal(false)
}

// ForceResetSessions 强制销毁并重新创建所有 ONNX Session，即使有活跃请求
// 用于推理完成后立即释放 ONNX Runtime 内存池
func (rt *OrtCpuRuntime) ForceResetSessions() error {
	return rt.resetSessionsInternal(true)
}

func (rt *OrtCpuRuntime) resetSessionsInternal(force bool) error {
	// 有活跃请求时不执行重置，避免销毁正在使用的 session（除非强制）
	if !force && rt.ActiveRequests.Load() > 0 {
		log.Printf("[ORT] 跳过 Session 重置：当前有 %d 个活跃请求", rt.ActiveRequests.Load())
		return nil
	}

	log.Printf("[ORT] 销毁旧 sessions...")

	rt.SessionMutex.Lock()
	rt.destroyInferenceSessions()
	rt.destroyCodecEncodeSession()
	rt.SessionMutex.Unlock()

	// 重新创建推理 sessions（阶段2）
	log.Printf("[ORT] 重新创建推理 sessions...")
	return rt.CreateSessions()
}

// DestroyAllSessions 销毁所有 ONNX Session（阶段1+阶段2），不重建
// 用于在编码参考音频前释放所有推理 session 内存，避免内存峰值叠加
func (rt *OrtCpuRuntime) DestroyAllSessions() {
	rt.SessionMutex.Lock()
	defer rt.SessionMutex.Unlock()

	log.Printf("[ORT] 销毁所有 sessions（阶段1+阶段2）...")
	rt.destroyInferenceSessions()
	rt.destroyCodecEncodeSession()
	log.Printf("[ORT] 所有 sessions 已销毁")
}

// DestroyCodecEncodeSession 销毁阶段1（音频编码）的 CodecEncode session
// 编码完成后立即调用，释放其 ONNX 内存池
func (rt *OrtCpuRuntime) DestroyCodecEncodeSession() {
	rt.SessionMutex.Lock()
	defer rt.SessionMutex.Unlock()

	log.Printf("[ORT] 销毁 CodecEncode session（阶段1）...")
	rt.destroyCodecEncodeSession()
}

// EnsureCodecEncodeSession 确保 CodecEncode session 存在，不存在则懒加载创建
func (rt *OrtCpuRuntime) EnsureCodecEncodeSession() error {
	rt.SessionMutex.Lock()
	defer rt.SessionMutex.Unlock()

	s := rt.Onnx
	if s == nil {
		return fmt.Errorf("OnnxSessions 未初始化")
	}
	if s.CodecEncode != nil {
		return nil // 已存在，无需创建
	}

	log.Printf("[ORT] 懒加载创建 CodecEncode session...")
	codecDir := filepath.Dir(rt.CodecMetaPath)
	codecFiles := rt.CodecMeta["files"].(map[string]interface{})

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("创建 SessionOptions 失败: %w", err)
	}
	defer sessionOptions.Destroy()

	if err := sessionOptions.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableExtended); err != nil {
		log.Printf("  警告: 设置图优化级别失败: %v", err)
	}
	if err := sessionOptions.AddSessionConfigEntry("session.enable_mem_pattern", "0"); err != nil {
		log.Printf("  警告: 设置 enable_mem_pattern 失败: %v", err)
	}
	if err := sessionOptions.AddSessionConfigEntry("session.enable_mem_reuse", "1"); err != nil {
		log.Printf("  警告: 设置 enable_mem_reuse 失败: %v", err)
	}
	if err := sessionOptions.AddSessionConfigEntry("session.use_arena", "0"); err != nil {
		log.Printf("  警告: 设置 use_arena 失败: %v", err)
	}
	if err := sessionOptions.SetIntraOpNumThreads(rt.ThreadCount); err != nil {
		log.Printf("  警告: 设置 IntraOp 线程数失败: %v", err)
	}
	if err := sessionOptions.SetInterOpNumThreads(1); err != nil {
		log.Printf("  警告: 设置 InterOp 线程数失败: %v", err)
	}

	onnxPath := filepath.Join(codecDir, fmt.Sprintf("%v", codecFiles["encode"]))
	sess, err := ort.NewDynamicAdvancedSession(onnxPath,
		[]string{"waveform", "input_lengths"},
		[]string{"audio_codes", "audio_code_lengths"},
		sessionOptions)
	if err != nil {
		return fmt.Errorf("创建 CodecEncode session 失败: %w", err)
	}
	s.CodecEncode = &TimedSession{Session: sess, LastUsed: time.Now()}
	log.Printf("[ORT] CodecEncode session 懒加载创建完成")
	return nil
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
