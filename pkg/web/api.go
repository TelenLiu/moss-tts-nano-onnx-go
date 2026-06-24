package web

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/audio"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/device"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
)

func (s *Server) handleProgress(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	ch := make(chan ProgressEvent, 64)
	s.mu.Lock()
	s.subscribers[ch] = struct{}{}
	existing := make([]ProgressEvent, len(s.Progress))
	copy(existing, s.Progress)
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		delete(s.subscribers, ch)
		s.mu.Unlock()
	}()

	for _, evt := range existing {
		data, _ := json.Marshal(evt)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		if evt.Done || evt.Error != "" {
			return
		}
	}

	for {
		select {
		case evt := <-ch:
			data, _ := json.Marshal(evt)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			if evt.Done || evt.Error != "" {
				return
			}
		case <-r.Context().Done():
			return
		case <-time.After(30 * time.Second):
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		}
	}
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	ready := s.Ready
	version := s.Version
	s.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ready":   ready,
		"version": version,
	})
}

func (s *Server) handleDeviceInfo(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	deviceInfo := s.DeviceInfo
	executionMode := s.ExecutionMode
	s.mu.RUnlock()

	// 构建响应
	response := map[string]interface{}{
		"cpu": map[string]interface{}{
			"num_cores":       deviceInfo.CPUInfo.NumCores,
			"num_threads":     deviceInfo.CPUInfo.NumThreads,
			"available_cores": deviceInfo.CPUInfo.AvailableCores,
			"core_freq_mhz":   deviceInfo.CPUInfo.CoreFreqMHz,
			"model_name":      deviceInfo.CPUInfo.ModelName,
		},
		"has_gpu":         deviceInfo.HasGPU,
		"has_cuda":        deviceInfo.HasCUDA,
		"has_coreml":      deviceInfo.HasCoreML,
		"execution_mode":  executionMode,
		"available_modes": device.GetAvailableModes(deviceInfo.HasGPU),
	}

	// 如果有GPU信息，添加到响应中
	if deviceInfo.HasGPU {
		response["gpu"] = map[string]interface{}{
			"available":     deviceInfo.GPUInfo.Available,
			"name":          deviceInfo.GPUInfo.Name,
			"vendor":        deviceInfo.GPUInfo.Vendor,
			"device_id":     deviceInfo.GPUInfo.DeviceID,
			"memory_mb":     deviceInfo.GPUInfo.MemoryMB,
			"compute_units": deviceInfo.GPUInfo.ComputeUnits,
		}
	}

	// 添加执行提供程序设备信息
	if len(deviceInfo.EpDevices) > 0 {
		epDevices := make([]map[string]string, len(deviceInfo.EpDevices))
		for i, ep := range deviceInfo.EpDevices {
			epDevices[i] = map[string]string{
				"name":   ep.Name,
				"vendor": ep.Vendor,
			}
		}
		response["ep_devices"] = epDevices
	}

	// 添加推理单元池信息
	s.mu.RLock()
	pool := s.Pool
	onnxCfg := s.OnnxConfig
	s.mu.RUnlock()
	if pool != nil {
		coreCPUs := 0
		if onnxCfg != nil {
			coreCPUs = onnxCfg.CoreCPUs
		}
		response["onnx_pool"] = map[string]interface{}{
			"work_cores":       pool.WorkCoreCount(),
			"reserve_cores":    pool.ReserveCoreCount(),
			"core_cpus":        coreCPUs,
			"pending_requests": pool.PendingCount(),
			"cores":            pool.Status(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleSynthesize(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	pool := s.Pool
	ready := s.Ready
	s.mu.RUnlock()

	if !ready || pool == nil {
		http.Error(w, "系统尚未就绪，请等待加载完成", http.StatusServiceUnavailable)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SynthesizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if req.Text == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}

	voice := req.Voice
	if voice == "" {
		voice = "Junhao"
	}
	sampleMode := req.SampleMode
	if sampleMode == "" {
		sampleMode = "fixed"
	}
	doSample := sampleMode != "greedy"
	maxNewFrames := req.MaxNewFrames
	if maxNewFrames <= 0 {
		// 与 Python 源码保持一致：默认 375 帧
		// 不同随机种子会影响模型语速，帧数不足会导致音频末尾被截断
		maxNewFrames = 375
		log.Debugf("[API] 使用默认 maxNewFrames: %d", maxNewFrames)
	}
	voiceCloneMaxTokens := req.VoiceCloneMaxTextTokens
	if voiceCloneMaxTokens <= 0 {
		voiceCloneMaxTokens = 75 // 与Python源码保持一致，保证长文本克隆质量
	}

	promptAudioPath := req.PromptAudioPath
	preloadId := req.PreloadID
	preloadAudioPath := ""

	// 处理 base64 参考音频：解码写入临时文件，合成完成后自动清理
	// audio_clone_gob 缓存基于音频内容 hash，相同内容重复上传会命中缓存
	var tempAudioPath string
	if req.PromptAudioB64 != "" {
		audioData, err := base64.StdEncoding.DecodeString(req.PromptAudioB64)
		if err != nil {
			http.Error(w, fmt.Sprintf("prompt_audio_b64 base64解码失败: %v", err), http.StatusBadRequest)
			return
		}
		if len(audioData) == 0 {
			http.Error(w, "prompt_audio_b64 音频数据为空", http.StatusBadRequest)
			return
		}

		tempFile, err := os.CreateTemp("", "prompt-speech-*.wav")
		if err != nil {
			http.Error(w, fmt.Sprintf("创建临时文件失败: %v", err), http.StatusInternalServerError)
			return
		}
		if _, err := tempFile.Write(audioData); err != nil {
			tempFile.Close()
			os.Remove(tempFile.Name())
			http.Error(w, fmt.Sprintf("写入临时文件失败: %v", err), http.StatusInternalServerError)
			return
		}
		tempFile.Close()

		tempAudioPath = tempFile.Name()
		promptAudioPath = tempAudioPath
		preloadId = ""
		preloadAudioPath = ""
		log.Infof("[API synthesize] base64参考音频已写入临时文件: %s (size=%d)", tempAudioPath, len(audioData))
	} else if req.DemoID != "" {
		// 选择 demo 时，以 demo 自身配置为准，忽略请求中的 preload_id
		s.mu.RLock()
		if demo, ok := s.DemoEntriesByID[req.DemoID]; ok {
			promptAudioPath = demo.Path
			if demo.PreloadID != "" {
				// demo 声明了 preloadId，使用 demo 自身的音频路径进行预加载
				preloadId = demo.PreloadID
				preloadAudioPath = demo.Path
			} else {
				// demo 未声明 preloadId，不使用 preload，直接使用 demo 音频
				preloadId = ""
				preloadAudioPath = ""
			}
		}
		s.mu.RUnlock()
	} else if preloadId != "" {
		// 未选择 demo 但指定了 preloadId，从 demo 列表中查找对应的音频路径
		s.mu.RLock()
		for _, demo := range s.DemoEntries {
			if demo.PreloadID == preloadId {
				preloadAudioPath = demo.Path
				promptAudioPath = demo.Path
				break
			}
		}
		s.mu.RUnlock()
	}

	// 清理 base64 参考音频临时文件（合成完成后自动删除）
	if tempAudioPath != "" {
		defer os.Remove(tempAudioPath)
	}

	seedVal := "nil"
	if req.Seed != nil {
		seedVal = fmt.Sprintf("%d", *req.Seed)
	}
	log.Infof("[API synthesize] text=%q voice=%q promptAudioPath=%q preloadId=%q sampleMode=%s maxNewFrames=%d voiceCloneMaxTokens=%d seed=%s stream=%v",
		req.Text, voice, promptAudioPath, preloadId, sampleMode, maxNewFrames, voiceCloneMaxTokens, seedVal, req.Stream)

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	go func() {
		<-ctx.Done()
		log.Infof("[API synthesize] 请求断开，取消合成")
	}()

	enableRobust := true
	enableWeText := false // 默认禁用WeTextProcessing，避免性能问题（首次调用耗时17秒）
	if req.EnableRobust != nil {
		enableRobust = *req.EnableRobust
	}
	if req.EnableWeText != nil {
		enableWeText = *req.EnableWeText
	}
	log.Debugf("[API synthesize] enableRobust=%v enableWeText=%v", enableRobust, enableWeText)

	// 构建采样参数覆盖
	overrides := &ortruntime.GenerationOverrides{
		TextTemperature:        req.TextTemperature,
		TextTopK:               req.TextTopK,
		TextTopP:               req.TextTopP,
		AudioTemperature:       req.AudioTemperature,
		AudioTopK:              req.AudioTopK,
		AudioTopP:              req.AudioTopP,
		AudioRepetitionPenalty: req.AudioRepetitionPenalty,
	}

	if req.Stream {
		s.handleStreamSynthesize(w, ctx, pool, req, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, maxNewFrames, voiceCloneMaxTokens, enableRobust, enableWeText, overrides)
		return
	}

	result, err := pool.SynthesizeWithContextEx(ctx,
		req.Text, voice, promptAudioPath, "",
		preloadId, preloadAudioPath,
		sampleMode, doSample, false,
		maxNewFrames, voiceCloneMaxTokens,
		enableRobust, enableWeText, req.Seed, overrides,
	)
	if err != nil {
		log.Errorf("[API synthesize] 合成失败: %v", err)
		http.Error(w, fmt.Sprintf("Synthesis failed: %v", err), http.StatusInternalServerError)
		return
	}

	log.Infof("[API synthesize] 合成成功: sampleRate=%d audioSamples=%d elapsed=%.2fs chunks=%d",
		result.SampleRate, result.AudioSamples, result.ElapsedSec, len(result.TextChunks))

	// 根据 format 参数选择编码格式
	outputFormat := req.Format
	if outputFormat == "" {
		outputFormat = "mp3"
	}

	var audioData []byte
	var actualFormat string
	if outputFormat == "mp3" {
		mp3Cfg := audio.DefaultMP3EncodeConfig
		if req.MP3SampleRate > 0 {
			mp3Cfg.SampleRate = req.MP3SampleRate
		}
		if req.MP3VBRQuality > 0 {
			mp3Cfg.VBRQuality = req.MP3VBRQuality
		}
		mp3Data, err := audio.EncodeMP3(result.Waveform, result.Channels, result.SampleRate, mp3Cfg)
		if err != nil {
			log.Warnf("[API synthesize] MP3 编码失败，回退 WAV: %v", err)
			audioData, _ = audio.EncodeWAV(result.Waveform, result.Channels, result.SampleRate)
			actualFormat = "wav"
		} else {
			audioData = mp3Data
			actualFormat = "mp3"
		}
	} else {
		audioData, _ = audio.EncodeWAV(result.Waveform, result.Channels, result.SampleRate)
		actualFormat = "wav"
	}

	resp := SynthesizeResponse{
		AudioPath:      "",
		AudioDataB64:   base64.StdEncoding.EncodeToString(audioData),
		SampleRate:     result.SampleRate,
		AudioSamples:   result.AudioSamples,
		ElapsedSeconds: result.ElapsedSec,
		Voice:          voice,
		TextChunks:     result.TextChunks,
		SampleMode:     result.SampleMode,
		DoSample:       result.DoSample,
		Format:         actualFormat,
		SeedUsed:       result.SeedUsed,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleStreamSynthesize(w http.ResponseWriter, ctx context.Context, pool *ttsruntime.Pool, req SynthesizeRequest, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode string, doSample bool, maxNewFrames, voiceCloneMaxTokens int, enableRobust, enableWeText bool, overrides *ortruntime.GenerationOverrides) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	chunkChan, err := pool.SynthesizeStreamEx(ctx,
		req.Text, voice, promptAudioPath,
		preloadId, preloadAudioPath,
		sampleMode, doSample,
		maxNewFrames, voiceCloneMaxTokens,
		enableRobust, enableWeText, req.Seed, overrides,
	)
	if err != nil {
		log.Errorf("[API synthesize stream] 流式合成启动失败: %v", err)
		return
	}

	// 判断输出格式：MP3 需要收集完整波形后编码，PCM 可以实时流式输出
	outputFormat := req.Format
	if outputFormat == "" {
		outputFormat = "mp3"
	}

	totalSamples := 0
	chunkCount := 0
	startTime := time.Now()

	if outputFormat == "mp3" {
		// MP3 模式：收集完整波形后编码返回
		var allWaveforms [][]float32
		var sampleRate, channels int
		var seedUsed int64
		for chunk := range chunkChan {
			select {
			case <-ctx.Done():
				log.Infof("[API synthesize stream] 请求断开，停止流式输出")
				return
			default:
			}
			if len(chunk.Waveform) == 0 {
				continue
			}
			if chunk.SeedUsed != 0 {
				seedUsed = chunk.SeedUsed
			}
			allWaveforms = append(allWaveforms, chunk.Waveform)
			sampleRate = chunk.SampleRate
			channels = chunk.Channels
			totalSamples += len(chunk.Waveform) / chunk.Channels
			chunkCount++
		}
		if len(allWaveforms) == 0 {
			return
		}
		waveform := audio.ConcatWaveforms(allWaveforms)
		mp3Cfg := audio.DefaultMP3EncodeConfig
		if req.MP3SampleRate > 0 {
			mp3Cfg.SampleRate = req.MP3SampleRate
		}
		if req.MP3VBRQuality > 0 {
			mp3Cfg.VBRQuality = req.MP3VBRQuality
		}
		mp3Data, err := audio.EncodeMP3(waveform, channels, sampleRate, mp3Cfg)
		if err != nil {
			log.Errorf("[API synthesize stream] MP3 编码失败: %v", err)
			http.Error(w, fmt.Sprintf("MP3 encoding failed: %v", err), http.StatusInternalServerError)
			return
		}
		elapsed := time.Since(startTime).Seconds()
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(mp3Data)))
		w.Header().Set("X-Audio-Sample-Rate", fmt.Sprintf("%d", sampleRate))
		w.Header().Set("X-Audio-Channels", fmt.Sprintf("%d", channels))
		w.Header().Set("X-Audio-Samples", fmt.Sprintf("%d", totalSamples))
		w.Header().Set("X-Elapsed-Seconds", fmt.Sprintf("%.2f", elapsed))
		w.Header().Set("X-Audio-Format", "mp3")
		w.Header().Set("X-Seed-Used", fmt.Sprintf("%d", seedUsed))
		w.Write(mp3Data)
		log.Infof("[API synthesize stream] MP3流式合成完成: chunks=%d totalSamples=%d elapsed=%.2fs mp3Size=%d",
			chunkCount, totalSamples, elapsed, len(mp3Data))
		return
	}

	// PCM 流式模式（默认 wav 格式走此路径）
	headersSent := false
	var seedUsed int64
	for chunk := range chunkChan {
		select {
		case <-ctx.Done():
			log.Infof("[API synthesize stream] 请求断开，停止流式输出")
			return
		default:
		}

		if len(chunk.Waveform) == 0 {
			continue
		}

		if chunk.SeedUsed != 0 {
			seedUsed = chunk.SeedUsed
		}

		if !headersSent {
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("Transfer-Encoding", "chunked")
			w.Header().Set("X-Audio-Codec", "pcm_f32le")
			w.Header().Set("X-Audio-Sample-Rate", fmt.Sprintf("%d", chunk.SampleRate))
			w.Header().Set("X-Audio-Channels", fmt.Sprintf("%d", chunk.Channels))
			w.Header().Set("X-Seed-Used", fmt.Sprintf("%d", seedUsed))
			w.WriteHeader(http.StatusOK)
			headersSent = true
		}

		pcmData := unsafe.Slice((*byte)(unsafe.Pointer(&chunk.Waveform[0])), len(chunk.Waveform)*4)

		if _, err := w.Write(pcmData); err != nil {
			log.Errorf("[API synthesize stream] 写入流失败: %v", err)
			return
		}
		flusher.Flush()

		totalSamples += len(chunk.Waveform) / chunk.Channels
		chunkCount++
	}

	elapsed := time.Since(startTime).Seconds()
	log.Infof("[API synthesize stream] 流式合成完成: chunks=%d totalSamples=%d elapsed=%.2fs",
		chunkCount, totalSamples, elapsed)
}

func (s *Server) handleVoices(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	pool := s.Pool
	ready := s.Ready
	s.mu.RUnlock()

	if !ready || pool == nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]map[string]string{})
		return
	}

	voices := pool.ListBuiltinVoices()
	type voiceInfo struct {
		Voice string `json:"voice"`
		Label string `json:"label"`
	}
	var infos []voiceInfo
	for _, v := range voices {
		label := fmt.Sprintf("%v", v["voice"])
		if l, ok := v["label"]; ok {
			label = fmt.Sprintf("%v", l)
		}
		infos = append(infos, voiceInfo{
			Voice: fmt.Sprintf("%v", v["voice"]),
			Label: label,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(infos)
}

func (s *Server) handleAudio(w http.ResponseWriter, r *http.Request) {
	filename := strings.TrimPrefix(r.URL.Path, "/api/audio/")
	filePath := filepath.Join(os.TempDir(), filename)
	log.Debugf("[API audio] 请求音频: %s (完整路径: %s)", filename, filePath)

	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		log.Debugf("[API audio] 音频文件不存在: %s", filePath)
		http.Error(w, "Audio file not found", http.StatusNotFound)
		return
	}
	if err != nil {
		log.Errorf("[API audio] 访问音频文件出错: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	log.Debugf("[API audio] 提供音频: %s (大小: %d bytes)", filePath, info.Size())
	// 根据文件扩展名设置 Content-Type
	if strings.HasSuffix(strings.ToLower(filename), ".mp3") {
		w.Header().Set("Content-Type", "audio/mpeg")
	} else {
		w.Header().Set("Content-Type", "audio/wav")
	}
	w.Header().Set("Content-Length", fmt.Sprintf("%d", info.Size()))
	http.ServeFile(w, r, filePath)
}

func (s *Server) handleDemos(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	demos := s.DemoEntries
	s.mu.RUnlock()

	type demoInfo struct {
		ID        string `json:"id"`
		Name      string `json:"name"`
		Text      string `json:"text"`
		PreloadID string `json:"preloadId,omitempty"`
	}
	var infos []demoInfo
	for _, d := range demos {
		infos = append(infos, demoInfo{
			ID:        d.ID,
			Name:      d.Name,
			Text:      d.Text,
			PreloadID: d.PreloadID,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(infos)
}

func (s *Server) handleDemoPromptAudio(w http.ResponseWriter, r *http.Request) {
	demoID := strings.TrimPrefix(r.URL.Path, "/api/demo-prompt-audio/")
	if demoID == "" {
		http.Error(w, "demo_id is required", http.StatusBadRequest)
		return
	}

	s.mu.RLock()
	demo, ok := s.DemoEntriesByID[demoID]
	s.mu.RUnlock()

	if !ok {
		http.Error(w, "demo not found", http.StatusNotFound)
		return
	}

	ext := filepath.Ext(demo.Path)
	if strings.ToLower(ext) == ".wav" {
		w.Header().Set("Content-Type", "audio/wav")
	} else {
		w.Header().Set("Content-Type", "application/octet-stream")
	}
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")

	http.ServeFile(w, r, demo.Path)
	log.Debugf("[Demo] 提供演示音频: %s -> %s", demoID, demo.Path)
}
