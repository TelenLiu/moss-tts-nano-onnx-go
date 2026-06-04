package web

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/http/pprof"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/audio"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/normalizer"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/runtime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type ProgressEvent struct {
	Phase      string  `json:"phase"`
	Message    string  `json:"message"`
	Percent    float64 `json:"percent"`
	Error      string  `json:"error,omitempty"`
	Done       bool    `json:"done"`
	File       string  `json:"file,omitempty"`
	BytesDone  int64   `json:"bytes_done,omitempty"`
	BytesTotal int64   `json:"bytes_total,omitempty"`
	SpeedMBps  float64 `json:"speed_mbps,omitempty"`
}

type DemoEntry struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Role      string `json:"role"`
	Text      string `json:"text"`
	Path      string `json:"-"`
	PreloadID string `json:"preloadId,omitempty"`
}

type Server struct {
	Cfg          *deps.Config
	CpuThreads   int
	MaxNewFrames int
	Host         string
	Port         int
	AppRoot      string

	mu              sync.RWMutex
	RuntimeManager  *runtime.RuntimeManager
	Runtime         *ttsruntime.OnnxTtsRuntime
	Ready           bool
	Progress        []ProgressEvent
	subscribers     map[chan ProgressEvent]struct{}
	DemoEntries     []DemoEntry
	DemoEntriesByID map[string]*DemoEntry
	AssetsDir       string
}

func NewServer(cfg *deps.Config, cpuThreads, maxNewFrames int, host string, port int, appRoot string) *Server {
	cwd, _ := os.Getwd()
	assetsDir := filepath.Join(cwd, "assets")
	demoPath := filepath.Join(assetsDir, "demo.jsonl")

	s := &Server{
		Cfg:             cfg,
		CpuThreads:      cpuThreads,
		MaxNewFrames:    maxNewFrames,
		Host:            host,
		Port:            port,
		AppRoot:         appRoot,
		subscribers:     make(map[chan ProgressEvent]struct{}),
		DemoEntriesByID: make(map[string]*DemoEntry),
		AssetsDir:       assetsDir,
	}

	if _, err := os.Stat(demoPath); err == nil {
		s.loadDemoEntries(demoPath, assetsDir)
		log.Printf("[Demo] 已加载 %d 个默认样例", len(s.DemoEntries))
	} else {
		log.Printf("[Demo] 未找到demo.jsonl，跳过加载: %v", err)
	}

	return s
}

func (s *Server) emit(evt ProgressEvent) {
	s.mu.Lock()
	s.Progress = append(s.Progress, evt)
	subs := make([]chan ProgressEvent, 0, len(s.subscribers))
	for ch := range s.subscribers {
		subs = append(subs, ch)
	}
	s.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- evt:
		default:
		}
	}
	log.Printf("[%s] %s (%.0f%%)", evt.Phase, evt.Message, evt.Percent)
}

func (s *Server) loadDemoEntries(demoPath, assetsDir string) {
	file, err := os.Open(demoPath)
	if err != nil {
		log.Printf("[Demo] 打开demo.jsonl失败: %v", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	demoIndex := 1
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var entry struct {
			Name      string `json:"name"`
			Role      string `json:"role"`
			Text      string `json:"text"`
			PreloadID string `json:"preloadId"`
		}
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			log.Printf("[Demo] 解析demo行失败: %v", err)
			continue
		}

		if entry.Role == "" || entry.Text == "" {
			continue
		}

		rolePath := entry.Role
		rolePath = strings.TrimPrefix(rolePath, "assets/")
		fullPath := filepath.Join(assetsDir, filepath.FromSlash(rolePath))
		if _, err := os.Stat(fullPath); err != nil {
			log.Printf("[Demo] 跳过demo，音频文件不存在: %s", fullPath)
			continue
		}

		demoID := fmt.Sprintf("demo-%d", demoIndex)
		demoEntry := DemoEntry{
			ID:        demoID,
			Name:      entry.Name,
			Role:      entry.Role,
			Text:      entry.Text,
			Path:      fullPath,
			PreloadID: entry.PreloadID,
		}
		s.DemoEntries = append(s.DemoEntries, demoEntry)
		s.DemoEntriesByID[demoID] = &s.DemoEntries[len(s.DemoEntries)-1]
		demoIndex++
	}

	if err := scanner.Err(); err != nil {
		log.Printf("[Demo] 读取demo.jsonl错误: %v", err)
	}
}

func (s *Server) Start() error {
	s.Cfg.OnProgress = func(p deps.DownloadProgress) {
		s.emit(ProgressEvent{
			Phase:      "download",
			Message:    p.Message,
			Percent:    p.Percent,
			File:       p.File,
			BytesDone:  p.BytesDone,
			BytesTotal: p.BytesTotal,
			SpeedMBps:  p.SpeedMBps,
		})
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/api/progress", s.handleProgress)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/synthesize", s.handleSynthesize)
	mux.HandleFunc("/api/voices", s.handleVoices)
	mux.HandleFunc("/api/audio/", s.handleAudio)
	mux.HandleFunc("/api/upload-prompt-audio", s.handleUploadPromptAudio)
	mux.HandleFunc("/api/demos", s.handleDemos)
	mux.HandleFunc("/api/demo-prompt-audio/", s.handleDemoPromptAudio)

	go s.backgroundInit()

	addr := fmt.Sprintf("%s:%d", s.Host, s.Port)
	displayAddr := addr
	if s.Host == "" {
		displayAddr = fmt.Sprintf("0.0.0.0:%d", s.Port)
	}
	log.Printf("Web 服务启动于 http://%s/ (监听所有网络接口)", displayAddr)
	return http.ListenAndServe(addr, mux)
}

func (s *Server) backgroundInit() {
	s.emit(ProgressEvent{Phase: "check", Message: "正在检测运行环境...", Percent: 5})

	s.emit(ProgressEvent{Phase: "load", Message: "正在初始化文本归一化引擎...", Percent: 8})
	s.emit(ProgressEvent{Phase: "load", Message: "文本归一化引擎后台加载中（如首次运行需构建 FST 缓存，约 5-10 分钟）", Percent: 9})

	s.emit(ProgressEvent{Phase: "download", Message: "正在检查 ONNX Runtime 本地依赖...", Percent: 10})
	deps.SetDynlibPath(s.Cfg.LibDir)
	if err := deps.EnsureNativeLibs(s.Cfg); err != nil {
		s.emit(ProgressEvent{Phase: "error", Message: fmt.Sprintf("ONNX Runtime 依赖准备失败: %v", err), Error: fmt.Sprintf("ONNX Runtime 依赖准备失败: %v", err)})
		return
	}
	s.emit(ProgressEvent{Phase: "download", Message: "ONNX Runtime 依赖已就绪", Percent: 25})

	s.emit(ProgressEvent{Phase: "download", Message: "正在检查模型文件...", Percent: 30})
	if err := deps.EnsureModels(s.Cfg); err != nil {
		s.emit(ProgressEvent{Phase: "error", Message: fmt.Sprintf("模型文件准备失败: %v", err), Error: fmt.Sprintf("模型文件准备失败: %v", err)})
		return
	}
	s.emit(ProgressEvent{Phase: "load", Message: "模型下载完成，正在加载...", Percent: 50})

	s.emit(ProgressEvent{Phase: "load", Message: "正在初始化 ONNX Runtime 环境...", Percent: 55})
	if err := ortruntime.InitializeORT(s.Cfg.LibDir); err != nil {
		s.emit(ProgressEvent{Phase: "error", Message: fmt.Sprintf("ONNX Runtime 初始化失败: %v", err), Error: fmt.Sprintf("ONNX Runtime 初始化失败: %v", err)})
		return
	}
	s.emit(ProgressEvent{Phase: "load", Message: "ONNX Runtime 环境初始化成功", Percent: 70})

	s.emit(ProgressEvent{Phase: "load", Message: "正在加载 TTS 模型...", Percent: 75})
	rt, err := ttsruntime.NewOnnxTtsRuntime(
		s.Cfg.ModelDir, s.CpuThreads,
		&s.MaxNewFrames, nil, nil,
	)
	if err != nil {
		s.emit(ProgressEvent{Phase: "error", Message: fmt.Sprintf("TTS 运行时初始化失败: %v", err), Error: fmt.Sprintf("TTS 运行时初始化失败: %v", err)})
		return
	}
	s.emit(ProgressEvent{Phase: "load", Message: "TTS 模型加载完成", Percent: 95})

	s.mu.Lock()
	s.Runtime = rt
	s.mu.Unlock()

	s.emit(ProgressEvent{Phase: "load", Message: "正在准备文本归一化引擎...", Percent: 96})
	normalizer.EnsureInitializedSync(func(msg string, pct float64) {
		s.emit(ProgressEvent{Phase: "load", Message: msg, Percent: pct})
	})
	s.emit(ProgressEvent{Phase: "load", Message: "文本归一化引擎就绪", Percent: 99})

	s.mu.Lock()
	s.Ready = true
	s.mu.Unlock()

	s.emit(ProgressEvent{Phase: "ready", Message: "系统就绪", Percent: 100, Done: true})
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")
	s.mu.RLock()
	ready := s.Ready
	s.mu.RUnlock()
	if ready {
		fmt.Fprint(w, ttsHTML)
	} else {
		fmt.Fprint(w, loadingHTML)
	}
}

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
	s.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ready": ready})
}

func (s *Server) handleSynthesize(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	rt := s.Runtime
	ready := s.Ready
	s.mu.RUnlock()

	if !ready || rt == nil {
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
		// 根据文本长度动态调整 max_new_frames
		// 经验值：中文文本每字约需 3-4 帧
		estimatedFrames := len(req.Text) * 4
		maxNewFrames = min(estimatedFrames+100, 2000) // 上限 2000 帧
		log.Printf("[API] 动态计算 maxNewFrames: textLen=%d estimated=%d final=%d", len(req.Text), estimatedFrames, maxNewFrames)
	}
	voiceCloneMaxTokens := req.VoiceCloneMaxTextTokens
	if voiceCloneMaxTokens <= 0 {
		voiceCloneMaxTokens = 300 // 默认值300，减少chunk数量，提高音频质量
	}

	promptAudioPath := req.PromptAudioPath
	preloadId := req.PreloadID
	preloadAudioPath := ""

	if req.UploadedPromptAudio != "" {
		promptAudioPath = req.UploadedPromptAudio
	} else if req.DemoID != "" {
		s.mu.RLock()
		if demo, ok := s.DemoEntriesByID[req.DemoID]; ok {
			promptAudioPath = demo.Path
			// 如果demo有preloadId，使用preloadId
			if demo.PreloadID != "" {
				preloadId = demo.PreloadID
				preloadAudioPath = demo.Path
			}
		}
		s.mu.RUnlock()
	}

	// 如果直接指定了preloadId，但没有preloadAudioPath，从demo中查找
	if preloadId != "" && preloadAudioPath == "" {
		s.mu.RLock()
		for _, demo := range s.DemoEntries {
			if demo.PreloadID == preloadId {
				preloadAudioPath = demo.Path
				break
			}
		}
		s.mu.RUnlock()
	}

	seedVal := "nil"
	if req.Seed != nil {
		seedVal = fmt.Sprintf("%d", *req.Seed)
	}
	log.Printf("[API synthesize] text=%q voice=%q promptAudioPath=%q preloadId=%q sampleMode=%s maxNewFrames=%d voiceCloneMaxTokens=%d seed=%s stream=%v",
		req.Text, voice, promptAudioPath, preloadId, sampleMode, maxNewFrames, voiceCloneMaxTokens, seedVal, req.Stream)

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	go func() {
		<-ctx.Done()
		log.Printf("[API synthesize] 请求断开，取消合成")
	}()

	enableRobust := true
	enableWeText := false // 默认禁用WeTextProcessing，避免性能问题（首次调用耗时17秒）
	if req.EnableRobust != nil {
		enableRobust = *req.EnableRobust
	}
	if req.EnableWeText != nil {
		enableWeText = *req.EnableWeText
	}

	if req.Stream {
		s.handleStreamSynthesize(w, ctx, rt, req, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode, doSample, maxNewFrames, voiceCloneMaxTokens, enableRobust, enableWeText)
		return
	}

	result, err := rt.SynthesizeWithContextEx(ctx,
		req.Text, voice, promptAudioPath, "",
		preloadId, preloadAudioPath,
		sampleMode, doSample, false,
		maxNewFrames, voiceCloneMaxTokens,
		enableRobust, enableWeText, req.Seed,
	)
	if err != nil {
		log.Printf("[API synthesize] 合成失败: %v", err)
		http.Error(w, fmt.Sprintf("Synthesis failed: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[API synthesize] 合成成功: sampleRate=%d audioSamples=%d elapsed=%.2fs chunks=%d",
		result.SampleRate, result.AudioSamples, result.ElapsedSec, len(result.TextChunks))

	resp := SynthesizeResponse{
		AudioPath:      "",
		AudioDataB64:   base64.StdEncoding.EncodeToString(result.AudioData),
		SampleRate:     result.SampleRate,
		AudioSamples:   result.AudioSamples,
		ElapsedSeconds: result.ElapsedSec,
		Voice:          voice,
		TextChunks:     result.TextChunks,
		SampleMode:     result.SampleMode,
		DoSample:       result.DoSample,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleStreamSynthesize(w http.ResponseWriter, ctx context.Context, rt *ttsruntime.OnnxTtsRuntime, req SynthesizeRequest, voice, promptAudioPath, preloadId, preloadAudioPath, sampleMode string, doSample bool, maxNewFrames, voiceCloneMaxTokens int, enableRobust, enableWeText bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	chunkChan, err := rt.SynthesizeStreamEx(ctx,
		req.Text, voice, promptAudioPath,
		preloadId, preloadAudioPath,
		sampleMode, doSample,
		maxNewFrames, voiceCloneMaxTokens,
		enableRobust, enableWeText, req.Seed,
	)
	if err != nil {
		log.Printf("[API synthesize stream] 流式合成启动失败: %v", err)
		return
	}

	totalSamples := 0
	chunkCount := 0
	startTime := time.Now()
	headersSent := false

	for chunk := range chunkChan {
		select {
		case <-ctx.Done():
			log.Printf("[API synthesize stream] 请求断开，停止流式输出")
			return
		default:
		}

		if len(chunk.Waveform) == 0 {
			continue
		}

		if !headersSent {
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("Transfer-Encoding", "chunked")
			w.Header().Set("X-Audio-Codec", "pcm_f32le")
			w.Header().Set("X-Audio-Sample-Rate", fmt.Sprintf("%d", chunk.SampleRate))
			w.Header().Set("X-Audio-Channels", fmt.Sprintf("%d", chunk.Channels))
			w.WriteHeader(http.StatusOK)
			headersSent = true
		}

		pcmData := make([]byte, len(chunk.Waveform)*4)
		for i, sample := range chunk.Waveform {
			bits := math.Float32bits(sample)
			binary.LittleEndian.PutUint32(pcmData[i*4:], bits)
		}

		if _, err := w.Write(pcmData); err != nil {
			log.Printf("[API synthesize stream] 写入流失败: %v", err)
			return
		}
		flusher.Flush()

		totalSamples += len(chunk.Waveform) / chunk.Channels
		chunkCount++
	}

	elapsed := time.Since(startTime).Seconds()
	log.Printf("[API synthesize stream] 流式合成完成: chunks=%d totalSamples=%d elapsed=%.2fs",
		chunkCount, totalSamples, elapsed)
}

func (s *Server) handleVoices(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	rt := s.Runtime
	ready := s.Ready
	s.mu.RUnlock()

	if !ready || rt == nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]map[string]string{})
		return
	}

	voices := rt.OrtRuntime.ListBuiltinVoices()
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
	log.Printf("[API audio] 请求音频: %s (完整路径: %s)", filename, filePath)

	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		log.Printf("[API audio] 音频文件不存在: %s", filePath)
		http.Error(w, "Audio file not found", http.StatusNotFound)
		return
	}
	if err != nil {
		log.Printf("[API audio] 访问音频文件出错: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	log.Printf("[API audio] 提供音频: %s (大小: %d bytes)", filePath, info.Size())
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", info.Size()))
	http.ServeFile(w, r, filePath)
}

func (s *Server) handleUploadPromptAudio(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		AudioDataB64 string `json:"audio_data_b64"`
		FileName     string `json:"file_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("解析请求失败: %v", err), http.StatusBadRequest)
		return
	}
	if req.AudioDataB64 == "" {
		http.Error(w, "audio_data_b64 不能为空", http.StatusBadRequest)
		return
	}

	audioData, err := base64.StdEncoding.DecodeString(req.AudioDataB64)
	if err != nil {
		http.Error(w, fmt.Sprintf("base64解码失败: %v", err), http.StatusBadRequest)
		return
	}
	if len(audioData) == 0 {
		http.Error(w, "音频数据为空", http.StatusBadRequest)
		return
	}

	ext := ".wav"
	if req.FileName != "" {
		e := filepath.Ext(req.FileName)
		if e != "" && len(e) <= 16 {
			ext = e
		}
	}

	tempFile, err := os.CreateTemp("", "prompt-speech-*"+ext)
	if err != nil {
		http.Error(w, fmt.Sprintf("创建临时文件失败：%v", err), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	if _, err := tempFile.Write(audioData); err != nil {
		os.Remove(tempFile.Name())
		http.Error(w, fmt.Sprintf("写入文件失败: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[API upload] 参考音频上传成功: name=%s size=%d path=%s", req.FileName, len(audioData), tempFile.Name())

	resp := map[string]string{
		"path":      tempFile.Name(),
		"name":      req.FileName,
		"file_size": fmt.Sprintf("%d", len(audioData)),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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
	log.Printf("[Demo] 提供演示音频: %s -> %s", demoID, demo.Path)
}

type SynthesizeRequest struct {
	Text                    string `json:"text"`
	Voice                   string `json:"voice"`
	DemoID                  string `json:"demo_id"`
	PreloadID               string `json:"preload_id"`
	PromptAudioPath         string `json:"prompt_audio_path"`
	UploadedPromptAudio     string `json:"uploaded_prompt_audio"`
	SampleMode              string `json:"sample_mode"`
	MaxNewFrames            int    `json:"max_new_frames"`
	VoiceCloneMaxTextTokens int    `json:"voice_clone_max_text_tokens"`
	Seed                    *int   `json:"seed"`
	Stream                  bool   `json:"stream"`
	EnableRobust            *bool  `json:"enable_robust"`
	EnableWeText            *bool  `json:"enable_wetext"`
}

type SynthesizeResponse struct {
	AudioPath      string   `json:"audio_path"`
	AudioDataB64   string   `json:"audio_data_b64,omitempty"`
	SampleRate     int      `json:"sample_rate"`
	AudioSamples   int      `json:"audio_samples"`
	ElapsedSeconds float64  `json:"elapsed_seconds"`
	Voice          string   `json:"voice"`
	TextChunks     []string `json:"text_chunks"`
	SampleMode     string   `json:"sample_mode"`
	DoSample       bool     `json:"do_sample"`
}

var _ = audio.WriteWAV

const loadingHTML = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MOSS-TTS-Nano - 加载中</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:640px;margin:60px auto;padding:0 20px;background:#f8f9fa;color:#333}
h1{color:#1a73e8;font-size:26px;margin-bottom:4px}
.subtitle{color:#888;font-size:14px;margin-bottom:28px}
.phase-card{background:#fff;border-radius:12px;padding:24px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.phase-title{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.phase-title.download{color:#e8710a}
.phase-title.load{color:#1a73e8}
.phase-title.ready{color:#0d652d}
.phase-title.error{color:#d93025}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot.download{background:#e8710a}
.dot.load{background:#1a73e8}
.dot.ready{background:#0d652d}
.dot.error{background:#d93025}
.bar-bg{background:#e9ecef;border-radius:6px;height:14px;overflow:hidden;margin:8px 0}
.bar-fill{height:100%;border-radius:6px;transition:width .4s ease}
.bar-fill.download{background:linear-gradient(90deg,#f9a825,#fb8c00)}
.bar-fill.load{background:linear-gradient(90deg,#42a5f5,#1a73e8)}
.bar-fill.ready{background:linear-gradient(90deg,#66bb6a,#2e7d32)}
.pct-row{display:flex;justify-content:space-between;font-size:12px;color:#888;margin-top:4px}
.speed{font-weight:600;color:#e8710a}
.log-area{margin-top:12px;max-height:160px;overflow-y:auto;font-size:11px;color:#999;font-family:'SF Mono',Menlo,monospace;line-height:1.6;background:#f8f9fa;border-radius:8px;padding:10px}
.log-area div{padding:1px 0;border-bottom:1px solid #eee}
.log-area div.error{color:#d93025;font-weight:600}
.spinner{display:inline-block;width:14px;height:14px;border:2.5px solid #e9ecef;border-top-color:#1a73e8;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<h1>MOSS-TTS-Nano ONNX</h1>
<p class="subtitle">语音合成系统正在初始化，请稍候...</p>

<div class="phase-card" id="dl-card">
<div class="phase-title download"><span class="dot download"></span> 下载阶段</div>
<div class="bar-bg"><div class="bar-fill download" id="dl-bar" style="width:0%"></div></div>
<div class="pct-row"><span id="dl-pct">0%</span><span id="dl-detail"></span></div>
</div>

<div class="phase-card" id="ld-card">
<div class="phase-title load"><span class="dot load"></span> 加载阶段</div>
<div class="bar-bg"><div class="bar-fill load" id="ld-bar" style="width:0%"></div></div>
<div class="pct-row"><span id="ld-pct">等待中...</span><span id="ld-detail"></span></div>
</div>

<div class="log-area" id="log"></div>

<script>
const dlBar=document.getElementById('dl-bar'),dlPct=document.getElementById('dl-pct'),dlDetail=document.getElementById('dl-detail');
const ldBar=document.getElementById('ld-bar'),ldPct=document.getElementById('ld-pct'),ldDetail=document.getElementById('ld-detail');
const logEl=document.getElementById('log');
function addLog(msg,cls){const d=document.createElement('div');if(cls)d.className=cls;d.textContent=msg;logEl.appendChild(d);logEl.scrollTop=logEl.scrollHeight}
function fmtBytes(b){if(b>=1073741824)return(b/1073741824).toFixed(2)+' GB';if(b>=1048576)return(b/1048576).toFixed(1)+' MB';if(b>=1024)return(b/1024).toFixed(1)+' KB';return b+' B'}

const es=new EventSource('/api/progress');
let dlPctVal=0,ldPctVal=0;
es.onmessage=function(e){
  const d=JSON.parse(e.data);
  addLog(d.message,d.error?'error':'');
  if(d.phase==='download'){
    dlPctVal=Math.max(dlPctVal,d.percent||0);
    dlBar.style.width=dlPctVal+'%';
    dlPct.textContent=dlPctVal.toFixed(0)+'%';
    let detail='';
    if(d.bytes_total>0)detail=fmtBytes(d.bytes_done||0)+' / '+fmtBytes(d.bytes_total);
    if(d.speed_mbps>0)detail+=(detail?' · ':'')+d.speed_mbps.toFixed(1)+' MB/s';
    if(d.file)detail=(detail?' · ':'')+d.file;
    dlDetail.textContent=detail;
  }else if(d.phase==='load'){
    ldPctVal=Math.max(ldPctVal,d.percent||0);
    ldBar.style.width=ldPctVal+'%';
    ldPct.textContent=d.message;
  }else if(d.phase==='ready'){
    es.close();
    dlBar.style.width='100%';dlPct.textContent='100%';dlDetail.textContent='完成';
    ldBar.style.width='100%';ldPct.textContent='完成';
    document.getElementById('dl-card').querySelector('.phase-title').className='phase-title ready';
    document.getElementById('ld-card').querySelector('.phase-title').className='phase-title ready';
    addLog('系统就绪，正在跳转...');
    setTimeout(()=>{window.location.reload()},1200);
  }else if(d.phase==='error'){
    es.close();
    document.getElementById('dl-card').querySelector('.phase-title').className='phase-title error';
    document.getElementById('ld-card').querySelector('.phase-title').className='phase-title error';
  }else{
    ldPct.textContent=d.message;
  }
};
es.onerror=function(){es.close();addLog('连接断开，3秒后重试...','error');setTimeout(()=>{window.location.reload()},3000)};
addLog('正在连接服务器...');
</script>
</body>
</html>
`

const ttsHTML = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>MOSS-TTS-Nano Demo</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#f8f9fa;color:#333}
h1{color:#1a73e8}
.field{margin-bottom:16px}
label{display:block;font-weight:600;margin-bottom:4px}
textarea,select,input{width:100%;padding:8px;border:1px solid #ddd;border-radius:4px;font-size:14px;box-sizing:border-box}
textarea{min-height:80px;resize:vertical}
button{background:#1a73e8;color:white;border:none;padding:10px 24px;border-radius:4px;cursor:pointer;font-size:16px}
button:hover{background:#1557b0}
button:disabled{background:#ccc;cursor:not-allowed}
button.secondary{background:#fff;color:#1a73e8;border:1px solid #1a73e8;padding:6px 16px;font-size:14px;margin-right:8px}
button.secondary:hover{background:#e8f0fe}
.result{margin-top:20px;padding:16px;background:white;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
audio{width:100%;margin-top:12px}
.meta{font-size:12px;color:#666;margin-top:4px}
.error{color:#d93025}
.badge{display:inline-block;background:#e8f0fe;color:#1a73e8;padding:2px 8px;border-radius:12px;font-size:11px;margin-left:8px}
.prompt-audio-box{background:#fff;border:1px solid #ddd;border-radius:4px;padding:12px}
.prompt-audio-box input[type="file"]{border:none;padding:4px 0}
.prompt-audio-box audio{margin-top:8px}
.prompt-audio-actions{margin-top:8px}

.row{display:flex;gap:12px}
.row .field{flex:1}
.details-summary{cursor:pointer;font-weight:600;color:#1a73e8}
details{background:#fff;border:1px solid #ddd;border-radius:4px;padding:12px}
</style>
</head>
<body>
<h1>MOSS-TTS-Nano ONNX Demo <span class="badge">Ready</span></h1>

<div class="field">
  <label for="demo">演示样例</label>
  <select id="demo" onchange="onDemoChange()"></select>
  <div class="meta">选择预置的演示样例，会自动填入文本并加载参考音频</div>
</div>

<div class="field"><label for="text">输入文本</label><textarea id="text" placeholder="请输入要合成的文本...">你好，欢迎使用MOSS语音合成系统。</textarea></div>

<div class="field">
  <label for="voice">内置音色</label>
  <select id="voice" onchange="onVoiceChange()"></select>
  <div class="meta">选择内置音色</div>
</div>

<div class="field">
  <label for="prompt-audio-upload">参考音频 (克隆音色)</label>
  <div class="prompt-audio-box">
    <input id="prompt-audio-upload" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.aac" onchange="onPromptAudioChange()">
    <audio id="prompt-audio-preview" controls hidden></audio>
    <div id="prompt-audio-source" class="meta">使用内置音色，未上传参考音频。</div>
    <div class="prompt-audio-actions">
      <button id="choose-prompt-btn" class="secondary" type="button" hidden onclick="choosePromptAudio()">选择文件</button>
      <button id="clear-prompt-btn" class="secondary" type="button" hidden onclick="clearPromptAudio()">使用默认</button>
    </div>
  </div>
</div>

<details>
  <summary class="details-summary">高级选项</summary>
  <div class="row" style="margin-top:12px;">
    <div class="field">
      <label for="sample-mode">采样模式</label>
      <select id="sample-mode"><option value="fixed">fixed</option><option value="full">full</option><option value="greedy">greedy</option></select>
      <div class="meta">fixed 使用内置ONNX采样常数; full 使用页面参数采样; greedy 禁用采样</div>
    </div>
    <div class="field">
      <label for="seed">随机种子</label>
      <input id="seed" type="number" step="1" value="0">
      <div class="meta">0 表示随机种子</div>
    </div>
  </div>
  <div class="row">
    <div class="field">
      <label for="max-new-frames">最大帧数</label>
      <input id="max-new-frames" type="number" value="2000" min="1">
    </div>
    <div class="field">
      <label for="voice-clone-max-text-tokens">最大文本Token数</label>
      <input id="voice-clone-max-text-tokens" type="number" value="300" min="1">
    </div>
  </div>
  <div class="row" style="margin-top:8px;">
    <div class="field">
      <label><input id="enable-robust" type="checkbox" checked> 开启鲁棒性文本归一化</label>
      <div class="meta">处理标点、空格、括号、重复符号等</div>
    </div>
    <div class="field">
      <label><input id="enable-wetext" type="checkbox"> 开启 WeTextProcessing</label>
      <div class="meta">数字/日期/金额等语义级展开（首次调用耗时17秒，建议仅在需要时启用）</div>
    </div>
  </div>
</details>

<div class="field">
  <label>播放模式</label>
  <select id="play-mode">
    <option value="stream">流式播放 (实时合成)</option>
    <option value="buffer">生成后播放 (完整音频)</option>
  </select>
  <div class="meta">生成后播放：等待全部合成完成后播放；流式播放：边合成边播放（低延迟）</div>
</div>

<button id="btn" onclick="doSynthesize()">开始合成</button>
<div id="result" class="result" style="display:none"></div>

<script>
let uploadedPromptAudioPath = '';
let uploadedPromptAudioName = '';
let currentDemoPromptAudioUrl = null;
let voiceDataMap = {};
let demosById = {};

async function loadDemos() {
  try {
    const r = await fetch('/api/demos');
    const demos = await r.json();
    const sel = document.getElementById('demo');
    sel.innerHTML = '';

    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = '(无) - 使用内置音色或上传';
    sel.appendChild(emptyOption);

    demos.forEach(d => {
      demosById[d.id] = d;
      const o = document.createElement('option');
      o.value = d.id;
      // 如果有preloadId，在名称后面添加标识
      if (d.preloadId) {
        o.textContent = d.name + ' [预加载]';
      } else {
        o.textContent = d.name;
      }
      sel.appendChild(o);
    });
  } catch (e) {
    console.error('Failed to load demos:', e);
  }
}

async function loadVoices() {
  try {
    const r = await fetch('/api/voices');
    const v = await r.json();
    const sel = document.getElementById('voice');
    sel.innerHTML = '';

    const builtinGroup = document.createElement('optgroup');
    builtinGroup.label = '内置音色';
    let hasBuiltin = false;

    v.forEach(x => {
      voiceDataMap[x.voice] = x;
      const o = document.createElement('option');
      o.value = x.voice;
      o.textContent = x.label || x.voice;
      builtinGroup.appendChild(o);
      hasBuiltin = true;
    });

    if (!hasBuiltin) {
      const o = document.createElement('option');
      o.value = 'Junhao';
      o.textContent = 'Junhao';
      sel.appendChild(o);
    }
    if (hasBuiltin) sel.appendChild(builtinGroup);
  } catch (e) {
    console.error(e);
  }
}

function onVoiceChange() {
}

function clearCurrentDemoPromptAudio() {
  if (currentDemoPromptAudioUrl) {
    URL.revokeObjectURL(currentDemoPromptAudioUrl);
    currentDemoPromptAudioUrl = null;
  }
}

function onDemoChange() {
  const demoId = document.getElementById('demo').value;
  const demo = demosById[demoId];
  clearCurrentDemoPromptAudio();
  clearUploadedPromptAudio();

  if (!demoId || !demo) {
    document.getElementById('prompt-audio-source').textContent = '使用内置音色，未上传参考音频。';
    document.getElementById('prompt-audio-preview').hidden = true;
    document.getElementById('choose-prompt-btn').hidden = true;
    document.getElementById('clear-prompt-btn').hidden = true;
    document.getElementById('prompt-audio-upload').hidden = false;
    return;
  }

  document.getElementById('text').value = demo.text;
  const preview = document.getElementById('prompt-audio-preview');
  const source = document.getElementById('prompt-audio-source');
  const chooseBtn = document.getElementById('choose-prompt-btn');
  const clearBtn = document.getElementById('clear-prompt-btn');
  const input = document.getElementById('prompt-audio-upload');

  currentDemoPromptAudioUrl = '/api/demo-prompt-audio/' + encodeURIComponent(demo.id);
  preview.src = currentDemoPromptAudioUrl;
  preview.hidden = false;
  input.hidden = true;
  // 显示demo信息和preloadId
  let sourceText = '使用演示样例音频: ' + demo.name;
  if (demo.preloadId) {
    sourceText += ' (预加载缓存: ' + demo.preloadId + ')';
  }
  source.textContent = sourceText;
  chooseBtn.hidden = false;
  clearBtn.hidden = false;
  uploadedPromptAudioPath = '';
  uploadedPromptAudioName = '';
}

function choosePromptAudio() {
  document.getElementById('prompt-audio-upload').click();
}

function onPromptAudioChange() {
  const input = document.getElementById('prompt-audio-upload');
  const preview = document.getElementById('prompt-audio-preview');
  const source = document.getElementById('prompt-audio-source');
  const chooseBtn = document.getElementById('choose-prompt-btn');
  const clearBtn = document.getElementById('clear-prompt-btn');
  const files = input.files;

  clearCurrentDemoPromptAudio();

  if (files && files.length > 0) {
    const file = files[0];
    currentDemoPromptAudioUrl = URL.createObjectURL(file);
    preview.src = currentDemoPromptAudioUrl;
    preview.hidden = false;
    input.hidden = true;
    source.textContent = '已选择: ' + file.name + ' (点击合成时将上传)';
    chooseBtn.hidden = true;
    clearBtn.hidden = false;
    uploadedPromptAudioName = file.name;
  }
}

function clearUploadedPromptAudio() {
  const input = document.getElementById('prompt-audio-upload');
  const preview = document.getElementById('prompt-audio-preview');
  const source = document.getElementById('prompt-audio-source');
  const chooseBtn = document.getElementById('choose-prompt-btn');
  const clearBtn = document.getElementById('clear-prompt-btn');

  input.value = '';
  preview.pause();
  preview.removeAttribute('src');
  preview.load();
  preview.hidden = true;
  input.hidden = false;
  source.textContent = '使用内置音色，未上传参考音频。';
  chooseBtn.hidden = true;
  clearBtn.hidden = true;
  uploadedPromptAudioPath = '';
  uploadedPromptAudioName = '';
}

function clearPromptAudio() {
  clearCurrentDemoPromptAudio();
  clearUploadedPromptAudio();
  onDemoChange();
}

function getConfig() {
  const seedVal = parseInt(document.getElementById('seed').value) || 0;
  const demoId = document.getElementById('demo').value;
  return {
    text: document.getElementById('text').value,
    voice: document.getElementById('voice').value,
    demo_id: demoId,
    sample_mode: document.getElementById('sample-mode').value,
    max_new_frames: parseInt(document.getElementById('max-new-frames').value) || 375,
    voice_clone_max_text_tokens: parseInt(document.getElementById('voice-clone-max-text-tokens').value) || 300,
    seed: seedVal === 0 ? null : seedVal,
    enable_robust: document.getElementById('enable-robust').checked,
    enable_wetext: document.getElementById('enable-wetext').checked
  };
}

async function doSynthesize() {
  const btn = document.getElementById('btn');
  const playMode = document.getElementById('play-mode').value;
  btn.disabled = true;
  btn.textContent = playMode === 'stream' ? '流式合成中...' : '合成中...';
  const result = document.getElementById('result');
  result.style.display = 'block';
  result.innerHTML = '<p>正在合成，请稍候...</p>';

  try {
    const input = document.getElementById('prompt-audio-upload');
    const files = input.files;
    let uploadedPath = '';

    if (files && files.length > 0) {
      result.innerHTML = '<p>正在上传参考音频...</p>';
      const file = files[0];
      const fileBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(fileBuffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64Data = btoa(binary);

      const uploadResp = await fetch('/api/upload-prompt-audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_data_b64: base64Data,
          file_name: file.name
        })
      });
      if (!uploadResp.ok) throw new Error('上传失败');
      const uploadData = await uploadResp.json();
      uploadedPath = uploadData.path;
      uploadedPromptAudioPath = uploadedPath;
    }

    const cfg = getConfig();
    const body = {
      text: cfg.text,
      voice: cfg.voice,
      demo_id: cfg.demo_id,
      prompt_audio_path: '',
      uploaded_prompt_audio: uploadedPath,
      sample_mode: cfg.sample_mode,
      max_new_frames: cfg.max_new_frames,
      voice_clone_max_text_tokens: cfg.voice_clone_max_text_tokens,
      seed: cfg.seed,
      stream: playMode === 'stream'
    };

    if (playMode === 'stream') {
      await doStreamSynthesize(body, result, btn);
    } else {
      await doBufferSynthesize(body, result, btn);
    }
  } catch (e) {
    result.innerHTML = '<p class="error">请求失败: ' + e.message + '</p>';
    btn.disabled = false;
    btn.textContent = '开始合成';
  }
}

async function doBufferSynthesize(body, result, btn) {
  result.innerHTML = '<p>正在合成语音...</p>';
  const r = await fetch('/api/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await r.json();
  if (!r.ok) {
    result.innerHTML = '<p class="error">错误: ' + (data.error || r.statusText) + '</p>';
    btn.disabled = false;
    btn.textContent = '开始合成';
    return;
  }
  const durationSec = data.sample_rate > 0 ? (data.audio_samples / data.sample_rate).toFixed(2) : '0.00';
  result.innerHTML = '<p><strong>合成完成!</strong> 耗时: ' + data.elapsed_seconds.toFixed(2) + '秒 音频时长: ' + durationSec + '秒 采样率: ' + data.sample_rate + 'Hz</p><p>音色: ' + data.voice + ' | 采样模式: ' + data.sample_mode + ' | 分块数: ' + (data.text_chunks ? data.text_chunks.length : 0) + '</p>';

  if (data.audio_data_b64) {
    const byteCharacters = atob(data.audio_data_b64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(blob);
    const audioEl = document.createElement('audio');
    audioEl.controls = true;
    audioEl.type = 'audio/wav';
    audioEl.src = audioUrl;
    result.appendChild(audioEl);
  }
  btn.disabled = false;
  btn.textContent = '开始合成';
}

async function doStreamSynthesize(body, result, btn) {
  result.innerHTML = '<p>正在流式合成语音...</p>';

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);

  try {
    const r = await fetch('/api/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      result.innerHTML = '<p class="error">错误: ' + (data.error || r.statusText) + '</p>';
      btn.disabled = false;
      btn.textContent = '开始合成';
      return;
    }

    const sampleRate = parseInt(r.headers.get('X-Audio-Sample-Rate') || '24000');
    const channels = parseInt(r.headers.get('X-Audio-Channels') || '2');

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextCtor({ sampleRate: sampleRate });
    await audioCtx.resume();

    let nextPlaybackTime = audioCtx.currentTime + 0.1;
    let totalSamplesReceived = 0;
    let chunkCount = 0;

    const reader = r.body.getReader();
    let remainder = new Uint8Array(0);
    const bytesPerFrame = channels * 4;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) continue;

      const merged = new Uint8Array(remainder.length + value.length);
      merged.set(remainder);
      merged.set(value, remainder.length);

      const alignedLength = Math.floor(merged.length / bytesPerFrame) * bytesPerFrame;
      if (alignedLength <= 0) {
        remainder = merged;
        continue;
      }

      const pcmChunk = merged.subarray(0, alignedLength);
      remainder = merged.subarray(alignedLength);

      const totalFrames = pcmChunk.length / bytesPerFrame;
      const audioBuffer = audioCtx.createBuffer(channels, totalFrames, sampleRate);
      const view = new DataView(pcmChunk.buffer, pcmChunk.byteOffset, pcmChunk.length);

      for (let ch = 0; ch < channels; ch++) {
        const channelData = audioBuffer.getChannelData(ch);
        for (let i = 0; i < totalFrames; i++) {
          const byteOffset = (i * channels + ch) * 4;
          channelData[i] = view.getFloat32(byteOffset, true);
        }
      }

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      const startAt = Math.max(nextPlaybackTime, audioCtx.currentTime + 0.02);
      source.start(startAt);
      nextPlaybackTime = startAt + audioBuffer.duration;

      totalSamplesReceived += totalFrames;
      chunkCount++;
    }

    result.querySelector('p').textContent = '流式合成完成! (共 ' + chunkCount + ' 个音频块)';
  } catch (e) {
    if (e.name === 'AbortError') {
      result.innerHTML = '<p class="error">请求超时</p>';
    } else {
      result.innerHTML = '<p class="error">请求失败: ' + e.message + '</p>';
    }
  } finally {
    clearTimeout(timeoutId);
    btn.disabled = false;
    btn.textContent = '开始合成';
  }
}

loadDemos();
loadVoices();
</script>
</body>
</html>
`
