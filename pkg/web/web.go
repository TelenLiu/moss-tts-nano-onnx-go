package web

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/audio"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
)

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

type Server struct {
	Cfg          *deps.Config
	CpuThreads   int
	MaxNewFrames int
	OutputDir    string
	Host         string
	Port         int

	mu              sync.RWMutex
	Runtime         *ttsruntime.OnnxTtsRuntime
	Ready           bool
	Progress        []ProgressEvent
	subscribers     map[chan ProgressEvent]struct{}
	PromptUploadDir string
}

func NewServer(cfg *deps.Config, cpuThreads, maxNewFrames int, outputDir, host string, port int) *Server {
	absOutputDir, _ := filepath.Abs(outputDir)
	os.MkdirAll(absOutputDir, 0755)
	promptUploadDir := filepath.Join(absOutputDir, ".prompt_uploads")
	os.MkdirAll(promptUploadDir, 0755)
	return &Server{
		Cfg:             cfg,
		CpuThreads:      cpuThreads,
		MaxNewFrames:    maxNewFrames,
		OutputDir:       absOutputDir,
		Host:            host,
		Port:            port,
		subscribers:     make(map[chan ProgressEvent]struct{}),
		PromptUploadDir: promptUploadDir,
	}
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
	mux.HandleFunc("/api/progress", s.handleProgress)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/synthesize", s.handleSynthesize)
	mux.HandleFunc("/api/voices", s.handleVoices)
	mux.HandleFunc("/api/audio/", s.handleAudio)
	mux.HandleFunc("/api/upload-prompt-audio", s.handleUploadPromptAudio)
	mux.HandleFunc("/api/config/export", s.handleConfigExport)
	mux.HandleFunc("/api/config/import", s.handleConfigImport)

	go s.backgroundInit()

	addr := fmt.Sprintf("%s:%d", s.Host, s.Port)
	log.Printf("Web 服务启动于 http://%s/", addr)
	return http.ListenAndServe(addr, mux)
}

func (s *Server) backgroundInit() {
	s.emit(ProgressEvent{Phase: "check", Message: "正在检测运行环境...", Percent: 5})

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
		s.OutputDir,
	)
	if err != nil {
		s.emit(ProgressEvent{Phase: "error", Message: fmt.Sprintf("TTS 运行时初始化失败: %v", err), Error: fmt.Sprintf("TTS 运行时初始化失败: %v", err)})
		return
	}
	s.emit(ProgressEvent{Phase: "load", Message: "TTS 模型加载完成", Percent: 95})

	s.mu.Lock()
	s.Runtime = rt
	s.Ready = true
	s.mu.Unlock()

	s.emit(ProgressEvent{Phase: "ready", Message: "系统就绪", Percent: 100, Done: true})
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
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
		maxNewFrames = 375
	}
	voiceCloneMaxTokens := req.VoiceCloneMaxTextTokens
	if voiceCloneMaxTokens <= 0 {
		voiceCloneMaxTokens = 75
	}

	promptAudioPath := req.PromptAudioPath
	if req.UploadedPromptAudio != "" {
		promptAudioPath = req.UploadedPromptAudio
	}

	seedVal := "nil"
	if req.Seed != nil {
		seedVal = fmt.Sprintf("%d", *req.Seed)
	}
	log.Printf("[API synthesize] text=%q voice=%q promptAudioPath=%q uploadedPromptAudio=%q sampleMode=%s maxNewFrames=%d voiceCloneMaxTokens=%d seed=%s",
		req.Text, voice, promptAudioPath, req.UploadedPromptAudio, sampleMode, maxNewFrames, voiceCloneMaxTokens, seedVal)

	result, err := rt.Synthesize(
		req.Text, voice, promptAudioPath, "",
		sampleMode, doSample, false,
		maxNewFrames, voiceCloneMaxTokens,
		true, req.Seed,
	)
	if err != nil {
		log.Printf("[API synthesize] 合成失败: %v", err)
		http.Error(w, fmt.Sprintf("Synthesis failed: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[API synthesize] 合成成功: audio=%s sampleRate=%d audioSamples=%d elapsed=%.2fs chunks=%d",
		result.AudioPath, result.SampleRate, result.AudioSamples, result.ElapsedSec, len(result.TextChunks))

	audioFilename := filepath.Base(result.AudioPath)
	resp := SynthesizeResponse{
		AudioPath:      "/api/audio/" + audioFilename,
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
	filePath := filepath.Join(s.OutputDir, filename)
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

	r.ParseMultipartForm(32 << 20)
	file, header, err := r.FormFile("prompt_audio")
	if err != nil {
		http.Error(w, fmt.Sprintf("读取上传文件失败: %v", err), http.StatusBadRequest)
		return
	}
	defer file.Close()

	ext := filepath.Ext(header.Filename)
	if ext == "" || len(ext) > 16 {
		ext = ".wav"
	}

	tempFile, err := os.CreateTemp(s.PromptUploadDir, "prompt-speech-*"+ext)
	if err != nil {
		http.Error(w, fmt.Sprintf("创建临时文件失败: %v", err), http.StatusInternalServerError)
		return
	}
	defer tempFile.Close()

	written, err := io.Copy(tempFile, file)
	if err != nil {
		os.Remove(tempFile.Name())
		http.Error(w, fmt.Sprintf("保存文件失败: %v", err), http.StatusInternalServerError)
		return
	}
	if written <= 0 {
		os.Remove(tempFile.Name())
		http.Error(w, "上传文件为空", http.StatusBadRequest)
		return
	}

	log.Printf("[API upload] 参考音频上传成功: name=%s path=%s size=%d", header.Filename, tempFile.Name(), written)

	resp := map[string]string{
		"path": tempFile.Name(),
		"name": header.Filename,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleConfigExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ConfigData
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", "attachment; filename=moss-tts-config.json")
	json.NewEncoder(w).Encode(req)
}

func (s *Server) handleConfigImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ConfigData
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid config: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(req)
}

type SynthesizeRequest struct {
	Text                    string `json:"text"`
	Voice                   string `json:"voice"`
	PromptAudioPath         string `json:"prompt_audio_path"`
	UploadedPromptAudio     string `json:"uploaded_prompt_audio"`
	SampleMode              string `json:"sample_mode"`
	MaxNewFrames            int    `json:"max_new_frames"`
	VoiceCloneMaxTextTokens int    `json:"voice_clone_max_text_tokens"`
	Seed                    *int   `json:"seed"`
}

type SynthesizeResponse struct {
	AudioPath      string   `json:"audio_path"`
	SampleRate     int      `json:"sample_rate"`
	AudioSamples   int      `json:"audio_samples"`
	ElapsedSeconds float64  `json:"elapsed_seconds"`
	Voice          string   `json:"voice"`
	TextChunks     []string `json:"text_chunks"`
	SampleMode     string   `json:"sample_mode"`
	DoSample       bool     `json:"do_sample"`
}

type ConfigData struct {
	Text                    string `json:"text"`
	Voice                   string `json:"voice"`
	PromptText              string `json:"prompt_text"`
	PromptAudioPath         string `json:"prompt_audio_path"`
	SampleMode              string `json:"sample_mode"`
	MaxNewFrames            int    `json:"max_new_frames"`
	VoiceCloneMaxTextTokens int    `json:"voice_clone_max_text_tokens"`
	Seed                    int    `json:"seed"`
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
.config-actions{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.config-actions button{font-size:14px;padding:8px 16px}
.row{display:flex;gap:12px}
.row .field{flex:1}
.details-summary{cursor:pointer;font-weight:600;color:#1a73e8}
details{background:#fff;border:1px solid #ddd;border-radius:4px;padding:12px}
</style>
</head>
<body>
<h1>MOSS-TTS-Nano ONNX Demo <span class="badge">Ready</span></h1>

<div class="config-actions">
  <button class="secondary" onclick="exportConfig()">导出配置</button>
  <button class="secondary" onclick="document.getElementById('config-import').click()">导入配置</button>
  <input type="file" id="config-import" accept=".json" style="display:none" onchange="importConfig(this)">
</div>

<div class="field"><label for="text">输入文本</label><textarea id="text" placeholder="请输入要合成的文本...">你好，欢迎使用MOSS语音合成系统。</textarea></div>

<div class="field">
  <label for="voice">内置音色 / Demo 预设</label>
  <select id="voice" onchange="onVoiceChange()"></select>
  <div class="meta">选择内置音色或 Demo 预设，Demo 会自动填充参考文本</div>
</div>

<div class="field">
  <label for="prompt-audio-upload">参考音频 (克隆音色)</label>
  <div class="prompt-audio-box">
    <input id="prompt-audio-upload" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.aac" onchange="onPromptAudioChange()">
    <audio id="prompt-audio-preview" controls hidden></audio>
    <div id="prompt-audio-source" class="meta">使用内置音色，未上传参考音频。</div>
    <div class="prompt-audio-actions">
      <button id="clear-prompt-btn" class="secondary" type="button" hidden onclick="clearPromptAudio()">清除参考音频</button>
    </div>
  </div>
</div>

<div class="field">
  <label for="prompt-text">参考文本 (可选)</label>
  <textarea id="prompt-text" placeholder="请输入参考音频对应的文本内容（用于辅助音色克隆，可选）"></textarea>
  <div class="meta">参考文本是参考音频对应的文字内容，有助于更准确地克隆音色。使用 Demo 预设时会自动填充。</div>
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
      <input id="max-new-frames" type="number" value="375" min="1">
    </div>
    <div class="field">
      <label for="voice-clone-max-text-tokens">最大文本Token数</label>
      <input id="voice-clone-max-text-tokens" type="number" value="75" min="1">
    </div>
  </div>
</details>

<button id="btn" onclick="doSynthesize()">开始合成</button>
<div id="result" class="result" style="display:none"></div>

<script>
let uploadedPromptAudioPath = '';
let uploadedPromptAudioName = '';
let voiceDataMap = {};

async function loadVoices() {
  try {
    const r = await fetch('/api/voices');
    const v = await r.json();
    const sel = document.getElementById('voice');
    sel.innerHTML = '';

    const builtinGroup = document.createElement('optgroup');
    builtinGroup.label = '内置音色';
    const demoGroup = document.createElement('optgroup');
    demoGroup.label = 'Demo 预设';

    let hasBuiltin = false;
    let hasDemo = false;

    v.forEach(x => {
      voiceDataMap[x.voice] = x;
      const o = document.createElement('option');
      o.value = x.voice;
      o.textContent = x.label || x.voice;
      if (x.is_demo) {
        demoGroup.appendChild(o);
        hasDemo = true;
      } else {
        builtinGroup.appendChild(o);
        hasBuiltin = true;
      }
    });

    if (!hasBuiltin && !hasDemo) {
      const o = document.createElement('option');
      o.value = 'Junhao';
      o.textContent = 'Junhao';
      sel.appendChild(o);
    }
    if (hasBuiltin) sel.appendChild(builtinGroup);
    if (hasDemo) sel.appendChild(demoGroup);
  } catch (e) {
    console.error(e);
  }
}

function onVoiceChange() {
  const voice = document.getElementById('voice').value;
  const data = voiceDataMap[voice];
  if (data && data.prompt_text) {
    document.getElementById('prompt-text').value = data.prompt_text;
  }
}

function onPromptAudioChange() {
  const input = document.getElementById('prompt-audio-upload');
  const preview = document.getElementById('prompt-audio-preview');
  const source = document.getElementById('prompt-audio-source');
  const clearBtn = document.getElementById('clear-prompt-btn');
  const files = input.files;

  if (files && files.length > 0) {
    const file = files[0];
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    input.hidden = true;
    source.textContent = '已选择: ' + file.name + ' (点击合成时将上传)';
    clearBtn.hidden = false;
    uploadedPromptAudioName = file.name;
  }
}

function clearPromptAudio() {
  const input = document.getElementById('prompt-audio-upload');
  const preview = document.getElementById('prompt-audio-preview');
  const source = document.getElementById('prompt-audio-source');
  const clearBtn = document.getElementById('clear-prompt-btn');

  input.value = '';
  preview.pause();
  preview.removeAttribute('src');
  preview.load();
  preview.hidden = true;
  input.hidden = false;
  source.textContent = '使用内置音色，未上传参考音频。';
  clearBtn.hidden = true;
  uploadedPromptAudioPath = '';
  uploadedPromptAudioName = '';
}

async function uploadPromptAudio(file) {
  const formData = new FormData();
  formData.append('prompt_audio', file);
  const r = await fetch('/api/upload-prompt-audio', { method: 'POST', body: formData });
  if (!r.ok) throw new Error('上传失败');
  const data = await r.json();
  return data.path;
}

function getConfig() {
  const seedVal = parseInt(document.getElementById('seed').value) || 0;
  return {
    text: document.getElementById('text').value,
    voice: document.getElementById('voice').value,
    prompt_text: document.getElementById('prompt-text').value,
    prompt_audio_path: uploadedPromptAudioPath,
    prompt_audio_name: uploadedPromptAudioName,
    sample_mode: document.getElementById('sample-mode').value,
    max_new_frames: parseInt(document.getElementById('max-new-frames').value) || 375,
    voice_clone_max_text_tokens: parseInt(document.getElementById('voice-clone-max-text-tokens').value) || 75,
    seed: seedVal === 0 ? null : seedVal
  };
}

function applyConfig(cfg) {
  if (cfg.text !== undefined) document.getElementById('text').value = cfg.text;
  if (cfg.voice !== undefined) document.getElementById('voice').value = cfg.voice;
  if (cfg.prompt_text !== undefined) document.getElementById('prompt-text').value = cfg.prompt_text;
  if (cfg.sample_mode !== undefined) document.getElementById('sample-mode').value = cfg.sample_mode;
  if (cfg.max_new_frames !== undefined) document.getElementById('max-new-frames').value = cfg.max_new_frames;
  if (cfg.voice_clone_max_text_tokens !== undefined) document.getElementById('voice-clone-max-text-tokens').value = cfg.voice_clone_max_text_tokens;
  if (cfg.seed !== undefined) document.getElementById('seed').value = cfg.seed || 0;
  // 恢复参考音频信息（仅显示，需要重新上传文件）
  if (cfg.prompt_audio_name) {
    uploadedPromptAudioName = cfg.prompt_audio_name;
    uploadedPromptAudioPath = cfg.prompt_audio_path || '';
    document.getElementById('prompt-audio-source').textContent = '配置中的参考音频: ' + cfg.prompt_audio_name + ' (请重新上传文件)';
  }
}

function exportConfig() {
  const cfg = getConfig();
  const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'moss-tts-config.json';
  a.click();
  URL.revokeObjectURL(url);
}

function importConfig(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    try {
      const cfg = JSON.parse(e.target.result);
      applyConfig(cfg);
      alert('配置导入成功');
    } catch (err) {
      alert('配置导入失败: ' + err.message);
    }
  };
  reader.readAsText(file);
  input.value = '';
}

async function doSynthesize() {
  const btn = document.getElementById('btn');
  btn.disabled = true;
  btn.textContent = '合成中...';
  const result = document.getElementById('result');
  result.style.display = 'block';
  result.innerHTML = '<p>正在合成，请稍候...</p>';

  try {
    const input = document.getElementById('prompt-audio-upload');
    const files = input.files;
    let uploadedPath = '';

    if (files && files.length > 0) {
      result.innerHTML = '<p>正在上传参考音频...</p>';
      uploadedPath = await uploadPromptAudio(files[0]);
      uploadedPromptAudioPath = uploadedPath;
    }

    const cfg = getConfig();
    const body = {
      text: cfg.text,
      voice: cfg.voice,
      prompt_audio_path: '',
      uploaded_prompt_audio: uploadedPath,
      sample_mode: cfg.sample_mode,
      max_new_frames: cfg.max_new_frames,
      voice_clone_max_text_tokens: cfg.voice_clone_max_text_tokens,
      seed: cfg.seed
    };

    result.innerHTML = '<p>正在合成语音...</p>';
    const r = await fetch('/api/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await r.json();
    if (!r.ok) {
      result.innerHTML = '<p class="error">错误: ' + (data.error || r.statusText) + '</p>';
      return;
    }
    const durationSec = data.sample_rate > 0 ? (data.audio_samples / data.sample_rate).toFixed(2) : '0.00';
    result.innerHTML = '<p><strong>合成完成!</strong> 耗时: ' + data.elapsed_seconds.toFixed(2) + '秒 音频时长: ' + durationSec + '秒 采样率: ' + data.sample_rate + 'Hz</p><p>音色: ' + data.voice + ' | 采样模式: ' + data.sample_mode + ' | 分块数: ' + (data.text_chunks ? data.text_chunks.length : 0) + '</p>';
    const audioEl = document.createElement('audio');
    audioEl.controls = true;
    audioEl.type = 'audio/wav';
    audioEl.preload = 'none';
    audioEl.src = data.audio_path;
    result.appendChild(audioEl);
  } catch (e) {
    result.innerHTML = '<p class="error">请求失败: ' + e.message + '</p>';
  } finally {
    btn.disabled = false;
    btn.textContent = '开始合成';
  }
}

loadVoices();
</script>
</body>
</html>
`
