package ttsruntime

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/worker"
)

// WorkerProcess 管理一个子进程推理单元
type WorkerProcess struct {
	Name string // 进程命名：workCore从1开始，reserveCore从r1开始
	cmd  *exec.Cmd
	conn net.Conn
	addr string // 子进程实际监听地址

	workerExePath string // 子进程可执行文件路径（硬链接/副本），Close 时需清理

	writeMu    sync.Mutex // 保护 conn 的并发写入
	mu         sync.Mutex
	activeReqs atomic.Int64
	nextReqID  atomic.Int64
	pending    map[int]chan *workerResponse // reqID -> response channel
	closed     bool
}

type workerResponse struct {
	resp       *worker.Response
	attachment []byte
}

// workerExeDir 存放子进程可执行文件副本的子目录名
const workerExeDir = ".workers"

// createWorkerExe 为子进程创建一个带序号的可执行文件副本，
// 使子进程在活动监视器/ps中显示独立的进程名（如 moss-tts-worker-1、moss-tts-worker-r1）。
// 优先使用符号链接（symlink），不占用磁盘空间；失败时回退到文件复制。
// 文件存放在 .workers 子目录中，Close 时清理。
func createWorkerExe(exePath, name string) string {
	exeDir := filepath.Dir(exePath)
	workersDir := filepath.Join(exeDir, workerExeDir)

	// 确保 .workers 目录存在
	if err := os.MkdirAll(workersDir, 0755); err != nil {
		log.Printf("[WorkerProcess] 创建worker目录失败: %v, 使用原始路径", err)
		return exePath
	}

	target := filepath.Join(workersDir, "moss-tts-worker-"+name)

	// 检查目标文件是否已存在
	fi2, err2 := os.Lstat(target)
	if err2 == nil {
		// 目标已存在，检查是否是符号链接指向正确的源文件
		if fi2.Mode()&os.ModeSymlink != 0 {
			linkTarget, err := os.Readlink(target)
			if err == nil && linkTarget == exePath {
				log.Printf("[WorkerProcess] 复用已有的 worker 符号链接: %s", target)
				return target
			}
		} else if fi2.Mode().IsRegular() {
			// 是普通文件（副本），检查是否与源文件相同
			fi1, err1 := os.Stat(exePath)
			if err1 == nil && !os.SameFile(fi1, fi2) && fi1.Size() == fi2.Size() {
				log.Printf("[WorkerProcess] 复用已有的 worker 副本: %s", target)
				return target
			}
		}
		// 文件存在但不匹配，删除重建
		os.Remove(target)
	}

	// 优先尝试符号链接（不占用磁盘空间）
	if err := os.Symlink(exePath, target); err == nil {
		log.Printf("[WorkerProcess] 创建 worker 符号链接: %s -> %s", target, exePath)
		return target
	}

	// 符号链接失败，回退到文件复制
	src, err := os.Open(exePath)
	if err != nil {
		log.Printf("[WorkerProcess] 无法打开源文件创建worker副本: %v, 使用原始路径", err)
		return exePath
	}
	defer src.Close()

	dst, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		log.Printf("[WorkerProcess] 无法创建worker副本文件: %v, 使用原始路径", err)
		return exePath
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		os.Remove(target)
		log.Printf("[WorkerProcess] 复制worker可执行文件失败: %v, 使用原始路径", err)
		return exePath
	}

	log.Printf("[WorkerProcess] 创建 worker 副本: %s", target)
	return target
}

// NewWorkerProcess 启动一个子进程推理单元
// name: 进程命名标识，如 "1","2" 或 "r1","r2"
func NewWorkerProcess(name string, modelDir string, threadCount int, coreMemMB int, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*WorkerProcess, error) {
	wp := &WorkerProcess{
		Name:    name,
		pending: make(map[int]chan *workerResponse),
	}

	// 获取当前可执行文件路径
	exePath, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("获取可执行文件路径失败: %w", err)
	}

	// 为子进程创建带序号的可执行文件副本，以便活动监视器显示独立进程名
	workerPath := createWorkerExe(exePath, name)
	wp.workerExePath = workerPath

	// 准备初始化参数
	maxFrames := 375
	if maxNewFrames != nil {
		maxFrames = *maxNewFrames
	}
	initReq := worker.InitRequest{
		ModelDir:      modelDir,
		ThreadCount:   threadCount,
		CoreMemMB:     coreMemMB,
		MaxNewFrames:  maxFrames,
		DoSample:      doSample,
		SampleMode:    sampleMode,
		ExecutionMode: executionMode,
		ListenAddr:    "127.0.0.1:0",
	}

	initJSON := marshalJSON(initReq)

	// 启动子进程（使用 worker 副本路径，使进程名显示为 moss-tts-worker）
	cmd := exec.Command(workerPath, "worker")
	cmd.Stderr = os.Stderr

	// 通过 stdin 传递初始化参数
	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("创建stdin管道失败: %w", err)
	}

	// 捕获子进程 stdout 以获取 LISTEN: 行
	addrReader, addrWriter, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("创建管道失败: %w", err)
	}
	cmd.Stdout = io.MultiWriter(os.Stdout, addrWriter)

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("启动子进程失败: %w", err)
	}

	// 发送初始化参数
	if _, err := stdinPipe.Write(initJSON); err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("发送初始化参数失败: %w", err)
	}
	stdinPipe.Close()

	// 等待子进程输出监听地址
	addrChan := make(chan string, 1)
	errChan := make(chan error, 1)
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := addrReader.Read(buf)
			if err != nil {
				errChan <- err
				return
			}
			output := string(buf[:n])
			for _, line := range strings.Split(output, "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "LISTEN:") {
					addrChan <- strings.TrimPrefix(line, "LISTEN:")
					return
				}
			}
		}
	}()

	var addr string
	select {
	case addr = <-addrChan:
	case err := <-errChan:
		cmd.Process.Kill()
		return nil, fmt.Errorf("读取子进程地址失败: %w", err)
	case <-time.After(120 * time.Second):
		cmd.Process.Kill()
		return nil, fmt.Errorf("等待子进程启动超时")
	}
	addrReader.Close()

	wp.cmd = cmd
	wp.addr = addr
	log.Printf("[WorkerProcess #%s] 子进程已启动, addr=%s, pid=%d", name, addr, cmd.Process.Pid)

	// 连接到子进程
	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("连接子进程失败: %w", err)
	}
	wp.conn = conn

	// 启动响应读取协程
	go wp.readLoop()

	return wp, nil
}

// readLoop 持续读取子进程响应并分发到对应的 pending channel
func (wp *WorkerProcess) readLoop() {
	for {
		resp, attachment, err := worker.ReadResponse(wp.conn)
		if err != nil {
			if !wp.isClosed() {
				log.Printf("[WorkerProcess #%s] 读取响应失败: %v", wp.Name, err)
			}
			wp.closePending(fmt.Errorf("连接断开: %w", err))
			return
		}

		wp.mu.Lock()
		ch, ok := wp.pending[resp.ID]
		// 终结性响应类型：删除 pending
		if ok && (resp.Type == worker.MsgDone || resp.Type == worker.MsgError || resp.Type == worker.MsgResult || resp.Type == worker.MsgPong || resp.Type == worker.MsgCancelled) {
			delete(wp.pending, resp.ID)
		}
		wp.mu.Unlock()

		if ok && ch != nil {
			ch <- &workerResponse{resp: resp, attachment: attachment}
		}
	}
}

// sendRequest 发送请求并等待响应，支持 ctx 取消
func (wp *WorkerProcess) sendRequest(ctx context.Context, req *worker.Request) (*worker.Response, []byte, error) {
	if wp.isClosed() {
		return nil, nil, fmt.Errorf("worker进程已关闭")
	}

	reqID := int(wp.nextReqID.Add(1))
	req.ID = reqID

	ch := make(chan *workerResponse, 1)
	wp.mu.Lock()
	wp.pending[reqID] = ch
	wp.mu.Unlock()

	defer func() {
		wp.mu.Lock()
		delete(wp.pending, reqID)
		wp.mu.Unlock()
	}()

	wp.writeMu.Lock()
	err := worker.WriteRequest(wp.conn, req, nil)
	wp.writeMu.Unlock()
	if err != nil {
		return nil, nil, fmt.Errorf("发送请求失败: %w", err)
	}

	select {
	case result := <-ch:
		if result.resp.Type == worker.MsgError {
			return nil, nil, fmt.Errorf("worker错误: %s", result.resp.Error)
		}
		if result.resp.Type == worker.MsgCancelled {
			return nil, nil, context.Canceled
		}
		return result.resp, result.attachment, nil
	case <-time.After(300 * time.Second):
		return nil, nil, fmt.Errorf("等待worker响应超时")
	case <-ctx.Done():
		// 上下文被取消，通知子进程终止推理
		log.Printf("[WorkerProcess #%s] 请求 #%d 上下文取消，发送cancel消息", wp.Name, reqID)
		wp.sendCancel(reqID)
		return nil, nil, ctx.Err()
	}
}

// sendStreamRequest 发送流式请求，返回响应channel，支持 ctx 取消
func (wp *WorkerProcess) sendStreamRequest(ctx context.Context, req *worker.Request) (<-chan *workerResponse, error) {
	if wp.isClosed() {
		return nil, fmt.Errorf("worker进程已关闭")
	}

	reqID := int(wp.nextReqID.Add(1))
	req.ID = reqID

	streamCh := make(chan *workerResponse, 64)

	wp.mu.Lock()
	wp.pending[reqID] = streamCh
	wp.mu.Unlock()

	wp.writeMu.Lock()
	err := worker.WriteRequest(wp.conn, req, nil)
	wp.writeMu.Unlock()
	if err != nil {
		wp.mu.Lock()
		delete(wp.pending, reqID)
		wp.mu.Unlock()
		close(streamCh)
		return nil, fmt.Errorf("发送流式请求失败: %w", err)
	}

	// 包装channel，在收到终结性响应或ctx取消后清理
	outCh := make(chan *workerResponse, 64)
	go func() {
		defer close(outCh)

		// 监听 ctx 取消，发送 cancel 消息
		ctxDone := ctx.Done()
		for {
			select {
			case resp, ok := <-streamCh:
				if !ok {
					return
				}
				outCh <- resp
				if resp.resp.Type == worker.MsgDone || resp.resp.Type == worker.MsgError || resp.resp.Type == worker.MsgCancelled {
					wp.mu.Lock()
					delete(wp.pending, reqID)
					wp.mu.Unlock()
					return
				}
			case <-ctxDone:
				// 上下文取消，发送 cancel 给子进程
				log.Printf("[WorkerProcess #%s] 流式请求 #%d 上下文取消，发送cancel消息", wp.Name, reqID)
				wp.sendCancel(reqID)
				ctxDone = nil // 避免重复发送
				// 继续读取子进程的 cancelled 响应
			}
		}
	}()

	return outCh, nil
}

// sendCancel 发送取消请求给子进程
func (wp *WorkerProcess) sendCancel(cancelReqID int) {
	cancelReq := &worker.Request{
		Type:        worker.MsgCancel,
		CancelReqID: cancelReqID,
	}
	wp.writeMu.Lock()
	err := worker.WriteRequest(wp.conn, cancelReq, nil)
	wp.writeMu.Unlock()
	if err != nil {
		log.Printf("[WorkerProcess #%s] 发送cancel消息失败: %v", wp.Name, err)
	}
}

// Close 关闭子进程
func (wp *WorkerProcess) Close() {
	wp.mu.Lock()
	wp.closed = true
	wp.mu.Unlock()

	if wp.conn != nil {
		wp.conn.Close()
	}
	if wp.cmd != nil && wp.cmd.Process != nil {
		wp.cmd.Process.Signal(os.Interrupt)
		done := make(chan error, 1)
		go func() {
			done <- wp.cmd.Wait()
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			wp.cmd.Process.Kill()
		}
	}

	// 清理子进程可执行文件副本，不影响主进程
	if wp.workerExePath != "" {
		exePath, _ := os.Executable()
		if wp.workerExePath != exePath {
			os.Remove(wp.workerExePath)
			// 尝试清理 .workers 目录（目录为空时删除）
			workersDir := filepath.Join(filepath.Dir(exePath), workerExeDir)
			os.Remove(workersDir) // 仅在目录为空时成功
		}
	}
}

func (wp *WorkerProcess) isClosed() bool {
	wp.mu.Lock()
	defer wp.mu.Unlock()
	return wp.closed
}

func (wp *WorkerProcess) closePending(err error) {
	wp.mu.Lock()
	defer wp.mu.Unlock()
	for id, ch := range wp.pending {
		ch <- &workerResponse{resp: &worker.Response{Type: worker.MsgError, Error: err.Error()}}
		delete(wp.pending, id)
	}
}

// ---- 高层 API ----

// Ping 检查子进程是否存活
func (wp *WorkerProcess) Ping() bool {
	resp, _, err := wp.sendRequest(context.Background(), &worker.Request{Type: worker.MsgPing})
	if err != nil {
		return false
	}
	return resp.Type == worker.MsgPong
}

// EncodeText 编码文本为 token IDs
func (wp *WorkerProcess) EncodeText(text string) []int {
	resp, _, err := wp.sendRequest(context.Background(), &worker.Request{
		Type:        worker.MsgEncodeText,
		TextContent: text,
	})
	if err != nil {
		log.Printf("[WorkerProcess #%s] EncodeText失败: %v", wp.Name, err)
		return nil
	}
	return resp.TokenIDs
}

// CountTextTokens 计算 token 数
func (wp *WorkerProcess) CountTextTokens(text string) int {
	resp, _, err := wp.sendRequest(context.Background(), &worker.Request{
		Type:        worker.MsgCountTokens,
		TextContent: text,
	})
	if err != nil {
		log.Printf("[WorkerProcess #%s] CountTextTokens失败: %v", wp.Name, err)
		return 0
	}
	return resp.TokenCount
}

// ListBuiltinVoices 列出内置音色
func (wp *WorkerProcess) ListBuiltinVoices() []map[string]interface{} {
	resp, _, err := wp.sendRequest(context.Background(), &worker.Request{Type: worker.MsgListVoices})
	if err != nil {
		log.Printf("[WorkerProcess #%s] ListBuiltinVoices失败: %v", wp.Name, err)
		return nil
	}
	return resp.Voices
}

// PreloadVoice 预加载音色
func (wp *WorkerProcess) PreloadVoice(preloadID, audioPath, voice string) error {
	resp, _, err := wp.sendRequest(context.Background(), &worker.Request{
		Type:      worker.MsgPreload,
		PreloadID: preloadID,
		AudioPath: audioPath,
	})
	if err != nil {
		return err
	}
	if resp.Type == worker.MsgError {
		return fmt.Errorf("preload失败: %s", resp.Error)
	}
	return nil
}

// SynthesizeWithContextEx 执行合成（兼容旧接口，主进程未预处理时使用）
func (wp *WorkerProcess) SynthesizeWithContextEx(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (*SynthesisResult, error) {
	resp, attachment, err := wp.sendRequest(ctx, &worker.Request{
		Type:                    worker.MsgSynthesize,
		Text:                    text,
		Voice:                   voice,
		PromptAudioPath:         promptAudioPath,
		OutputAudioPath:         outputAudioPath,
		PreloadID:               preloadId,
		PreloadAudioPath:        preloadAudioPath,
		SampleMode:              sampleMode,
		DoSample:                doSample,
		Streaming:               streaming,
		MaxNewFrames:            maxNewFrames,
		VoiceCloneMaxTextTokens: voiceCloneMaxTextTokens,
		EnableRobust:            enableRobust,
		EnableWeText:            enableWeText,
		Seed:                    seed,
	})
	if err != nil {
		return nil, err
	}

	result := &SynthesisResult{
		SampleRate:   resp.SampleRate,
		Channels:     resp.Channels,
		AudioSamples: resp.AudioSamples,
		ElapsedSec:   resp.ElapsedSec,
		AudioPath:    resp.AudioPath,
		SampleMode:   resp.SampleMode,
		DoSample:     resp.DoSample,
		Streaming:    resp.Streaming,
		TextChunks:   resp.TextChunks,
	}

	if len(attachment) > 0 {
		result.Waveform = bytesToFloat32s(attachment)
	}

	return result, nil
}

// SynthesizeWithPreparedText 执行合成，使用主进程已预处理的文本，子进程不再调用 PrepareSynthesisTextEx
func (wp *WorkerProcess) SynthesizeWithPreparedText(ctx context.Context, preparedText string, voice string, promptAudioPath string, outputAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, seed *int) (*SynthesisResult, error) {
	resp, attachment, err := wp.sendRequest(ctx, &worker.Request{
		Type:                    worker.MsgSynthesize,
		Text:                    preparedText,
		Voice:                   voice,
		PromptAudioPath:         promptAudioPath,
		OutputAudioPath:         outputAudioPath,
		PreloadID:               preloadId,
		PreloadAudioPath:        preloadAudioPath,
		SampleMode:              sampleMode,
		DoSample:                doSample,
		Streaming:               streaming,
		MaxNewFrames:            maxNewFrames,
		VoiceCloneMaxTextTokens: voiceCloneMaxTextTokens,
		PreparedText:            preparedText,
		Seed:                    seed,
	})
	if err != nil {
		return nil, err
	}

	result := &SynthesisResult{
		SampleRate:   resp.SampleRate,
		Channels:     resp.Channels,
		AudioSamples: resp.AudioSamples,
		ElapsedSec:   resp.ElapsedSec,
		AudioPath:    resp.AudioPath,
		SampleMode:   resp.SampleMode,
		DoSample:     resp.DoSample,
		Streaming:    resp.Streaming,
		TextChunks:   resp.TextChunks,
	}

	if len(attachment) > 0 {
		result.Waveform = bytesToFloat32s(attachment)
	}

	return result, nil
}

// SynthesizeStreamEx 执行流式合成（兼容旧接口，主进程未预处理时使用）
func (wp *WorkerProcess) SynthesizeStreamEx(ctx context.Context, text string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (<-chan StreamChunk, error) {
	streamCh, err := wp.sendStreamRequest(ctx, &worker.Request{
		Type:                    worker.MsgSynthesizeStream,
		Text:                    text,
		Voice:                   voice,
		PromptAudioPath:         promptAudioPath,
		PreloadID:               preloadId,
		PreloadAudioPath:        preloadAudioPath,
		SampleMode:              sampleMode,
		DoSample:                doSample,
		MaxNewFrames:            maxNewFrames,
		VoiceCloneMaxTextTokens: voiceCloneMaxTextTokens,
		EnableRobust:            enableRobust,
		EnableWeText:            enableWeText,
		Seed:                    seed,
	})
	if err != nil {
		return nil, err
	}

	chunkChan := make(chan StreamChunk, 64)
	go func() {
		defer close(chunkChan)
		for resp := range streamCh {
			if resp.resp.Type == worker.MsgDone || resp.resp.Type == worker.MsgCancelled {
				return
			}
			if resp.resp.Type == worker.MsgError {
				log.Printf("[WorkerProcess #%s] 流式错误: %s", wp.Name, resp.resp.Error)
				return
			}
			if resp.resp.Type == worker.MsgChunk {
				var waveform []float32
				if len(resp.attachment) > 0 {
					waveform = bytesToFloat32s(resp.attachment)
				}
				chunkChan <- StreamChunk{
					Waveform:   waveform,
					SampleRate: resp.resp.SampleRate,
					Channels:   resp.resp.Channels,
					ChunkIndex: resp.resp.ChunkIndex,
					IsPause:    resp.resp.IsPause,
				}
			}
		}
	}()

	return chunkChan, nil
}

// SynthesizeStreamWithPreparedText 执行流式合成，使用主进程已预处理的文本，子进程不再调用 PrepareSynthesisTextEx
func (wp *WorkerProcess) SynthesizeStreamWithPreparedText(ctx context.Context, preparedText string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, seed *int) (<-chan StreamChunk, error) {
	streamCh, err := wp.sendStreamRequest(ctx, &worker.Request{
		Type:                    worker.MsgSynthesizeStream,
		Text:                    preparedText,
		Voice:                   voice,
		PromptAudioPath:         promptAudioPath,
		PreloadID:               preloadId,
		PreloadAudioPath:        preloadAudioPath,
		SampleMode:              sampleMode,
		DoSample:                doSample,
		MaxNewFrames:            maxNewFrames,
		VoiceCloneMaxTextTokens: voiceCloneMaxTextTokens,
		PreparedText:            preparedText,
		Seed:                    seed,
	})
	if err != nil {
		return nil, err
	}

	chunkChan := make(chan StreamChunk, 64)
	go func() {
		defer close(chunkChan)
		for resp := range streamCh {
			if resp.resp.Type == worker.MsgDone || resp.resp.Type == worker.MsgCancelled {
				return
			}
			if resp.resp.Type == worker.MsgError {
				log.Printf("[WorkerProcess #%s] 流式错误: %s", wp.Name, resp.resp.Error)
				return
			}
			if resp.resp.Type == worker.MsgChunk {
				var waveform []float32
				if len(resp.attachment) > 0 {
					waveform = bytesToFloat32s(resp.attachment)
				}
				chunkChan <- StreamChunk{
					Waveform:   waveform,
					SampleRate: resp.resp.SampleRate,
					Channels:   resp.resp.Channels,
					ChunkIndex: resp.resp.ChunkIndex,
					IsPause:    resp.resp.IsPause,
				}
			}
		}
	}()

	return chunkChan, nil
}

// ---- 工具函数 ----

func bytesToFloat32s(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	if len(data)%4 != 0 {
		// 非对齐数据回退到逐元素转换
		count := len(data) / 4
		result := make([]float32, count)
		for i := range result {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return result
	}
	// 零拷贝：直接将 []byte 底层内存解释为 []float32
	count := len(data) / 4
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), count)
}

func encodeWAV(waveform []float32, channels, sampleRate int) []byte {
	numSamples := len(waveform)
	bytesPerSample := 2
	dataSize := numSamples * bytesPerSample

	// normalizeVolume: 防止削波，保证 WAV 格式合规
	maxAbs := float32(0)
	for _, s := range waveform {
		if s < 0 {
			s = -s
		}
		if s > maxAbs {
			maxAbs = s
		}
	}
	normFactor := float32(1.0)
	if maxAbs > 1.0 {
		normFactor = 1.0 / maxAbs
	}

	buf := make([]byte, 44+dataSize)
	copy(buf[0:4], []byte("RIFF"))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(36+dataSize))
	copy(buf[8:12], []byte("WAVE"))
	copy(buf[12:16], []byte("fmt "))
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], 1)
	binary.LittleEndian.PutUint16(buf[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:32], uint32(sampleRate*channels*bytesPerSample))
	binary.LittleEndian.PutUint16(buf[32:34], uint16(channels*bytesPerSample))
	binary.LittleEndian.PutUint16(buf[34:36], uint16(bytesPerSample*8))
	copy(buf[36:40], []byte("data"))
	binary.LittleEndian.PutUint32(buf[40:44], uint32(dataSize))

	for i, s := range waveform {
		s *= normFactor
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		val := int16(s * 32767)
		binary.LittleEndian.PutUint16(buf[44+i*2:], uint16(val))
	}

	return buf
}

func marshalJSON(v interface{}) []byte {
	data, _ := json.Marshal(v)
	return data
}
