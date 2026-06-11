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
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/worker"
)

// WorkerProcess 管理一个子进程推理单元
type WorkerProcess struct {
	ID   int
	cmd  *exec.Cmd
	conn net.Conn
	addr string // 子进程实际监听地址

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

// NewWorkerProcess 启动一个子进程推理单元
func NewWorkerProcess(id int, modelDir string, threadCount int, coreMemMB int, maxNewFrames *int, doSample *bool, sampleMode *string, executionMode string) (*WorkerProcess, error) {
	wp := &WorkerProcess{
		ID:      id,
		pending: make(map[int]chan *workerResponse),
	}

	// 获取当前可执行文件路径
	exePath, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("获取可执行文件路径失败: %w", err)
	}

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

	// 启动子进程
	cmd := exec.Command(exePath, "worker")
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
	log.Printf("[WorkerProcess #%d] 子进程已启动, addr=%s, pid=%d", id, addr, cmd.Process.Pid)

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
				log.Printf("[WorkerProcess #%d] 读取响应失败: %v", wp.ID, err)
			}
			wp.closePending(fmt.Errorf("连接断开: %w", err))
			return
		}

		wp.mu.Lock()
		ch, ok := wp.pending[resp.ID]
		// 流式请求（MsgChunk）不删除 pending，等 MsgDone/MsgError 时才删除
		if ok && (resp.Type == worker.MsgDone || resp.Type == worker.MsgError || resp.Type == worker.MsgResult || resp.Type == worker.MsgPong) {
			delete(wp.pending, resp.ID)
		}
		wp.mu.Unlock()

		if ok && ch != nil {
			ch <- &workerResponse{resp: resp, attachment: attachment}
		}
	}
}

// sendRequest 发送请求并等待响应
func (wp *WorkerProcess) sendRequest(req *worker.Request) (*worker.Response, []byte, error) {
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

	if err := worker.WriteRequest(wp.conn, req, nil); err != nil {
		return nil, nil, fmt.Errorf("发送请求失败: %w", err)
	}

	select {
	case result := <-ch:
		if result.resp.Type == worker.MsgError {
			return nil, nil, fmt.Errorf("worker错误: %s", result.resp.Error)
		}
		return result.resp, result.attachment, nil
	case <-time.After(300 * time.Second):
		return nil, nil, fmt.Errorf("等待worker响应超时")
	}
}

// sendStreamRequest 发送流式请求，返回响应channel
func (wp *WorkerProcess) sendStreamRequest(req *worker.Request) (<-chan *workerResponse, error) {
	if wp.isClosed() {
		return nil, fmt.Errorf("worker进程已关闭")
	}

	reqID := int(wp.nextReqID.Add(1))
	req.ID = reqID

	streamCh := make(chan *workerResponse, 64)

	wp.mu.Lock()
	wp.pending[reqID] = streamCh
	wp.mu.Unlock()

	if err := worker.WriteRequest(wp.conn, req, nil); err != nil {
		wp.mu.Lock()
		delete(wp.pending, reqID)
		wp.mu.Unlock()
		close(streamCh)
		return nil, fmt.Errorf("发送流式请求失败: %w", err)
	}

	// 包装channel，在收到 Done 后清理
	outCh := make(chan *workerResponse, 64)
	go func() {
		defer close(outCh)
		for resp := range streamCh {
			outCh <- resp
			if resp.resp.Type == worker.MsgDone || resp.resp.Type == worker.MsgError {
				wp.mu.Lock()
				delete(wp.pending, reqID)
				wp.mu.Unlock()
				return
			}
		}
	}()

	return outCh, nil
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
	resp, _, err := wp.sendRequest(&worker.Request{Type: worker.MsgPing})
	if err != nil {
		return false
	}
	return resp.Type == worker.MsgPong
}

// EncodeText 编码文本为 token IDs
func (wp *WorkerProcess) EncodeText(text string) []int {
	resp, _, err := wp.sendRequest(&worker.Request{
		Type:        worker.MsgEncodeText,
		TextContent: text,
	})
	if err != nil {
		log.Printf("[WorkerProcess #%d] EncodeText失败: %v", wp.ID, err)
		return nil
	}
	return resp.TokenIDs
}

// CountTextTokens 计算 token 数
func (wp *WorkerProcess) CountTextTokens(text string) int {
	resp, _, err := wp.sendRequest(&worker.Request{
		Type:        worker.MsgCountTokens,
		TextContent: text,
	})
	if err != nil {
		log.Printf("[WorkerProcess #%d] CountTextTokens失败: %v", wp.ID, err)
		return 0
	}
	return resp.TokenCount
}

// ListBuiltinVoices 列出内置音色
func (wp *WorkerProcess) ListBuiltinVoices() []map[string]interface{} {
	resp, _, err := wp.sendRequest(&worker.Request{Type: worker.MsgListVoices})
	if err != nil {
		log.Printf("[WorkerProcess #%d] ListBuiltinVoices失败: %v", wp.ID, err)
		return nil
	}
	return resp.Voices
}

// PreloadVoice 预加载音色
func (wp *WorkerProcess) PreloadVoice(preloadID, audioPath, voice string) error {
	resp, _, err := wp.sendRequest(&worker.Request{
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

// SynthesizeWithContextEx 执行合成
func (wp *WorkerProcess) SynthesizeWithContextEx(ctx context.Context, text string, voice string, promptAudioPath string, outputAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, streaming bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (*SynthesisResult, error) {
	resp, attachment, err := wp.sendRequest(&worker.Request{
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
		result.AudioData = encodeWAV(result.Waveform, result.Channels, result.SampleRate)
	}

	return result, nil
}

// SynthesizeStreamEx 执行流式合成
func (wp *WorkerProcess) SynthesizeStreamEx(ctx context.Context, text string, voice string, promptAudioPath string, preloadId string, preloadAudioPath string, sampleMode string, doSample bool, maxNewFrames int, voiceCloneMaxTextTokens int, enableRobust bool, enableWeText bool, seed *int) (<-chan StreamChunk, error) {
	streamCh, err := wp.sendStreamRequest(&worker.Request{
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
			if resp.resp.Type == worker.MsgDone {
				return
			}
			if resp.resp.Type == worker.MsgError {
				log.Printf("[WorkerProcess #%d] 流式错误: %s", wp.ID, resp.resp.Error)
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
	count := len(data) / 4
	result := make([]float32, count)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}

func encodeWAV(waveform []float32, channels, sampleRate int) []byte {
	numSamples := len(waveform)
	bytesPerSample := 2
	dataSize := numSamples * bytesPerSample

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
		val := int16(s * 32767)
		if val > 32767 {
			val = 32767
		}
		if val < -32768 {
			val = -32768
		}
		binary.LittleEndian.PutUint16(buf[44+i*2:], uint16(val))
	}

	return buf
}

func marshalJSON(v interface{}) []byte {
	data, _ := json.Marshal(v)
	return data
}
