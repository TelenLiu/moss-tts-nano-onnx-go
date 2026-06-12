package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/worker"
)

// workerConnState 维护子进程连接的状态，包括活跃请求的取消函数和写锁
type workerConnState struct {
	writeMu sync.Mutex // 保护 conn 的并发写入

	mu          sync.Mutex
	activeTasks map[int]context.CancelFunc // reqID -> cancelFunc
}

func newWorkerConnState() *workerConnState {
	return &workerConnState{
		activeTasks: make(map[int]context.CancelFunc),
	}
}

func (s *workerConnState) registerTask(reqID int, cancel context.CancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeTasks[reqID] = cancel
}

func (s *workerConnState) cancelTask(reqID int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	cancel, ok := s.activeTasks[reqID]
	if ok {
		cancel()
		delete(s.activeTasks, reqID)
	}
	return ok
}

func (s *workerConnState) unregisterTask(reqID int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.activeTasks, reqID)
}

func (s *workerConnState) safeWriteResponse(conn net.Conn, id int, resp *worker.Response, attachment []byte) {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	resp.ID = id
	if err := worker.WriteResponse(conn, resp, attachment); err != nil {
		log.Printf("[Worker] 写响应失败: %v", err)
	}
}

func workerMain() {
	// 从 stdin 读取初始化参数
	var initReq worker.InitRequest
	if err := json.NewDecoder(os.Stdin).Decode(&initReq); err != nil {
		log.Fatalf("[Worker] 读取初始化参数失败: %v", err)
	}

	log.Printf("[Worker] 初始化参数: modelDir=%s threads=%d coreMemMB=%d listenAddr=%s",
		initReq.ModelDir, initReq.ThreadCount, initReq.CoreMemMB, initReq.ListenAddr)

	// 初始化依赖
	cfg := deps.DefaultConfig()
	deps.SetDynlibPath(cfg.LibDir)
	if err := ortruntime.InitializeORT(cfg.LibDir); err != nil {
		log.Fatalf("[Worker] 初始化 ONNX Runtime 失败: %v", err)
	}

	// 创建 TTS Runtime
	rt, err := ttsruntime.NewOnnxTtsRuntime(
		initReq.ModelDir, initReq.ThreadCount, initReq.CoreMemMB,
		&initReq.MaxNewFrames, initReq.DoSample, initReq.SampleMode, initReq.ExecutionMode,
	)
	if err != nil {
		log.Fatalf("[Worker] 创建 TTS Runtime 失败: %v", err)
	}
	defer rt.Close()

	// 监听 TCP
	listener, err := net.Listen("tcp", initReq.ListenAddr)
	if err != nil {
		log.Fatalf("[Worker] 监听失败: %v", err)
	}

	// 输出实际监听地址给主进程（通过 stdout）
	addr := listener.Addr().String()
	fmt.Printf("LISTEN:%s\n", addr)

	log.Printf("[Worker] 开始监听: %s", addr)

	// 信号处理：主进程退出时子进程也退出
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("[Worker] 收到退出信号")
		listener.Close()
		os.Exit(0)
	}()

	// 只接受一个连接（主进程）
	conn, err := listener.Accept()
	if err != nil {
		log.Fatalf("[Worker] 接受连接失败: %v", err)
	}
	defer conn.Close()
	listener.Close() // 不再接受新连接

	log.Printf("[Worker] 主进程已连接: %s", conn.RemoteAddr())

	// 处理请求
	state := newWorkerConnState()
	workerHandleConnection(conn, rt, state)
}

func workerHandleConnection(conn net.Conn, rt *ttsruntime.OnnxTtsRuntime, state *workerConnState) {
	for {
		req, _, err := worker.ReadRequest(conn)
		if err != nil {
			if strings.Contains(err.Error(), "closed") || strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "reset") {
				log.Printf("[Worker] 连接关闭")
				return
			}
			log.Printf("[Worker] 读取请求失败: %v", err)
			return
		}

		switch req.Type {
		case worker.MsgPing:
			state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgPong, Pong: true}, nil)

		case worker.MsgEncodeText:
			tokenIDs := rt.EncodeText(req.TextContent)
			state.safeWriteResponse(conn, req.ID, &worker.Response{
				Type:     worker.MsgResult,
				TokenIDs: tokenIDs,
			}, nil)

		case worker.MsgCountTokens:
			count := rt.CountTextTokens(req.TextContent)
			state.safeWriteResponse(conn, req.ID, &worker.Response{
				Type:       worker.MsgResult,
				TokenCount: count,
			}, nil)

		case worker.MsgListVoices:
			voices := rt.OrtRuntime.ListBuiltinVoices()
			state.safeWriteResponse(conn, req.ID, &worker.Response{
				Type:   worker.MsgResult,
				Voices: voices,
			}, nil)

		case worker.MsgPreload:
			if rt.PreloadCache != nil {
				err := rt.PreloadCache.Preload(req.PreloadID, req.AudioPath, "")
				if err != nil {
					state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: err.Error()}, nil)
				} else {
					state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgResult}, nil)
				}
			} else {
				state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: "PreloadCache未初始化"}, nil)
			}

		case worker.MsgSynthesize:
			// 在 goroutine 中执行推理，以便主循环能继续读取 cancel 消息
			go workerHandleSynthesize(conn, req, rt, state)

		case worker.MsgSynthesizeStream:
			// 在 goroutine 中执行流式推理
			go workerHandleSynthesizeStream(conn, req, rt, state)

		case worker.MsgCancel:
			// 取消指定请求的推理
			cancelReqID := req.CancelReqID
			if cancelReqID == 0 {
				// 兼容：如果 CancelReqID 为 0，尝试使用 req.ID（旧方式）
				cancelReqID = req.ID
			}
			cancelled := state.cancelTask(cancelReqID)
			if cancelled {
				log.Printf("[Worker] 已取消请求 #%d 的推理", cancelReqID)
				state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgCancelled}, nil)
			} else {
				log.Printf("[Worker] 未找到请求 #%d，无法取消", cancelReqID)
				state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: "未找到请求"}, nil)
			}

		default:
			state.safeWriteResponse(conn, req.ID, &worker.Response{
				Type:  worker.MsgError,
				Error: fmt.Sprintf("未知请求类型: %s", req.Type),
			}, nil)
		}
	}
}

func workerHandleSynthesize(conn net.Conn, req *worker.Request, rt *ttsruntime.OnnxTtsRuntime, state *workerConnState) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	state.registerTask(req.ID, cancel)
	defer state.unregisterTask(req.ID)

	result, err := rt.SynthesizeWithContextEx(
		ctx,
		req.Text, req.Voice, req.PromptAudioPath, req.OutputAudioPath,
		req.PreloadID, req.PreloadAudioPath,
		req.SampleMode, req.DoSample, req.Streaming,
		req.MaxNewFrames, req.VoiceCloneMaxTextTokens,
		req.EnableRobust, req.EnableWeText, req.Seed,
	)
	if err != nil {
		// 检查是否因取消导致
		select {
		case <-ctx.Done():
			log.Printf("[Worker] 请求 #%d 已被取消", req.ID)
			state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgCancelled}, nil)
			return
		default:
		}
		state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: err.Error()}, nil)
		return
	}

	// 检查是否在推理完成后被取消
	select {
	case <-ctx.Done():
		log.Printf("[Worker] 请求 #%d 推理完成但已被取消，丢弃结果", req.ID)
		return
	default:
	}

	// 将 float32 波形转为 bytes attachment
	attachment := workerFloat32sToBytes(result.Waveform)

	state.safeWriteResponse(conn, req.ID, &worker.Response{
		Type:         worker.MsgResult,
		SampleRate:   result.SampleRate,
		Channels:     result.Channels,
		AudioSamples: result.AudioSamples,
		ElapsedSec:   result.ElapsedSec,
		AudioPath:    result.AudioPath,
		SampleMode:   result.SampleMode,
		DoSample:     result.DoSample,
		Streaming:    result.Streaming,
		TextChunks:   result.TextChunks,
		HasAudioData: len(attachment) > 0,
	}, attachment)
}

func workerHandleSynthesizeStream(conn net.Conn, req *worker.Request, rt *ttsruntime.OnnxTtsRuntime, state *workerConnState) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	state.registerTask(req.ID, cancel)
	defer state.unregisterTask(req.ID)

	chunkChan, err := rt.SynthesizeStreamEx(
		ctx, req.Text, req.Voice, req.PromptAudioPath,
		req.PreloadID, req.PreloadAudioPath,
		req.SampleMode, req.DoSample,
		req.MaxNewFrames, req.VoiceCloneMaxTextTokens,
		req.EnableRobust, req.EnableWeText, req.Seed,
	)
	if err != nil {
		state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: err.Error()}, nil)
		return
	}

	for chunk := range chunkChan {
		// 检查是否被取消
		select {
		case <-ctx.Done():
			log.Printf("[Worker] 流式请求 #%d 已被取消", req.ID)
			state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgCancelled}, nil)
			return
		default:
		}

		attachment := workerFloat32sToBytes(chunk.Waveform)
		state.safeWriteResponse(conn, req.ID, &worker.Response{
			Type:         worker.MsgChunk,
			SampleRate:   chunk.SampleRate,
			Channels:     chunk.Channels,
			AudioSamples: len(chunk.Waveform) / chunk.Channels,
			ChunkIndex:   chunk.ChunkIndex,
			IsPause:      chunk.IsPause,
			HasAudioData: len(attachment) > 0,
		}, attachment)
	}

	// 发送流结束标记
	state.safeWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgDone}, nil)
}

func workerFloat32sToBytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// 保留 unused import 引用
var (
	_ = rand.New
	_ = runtime.GC
	_ = debug.FreeOSMemory
	_ = time.Now
)
