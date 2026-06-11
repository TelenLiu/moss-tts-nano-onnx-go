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
	"syscall"
	"time"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/worker"
)

func workerMain() {
	// 设置进程标题，便于在活动监视器中区分主进程和子进程
	if len(os.Args) > 0 {
		os.Args[0] = "moss-tts-worker"
	}

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
	workerHandleConnection(conn, rt)
}

func workerHandleConnection(conn net.Conn, rt *ttsruntime.OnnxTtsRuntime) {
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
			workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgPong, Pong: true}, nil)

		case worker.MsgEncodeText:
			tokenIDs := rt.EncodeText(req.TextContent)
			workerWriteResponse(conn, req.ID, &worker.Response{
				Type:     worker.MsgResult,
				TokenIDs: tokenIDs,
			}, nil)

		case worker.MsgCountTokens:
			count := rt.CountTextTokens(req.TextContent)
			workerWriteResponse(conn, req.ID, &worker.Response{
				Type:       worker.MsgResult,
				TokenCount: count,
			}, nil)

		case worker.MsgListVoices:
			voices := rt.OrtRuntime.ListBuiltinVoices()
			workerWriteResponse(conn, req.ID, &worker.Response{
				Type:   worker.MsgResult,
				Voices: voices,
			}, nil)

		case worker.MsgPreload:
			if rt.PreloadCache != nil {
				err := rt.PreloadCache.Preload(req.PreloadID, req.AudioPath, "")
				if err != nil {
					workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: err.Error()}, nil)
				} else {
					workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgResult}, nil)
				}
			} else {
				workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: "PreloadCache未初始化"}, nil)
			}

		case worker.MsgSynthesize:
			workerHandleSynthesize(conn, req, rt)

		case worker.MsgSynthesizeStream:
			workerHandleSynthesizeStream(conn, req, rt)

		default:
			workerWriteResponse(conn, req.ID, &worker.Response{
				Type:  worker.MsgError,
				Error: fmt.Sprintf("未知请求类型: %s", req.Type),
			}, nil)
		}
	}
}

func workerHandleSynthesize(conn net.Conn, req *worker.Request, rt *ttsruntime.OnnxTtsRuntime) {
	result, err := rt.SynthesizeWithContextEx(
		context.Background(),
		req.Text, req.Voice, req.PromptAudioPath, req.OutputAudioPath,
		req.PreloadID, req.PreloadAudioPath,
		req.SampleMode, req.DoSample, req.Streaming,
		req.MaxNewFrames, req.VoiceCloneMaxTextTokens,
		req.EnableRobust, req.EnableWeText, req.Seed,
	)
	if err != nil {
		workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: err.Error()}, nil)
		return
	}

	// 将 float32 波形转为 bytes attachment
	attachment := workerFloat32sToBytes(result.Waveform)

	workerWriteResponse(conn, req.ID, &worker.Response{
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

func workerHandleSynthesizeStream(conn net.Conn, req *worker.Request, rt *ttsruntime.OnnxTtsRuntime) {
	ctx := context.Background()
	chunkChan, err := rt.SynthesizeStreamEx(
		ctx, req.Text, req.Voice, req.PromptAudioPath,
		req.PreloadID, req.PreloadAudioPath,
		req.SampleMode, req.DoSample,
		req.MaxNewFrames, req.VoiceCloneMaxTextTokens,
		req.EnableRobust, req.EnableWeText, req.Seed,
	)
	if err != nil {
		workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgError, Error: err.Error()}, nil)
		return
	}

	for chunk := range chunkChan {
		attachment := workerFloat32sToBytes(chunk.Waveform)
		workerWriteResponse(conn, req.ID, &worker.Response{
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
	workerWriteResponse(conn, req.ID, &worker.Response{Type: worker.MsgDone}, nil)
}

func workerWriteResponse(conn net.Conn, id int, resp *worker.Response, attachment []byte) {
	resp.ID = id
	if err := worker.WriteResponse(conn, resp, attachment); err != nil {
		log.Printf("[Worker] 写响应失败: %v", err)
	}
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
