package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/web"
)

func main() {
	if len(os.Args) < 2 {
		runServe(os.Args[1:])
		return
	}

	switch os.Args[1] {
	case "infer":
		runInfer(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	case "download":
		runDownload(os.Args[2:])
	case "help", "-h", "--help":
		printUsage()
	default:
		if strings.HasPrefix(os.Args[1], "-") {
			runServe(os.Args[1:])
		} else {
			printUsage()
			os.Exit(1)
		}
	}
}

func printUsage() {
	fmt.Println("MOSS-TTS-Nano ONNX Go - 语音合成工具")
	fmt.Println()
	fmt.Println("用法:")
	fmt.Println("  moss-tts-nano-onnx-go              启动Web体验服务 (默认)")
	fmt.Println("  moss-tts-nano-onnx-go serve [选项]  启动Web体验服务")
	fmt.Println("  moss-tts-nano-onnx-go infer [选项]  运行语音合成推理")
	fmt.Println("  moss-tts-nano-onnx-go download [选项] 下载模型和依赖")
	fmt.Println()
	fmt.Println("使用 'serve -h', 'infer -h', 'download -h' 查看子命令详细帮助")
}

func runInfer(args []string) {
	fs := flag.NewFlagSet("infer", flag.ExitOnError)
	modelDir := fs.String("model-dir", "", "模型目录 (默认自动下载)")
	outputPath := fs.String("output-audio-path", "", "输出音频路径")
	text := fs.String("text", "", "要合成的文本")
	textFile := fs.String("text-file", "", "包含合成文本的文件路径")
	voice := fs.String("voice", "Junhao", "内置音色名称")
	promptAudioPath := fs.String("prompt-audio-path", "", "参考音频路径 (覆盖 --voice)")
	sampleMode := fs.String("sample-mode", "fixed", "采样模式: greedy, fixed, full")
	doSample := fs.Int("do-sample", 1, "是否采样 (0/1)")
	cpuThreads := fs.Int("cpu-threads", defaultCpuThreads(), "ONNX Runtime 线程数 (默认: CPU核心数-1, 至少为1)")
	maxNewFrames := fs.Int("max-new-frames", 375, "最大生成音频帧数")
	voiceCloneMaxTokens := fs.Int("voice-clone-max-text-tokens", 75, "文本分块token预算")
	textTemp := fs.Float64("text-temperature", 1.0, "文本层采样温度")
	textTopP := fs.Float64("text-top-p", 1.0, "文本层top-p")
	textTopK := fs.Int("text-top-k", 50, "文本层top-k")
	audioTemp := fs.Float64("audio-temperature", 0.8, "音频层采样温度")
	audioTopP := fs.Float64("audio-top-p", 0.95, "音频层top-p")
	audioTopK := fs.Int("audio-top-k", 25, "音频层top-k")
	audioRepPenalty := fs.Float64("audio-repetition-penalty", 1.2, "音频层重复惩罚")
	enableNormalize := fs.Bool("enable-normalize-tts-text", true, "启用TTS文本正则化")
	disableNormalize := fs.Bool("disable-normalize-tts-text", false, "禁用TTS文本正则化")
	seed := fs.Int("seed", -1, "随机种子 (-1 表示不设置)")
	useMirror := fs.Bool("mirror", false, "使用国内加速镜像源下载依赖和模型")
	fs.Parse(args)

	validateText(*text, *textFile)

	cfg := deps.DefaultConfig()
	cfg.UseMirror = *useMirror
	if *modelDir != "" {
		cfg.ModelDir = *modelDir
	}

	log.Println("确保本地依赖已就绪...")
	deps.SetDynlibPath(cfg.LibDir)
	if err := deps.EnsureNativeLibs(cfg); err != nil {
		log.Fatalf("本地依赖准备失败: %v", err)
	}
	if err := deps.EnsureModels(cfg); err != nil {
		log.Fatalf("模型准备失败: %v", err)
	}

	resolvedText := resolveTextContent(*text, *textFile)
	doSampleBool := *doSample != 0
	normalizeEnabled := *enableNormalize && !*disableNormalize
	maxFrames := *maxNewFrames
	seedOpt := (*int)(nil)
	if *seed >= 0 {
		seedOpt = seed
	}

	_ = textTemp
	_ = textTopP
	_ = textTopK
	_ = audioTemp
	_ = audioTopP
	_ = audioTopK
	_ = audioRepPenalty

	if err := ortruntime.InitializeORT(cfg.LibDir); err != nil {
		log.Fatalf("初始化 ONNX Runtime 环境失败: %v", err)
	}

	cwd, _ := os.Getwd()
	if *outputPath == "" {
		*outputPath = filepath.Join(cwd, "infer_onnx_output.wav")
	}

	log.Printf("初始化 TTS 运行时 (model_dir=%s threads=%d)...", cfg.ModelDir, *cpuThreads)
	rt, err := ttsruntime.NewOnnxTtsRuntime(cfg.ModelDir, *cpuThreads, &maxFrames, &doSampleBool, sampleMode)
	if err != nil {
		log.Fatalf("初始化 TTS 运行时失败: %v", err)
	}
	defer rt.Close()

	log.Printf("开始合成: text=%q voice=%s sample_mode=%s", resolvedText, *voice, *sampleMode)
	result, err := rt.Synthesize(
		resolvedText, *voice, *promptAudioPath, *outputPath,
		*sampleMode, doSampleBool, false,
		*maxNewFrames, *voiceCloneMaxTokens,
		normalizeEnabled, seedOpt,
	)
	if err != nil {
		log.Fatalf("合成失败: %v", err)
	}
	log.Printf("合成完成: audio_path=%s sample_rate=%d elapsed=%.2fs frames=%d",
		result.AudioPath, result.SampleRate, result.ElapsedSec, len(result.TextChunks))
}

func defaultCpuThreads() int {
	n := runtime.NumCPU()
	if n > 1 {
		return n - 1
	}
	return 1
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelDir := fs.String("model-dir", "", "模型目录 (默认自动下载)")
	host := fs.String("host", "localhost", "监听地址")
	port := fs.Int("port", 18083, "监听端口")
	cpuThreads := fs.Int("cpu-threads", defaultCpuThreads(), "ONNX Runtime 线程数 (默认: CPU核心数-1, 至少为1)")
	maxNewFrames := fs.Int("max-new-frames", 375, "最大生成音频帧数")
	useMirror := fs.Bool("mirror", false, "使用国内加速镜像源下载依赖和模型")
	fs.Parse(args)

	cfg := deps.DefaultConfig()
	cfg.UseMirror = *useMirror
	if *modelDir != "" {
		cfg.ModelDir = *modelDir
	}

	appRoot := fmt.Sprintf("http://%s:%d", *host, *port)
	srv := web.NewServer(cfg, *cpuThreads, *maxNewFrames, *host, *port, appRoot)
	if err := srv.Start(); err != nil {
		log.Fatalf("Web 服务启动失败: %v", err)
	}
}

func runDownload(args []string) {
	fs := flag.NewFlagSet("download", flag.ExitOnError)
	useMirror := fs.Bool("mirror", false, "使用国内加速镜像源下载")
	modelDir := fs.String("model-dir", "", "模型保存目录")
	libDir := fs.String("lib-dir", "", "本地依赖保存目录")
	fs.Parse(args)

	cfg := deps.DefaultConfig()
	cfg.UseMirror = *useMirror
	if *modelDir != "" {
		cfg.ModelDir = *modelDir
	}
	if *libDir != "" {
		cfg.LibDir = *libDir
	}

	log.Println("下载本地依赖...")
	if err := deps.EnsureNativeLibs(cfg); err != nil {
		log.Fatalf("下载本地依赖失败: %v", err)
	}
	log.Println("下载模型文件...")
	if err := deps.EnsureModels(cfg); err != nil {
		log.Fatalf("下载模型文件失败: %v", err)
	}
	log.Println("所有依赖和模型下载完成!")
}

func validateText(text, textFile string) {
	if text == "" && textFile == "" {
		fmt.Println("错误: 必须指定 --text 或 --text-file")
		os.Exit(1)
	}
}

func resolveTextContent(text, textFile string) string {
	if text != "" {
		return text
	}
	data, err := os.ReadFile(textFile)
	if err != nil {
		log.Fatalf("读取文本文件失败: %v", err)
	}
	return string(data)
}
