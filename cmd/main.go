package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/deps"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/normalizer"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/onnxconfig"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ortruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/proxy"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/ttsruntime"
	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/web"
)

var (
	Version = "0"
)

func init() {
	// 启动时尝试加载代理配置（conf/proxy.json）。
	// 加载失败或不存在时不会影响对外 API/Web 服务的监听。
	if cfg, err := proxy.Load(proxy.DefaultConfigPath()); err != nil {
		//log.Printf("[Proxy] 警告: 加载代理配置失败, 已忽略: %v", err)
	} else if cfg != nil {
		log.Printf("[Proxy] 代理已启用: type=%s addr=%s:%s", cfg.Type, cfg.IP, cfg.Port)
	}
}

func main() {
	//版本输出
	fmt.Println("构建版本：", Version)

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
	executionMode := fs.String("execution-mode", "hybrid", "推理执行模式: hybrid(CPU+GPU混合), cpu(仅CPU), gpu(仅GPU)")
	maxNewFrames := fs.Int("max-new-frames", 375, "最大生成音频帧数")
	voiceCloneMaxTokens := fs.Int("voice-clone-max-text-tokens", 75, "文本分块token预算")
	textTemp := fs.Float64("text-temperature", 1.0, "文本层采样温度")
	textTopP := fs.Float64("text-top-p", 1.0, "文本层top-p")
	textTopK := fs.Int("text-top-k", 50, "文本层top-k")
	audioTemp := fs.Float64("audio-temperature", 0.8, "音频层采样温度")
	audioTopP := fs.Float64("audio-top-p", 0.95, "音频层top-p")
	audioTopK := fs.Int("audio-top-k", 25, "音频层top-k")
	audioRepPenalty := fs.Float64("audio-repetition-penalty", 1.2, "音频层重复惩罚")
	enableRobust := fs.Bool("enable-robust", true, "启用鲁棒性文本归一化 (标点/空格/括号处理)")
	disableRobust := fs.Bool("disable-robust", false, "禁用鲁棒性文本归一化")
	enableWeText := fs.Bool("enable-wetext", true, "启用WeTextProcessing (数字/日期/金额展开)")
	disableWeText := fs.Bool("disable-wetext", false, "禁用WeTextProcessing")
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
	robustEnabled := *enableRobust && !*disableRobust
	wetextEnabled := *enableWeText && !*disableWeText
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

	log.Printf("初始化 TTS 运行时 (model_dir=%s threads=%d mode=%s)...", cfg.ModelDir, *cpuThreads, *executionMode)
	rt, err := ttsruntime.NewOnnxTtsRuntime(cfg.ModelDir, *cpuThreads, &maxFrames, &doSampleBool, sampleMode, *executionMode)
	if err != nil {
		log.Fatalf("初始化 TTS 运行时失败: %v", err)
	}
	defer rt.Close()

	log.Printf("开始合成: text=%q voice=%s sample_mode=%s robust=%v wetext=%v", resolvedText, *voice, *sampleMode, robustEnabled, wetextEnabled)
	result, err := rt.SynthesizeEx(
		resolvedText, *voice, *promptAudioPath, *outputPath,
		*sampleMode, doSampleBool, false,
		*maxNewFrames, *voiceCloneMaxTokens,
		robustEnabled, wetextEnabled, seedOpt,
	)
	if err != nil {
		log.Fatalf("合成失败: %v", err)
	}
	log.Printf("合成完成: audio_path=%s sample_rate=%d elapsed=%.2fs frames=%d",
		result.AudioPath, result.SampleRate, result.ElapsedSec, len(result.TextChunks))
}

func defaultCpuThreads() int {
	// 优先从 onnx.json 读取 coreCPUs
	if cfg, err := onnxconfig.Load(""); err == nil && cfg.CoreCPUs > 0 {
		return cfg.CoreCPUs
	}
	n := runtime.NumCPU()
	if n > 1 {
		return n - 1
	}
	return 1
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelDir := fs.String("model-dir", "", "模型目录 (默认自动下载)")
	host := fs.String("host", "", "监听地址 (留空监听所有网络接口)")
	port := fs.Int("port", 18083, "监听端口")
	cpuThreads := fs.Int("cpu-threads", defaultCpuThreads(), "ONNX Runtime 线程数 (默认: CPU核心数-1, 至少为1)")
	executionMode := fs.String("execution-mode", "hybrid", "推理执行模式: hybrid(CPU+GPU混合), cpu(仅CPU), gpu(仅GPU)")
	maxNewFrames := fs.Int("max-new-frames", 375, "最大生成音频帧数")
	useMirror := fs.Bool("mirror", false, "使用国内加速镜像源下载依赖和模型")
	fs.Parse(args)

	cfg := deps.DefaultConfig()
	cfg.UseMirror = *useMirror
	if *modelDir != "" {
		cfg.ModelDir = *modelDir
	}

	displayHost := *host
	if displayHost == "" {
		displayHost = "0.0.0.0"
	}
	appRoot := fmt.Sprintf("http://%s:%d", displayHost, *port)
	srv := web.NewServer(cfg, *cpuThreads, *maxNewFrames, *executionMode, *host, *port, appRoot)

	// 设置优雅退出：捕获中断信号并释放资源
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		log.Println("\n收到退出信号，正在清理资源...")
		srv.Close()
		os.Exit(0)
	}()

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

	log.Println("构建文本归一化 FST 缓存（首次运行需要 5-10 分钟）...")
	if err := normalizer.BuildCache(); err != nil {
		log.Printf("警告: 文本归一化缓存构建异常: %v", err)
	} else {
		log.Println("文本归一化 FST 缓存构建完成!")
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
