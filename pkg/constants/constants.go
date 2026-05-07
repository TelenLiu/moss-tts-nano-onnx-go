package constants

const (
	// 音频生成参数
	DefaultMaxNewFrames     = 375   // 默认最大生成音频帧数（每帧 10ms，约 3.75 秒）
	MinMaxNewFrames         = 50    // 最小生成帧数（0.5 秒）
	MaxMaxNewFrames         = 2000  // 最大生成帧数（20 秒）
	
	// 文本分块参数
	DefaultVoiceCloneTokens = 75    // 默认文本分块 token 预算
	MinVoiceCloneTokens     = 10    // 最小 token 预算
	MaxVoiceCloneTokens     = 500   // 最大 token 预算
	
	// 采样参数
	DefaultTextTemperature  = 1.0   // 文本层采样温度（1.0=原始分布）
	DefaultTextTopP         = 1.0   // 文本层 nucleus sampling 阈值
	DefaultTextTopK         = 50    // 文本层 top-k 采样数量
	DefaultAudioTemperature = 0.8   // 音频层采样温度（较低=更保守）
	DefaultAudioTopP        = 0.95  // 音频层 nucleus sampling 阈值
	DefaultAudioTopK        = 25    // 音频层 top-k 采样数量
	DefaultAudioRepPenalty  = 1.2   // 音频层重复惩罚（>1 抑制重复）
	
	// 分块间隔参数
	DefaultInterChunkPauseShortSec = 0.15 // 短分块间隔（秒）
	DefaultInterChunkPauseLongSec  = 0.10 // 长分块间隔（秒）
	
	// CPU 线程参数
	DefaultCpuThreads = -1 // -1 表示自动检测（CPU 核心数 -1）
	MinCpuThreads     = 1  // 最小线程数
	MaxCpuThreads     = 64 // 最大线程数
	
	// 随机种子
	DefaultSeed = -1 // -1 表示不设置随机种子
	
	// 流式解码参数
	StreamDecodeBudgetMin = 1   // 最小流式解码帧预算
	StreamDecodeBudgetMax = 8   // 最大流式解码帧预算
	StreamLeadThreshold1  = 0.20 // 领先阈值 1（秒）
	StreamLeadThreshold2  = 0.55 // 领先阈值 2（秒）
	StreamLeadThreshold3  = 1.10 // 领先阈值 3（秒）
	
	// 音频处理参数
	AudioSampleRate    = 24000  // 音频采样率 (Hz)
	AudioChannels      = 1      // 音频通道数
	AudioBitsPerSample = 16     // 每样本位数
	
	// 下载参数
	DownloadBufferSize = 32 * 1024 // 下载缓冲区大小 (32KB)
	DownloadReportInterval = 500  // 下载进度报告间隔 (ms)
)
