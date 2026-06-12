package worker

// MessageType 请求/响应类型
type MessageType string

const (
	// 请求类型
	MsgSynthesize       MessageType = "synthesize"
	MsgSynthesizeStream MessageType = "synthesize_stream"
	MsgEncodeText       MessageType = "encode_text"
	MsgCountTokens      MessageType = "count_tokens"
	MsgListVoices       MessageType = "list_voices"
	MsgPreload          MessageType = "preload"
	MsgPing             MessageType = "ping"
	MsgCancel           MessageType = "cancel" // 取消正在进行的推理请求

	// 响应类型
	MsgResult  MessageType = "result"
	MsgChunk   MessageType = "chunk"
	MsgError   MessageType = "error"
	MsgPong    MessageType = "pong"
	MsgDone    MessageType = "done" // 流式结束标记
	MsgCancelled MessageType = "cancelled" // 请求已被取消
)

// Request 主进程发给子进程的请求
type Request struct {
	ID   int         `json:"id"`
	Type MessageType `json:"type"`

	// Synthesize / SynthesizeStream 参数
	Text                  string `json:"text,omitempty"`
	Voice                 string `json:"voice,omitempty"`
	PromptAudioPath       string `json:"promptAudioPath,omitempty"`
	OutputAudioPath       string `json:"outputAudioPath,omitempty"`
	PreloadID             string `json:"preloadId,omitempty"`
	PreloadAudioPath      string `json:"preloadAudioPath,omitempty"`
	SampleMode            string `json:"sampleMode,omitempty"`
	DoSample              bool   `json:"doSample,omitempty"`
	Streaming             bool   `json:"streaming,omitempty"`
	MaxNewFrames          int    `json:"maxNewFrames,omitempty"`
	VoiceCloneMaxTextTokens int  `json:"voiceCloneMaxTextTokens,omitempty"`
	EnableRobust          bool   `json:"enableRobust,omitempty"`
	EnableWeText          bool   `json:"enableWeText,omitempty"`
	Seed                  *int   `json:"seed,omitempty"`

	// PreparedText 主进程已预处理好的文本，子进程直接使用不再调用 PrepareSynthesisTextEx
	PreparedText         string `json:"preparedText,omitempty"`

	// EncodeText / CountTokens 参数
	TextContent string `json:"textContent,omitempty"`

	// Preload 参数
	AudioPath string `json:"audioPath,omitempty"`

	// Cancel 参数：取消指定 reqID 的推理
	CancelReqID int `json:"cancelReqId,omitempty"`

	// PromptAudioCodes 二进制数据（通过 attachment 发送）
	HasPromptAudioCodes bool `json:"hasPromptAudioCodes,omitempty"`
}

// Response 子进程返回给主进程的响应
type Response struct {
	ID   int         `json:"id"`
	Type MessageType `json:"type"`

	// 错误信息
	Error string `json:"error,omitempty"`

	// Synthesize 结果
	SampleRate   int     `json:"sampleRate,omitempty"`
	Channels     int     `json:"channels,omitempty"`
	AudioSamples int     `json:"audioSamples,omitempty"`
	ElapsedSec   float64 `json:"elapsedSec,omitempty"`
	AudioPath    string  `json:"audioPath,omitempty"`
	SampleMode   string  `json:"sampleMode,omitempty"`
	DoSample     bool    `json:"doSample,omitempty"`
	Streaming    bool    `json:"streaming,omitempty"`

	// TextChunks
	TextChunks []string `json:"textChunks,omitempty"`

	// EncodeText 结果
	TokenIDs []int `json:"tokenIds,omitempty"`

	// CountTokens 结果
	TokenCount int `json:"tokenCount,omitempty"`

	// ListVoices 结果
	Voices []map[string]interface{} `json:"voices,omitempty"`

	// Pong
	Pong bool `json:"pong,omitempty"`

	// 音频数据通过 attachment 发送（float32 波形）
	HasAudioData bool `json:"hasAudioData,omitempty"`

	// 流式 chunk 的 ChunkIndex
	ChunkIndex int  `json:"chunkIndex,omitempty"`
	IsPause    bool `json:"isPause,omitempty"`
}

// InitRequest 子进程启动时的初始化参数（通过命令行参数或 stdin 传入）
type InitRequest struct {
	ModelDir      string `json:"modelDir"`
	ThreadCount   int    `json:"threadCount"`
	CoreMemMB     int    `json:"coreMemMB"`
	MaxNewFrames  int    `json:"maxNewFrames"`
	DoSample      *bool  `json:"doSample,omitempty"`
	SampleMode    *string `json:"sampleMode,omitempty"`
	ExecutionMode string `json:"executionMode"`
	ListenAddr    string `json:"listenAddr"` // 子进程监听地址，如 "127.0.0.1:0"
}
