package web

type ProgressEvent struct {
	Phase             string  `json:"phase"`
	Message           string  `json:"message"`
	Percent           float64 `json:"percent"`
	Error             string  `json:"error,omitempty"`
	Done              bool    `json:"done"`
	File              string  `json:"file,omitempty"`
	BytesDone         int64   `json:"bytes_done,omitempty"`
	BytesTotal        int64   `json:"bytes_total,omitempty"`
	SpeedMBps         float64 `json:"speed_mbps,omitempty"`
	ElapsedMs         int64   `json:"elapsed_ms,omitempty"`
	EstimatedRemainMs int64   `json:"estimated_remain_ms,omitempty"`
}

type DemoEntry struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Role      string `json:"role"`
	Text      string `json:"text"`
	Path      string `json:"-"`
	PreloadID string `json:"preloadId,omitempty"`
}

type SynthesizeRequest struct {
	Text                    string   `json:"text"`
	Voice                   string   `json:"voice"`
	DemoID                  string   `json:"demo_id"`
	PreloadID               string   `json:"preload_id"`
	PromptAudioPath         string   `json:"prompt_audio_path"`
	UploadedPromptAudio     string   `json:"uploaded_prompt_audio"`
	SampleMode              string   `json:"sample_mode"`
	MaxNewFrames            int      `json:"max_new_frames"`
	VoiceCloneMaxTextTokens int      `json:"voice_clone_max_text_tokens"`
	Seed                    *int     `json:"seed"`
	Stream                  bool     `json:"stream"`
	EnableRobust            *bool    `json:"enable_robust"`
	EnableWeText            *bool    `json:"enable_wetext"`
	Format                  string   `json:"format"`
	MP3SampleRate           int      `json:"mp3_sample_rate"`
	MP3VBRQuality           float64  `json:"mp3_vbr_quality"`
	TextTemperature         *float64 `json:"text_temperature"`
	TextTopK                *int     `json:"text_top_k"`
	TextTopP                *float64 `json:"text_top_p"`
	AudioTemperature        *float64 `json:"audio_temperature"`
	AudioTopK               *int     `json:"audio_top_k"`
	AudioTopP               *float64 `json:"audio_top_p"`
	AudioRepetitionPenalty  *float64 `json:"audio_repetition_penalty"`
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
	Format         string   `json:"format"`
	SeedUsed       int64    `json:"seed_used"`
}
