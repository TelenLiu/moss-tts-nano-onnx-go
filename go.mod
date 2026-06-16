module github.com/TelenLiu/moss-tts-nano-onnx-go

go 1.23.2

require (
	github.com/TelenLiu/WeTextProcessing-go v0.0.2
	github.com/gunter-q12/resample v1.0.0
	github.com/sirupsen/logrus v1.9.4
	github.com/yalue/onnxruntime_go v1.31.0
	golang.org/x/net v0.43.0
)

require (
	golang.org/x/exp v0.0.0-20250106191152-7588d65b2ba8 // indirect
	golang.org/x/sys v0.35.0 // indirect
)

//replace github.com/TelenLiu/WeTextProcessing-go => ../WeTextProcessing-go
