package device

import (
	"fmt"
	"runtime"

	"github.com/TelenLiu/moss-tts-nano-onnx-go/pkg/log"
	"github.com/shirou/gopsutil/v4/cpu"
	ort "github.com/yalue/onnxruntime_go"
)

// ExecutionMode 推理执行模式
type ExecutionMode string

const (
	// ModeHybrid CPU+GPU混合推理（默认）
	ModeHybrid ExecutionMode = "hybrid"
	// ModeCPU 仅CPU推理
	ModeCPU ExecutionMode = "cpu"
	// ModeGPU 仅GPU推理
	ModeGPU ExecutionMode = "gpu"
)

// DeviceInfo 设备信息
type DeviceInfo struct {
	CPUInfo   CPUInfo        `json:"cpu"`
	GPUInfo   GPUInfo        `json:"gpu"`
	HasGPU    bool           `json:"has_gpu"`
	HasCUDA   bool           `json:"has_cuda"`
	HasCoreML bool           `json:"has_coreml"` // Apple M1/M2 GPU支持
	EpDevices []EpDeviceInfo `json:"ep_devices"`
}

// CPUInfo CPU信息
type CPUInfo struct {
	NumCores       int    `json:"num_cores"`
	NumThreads     int    `json:"num_threads"`
	AvailableCores int    `json:"available_cores"`
	CoreFreqMHz    int    `json:"core_freq_mhz"` // 单核CPU频率（MHz），0表示无法获取
	ModelName      string `json:"model_name"`    // CPU型号名称
}

// GPUInfo GPU信息
type GPUInfo struct {
	Available    bool   `json:"available"`
	Name         string `json:"name"`
	Vendor       string `json:"vendor"`
	DeviceID     int    `json:"device_id"`
	MemoryMB     int    `json:"memory_mb"`
	ComputeUnits int    `json:"compute_units"`
}

// EpDeviceInfo 执行提供程序设备信息
type EpDeviceInfo struct {
	Name   string `json:"name"`
	Vendor string `json:"vendor"`
}

// GetDeviceInfo 获取设备信息
func GetDeviceInfo() *DeviceInfo {
	info := &DeviceInfo{
		CPUInfo: CPUInfo{
			NumCores:       runtime.NumCPU(),
			NumThreads:     runtime.NumCPU(),
			AvailableCores: runtime.NumCPU(),
		},
		HasGPU:    false,
		HasCUDA:   false,
		HasCoreML: false,
	}

	// 使用 gopsutil 获取 CPU 详细信息（频率、型号等）
	if cpuInfos, err := cpu.Info(); err == nil && len(cpuInfos) > 0 {
		ci := cpuInfos[0]
		info.CPUInfo.CoreFreqMHz = int(ci.Mhz)
		info.CPUInfo.ModelName = ci.ModelName
		// gopsutil 的 Cores 字段表示每个物理 CPU 的核心数，用于更精确的物理核心数
		if ci.Cores > 0 {
			info.CPUInfo.NumCores = int(ci.Cores)
		}
		log.Debugf("[GetDeviceInfo] CPU: %s, 频率: %d MHz", ci.ModelName, int(ci.Mhz))
	} else {
		log.Debugf("[GetDeviceInfo] gopsutil cpu.Info 失败: %v", err)
	}

	// 尝试获取执行提供程序设备信息
	// 注意：GetEpDevices函数可能在某些版本的onnxruntime_go中不存在
	// 如果不存在，我们只能依赖其他方式检测GPU
	if ort.IsInitialized() {
		// 尝试创建CUDA选项来检测CUDA是否可用
		cudaOptions, err := ort.NewCUDAProviderOptions()
		if err == nil {
			cudaOptions.Destroy()
			info.HasCUDA = true
			info.HasGPU = true
			info.GPUInfo.Available = true
			info.GPUInfo.Name = "CUDA"
			info.GPUInfo.Vendor = "NVIDIA"
			log.Debugf("[GetDeviceInfo] CUDA设备可用")
		} else {
			log.Debugf("[GetDeviceInfo] CUDA不可用: %v", err)
		}

		// 尝试创建CoreML选项来检测CoreML是否可用（Apple M1/M2 GPU）
		coremlFlags := uint32(0) // 默认flags
		sessionOpts, err := ort.NewSessionOptions()
		if err == nil {
			err = sessionOpts.AppendExecutionProviderCoreML(coremlFlags)
			sessionOpts.Destroy()
			if err == nil {
				info.HasCoreML = true
				info.HasGPU = true
				info.GPUInfo.Available = true
				info.GPUInfo.Name = "CoreML"
				info.GPUInfo.Vendor = "Apple"
				log.Debugf("[GetDeviceInfo] CoreML设备可用 (Apple M1/M2 GPU)")
			} else {
				log.Debugf("[GetDeviceInfo] CoreML不可用: %v", err)
			}
		} else {
			log.Debugf("[GetDeviceInfo] 创建SessionOptions失败: %v", err)
		}
	}

	return info
}

// DetectGPU 检测GPU可用性（需要在ONNX Runtime初始化后调用）
func DetectGPU() bool {
	if !ort.IsInitialized() {
		log.Debugf("[DetectGPU] ONNX Runtime 未初始化，无法检测GPU")
		return false
	}

	// 尝试创建CUDA选项来检测CUDA是否可用
	cudaOptions, err := ort.NewCUDAProviderOptions()
	if err != nil {
		log.Debugf("[DetectGPU] 创建CUDA选项失败: %v", err)
		return false
	}
	cudaOptions.Destroy()
	log.Debugf("[DetectGPU] 检测到CUDA设备")
	return true
}

// GetAvailableModes 获取可用的推理模式
func GetAvailableModes(hasGPU bool) []ExecutionMode {
	modes := []ExecutionMode{ModeHybrid, ModeCPU}
	if hasGPU {
		modes = append(modes, ModeGPU)
	}
	return modes
}

// GetModeLabel 获取推理模式的显示标签
func GetModeLabel(mode ExecutionMode) string {
	switch mode {
	case ModeHybrid:
		return "CPU+GPU混合推理（推荐）"
	case ModeCPU:
		return "仅CPU推理"
	case ModeGPU:
		return "仅GPU推理"
	default:
		return fmt.Sprintf("未知模式: %s", mode)
	}
}

// GetRecommendedThreads 获取推荐的CPU线程数
func GetRecommendedThreads(mode ExecutionMode, cpuCores int) int {
	switch mode {
	case ModeHybrid:
		// 混合模式：使用CPU核心数-1，留一个核心给GPU调度
		if cpuCores > 1 {
			return cpuCores - 1
		}
		return 1
	case ModeCPU:
		// 仅CPU模式：使用全部CPU核心
		return cpuCores
	case ModeGPU:
		// 仅GPU模式：使用最少CPU线程（仅用于数据预处理）
		return 1
	default:
		return cpuCores
	}
}

// ValidateMode 验证推理模式是否可用
func ValidateMode(mode ExecutionMode, hasGPU bool) error {
	if mode == ModeGPU && !hasGPU {
		return fmt.Errorf("GPU推理模式不可用：未检测到CUDA设备")
	}
	return nil
}
