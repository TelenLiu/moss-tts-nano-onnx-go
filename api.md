# MOSS-TTS-Nano ONNX API 文档

## 概述

MOSS-TTS-Nano ONNX 提供基于 HTTP 的 RESTful API 和 SSE（Server-Sent Events）接口，支持语音合成、流式合成、音色克隆、演示样例等功能。

**基础地址**: `http://{host}:{port}`

---

## 目录

- [1. 系统状态](#1-系统状态)
- [2. 初始化进度](#2-初始化进度)
- [3. 设备信息](#3-设备信息)
- [4. 语音合成（非流式）](#4-语音合成非流式)
- [5. 语音合成（流式）](#5-语音合成流式)
- [6. 内置音色列表](#6-内置音色列表)
- [7. 上传参考音频](#7-上传参考音频)
- [8. 获取音频文件](#8-获取音频文件)
- [9. 演示样例列表](#9-演示样例列表)
- [10. 获取演示音频](#10-获取演示音频)
- [附录：错误码说明](#附录错误码说明)

---

## 1. 系统状态

查询系统是否已初始化就绪，可正常接受合成请求。

```
GET /api/status
```

### 响应

#### ready=true 时（系统就绪）

| 字段 | 类型 | 说明 |
|------|------|------|
| ready | bool | 固定为 `true`，表示系统就绪，可接受合成请求 |
| version | string | 引擎编译版本号（通过 `-ldflags` 注入，默认为 `"0"`） |

#### ready=false 时（初始化中或异常）

| 字段 | 类型 | 说明 |
|------|------|------|
| ready | bool | 固定为 `false`，表示系统尚未就绪 |
| version | string | 引擎编译版本号 |
| phase | string | 当前初始化阶段：`check`（环境检测）、`download`（下载阶段）、`load`（加载阶段）、`ready`（即将就绪）、`error`（初始化失败） |
| message | string | 当前状态描述信息 |
| percent | float | 总体进度百分比（0-100） |
| error | string | 错误详情，仅 `phase=error` 时存在 |
| estimated_remain_ms | int64 | 预计剩余时间（毫秒），仅加载阶段且可估算时存在 |
| action | string | 推荐操作：`wait`（等待，稍后轮询）、`retry`（需处理后重试） |
| action_hint | string | 操作建议说明，帮助第三方判断如何处理当前状态 |

### action 取值说明

| 值 | 含义 | 第三方建议处理方式 |
|------|------|------|
| `wait` | 系统正在初始化中 | 间隔一段时间（建议 3-5 秒）后再次轮询 `/api/status`，或连接 `/api/progress` (SSE) 获取实时进度 |
| `retry` | 初始化失败 | 根据 `error` 字段排查原因，解决问题后重启服务或重新请求 |

### 示例

**系统就绪**：
```json
{"ready": true, "version": "0"}
```

**初始化中（下载阶段）**：
```json
{
  "ready": false,
  "version": "0",
  "phase": "download",
  "message": "正在检查模型文件...",
  "percent": 30,
  "action": "wait",
  "action_hint": "系统正在初始化（download，已完成 30%），请等待 ready=true 后再发起合成请求。可连接 /api/progress (SSE) 获取实时进度。"
}
```

**初始化中（加载阶段，含预计剩余时间）**：
```json
{
  "ready": false,
  "version": "0",
  "phase": "load",
  "message": "正在构建文本归一化 FST 缓存...",
  "percent": 96,
  "estimated_remain_ms": 120000,
  "action": "wait",
  "action_hint": "系统正在初始化（load，已完成 96%），请等待 ready=true 后再发起合成请求。可连接 /api/progress (SSE) 获取实时进度。"
}
```

**初始化失败**：
```json
{
  "ready": false,
  "version": "0",
  "phase": "error",
  "message": "ONNX Runtime 依赖准备失败: download timeout",
  "percent": 25,
  "error": "ONNX Runtime 依赖准备失败: download timeout",
  "action": "retry",
  "action_hint": "初始化失败，请检查错误信息后重试。可尝试：1) 确认网络可访问模型/依赖下载源（或使用 -mirror 参数切换国内镜像）；2) 确认磁盘空间充足；3) 确认模型文件完整后重启服务。"
}
```

---

## 2. 初始化进度

通过 SSE（Server-Sent Events）实时获取系统初始化进度，包括依赖下载、模型加载等阶段。系统就绪后连接自动关闭。

```
GET /api/progress
```

### 响应格式

SSE 事件流，每个事件格式为 `data: {JSON}\n\n`，30 秒无事件时发送 keepalive 注释行 `: keepalive\n\n`。

### 事件数据结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| phase | string | 是 | 当前阶段：`check`（环境检测）、`download`（下载阶段）、`load`（加载阶段）、`ready`（就绪）、`error`（错误） |
| message | string | 是 | 进度描述信息 |
| percent | float | 是 | 总体进度百分比（0-100） |
| error | string | 否 | 错误信息，仅 `phase=error` 时存在 |
| done | bool | 是 | 是否完成，仅 `phase=ready` 时为 `true` |
| file | string | 否 | 当前下载的文件名，仅 `phase=download` |
| bytes_done | int64 | 否 | 已下载字节数，仅 `phase=download` |
| bytes_total | int64 | 否 | 总字节数，仅 `phase=download` |
| speed_mbps | float | 否 | 下载速度（MB/s），仅 `phase=download` |
| elapsed_ms | int64 | 否 | 已用时间（毫秒），仅 `phase=load` |
| estimated_remain_ms | int64 | 否 | 预计剩余时间（毫秒），仅 `phase=load` |

### 示例

```
data: {"phase":"download","message":"正在检查 ONNX Runtime 本地依赖...","percent":10,"done":false}

data: {"phase":"download","message":"ONNX Runtime 依赖已就绪","percent":25,"done":false}

data: {"phase":"load","message":"TTS 模型加载完成","percent":95,"done":false}

data: {"phase":"ready","message":"系统就绪","percent":100,"done":true}
```

---

## 3. 设备信息

获取当前运行环境的 CPU、GPU 信息及推理单元池状态。

```
GET /api/device-info
```

### 响应

| 字段 | 类型 | 说明 |
|------|------|------|
| cpu | object | CPU 信息 |
| cpu.num_cores | int | CPU 物理核心数 |
| cpu.num_threads | int | CPU 逻辑线程数 |
| cpu.available_cores | int | 可用核心数 |
| cpu.core_freq_mhz | int | 单核 CPU 频率（MHz），0 表示无法获取 |
| cpu.model_name | string | CPU 型号名称 |
| has_gpu | bool | 是否有 GPU 可用 |
| has_cuda | bool | 是否支持 CUDA |
| has_coreml | bool | 是否支持 CoreML（macOS） |
| execution_mode | string | 当前推理模式：`hybrid`（混合）、`cpu`（仅CPU）、`gpu`（仅GPU） |
| available_modes | string[] | 可用的推理模式列表 |
| gpu | object | GPU 信息（仅 `has_gpu=true` 时存在） |
| gpu.available | bool | GPU 是否可用 |
| gpu.name | string | GPU 名称 |
| gpu.vendor | string | GPU 厂商 |
| gpu.device_id | string | GPU 设备 ID |
| gpu.memory_mb | int | GPU 显存大小（MB） |
| gpu.compute_units | int | GPU 计算单元数 |
| ep_devices | object[] | 执行提供程序设备信息 |
| ep_devices[].name | string | 设备名称 |
| ep_devices[].vendor | string | 设备厂商 |
| onnx_pool | object | 推理单元池状态（系统就绪后存在） |
| onnx_pool.work_cores | int | 常驻推理核心数 |
| onnx_pool.reserve_cores | int | 预留推理核心数 |
| onnx_pool.core_cpus | int | 每个推理单元使用的 CPU 核心数 |
| onnx_pool.pending_requests | int | 等待中的请求数 |
| onnx_pool.cores | object[] | 各核心状态详情 |

### 示例

```json
{
  "cpu": {"num_cores": 8, "num_threads": 8, "available_cores": 8, "core_freq_mhz": 3200, "model_name": "Apple M1"},
  "has_gpu": true,
  "has_cuda": false,
  "has_coreml": true,
  "execution_mode": "hybrid",
  "available_modes": ["hybrid", "cpu"],
  "gpu": {
    "available": true,
    "name": "Apple M1",
    "vendor": "Apple",
    "device_id": "0",
    "memory_mb": 8192,
    "compute_units": 8
  },
  "onnx_pool": {
    "work_cores": 1,
    "reserve_cores": 0,
    "core_cpus": 4,
    "pending_requests": 0,
    "cores": [{"name": "workCore1", "type": "work", "active_reqs": 0}]
  }
}
```

---

## 4. 语音合成（非流式）

提交文本合成语音，等待全部合成完成后返回完整的音频数据（Base64 编码）。

```
POST /api/synthesize
Content-Type: application/json
```

### 请求参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | **是** | - | 要合成的文本内容 |
| voice | string | 否 | `"Junhao"` | 内置音色名称，可通过 `/api/voices` 获取可用列表 |
| demo_id | string | 否 | `""` | 演示样例 ID，指定后自动使用该样例的参考音频。可通过 `/api/demos` 获取列表 |
| preload_id | string | 否 | `""` | 预加载缓存 ID，用于复用已加载的参考音频特征，减少重复计算 |
| prompt_audio_path | string | 否 | `""` | 参考音频的服务器本地路径（服务端路径，非客户端路径） |
| prompt_audio_b64 | string | 否 | `""` | Base64 编码的参考音频数据，支持 WAV/MP3/FLAC/M4A/OGG/OPUS/AAC 格式。优先级高于 `demo_id`，相同内容会命中 audio_clone_gob 缓存 |
| sample_mode | string | 否 | `"fixed"` | 采样模式，见下方说明 |
| max_new_frames | int | 否 | 动态计算 | 最大生成帧数。默认按 `文本长度×4+100` 动态计算，上限 2000 |
| voice_clone_max_text_tokens | int | 否 | `75` | 音色克隆时的最大文本 Token 数，影响克隆质量，与 Python 源码保持一致 |
| seed | int | 否 | `null`（随机） | 随机种子。`null` 或不传表示随机种子，传入整数则使用固定种子以保证可复现性 |
| stream | bool | 否 | `false` | 是否使用流式输出。非流式请求中设为 `true` 时将走流式逻辑 |
| enable_robust | bool | 否 | `true` | 是否启用鲁棒性文本归一化。处理标点、空格、括号、重复符号等异常输入 |
| enable_wetext | bool | 否 | `false` | 是否启用 WeTextProcessing 语义级文本归一化。处理数字/日期/金额等语义展开。首次调用耗时约 17 秒 |
| format | string | 否 | `"mp3"` | 输出音频格式：`"mp3"` 或 `"wav"` |
| mp3_sample_rate | int | 否 | 默认配置 | MP3 编码采样率（Hz），仅 `format=mp3` 时有效 |
| mp3_vbr_quality | float | 否 | 默认配置 | MP3 VBR 质量参数，仅 `format=mp3` 时有效 |
| text_temperature | float | 否 | `null`（使用内置值） | 文本生成温度，覆盖 ONNX 模型内置常数。仅 `sample_mode=full` 时生效 |
| text_top_k | int | 否 | `null`（使用内置值） | 文本生成 Top-K 采样参数。仅 `sample_mode=full` 时生效 |
| text_top_p | float | 否 | `null`（使用内置值） | 文本生成 Top-P 采样参数。仅 `sample_mode=full` 时生效 |
| audio_temperature | float | 否 | `null`（使用内置值） | 音频生成温度，覆盖 ONNX 模型内置常数。仅 `sample_mode=full` 时生效 |
| audio_top_k | int | 否 | `null`（使用内置值） | 音频生成 Top-K 采样参数。仅 `sample_mode=full` 时生效 |
| audio_top_p | float | 否 | `null`（使用内置值） | 音频生成 Top-P 采样参数。仅 `sample_mode=full` 时生效 |
| audio_repetition_penalty | float | 否 | `null`（使用内置值） | 音频重复惩罚系数。仅 `sample_mode=full` 时生效 |

#### 采样模式说明

| 值 | 说明 |
|------|------|
| `fixed` | 使用 ONNX 模型内置的固定采样常数，稳定可复现 |
| `full` | 使用请求中提供的采样参数（temperature/top_k/top_p 等），灵活控制生成多样性 |
| `greedy` | 贪心解码，不使用采样，每次选择概率最高的 Token，确定性最强 |

### 参考音频优先级

当需要指定参考音频（用于音色克隆）时，按以下优先级选择：

1. **`prompt_audio_b64`** — Base64 编码的参考音频数据（最高优先级）
2. **`demo_id`** — 演示样例的参考音频
3. **`prompt_audio_path`** — 服务器本地路径

### 响应

| 字段 | 类型 | 说明 |
|------|------|------|
| audio_path | string | 音频文件路径（当前始终为空） |
| audio_data_b64 | string | Base64 编码的音频数据（MP3 或 WAV） |
| sample_rate | int | 音频采样率（Hz） |
| audio_samples | int | 音频采样点数 |
| elapsed_seconds | float | 合成耗时（秒） |
| voice | string | 实际使用的音色名称 |
| text_chunks | string[] | 文本分块信息 |
| sample_mode | string | 实际使用的采样模式 |
| do_sample | bool | 是否使用了采样 |
| format | string | 实际输出的音频格式（`mp3` 或 `wav`）。若请求 MP3 但编码失败，会自动回退为 `wav` |
| seed_used | int64 | 实际使用的随机种子 |

### 示例

**请求**：
```json
{
  "text": "你好，欢迎使用MOSS语音合成系统。",
  "voice": "Junhao",
  "format": "mp3"
}
```

**响应**：
```json
{
  "audio_path": "",
  "audio_data_b64": "//uQxAAAAAANIAAAAAExBTUUzLjEwMFVV...",
  "sample_rate": 24000,
  "audio_samples": 48000,
  "elapsed_seconds": 1.23,
  "voice": "Junhao",
  "text_chunks": ["你好，", "欢迎使用", "MOSS语音合成系统。"],
  "sample_mode": "fixed",
  "do_sample": true,
  "format": "mp3",
  "seed_used": 123456789
}
```

### 错误响应

| HTTP 状态码 | 说明 |
|-------------|------|
| 400 | 请求参数错误（如 `text` 为空） |
| 500 | 合成失败 |
| 503 | 系统尚未就绪 |

---

## 5. 语音合成（流式）

流式合成语音，边合成边返回音频数据，实现低延迟播放。

与非流式合成使用相同的 `POST /api/synthesize` 接口，设置 `stream=true` 即可。

```
POST /api/synthesize
Content-Type: application/json

{"text": "你好", "stream": true, "format": "wav"}
```

### 流式输出行为

根据 `format` 参数不同，流式输出行为不同：

#### format=wav（PCM 流式）

- 实时流式输出 PCM 数据，低延迟
- Content-Type: `application/octet-stream`
- Transfer-Encoding: `chunked`
- 每个 chunk 为 `pcm_f32le` 格式（32 位浮点小端序）的原始 PCM 数据

**响应头**：

| 头部 | 说明 |
|------|------|
| X-Audio-Codec | 音频编码格式，固定为 `pcm_f32le` |
| X-Audio-Sample-Rate | 采样率（Hz） |
| X-Audio-Channels | 声道数 |
| X-Seed-Used | 实际使用的随机种子 |

**播放示例**（JavaScript）：
```javascript
const ctx = new AudioContext({ sampleRate: 24000 });
const response = await fetch('/api/synthesize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: '你好', stream: true, format: 'wav'})
});
const reader = response.body.getReader();
// 读取 PCM float32 数据并实时播放
```

#### format=mp3（MP3 缓冲式）

- 等待全部合成完成后，一次性返回 MP3 编码数据
- Content-Type: `audio/mpeg`

**响应头**：

| 头部 | 说明 |
|------|------|
| Content-Length | MP3 数据总字节数 |
| X-Audio-Sample-Rate | 采样率（Hz） |
| X-Audio-Channels | 声道数 |
| X-Audio-Samples | 总采样点数 |
| X-Elapsed-Seconds | 合成耗时（秒） |
| X-Audio-Format | 固定为 `mp3` |
| X-Seed-Used | 实际使用的随机种子 |

---

## 6. 内置音色列表

获取所有可用的内置音色。

```
GET /api/voices
```

### 响应

返回音色信息数组。

| 字段 | 类型 | 说明 |
|------|------|------|
| voice | string | 音色标识符，用于合成请求的 `voice` 参数 |
| label | string | 音色显示名称 |

### 示例

```json
[
  {"voice": "Junhao", "label": "Junhao"},
  {"voice": "Xiaobei", "label": "Xiaobei"}
]
```

---

## 7. 获取音频文件

通过文件名获取系统临时目录中的音频文件。

```
GET /api/audio/{filename}
```

### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| filename | string | 音频文件名（位于系统临时目录中） |

### 响应

- Content-Type 根据文件扩展名自动设置（`.mp3` → `audio/mpeg`，其他 → `audio/wav`）
- 直接返回音频文件内容

### 错误响应

| HTTP 状态码 | 说明 |
|-------------|------|
| 404 | 音频文件不存在 |
| 500 | 访问文件出错 |

---

## 8. 演示样例列表

获取所有预置的演示样例，每个样例包含预设文本和参考音频。

```
GET /api/demos
```

### 响应

返回演示样例数组。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 样例唯一标识符，格式为 `demo-{N}` |
| name | string | 样例名称 |
| text | string | 预设合成文本 |
| preloadId | string | 预加载缓存 ID（可选）。如有值，表示该样例的参考音频特征已预加载到缓存中，使用时可跳过特征提取步骤 |

### 示例

```json
[
  {"id": "demo-1", "name": "新闻播报", "text": "今天天气晴朗，适合出行。", "preloadId": "news_anchor"},
  {"id": "demo-2", "name": "故事讲述", "text": "从前有一座山，山里有座庙。"}
]
```

---

## 9. 获取演示音频

获取指定演示样例的参考音频文件。

```
GET /api/demo-prompt-audio/{demo_id}
```

### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| demo_id | string | 演示样例 ID，可通过 `/api/demos` 获取 |

### 响应

- Content-Type 根据文件扩展名设置（`.wav` → `audio/wav`，其他 → `application/octet-stream`）
- 禁用缓存（Cache-Control: no-cache）
- 直接返回音频文件内容

### 错误响应

| HTTP 状态码 | 说明 |
|-------------|------|
| 400 | `demo_id` 为空 |
| 404 | 指定的演示样例不存在 |

---

## 附录：错误码说明

| HTTP 状态码 | 含义 | 触发场景 |
|-------------|------|----------|
| 200 | 成功 | 请求成功处理 |
| 400 | 请求错误 | 参数缺失、格式错误、Base64 解码失败等 |
| 404 | 未找到 | 音频文件或演示样例不存在 |
| 405 | 方法不允许 | 使用了错误的 HTTP 方法（如 GET 代替 POST） |
| 500 | 服务器内部错误 | 合成失败、编码失败、文件写入失败等 |
| 503 | 服务不可用 | 系统尚未初始化就绪，无法接受合成请求 |

---

## 附录：典型调用流程

### 基础语音合成

```
1. GET  /api/status          → 确认系统就绪
2. GET  /api/voices          → 获取可用音色列表
3. POST /api/synthesize      → 提交合成请求，获取 Base64 音频数据
```

### 音色克隆

```
1. GET  /api/status                       → 确认系统就绪
2. POST /api/synthesize                   → 使用 prompt_audio_b64 参数直接传入 Base64 参考音频数据
```

### 使用演示样例

```
1. GET  /api/status                       → 确认系统就绪
2. GET  /api/demos                        → 获取演示样例列表
3. GET  /api/demo-prompt-audio/{demo_id}  → 获取参考音频（可选，用于预览）
4. POST /api/synthesize                   → 使用 demo_id 参数提交合成请求
```

### 流式低延迟播放

```
1. GET  /api/status          → 确认系统就绪
2. POST /api/synthesize      → 设置 stream=true, format="wav"
3. 实时读取 PCM 流并播放
```
