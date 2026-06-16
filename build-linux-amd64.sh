#!/bin/bash
set -euo pipefail

# 使用 Docker 构建 Linux amd64 二进制文件
# 用法: ./build-linux-amd64.sh [镜像源]
#   镜像源: 可选参数，默认 https://goproxy.cn，可设为 direct 使用官方源

PROXY="${1:-https://goproxy.cn}"
BINARY="moss-tts-nano-onnx-go"
OUTPUT_DIR="./build"

echo "=== 构建 Linux amd64 二进制 ==="
echo "Go Proxy: ${PROXY}"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 使用 golang 官方镜像编译（强制 linux/amd64 平台，确保 GCC 支持 -m64）
docker run --rm \
    --platform=linux/amd64 \
    -v "$(pwd):/app" \
    -w /app \
    -e GOPROXY="${PROXY}" \
    -e GOOS=linux \
    -e GOARCH=amd64 \
    -e CGO_ENABLED=1 \
    -e TZ=Asia/Shanghai \
    golang:1.24.6 \
    sh -c "
        go mod tidy && \
        version=\$(date +'%y.%-m.%-d%H') && \
        echo \"版本: \${version}\" && \
        go build -ldflags=\"-X main.Version=\${version}\" -o /app/${OUTPUT_DIR}/${BINARY} ./cmd/ && \
        echo '=== 编译完成 ==='
    "

echo "=== 二进制文件位置 ==="
ls -lh "${OUTPUT_DIR}/${BINARY}"
echo "=== 完成 ==="
