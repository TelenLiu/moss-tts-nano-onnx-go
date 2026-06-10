FROM golang:1.24.6 AS builder

WORKDIR /app
ENV GOPROXY https://goproxy.cn
COPY go.mod go.sum ./
RUN go mod tidy
COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -o moss-tts-nano-onnx-go ./cmd/

FROM debian:12.7
ARG DEBIAN_FRONTEND=noninteractive
RUN sed -i 's@http://deb.debian.org@http://mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources && sed -i 's@http://security.debian.org@http://mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources && apt-get -qq update && apt-get -qq install -y --no-install-recommends ca-certificates curl ffmpeg libavcodec-extra
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN mkdir -p /app/conf
WORKDIR /app
COPY --from=builder /app/moss-tts-nano-onnx-go .
COPY Dockerfile .

EXPOSE 18083

ENTRYPOINT ["/app/moss-tts-nano-onnx-go"]
CMD ["serve", "--host", "0.0.0.0", "--port", "18083"]
