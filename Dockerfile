FROM golang:1.23.2-bookworm AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o moss-tts-nano-onnx-go ./cmd/

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/moss-tts-nano-onnx-go .
COPY Dockerfile .

ENV HOME=/root
EXPOSE 18083

ENTRYPOINT ["/app/moss-tts-nano-onnx-go"]
CMD ["serve", "--host", "0.0.0.0", "--port", "18083"]
