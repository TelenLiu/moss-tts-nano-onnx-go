

### Build

```sh
sudo docker buildx build --build-arg version=$(date +'%y.%m.%d%H') --platform linux/amd64,linux/arm64 -f Dockerfile_update . -t telenliu/moss-tts-nano-onnx-go:v$(date +'%y.%m.%d%H') --push
```

```sh
sudo docker buildx build --build-arg version=$(date +'%y.%m.%d%H') --platform linux/amd64,linux/arm64 . -t telenliu/moss-tts-nano-onnx-go:v$(date +'%y.%m.%d%H') --push
```



### RUN

```sh
docker run -d \
  --name=moss_tts_nano_onnx_go \
  --restart=unless-stopped \
  -p 18083:18083 \
  -e TZ="Asia/Shanghai" \
  -m 6g \
  --memory-swap=16g \
  -v ./assets:/app/assets \
  -v ./conf:/app/conf \
  -v ./models:/app/models \
  -v ./lib/cache:/app/lib/cache \
  -v ./lib/onnxruntime-linux:/app/lib/onnxruntime \
  --log-driver json-file \
  --log-opt max-size=100m \
  --log-opt max-file=7 \
  telenliu/moss-tts-nano-onnx-go:v26.06.1013
```

```yaml
services:
  moss_tts_nano_onnx_go:
    image: telenliu/moss-tts-nano-onnx-go:v26.06.1013
    container_name: moss_tts_nano_onnx_go
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ports:
      - "18083:18083"
    environment:
      - TZ=Asia/Shanghai
    deploy: #集群
      resources:
        limits:
          cpus: '24'
          memory: 6G 
        reservations:
          cpus: '1'
          memory: 1G
    volumes:
      - /data/moss_tts_nano_onnx_go/app/assets:/app/assets
      - /data/moss_tts_nano_onnx_go/app/conf:/app/conf
      - /data/moss_tts_nano_onnx_go/app/models:/app/models
      - /data/moss_tts_nano_onnx_go/app/lib/cache:/app/lib/cache
      - /data/moss_tts_nano_onnx_go/app/lib/onnxruntime:/app/lib/onnxruntime
    mem_limit: 6g  #单机优先
    mem_reservation: 1g
    memswap_limit: 16g
    logging:
      driver: "json-file" 
      options:
        max-size: "100m" #单文件最大 100MB
        max-file: "7"  # 最多保存7个文件
```

