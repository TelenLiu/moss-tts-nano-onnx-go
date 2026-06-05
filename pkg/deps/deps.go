package deps

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	OnnxRuntimeVersion = "1.26.0"
	DefaultHFBaseURL   = "https://huggingface.co"
	MirrorHFBaseURL    = "https://hf-mirror.com"
	OnnxRuntimeBaseURL = "https://github.com/microsoft/onnxruntime/releases/download"
)

type DownloadProgress struct {
	Phase      string  `json:"phase"`
	File       string  `json:"file"`
	BytesDone  int64   `json:"bytes_done"`
	BytesTotal int64   `json:"bytes_total"`
	Percent    float64 `json:"percent"`
	SpeedMBps  float64 `json:"speed_mbps"`
	Message    string  `json:"message"`
}

type ProgressCallback func(progress DownloadProgress)

type Config struct {
	UseMirror     bool
	LibDir        string
	ModelDir      string
	HFBaseURL     string
	TTSRepoID     string
	CodecRepoID   string
	OnProgress    ProgressCallback
}

func DefaultConfig() *Config {
	cwd, _ := os.Getwd()
	return &Config{
		LibDir:      filepath.Join(cwd, "lib"),
		ModelDir:    filepath.Join(cwd, "models"),
		HFBaseURL:   DefaultHFBaseURL,
		TTSRepoID:   "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
		CodecRepoID: "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
	}
}

func (c *Config) ApplyMirror() {
	if c.UseMirror {
		c.HFBaseURL = MirrorHFBaseURL
	}
}

func (c *Config) reportProgress(p DownloadProgress) {
	if c.OnProgress != nil {
		c.OnProgress(p)
	}
}

func formatBytes(b int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case b >= GB:
		return fmt.Sprintf("%.2f GB", float64(b)/float64(GB))
	case b >= MB:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(MB))
	case b >= KB:
		return fmt.Sprintf("%.1f KB", float64(b)/float64(KB))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func ortPlatform() string {
	switch runtime.GOOS {
	case "darwin":
		return "osx"
	case "linux":
		return "linux"
	case "windows":
		return "win"
	default:
		return runtime.GOOS
	}
}

func ortArch() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "arm64"
		}
		return "x86_64"
	case "linux":
		if runtime.GOARCH == "arm64" {
			return "aarch64"
		}
		return "x64"
	case "windows":
		if runtime.GOARCH == "arm64" {
			return "arm64"
		}
		return "x64"
	default:
		if runtime.GOARCH == "arm64" {
			return "aarch64"
		}
		return "x64"
	}
}

func ortLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "linux":
		return "libonnxruntime.so"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "libonnxruntime.so"
	}
}

func ortLibGlobPattern() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime*.dylib"
	case "linux":
		return "libonnxruntime.so*"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "libonnxruntime.so*"
	}
}

func findOrtLib(ortDir string) string {
	searchPaths := []string{
		ortDir,
		filepath.Join(ortDir, "lib"),
	}
	for _, dir := range searchPaths {
		directPath := filepath.Join(dir, ortLibName())
		if fileExists(directPath) {
			return directPath
		}
	}
	for _, dir := range searchPaths {
		matches, err := filepath.Glob(filepath.Join(dir, ortLibGlobPattern()))
		if err == nil && len(matches) > 0 {
			for _, m := range matches {
				if !strings.Contains(m, ".dSYM") && !strings.Contains(m, ".so.") {
					return m
				}
			}
			return matches[0]
		}
	}
	return filepath.Join(ortDir, ortLibName())
}

func (c *Config) ortDownloadURL() string {
	plat := ortPlatform()
	arch := ortArch()
	pkgName := fmt.Sprintf("onnxruntime-%s-%s-%s", plat, arch, OnnxRuntimeVersion)
	if runtime.GOOS == "windows" {
		return fmt.Sprintf("%s/v%s/%s.zip", OnnxRuntimeBaseURL, OnnxRuntimeVersion, pkgName)
	}
	return fmt.Sprintf("%s/v%s/%s.tgz", OnnxRuntimeBaseURL, OnnxRuntimeVersion, pkgName)
}

func isOrtLibFile(name string) bool {
	base := filepath.Base(name)
	switch {
	case strings.HasPrefix(base, "libonnxruntime") && (strings.HasSuffix(base, ".dylib") || strings.HasSuffix(base, ".so") || strings.Contains(base, ".so.")):
		return true
	case base == "onnxruntime.dll":
		return true
	}
	return false
}

// stripArchiveTopDir 规范化归档内路径：去除前导 "/", "./" 以及顶层目录。
// onnxruntime 自 1.25.1 起在 tar/zip 中使用 "./onnxruntime-osx-arm64-1.x.x/..." 形式，
// 因此单纯用 strings.Split 然后丢弃 parts[0] 会留下 ".onnxruntime-..." 这样的相对路径，
// 导致后续 lib/ include/ 匹配全部失败。
func stripArchiveTopDir(name string) string {
	normalized := strings.ReplaceAll(name, "\\", "/")
	normalized = strings.TrimLeft(normalized, "/")
	for strings.HasPrefix(normalized, "./") {
		normalized = strings.TrimPrefix(normalized, "./")
	}
	if idx := strings.Index(normalized, "/"); idx >= 0 {
		return normalized[idx+1:]
	}
	return normalized
}

func isOrtNeededFile(name string) bool {
	relativePath := stripArchiveTopDir(name)
	switch {
	case strings.HasPrefix(relativePath, "lib/") && isOrtLibFile(name):
		return true
	case strings.HasPrefix(relativePath, "include/"):
		return true
	}
	return false
}

func EnsureNativeLibs(cfg *Config) error {
	cfg.ApplyMirror()
	ortDir := filepath.Join(cfg.LibDir, "onnxruntime")
	ortTarget := findOrtLib(ortDir)

	if fileExists(ortTarget) {
		log.Printf("ONNX Runtime 本地依赖已就绪: %s", ortTarget)
		cfg.reportProgress(DownloadProgress{Phase: "download", File: "ONNX Runtime", Message: "ONNX Runtime 已就绪 (跳过下载)", Percent: 100})
		return nil
	}

	log.Printf("开始下载 ONNX Runtime v%s ...", OnnxRuntimeVersion)
	ortURL := cfg.ortDownloadURL()
	if err := downloadAndExtractOrtLibs(cfg, ortURL, ortDir); err != nil {
		return fmt.Errorf("下载 ONNX Runtime 失败: %w", err)
	}

	ortTarget = findOrtLib(ortDir)
	if !fileExists(ortTarget) {
		return fmt.Errorf("ONNX Runtime 下载完成但未找到库文件 (搜索目录: %s)", ortDir)
	}

	log.Printf("ONNX Runtime 下载完成: %s", ortTarget)
	return nil
}

func EnsureModels(cfg *Config) error {
	cfg.ApplyMirror()
	ttsDir := filepath.Join(cfg.ModelDir, "MOSS-TTS-Nano-100M-ONNX")
	codecDir := filepath.Join(cfg.ModelDir, "MOSS-Audio-Tokenizer-Nano-ONNX")

	for _, candidate := range []string{
		"browser_poc_manifest.json",
		"MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
	} {
		p := filepath.Join(cfg.ModelDir, candidate)
		if fileExists(p) {
			log.Printf("模型已就绪: %s", p)
			cfg.reportProgress(DownloadProgress{Phase: "download", File: "TTS Model", Message: "模型文件已就绪 (跳过下载)", Percent: 100})
			return nil
		}
	}

	log.Printf("开始下载模型文件 (useMirror=%v, hfBaseURL=%s)...", cfg.UseMirror, cfg.HFBaseURL)

	if err := downloadHFRepo(cfg, cfg.HFBaseURL, cfg.TTSRepoID, ttsDir, "TTS"); err != nil {
		if cfg.UseMirror {
			log.Printf("镜像源下载TTS模型失败，尝试默认源: %v", err)
			cfg2 := *cfg
			cfg2.HFBaseURL = DefaultHFBaseURL
			if err2 := downloadHFRepo(&cfg2, DefaultHFBaseURL, cfg.TTSRepoID, ttsDir, "TTS"); err2 != nil {
				return fmt.Errorf("下载 TTS 模型失败 (镜像: %v, 默认: %v)", err, err2)
			}
		} else {
			return fmt.Errorf("下载 TTS 模型失败: %w", err)
		}
	}
	if err := downloadHFRepo(cfg, cfg.HFBaseURL, cfg.CodecRepoID, codecDir, "Codec"); err != nil {
		if cfg.UseMirror {
			log.Printf("镜像源下载Codec模型失败，尝试默认源: %v", err)
			cfg2 := *cfg
			cfg2.HFBaseURL = DefaultHFBaseURL
			if err2 := downloadHFRepo(&cfg2, DefaultHFBaseURL, cfg.CodecRepoID, codecDir, "Codec"); err2 != nil {
				return fmt.Errorf("下载 Codec 模型失败 (镜像: %v, 默认: %v)", err, err2)
			}
		} else {
			return fmt.Errorf("下载 Codec 模型失败: %w", err)
		}
	}

	log.Printf("模型下载完成: %s", cfg.ModelDir)
	return nil
}

func downloadHFRepo(cfg *Config, baseURL, repoID, localDir, label string) error {
	os.MkdirAll(localDir, 0755)
	apiURL := fmt.Sprintf("%s/api/models/%s/tree/main", baseURL, repoID)
	resp, err := http.Get(apiURL)
	if err != nil {
		return fmt.Errorf("获取 HuggingFace 仓库文件列表失败: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HuggingFace API 返回 %d: %s", resp.StatusCode, string(body[:min(len(body), 200)]))
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	var files []struct {
		Type string `json:"type"`
		Path string `json:"path"`
		Size int64  `json:"size"`
	}
	if err := json.Unmarshal(body, &files); err != nil {
		return fmt.Errorf("解析 HuggingFace API 响应失败: %w", err)
	}
	allowedExts := map[string]bool{".onnx": true, ".data": true, ".json": true}
	allowedNames := map[string]bool{"tokenizer.model": true}

	var toDownload []struct {
		Path string
		Size int64
		URL  string
	}
	var totalSize int64
	for _, f := range files {
		if f.Type != "file" && f.Type != "blob" {
			continue
		}
		ext := strings.ToLower(filepath.Ext(f.Path))
		base := filepath.Base(f.Path)
		if !allowedExts[ext] && !allowedNames[base] {
			continue
		}
		targetPath := filepath.Join(localDir, f.Path)
		if fileExists(targetPath) {
			fi, _ := os.Stat(targetPath)
			if fi != nil && fi.Size() > 0 {
				continue
			}
		}
		downloadURL := fmt.Sprintf("%s/%s/resolve/main/%s", baseURL, repoID, f.Path)
		toDownload = append(toDownload, struct {
			Path string
			Size int64
			URL  string
		}{f.Path, f.Size, downloadURL})
		totalSize += f.Size
	}

	if len(toDownload) == 0 {
		cfg.reportProgress(DownloadProgress{Phase: "download", File: label, Message: fmt.Sprintf("%s 模型已就绪", label), Percent: 100})
		return nil
	}

	cfg.reportProgress(DownloadProgress{Phase: "download", File: label, Message: fmt.Sprintf("需下载 %d 个文件, 共 %s", len(toDownload), formatBytes(totalSize)), BytesTotal: totalSize})

	var totalDownloaded int64
	for i, f := range toDownload {
		targetPath := filepath.Join(localDir, f.Path)
		os.MkdirAll(filepath.Dir(targetPath), 0755)
		prefix := fmt.Sprintf("[%d/%d] %s", i+1, len(toDownload), f.Path)
		cfg.reportProgress(DownloadProgress{
			Phase:   "download",
			File:    f.Path,
			Message: fmt.Sprintf("%s 下载中...", prefix),
		})
		if err := downloadFileTracked(cfg, f.URL, targetPath, f.Size, prefix, totalDownloaded, totalSize, label); err != nil {
			log.Printf("  警告: 下载 %s 失败: %v", f.Path, err)
		}
		fi, _ := os.Stat(targetPath)
		if fi != nil {
			totalDownloaded += fi.Size()
		} else {
			totalDownloaded += f.Size
		}
	}

	cfg.reportProgress(DownloadProgress{Phase: "download", File: label, Message: fmt.Sprintf("%s 模型下载完成 (%s)", label, formatBytes(totalSize)), Percent: 100, BytesDone: totalSize, BytesTotal: totalSize})
	return nil
}

func downloadAndExtractOrtLibs(cfg *Config, url, destDir string) error {
	os.MkdirAll(destDir, 0755)
	tmpFile, err := os.CreateTemp("", "moss-tts-ort-*.download")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	log.Printf("  下载: %s", url)
	cfg.reportProgress(DownloadProgress{Phase: "download", File: "ONNX Runtime", Message: "下载 ONNX Runtime..."})
	if err := downloadFileTracked(cfg, url, tmpPath, 0, "ONNX Runtime", 0, 0, "ORT"); err != nil {
		return err
	}

	cfg.reportProgress(DownloadProgress{Phase: "download", File: "ONNX Runtime", Message: "解压 ONNX Runtime..."})

	if strings.HasSuffix(url, ".tgz") || strings.HasSuffix(url, ".tar.gz") {
		return extractOrtTgz(tmpPath, destDir)
	} else if strings.HasSuffix(url, ".zip") {
		return extractOrtZip(tmpPath, destDir)
	}
	return fmt.Errorf("不支持的下载格式: %s", url)
}

func downloadFileTracked(cfg *Config, url, dest string, expectedSize int64, prefix string, groupDone, groupTotal int64, label string) error {
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil
		},
	}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, url)
	}

	contentSize := resp.ContentLength
	if contentSize <= 0 {
		contentSize = expectedSize
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	buf := make([]byte, 32*1024)
	var written int64
	startTime := time.Now()
	lastReport := startTime
	lastWritten := int64(0)

	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			w, writeErr := out.Write(buf[:n])
			written += int64(w)
			if writeErr != nil {
				return writeErr
			}
		}
		now := time.Now()
		elapsed := now.Sub(lastReport)
		if elapsed >= 500*time.Millisecond || readErr != nil {
			var speedMBps float64
			if elapsed.Seconds() > 0 {
				chunkBytes := written - lastWritten
				speedMBps = float64(chunkBytes) / elapsed.Seconds() / (1024 * 1024)
			}
			var filePct float64
			if contentSize > 0 {
				filePct = float64(written) / float64(contentSize) * 100
			}
			msg := fmt.Sprintf("%s %s / %s (%.1f%%, %.1f MB/s)", prefix, formatBytes(written), formatBytes(contentSize), filePct, speedMBps)
			p := DownloadProgress{
				Phase:     "download",
				File:      label,
				BytesDone: groupDone + written,
				BytesTotal: groupTotal,
				Percent:   filePct,
				SpeedMBps: speedMBps,
				Message:   msg,
			}
			if groupTotal > 0 {
				p.Percent = float64(groupDone+written) / float64(groupTotal) * 100
			}
			cfg.reportProgress(p)
			lastReport = now
			lastWritten = written
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return readErr
		}
	}

	if expectedSize > 0 && written != expectedSize {
		log.Printf("  警告: 下载大小不匹配 expected=%d actual=%d", expectedSize, written)
	}
	return nil
}

func extractOrtTgz(src, dest string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if !isOrtNeededFile(hdr.Name) {
			continue
		}
		target := filepath.Join(dest, stripArchiveTopDir(hdr.Name))
		switch hdr.Typeflag {
		case tar.TypeDir:
			os.MkdirAll(target, 0755)
		case tar.TypeReg:
			os.MkdirAll(filepath.Dir(target), 0755)
			out, err := os.Create(target)
			if err != nil {
				continue
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				continue
			}
			out.Close()
		case tar.TypeSymlink:
			// 1.26.0+ 在 macOS 上额外增加了 libonnxruntime.1.dylib -> libonnxruntime.1.26.0.dylib
			// 等符号链接，需要保留以便 dlopen 找到主库。
			os.MkdirAll(filepath.Dir(target), 0755)
			_ = os.Remove(target)
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				continue
			}
		}
	}
	return nil
}

func extractOrtZip(src, dest string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, f := range r.File {
		if !isOrtNeededFile(f.Name) {
			continue
		}
		target := filepath.Join(dest, stripArchiveTopDir(f.Name))
		if f.FileInfo().IsDir() {
			os.MkdirAll(target, 0755)
			continue
		}
		os.MkdirAll(filepath.Dir(target), 0755)
		rc, err := f.Open()
		if err != nil {
			continue
		}
		out, err := os.Create(target)
		if err != nil {
			rc.Close()
			continue
		}
		io.Copy(out, rc)
		out.Close()
		rc.Close()
	}
	return nil
}

func downloadFile(url, dest string) error {
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil
		},
	}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, url)
	}
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func SetDynlibPath(libDir string) {
	ortDir := filepath.Join(libDir, "onnxruntime")
	libDirPath := filepath.Join(ortDir, "lib")
	paths := []string{ortDir, libDirPath}
	switch runtime.GOOS {
	case "darwin":
		existing := os.Getenv("DYLD_LIBRARY_PATH")
		if existing != "" {
			paths = append(paths, existing)
		}
		os.Setenv("DYLD_LIBRARY_PATH", strings.Join(paths, ":"))
	case "linux":
		existing := os.Getenv("LD_LIBRARY_PATH")
		if existing != "" {
			paths = append(paths, existing)
		}
		os.Setenv("LD_LIBRARY_PATH", strings.Join(paths, ":"))
	case "windows":
		existing := os.Getenv("PATH")
		if existing != "" {
			paths = append(paths, existing)
		}
		os.Setenv("PATH", strings.Join(paths, ";"))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
