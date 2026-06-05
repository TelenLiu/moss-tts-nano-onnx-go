// Package proxy 提供网络代理配置加载与 http.Client 构造能力。
//
// 配置从 conf/proxy.json 读取（路径可通过 PROXY_CONFIG 环境变量覆盖）。
// 支持 socks5 与 http 两种代理方式，可对接 ClashX 等本地代理。
// 仅影响出向网络请求，不会修改对外提供服务的 API/Web 监听。
package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/proxy"
)

// 代理类型常量
const (
	TypeSOCKS5 = "socks5"
	TypeHTTP   = "http"
)

// 常见环境变量名
const (
	EnvHTTPProxy  = "HTTP_PROXY"
	EnvHTTPSProxy = "HTTPS_PROXY"
	EnvAllProxy   = "ALL_PROXY"
	EnvNoProxy    = "NO_PROXY"
)

// Config 表示 conf/proxy.json 的内容。
// type: "socks5" 或 "http"
// ip/port: 代理服务器地址
// username/password: 可选, 代理认证
type Config struct {
	Type     string `json:"type"`
	IP       string `json:"ip"`
	Port     string `json:"port"`
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
}

// Global 当前生效的代理配置（包级单例）。
var (
	global   *Config
	globalMu sync.RWMutex
)

// DefaultConfigPath 默认配置文件路径。
func DefaultConfigPath() string {
	cwd, _ := os.Getwd()
	return filepath.Join(cwd, "conf", "proxy.json")
}

// Load 从指定路径加载代理配置。文件不存在或字段为空时, 返回 nil（表示不使用代理）。
// 加载成功后写入包级 Global, 供其他模块查询。
func Load(path string) (*Config, error) {
	if path == "" {
		path = DefaultConfigPath()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("[Proxy] 配置文件不存在, 跳过代理设置: %s", path)
			return nil, nil
		}
		return nil, fmt.Errorf("读取代理配置失败: %w", err)
	}

	cfg := &Config{}
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("解析代理配置失败: %w", err)
	}

	cfg = normalize(cfg)
	if cfg == nil {
		log.Printf("[Proxy] 配置文件内容为空, 不启用代理: %s", path)
		return nil, nil
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	SetGlobal(cfg)
	log.Printf("[Proxy] 已加载代理配置: type=%s addr=%s:%s", cfg.Type, cfg.IP, cfg.Port)
	return cfg, nil
}

// LoadFromDefault 加载默认路径的代理配置。
func LoadFromDefault() (*Config, error) {
	return Load(DefaultConfigPath())
}

// normalize 规范化配置: 去除空白, 缺失关键字段视为不启用代理。
func normalize(cfg *Config) *Config {
	if cfg == nil {
		return nil
	}
	cfg.Type = strings.ToLower(strings.TrimSpace(cfg.Type))
	cfg.IP = strings.TrimSpace(cfg.IP)
	cfg.Port = strings.TrimSpace(cfg.Port)
	if cfg.Type == "" || cfg.IP == "" || cfg.Port == "" {
		return nil
	}
	return cfg
}

// Validate 校验配置合法性。
func (c *Config) Validate() error {
	if c.Type != TypeSOCKS5 && c.Type != TypeHTTP {
		return fmt.Errorf("不支持的代理类型: %s (仅支持 socks5/http)", c.Type)
	}
	if net.ParseIP(c.IP) == nil {
		// 允许域名形式, 这里只做非空校验
		if c.IP == "" {
			return fmt.Errorf("代理 ip 不能为空")
		}
	}
	if _, err := c.PortInt(); err != nil {
		return err
	}
	return nil
}

// PortInt 返回端口号整型。
func (c *Config) PortInt() (int, error) {
	var p int
	if _, err := fmt.Sscanf(c.Port, "%d", &p); err != nil {
		return 0, fmt.Errorf("代理端口格式错误: %s", c.Port)
	}
	if p <= 0 || p > 65535 {
		return 0, fmt.Errorf("代理端口超出范围: %s", c.Port)
	}
	return p, nil
}

// ProxyURL 返回代理 URL (socks5://user:pass@ip:port 或 http://user:pass@ip:port)。
func (c *Config) ProxyURL() (*url.URL, error) {
	scheme := c.Type
	if scheme == TypeSOCKS5 {
		scheme = "socks5"
	}
	u := &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(c.IP, c.Port),
	}
	if c.Username != "" {
		u.User = url.UserPassword(c.Username, c.Password)
	}
	return u, nil
}

// SetGlobal 设置全局代理配置。
func SetGlobal(c *Config) {
	globalMu.Lock()
	defer globalMu.Unlock()
	global = c
}

// GetGlobal 获取全局代理配置（可能为 nil）。
func GetGlobal() *Config {
	globalMu.RLock()
	defer globalMu.RUnlock()
	return global
}

// Enabled 返回是否启用了代理。
func (c *Config) Enabled() bool {
	return c != nil && c.IP != "" && c.Port != "" && c.Type != ""
}

// buildSOCKS5Dialer 构造 socks5 dialer。
func (c *Config) buildSOCKS5Dialer() (proxy.Dialer, error) {
	addr := net.JoinHostPort(c.IP, c.Port)
	var auth *proxy.Auth
	if c.Username != "" {
		auth = &proxy.Auth{User: c.Username, Password: c.Password}
	}
	d, err := proxy.SOCKS5("tcp", addr, auth, proxy.Direct)
	if err != nil {
		return nil, fmt.Errorf("构建 SOCKS5 dialer 失败: %w", err)
	}
	return d, nil
}

// NewHTTPClient 根据代理配置构造一个 *http.Client。
// 当 c 为 nil 时, 返回使用默认代理环境变量 (HTTP_PROXY/HTTPS_PROXY) 的客户端，
// 行为与直接使用 http.DefaultClient 一致。
func (c *Config) NewHTTPClient() (*http.Client, error) {
	transport := &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   30 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	if c == nil || !c.Enabled() {
		// 回退到系统代理环境变量
		transport.Proxy = http.ProxyFromEnvironment
		return &http.Client{Transport: transport, Timeout: 0}, nil
	}

	switch c.Type {
	case TypeHTTP:
		pu, err := c.ProxyURL()
		if err != nil {
			return nil, err
		}
		transport.Proxy = http.ProxyURL(pu)
	case TypeSOCKS5:
		dialer, err := c.buildSOCKS5Dialer()
		if err != nil {
			return nil, err
		}
		// 通过 context 将 socks5 dialer 注入到 transport
		transport.Proxy = nil
		transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.Dial(network, addr)
		}
	default:
		return nil, fmt.Errorf("不支持的代理类型: %s", c.Type)
	}

	return &http.Client{Transport: transport, Timeout: 0}, nil
}

// NewClientFromGlobal 基于全局代理配置构造 http.Client。
func NewClientFromGlobal() (*http.Client, error) {
	return GetGlobal().NewHTTPClient()
}

// ApplyToTransport 将代理配置应用到传入的 *http.Transport（便于复用现有 transport）。
// 如果未配置代理, 则保留 transport 原状。
func ApplyToTransport(t *http.Transport) error {
	c := GetGlobal()
	if c == nil || !c.Enabled() {
		return nil
	}
	switch c.Type {
	case TypeHTTP:
		pu, err := c.ProxyURL()
		if err != nil {
			return err
		}
		t.Proxy = http.ProxyURL(pu)
	case TypeSOCKS5:
		dialer, err := c.buildSOCKS5Dialer()
		if err != nil {
			return err
		}
		t.Proxy = nil
		t.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.Dial(network, addr)
		}
	}
	return nil
}

// ApplyToDefaultClient 将代理配置注入到 http.DefaultClient.DefaultTransport。
// 这是便捷方法, 适合无法直接拿到 *http.Client 的场景。
func ApplyToDefaultClient() error {
	transport, ok := http.DefaultTransport.(*http.Transport)
	if !ok || transport == nil {
		transport = &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   30 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		}
		http.DefaultTransport = transport
	}
	return ApplyToTransport(transport)
}
