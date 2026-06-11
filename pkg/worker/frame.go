package worker

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
)

// Frame 格式: [4字节 header长度][JSON header][attachment bytes]
// header 中可包含 HasAudioData / HasPromptAudioCodes 等标记

const frameHeaderLen = 4

// WriteFrame 写一帧数据到连接
func WriteFrame(conn net.Conn, header interface{}, attachment []byte) error {
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("序列化header失败: %w", err)
	}

	// 写 header 长度
	if err := binary.Write(conn, binary.BigEndian, uint32(len(headerJSON))); err != nil {
		return fmt.Errorf("写header长度失败: %w", err)
	}

	// 写 header
	if _, err := conn.Write(headerJSON); err != nil {
		return fmt.Errorf("写header失败: %w", err)
	}

	// 写 attachment
	if len(attachment) > 0 {
		if _, err := conn.Write(attachment); err != nil {
			return fmt.Errorf("写attachment失败: %w", err)
		}
	}

	return nil
}

// ReadFrame 从连接读一帧数据
func ReadFrame(conn net.Conn) (headerJSON []byte, attachment []byte, err error) {
	// 读 header 长度
	var headerLen uint32
	if err := binary.Read(conn, binary.BigEndian, &headerLen); err != nil {
		return nil, nil, fmt.Errorf("读header长度失败: %w", err)
	}

	if headerLen > 10*1024*1024 { // 10MB 上限
		return nil, nil, fmt.Errorf("header过长: %d", headerLen)
	}

	// 读 header
	headerJSON = make([]byte, headerLen)
	if _, err := io.ReadFull(conn, headerJSON); err != nil {
		return nil, nil, fmt.Errorf("读header失败: %w", err)
	}

	return headerJSON, nil, nil
}

// ReadRequest 从连接读取请求
func ReadRequest(conn net.Conn) (*Request, []byte, error) {
	headerJSON, _, err := ReadFrame(conn)
	if err != nil {
		return nil, nil, err
	}

	var req Request
	if err := json.Unmarshal(headerJSON, &req); err != nil {
		return nil, nil, fmt.Errorf("解析请求失败: %w", err)
	}

	// TODO: 如果 HasPromptAudioCodes，读取 attachment
	return &req, nil, nil
}

// WriteRequest 写请求到连接
func WriteRequest(conn net.Conn, req *Request, attachment []byte) error {
	return WriteFrame(conn, req, attachment)
}

// ReadResponse 从连接读取响应
func ReadResponse(conn net.Conn) (*Response, []byte, error) {
	headerJSON, _, err := ReadFrame(conn)
	if err != nil {
		return nil, nil, err
	}

	var resp Response
	if err := json.Unmarshal(headerJSON, &resp); err != nil {
		return nil, nil, fmt.Errorf("解析响应失败: %w", err)
	}

	// 如果有音频数据，读取 attachment
	var attachment []byte
	if resp.HasAudioData {
		// 音频数据长度 = AudioSamples * Channels * 4 (float32)
		audioBytes := resp.AudioSamples * resp.Channels * 4
		if audioBytes > 0 {
			attachment = make([]byte, audioBytes)
			if _, err := io.ReadFull(conn, attachment); err != nil {
				return nil, nil, fmt.Errorf("读音频attachment失败: %w", err)
			}
		}
	}

	return &resp, attachment, nil
}

// WriteResponse 写响应到连接
func WriteResponse(conn net.Conn, resp *Response, attachment []byte) error {
	return WriteFrame(conn, resp, attachment)
}
