package tokenizer

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	PieceNormal     = 1
	PieceUnknown    = 2
	PieceControl    = 3
	PieceUserDefined = 4
	PieceUnused     = 5
	PieceByte       = 6
)

type Piece struct {
	Text  string
	Score float32
	Type  int
}

type Processor struct {
	Pieces    []Piece
	PieceToID map[string]int
	Scores    []float32
	Types     []int
	UnkID     int
	BosID     int
	EosID     int
	PadID     int
	VocabSize int
}

func NewProcessor(modelPath string) (*Processor, error) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("读取 SentencePiece 模型文件失败: %w", err)
	}
	return parseModel(data)
}

func parseModel(data []byte) (*Processor, error) {
	p := &Processor{
		PieceToID: make(map[string]int),
		UnkID:     0,
		BosID:     -1,
		EosID:     -1,
		PadID:     -1,
	}
	sr := &sliceReader{data: data}
	for !sr.done() {
		fieldNum, wireType, err := sr.readTag()
		if err != nil {
			break
		}
		switch fieldNum {
		case 1:
			// Field 1 could be either trainer_spec or pieces depending on model format
			val, err := sr.readBytes(wireType)
			if err != nil {
				continue
			}
			// Try to parse as SentencePiece first (newer model format)
			// If it looks like a piece (has piece text field), treat it as piece
			piece, isPiece := tryParseSentencePiece(val)
			if isPiece {
				p.Pieces = append(p.Pieces, piece)
			} else {
				// Otherwise treat as trainer_spec (standard format)
				parseTrainerSpec(val, p)
			}
		case 5, 4:
			val, err := sr.readBytes(wireType)
			if err != nil {
				continue
			}
			piece := parseSentencePiece(val)
			p.Pieces = append(p.Pieces, piece)
		default:
			sr.skipField(wireType)
		}
	}
	p.VocabSize = len(p.Pieces)
	p.Scores = make([]float32, p.VocabSize)
	p.Types = make([]int, p.VocabSize)
	for i, piece := range p.Pieces {
		p.PieceToID[piece.Text] = i
		p.Scores[i] = piece.Score
		p.Types[i] = piece.Type
		switch piece.Type {
		case PieceUnknown:
			if p.UnkID == 0 || p.UnkID > i {
				p.UnkID = i
			}
		case PieceControl:
			switch piece.Text {
			case "<s>":
				p.BosID = i
			case "</s>":
				p.EosID = i
			case "<pad>":
				p.PadID = i
			}
		}
	}
	return p, nil
}

func tryParseSentencePiece(data []byte) (Piece, bool) {
	sr := &sliceReader{data: data}
	p := Piece{Type: PieceNormal}
	hasPiece := false
	for !sr.done() {
		fieldNum, wireType, err := sr.readTag()
		if err != nil {
			break
		}
		switch fieldNum {
		case 1:
			val, err := sr.readString(wireType)
			if err == nil {
				p.Text = val
				hasPiece = true
			}
		case 2:
			val, err := sr.readFloat32(wireType)
			if err == nil {
				p.Score = val
			}
		case 3:
			val, err := sr.readVarint(wireType)
			if err == nil {
				p.Type = int(val)
			}
		default:
			sr.skipField(wireType)
		}
	}
	return p, hasPiece
}

func parseTrainerSpec(data []byte, p *Processor) {
	sr := &sliceReader{data: data}
	for !sr.done() {
		fieldNum, wireType, err := sr.readTag()
		if err != nil {
			break
		}
		switch fieldNum {
		case 9:
			val, err := sr.readVarint(wireType)
			if err == nil {
				p.UnkID = int(val)
			}
		case 7:
			val, err := sr.readVarint(wireType)
			if err == nil {
				p.BosID = int(val)
			}
		case 8:
			val, err := sr.readVarint(wireType)
			if err == nil {
				p.EosID = int(val)
			}
		case 17:
			val, err := sr.readVarint(wireType)
			if err == nil {
				p.PadID = int(val)
			}
		default:
			sr.skipField(wireType)
		}
	}
}

func parseSentencePiece(data []byte) Piece {
	p := Piece{Type: PieceNormal}
	sr := &sliceReader{data: data}
	for !sr.done() {
		fieldNum, wireType, err := sr.readTag()
		if err != nil {
			break
		}
		switch fieldNum {
		case 1:
			val, err := sr.readString(wireType)
			if err == nil {
				p.Text = val
			}
		case 2:
			val, err := sr.readFloat32(wireType)
			if err == nil {
				p.Score = val
			}
		case 3:
			val, err := sr.readVarint(wireType)
			if err == nil {
				p.Type = int(val)
			}
		default:
			sr.skipField(wireType)
		}
	}
	return p
}

func (p *Processor) Encode(text string) []int {
	if text == "" {
		return nil
	}
	text = normalizeNFKC(text)
	// SentencePiece 在编码前会给文本前导添加一个空格，归一化后即元符号 ▁（U+2581），
	// 因此非空文本的首个 token 通常是 ▁。这里补齐该前导符号，以与 Python sentencepiece
	// 的 EncodeAsIds 行为对齐，避免 token 序列整体偏移。
	normalized := "\xe2\x96\x81" + strings.ReplaceAll(text, " ", "\xe2\x96\x81")
	runes := []rune(normalized)
	n := len(runes)

	// SentencePiece unigram 模型使用 Viterbi 动态规划求整条路径分数和最大的切分，
	// 而不是在每个位置贪心取单个 piece 的最高分。dp[i] 表示覆盖 runes[:i] 的最优路径。
	negInf := float32(math.Inf(-1))
	dp := make([]float32, n+1)
	back := make([]int, n+1)
	pieceID := make([]int, n+1)
	for i := 1; i <= n; i++ {
		dp[i] = negInf
		back[i] = -1
		pieceID[i] = -1
	}
	dp[0] = 0

	for i := 0; i < n; i++ {
		if dp[i] == negInf {
			continue
		}
		matched := false
		maxLen := i + 64
		if maxLen > n {
			maxLen = n
		}
		for end := i + 1; end <= maxLen; end++ {
			candidate := string(runes[i:end])
			if id, ok := p.PieceToID[candidate]; ok {
				t := p.Types[id]
				if t == PieceNormal || t == PieceUserDefined || t == PieceByte {
					score := dp[i] + p.Scores[id]
					if score > dp[end] {
						dp[end] = score
						back[end] = i
						pieceID[end] = id
					}
					matched = true
				}
			}
		}
		if !matched {
			// 单字符回退：ASCII 字节用 <0xXX> piece，其余用 UNK。
			ch := runes[i]
			var id int
			if ch < 0x100 {
				hex := fmt.Sprintf("<0x%02X>", ch)
				if hid, ok := p.PieceToID[hex]; ok {
					id = hid
				} else {
					id = p.UnkID
				}
			} else {
				id = p.UnkID
			}
			score := dp[i]
			if score > dp[i+1] {
				dp[i+1] = score
				back[i+1] = i
				pieceID[i+1] = id
			}
		}
	}

	// 回溯还原 token 顺序
	var rev []int
	for pos := n; pos > 0; pos = back[pos] {
		if pieceID[pos] < 0 {
			break
		}
		rev = append(rev, pieceID[pos])
	}
	tokens := make([]int, len(rev))
	for i, id := range rev {
		tokens[len(rev)-1-i] = id
	}
	return tokens
}

func (p *Processor) EncodeAsIDs(text string) []int {
	return p.Encode(text)
}

func (p *Processor) Decode(tokenIDs []int) string {
	var sb strings.Builder
	for _, id := range tokenIDs {
		if id < 0 || id >= p.VocabSize {
			continue
		}
		piece := p.Pieces[id].Text
		if p.Types[id] == PieceControl {
			continue
		}
		if p.Types[id] == PieceByte {
			if len(piece) >= 5 && piece[:4] == "<0x" && piece[len(piece)-1] == '>' {
				var b byte
				fmt.Sscanf(piece[3:len(piece)-1], "%02X", &b)
				sb.WriteByte(b)
			}
			continue
		}
		sb.WriteString(piece)
	}
	result := sb.String()
	result = strings.ReplaceAll(result, "\xe2\x96\x81", " ")
	return strings.TrimSpace(result)
}

func (p *Processor) CountTokens(text string) int {
	return len(p.Encode(text))
}

func (p *Processor) GetPieceID(piece string) int {
	if id, ok := p.PieceToID[piece]; ok {
		return id
	}
	return p.UnkID
}

func (p *Processor) VocabSize_() int {
	return p.VocabSize
}

// normalizeNFKC normalizes fullwidth characters to their ASCII equivalents,
// matching SentencePiece's internal NFKC normalization behavior.
// SentencePiece normalizes the fullwidth block (U+FF01-U+FF5E) to ASCII,
// but keeps smart quotes (U+201C/U+201D) and dashes (U+2013/U+2014) as-is.
func normalizeNFKC(text string) string {
	runes := []rune(text)
	result := make([]rune, 0, len(runes))
	for _, r := range runes {
		if r >= 0xFF01 && r <= 0xFF5E {
			result = append(result, rune(r-0xFEE0))
		} else {
			result = append(result, r)
		}
	}
	return string(result)
}

type sliceReader struct {
	data []byte
	pos  int
}

func (r *sliceReader) done() bool {
	return r.pos >= len(r.data)
}

func (r *sliceReader) readTag() (fieldNum, wireType int, err error) {
	v, err := r.readVarintRaw()
	if err != nil {
		return 0, 0, err
	}
	wireType = int(v & 0x07)
	fieldNum = int(v >> 3)
	return
}

func (r *sliceReader) readVarint(wireType int) (uint64, error) {
	if wireType != 0 {
		return 0, fmt.Errorf("expected varint wire type, got %d", wireType)
	}
	return r.readVarintRaw()
}

func (r *sliceReader) readVarintRaw() (uint64, error) {
	var result uint64
	var shift uint
	for r.pos < len(r.data) {
		b := r.data[r.pos]
		r.pos++
		result |= uint64(b&0x7F) << shift
		if b&0x80 == 0 {
			return result, nil
		}
		shift += 7
	}
	return result, io.ErrUnexpectedEOF
}

func (r *sliceReader) readBytes(wireType int) ([]byte, error) {
	if wireType != 2 {
		return nil, fmt.Errorf("expected length-delimited wire type, got %d", wireType)
	}
	length, err := r.readVarintRaw()
	if err != nil {
		return nil, err
	}
	if r.pos+int(length) > len(r.data) {
		return nil, io.ErrUnexpectedEOF
	}
	result := r.data[r.pos : r.pos+int(length)]
	r.pos += int(length)
	return result, nil
}

func (r *sliceReader) readString(wireType int) (string, error) {
	data, err := r.readBytes(wireType)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (r *sliceReader) readFloat32(wireType int) (float32, error) {
	switch wireType {
	case 5:
		if r.pos+4 > len(r.data) {
			return 0, io.ErrUnexpectedEOF
		}
		bits := binary.LittleEndian.Uint32(r.data[r.pos : r.pos+4])
		r.pos += 4
		return math.Float32frombits(bits), nil
	case 1:
		if r.pos+8 > len(r.data) {
			return 0, io.ErrUnexpectedEOF
		}
		bits := binary.LittleEndian.Uint64(r.data[r.pos : r.pos+8])
		r.pos += 8
		return float32(math.Float64frombits(bits)), nil
	default:
		return 0, fmt.Errorf("expected fixed32 wire type, got %d", wireType)
	}
}

func (r *sliceReader) skipField(wireType int) {
	switch wireType {
	case 0:
		r.readVarintRaw()
	case 1:
		r.pos += 8
	case 2:
		length, _ := r.readVarintRaw()
		r.pos += int(length)
	case 5:
		r.pos += 4
	}
}

type byScore []struct {
	text  string
	score float32
	id    int
}

func (s byScore) Len() int           { return len(s) }
func (s byScore) Less(i, j int) bool { return s[i].score > s[j].score }
func (s byScore) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

var _ = utf8.RuneLen
var _ = sort.Sort
