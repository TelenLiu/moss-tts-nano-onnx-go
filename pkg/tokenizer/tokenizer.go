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
			val, err := sr.readBytes(wireType)
			if err != nil {
				continue
			}
			parseTrainerSpec(val, p)
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
	normalized := strings.ReplaceAll(text, " ", "\xe2\x96\x81")
	runes := []rune(normalized)
	var tokens []int
	i := 0
	for i < len(runes) {
		bestLen := 0
		bestID := -1
		bestScore := float32(math.Inf(-1))
		for end := i + 1; end <= len(runes) && end-i <= 64; end++ {
			candidate := string(runes[i:end])
			if id, ok := p.PieceToID[candidate]; ok {
				if p.Types[id] == PieceNormal || p.Types[id] == PieceUserDefined {
					if p.Scores[id] > bestScore {
						bestScore = p.Scores[id]
						bestLen = end - i
						bestID = id
					}
				}
			}
		}
		if bestID >= 0 && bestLen > 0 {
			tokens = append(tokens, bestID)
			i += bestLen
		} else {
			ch := runes[i]
			hex := fmt.Sprintf("<0x%02X>", ch)
			if id, ok := p.PieceToID[hex]; ok {
				tokens = append(tokens, id)
			} else {
				tokens = append(tokens, p.UnkID)
			}
			i++
		}
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
	return strings.TrimPrefix(result, " ")
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
