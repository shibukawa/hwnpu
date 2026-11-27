package soundwire

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// DecodeWAV decodes a WAV file (PCM of 8, 16, 24 or 32 bits, or 32 bit float) into its channels
// at its own rate, and the rate (internal/llm/lfm3audio.DecodeWAV).
func DecodeWAV(data []byte) ([][]float32, int, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errors.New("audio: not a WAV file")
	}
	var format, channels, bits int
	var rate int
	var pcm []byte
	for p := 12; p+8 <= len(data); {
		id, size := string(data[p:p+4]), int(binary.LittleEndian.Uint32(data[p+4:]))
		body := data[p+8 : min(p+8+size, len(data))]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, errors.New("audio: short fmt chunk")
			}
			format, channels = int(binary.LittleEndian.Uint16(body)), int(binary.LittleEndian.Uint16(body[2:]))
			rate, bits = int(binary.LittleEndian.Uint32(body[4:])), int(binary.LittleEndian.Uint16(body[14:]))
			if format == 0xFFFE && len(body) >= 26 { // WAVE_FORMAT_EXTENSIBLE: the sub format's first two bytes
				format = int(binary.LittleEndian.Uint16(body[24:]))
			}
		case "data":
			pcm = body
		}
		p += 8 + size + size%2
	}
	if channels == 0 || rate == 0 || pcm == nil {
		return nil, 0, errors.New("audio: WAV without fmt or data")
	}
	width := bits / 8
	if !(format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || format == 3 && bits == 32) {
		return nil, 0, fmt.Errorf("audio: WAV format %d with %d bits is not supported", format, bits)
	}
	n := len(pcm) / (width * channels)
	out := make([][]float32, channels)
	for c := range out {
		out[c] = make([]float32, n)
	}
	for i := 0; i < n; i++ {
		for c := 0; c < channels; c++ {
			b := pcm[(i*channels+c)*width:]
			var v float64
			switch {
			case format == 3:
				v = float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
			case bits == 8:
				v = (float64(b[0]) - 128) / 128
			case bits == 16:
				v = float64(int16(binary.LittleEndian.Uint16(b))) / (1 << 15)
			case bits == 24:
				v = float64(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / (1 << 23)
			default:
				v = float64(int32(binary.LittleEndian.Uint32(b))) / (1 << 31)
			}
			out[c][i] = float32(v)
		}
	}
	return out, rate, nil
}
