package pitch

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"google.golang.org/protobuf/encoding/protowire"
)

// tensor is an initializer of an ONNX graph: its shape and its values as float64 (the model's are
// float32, float64, int32 and int64).
type tensor struct {
	shape []int
	vals  []float64
}

// ONNX's TensorProto.DataType values that the reader takes.
const (
	onnxFloat  = 1
	onnxInt32  = 6
	onnxInt64  = 7
	onnxDouble = 11
)

// readInitializers reads the initializers of an ONNX model's graph, by name: ModelProto.graph (7)
// is a GraphProto whose initializer (5) are TensorProtos of dims (1), data_type (2), name (8) and
// the values in raw_data (9, little endian) or in float_data (4), int32_data (5), int64_data (7)
// or double_data (10). Nothing else of the file is read.
func readInitializers(path string) (map[string]tensor, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	graph, err := field(raw, 7)
	if err != nil {
		return nil, fmt.Errorf("pitch: %s: %w", path, err)
	}
	out := map[string]tensor{}
	err = fields(graph, func(num protowire.Number, b []byte) error {
		if num != 5 {
			return nil
		}
		name, t, err := parseTensor(b)
		if err != nil {
			return err
		}
		out[name] = t
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("pitch: %s: %w", path, err)
	}
	return out, nil
}

// fields calls f with each length-delimited field of a message; other wire types are skipped.
func fields(msg []byte, f func(num protowire.Number, b []byte) error) error {
	for len(msg) > 0 {
		num, typ, n := protowire.ConsumeTag(msg)
		if n < 0 {
			return protowire.ParseError(n)
		}
		msg = msg[n:]
		if typ == protowire.BytesType {
			b, m := protowire.ConsumeBytes(msg)
			if m < 0 {
				return protowire.ParseError(m)
			}
			if err := f(num, b); err != nil {
				return err
			}
			msg = msg[m:]
			continue
		}
		m := protowire.ConsumeFieldValue(num, typ, msg)
		if m < 0 {
			return protowire.ParseError(m)
		}
		msg = msg[m:]
	}
	return nil
}

// field returns the first length-delimited field num of a message.
func field(msg []byte, want protowire.Number) ([]byte, error) {
	var out []byte
	found := false
	err := fields(msg, func(num protowire.Number, b []byte) error {
		if num == want && !found {
			out, found = b, true
		}
		return nil
	})
	if err == nil && !found {
		err = fmt.Errorf("no field %d", want)
	}
	return out, err
}

func parseTensor(msg []byte) (string, tensor, error) {
	var t tensor
	var name string
	var dtype int
	var raw []byte
	var packed [][]byte // float_data, int32_data, int64_data or double_data, by field
	var packedNum protowire.Number
	for b := msg; len(b) > 0; {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return "", t, protowire.ParseError(n)
		}
		b = b[n:]
		switch {
		case typ == protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			if m < 0 {
				return "", t, protowire.ParseError(m)
			}
			switch num {
			case 1:
				t.shape = append(t.shape, int(int64(v)))
			case 2:
				dtype = int(v)
			}
			b = b[m:]
		case typ == protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			if m < 0 {
				return "", t, protowire.ParseError(m)
			}
			switch num {
			case 1: // packed dims
				for p := v; len(p) > 0; {
					d, k := protowire.ConsumeVarint(p)
					if k < 0 {
						return "", t, protowire.ParseError(k)
					}
					t.shape = append(t.shape, int(int64(d)))
					p = p[k:]
				}
			case 8:
				name = string(v)
			case 9:
				raw = v
			case 4, 5, 7, 10:
				packed, packedNum = append(packed, v), num
			}
			b = b[m:]
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			if m < 0 {
				return "", t, protowire.ParseError(m)
			}
			b = b[m:]
		}
	}
	count := 1
	for _, d := range t.shape {
		count *= d
	}
	switch {
	case raw != nil:
		size := map[int]int{onnxFloat: 4, onnxInt32: 4, onnxInt64: 8, onnxDouble: 8}[dtype]
		if size == 0 || len(raw) != size*count {
			return "", t, fmt.Errorf("initializer %s: %d bytes of type %d for shape %v", name, len(raw), dtype, t.shape)
		}
		t.vals = make([]float64, count)
		for i := range t.vals {
			p := raw[i*size:]
			switch dtype {
			case onnxFloat:
				t.vals[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(p)))
			case onnxInt32:
				t.vals[i] = float64(int32(binary.LittleEndian.Uint32(p)))
			case onnxInt64:
				t.vals[i] = float64(int64(binary.LittleEndian.Uint64(p)))
			case onnxDouble:
				t.vals[i] = math.Float64frombits(binary.LittleEndian.Uint64(p))
			}
		}
	default:
		for _, p := range packed {
			for len(p) > 0 {
				switch packedNum {
				case 4: // float, fixed 32
					t.vals = append(t.vals, float64(math.Float32frombits(binary.LittleEndian.Uint32(p))))
					p = p[4:]
				case 10: // double, fixed 64
					t.vals = append(t.vals, math.Float64frombits(binary.LittleEndian.Uint64(p)))
					p = p[8:]
				default: // int32, int64 as varints
					v, k := protowire.ConsumeVarint(p)
					if k < 0 {
						return "", t, protowire.ParseError(k)
					}
					t.vals = append(t.vals, float64(int64(v)))
					p = p[k:]
				}
			}
		}
		if len(t.vals) != count {
			return "", t, fmt.Errorf("initializer %s: %d values for shape %v", name, len(t.vals), t.shape)
		}
	}
	return name, t, nil
}
