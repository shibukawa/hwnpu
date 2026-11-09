package onnxrt

import (
	"errors"

	"google.golang.org/protobuf/encoding/protowire"
)

// The fields of onnx.proto this file edits.
const (
	modelGraph     = 7  // ModelProto.graph
	graphInput     = 11 // GraphProto.input
	graphOutput    = 12 // GraphProto.output
	graphValueInfo = 13 // GraphProto.value_info
	valueType      = 2  // ValueInfoProto.type
	typeTensor     = 1  // TypeProto.tensor_type
	tensorShape    = 2  // TypeProto.Tensor.shape
)

// FreeShapes is the ONNX model with its graph's inputs and outputs of no fixed shape and no
// value_info: a model exported for one picture size (YuNet's 2023mar, 640 x 640) then runs at
// any size its operators allow (shapes inferred from the input).
func FreeShapes(model []byte) ([]byte, error) {
	return editField(model, modelGraph, func(graph []byte) ([]byte, error) {
		var out []byte
		for b := graph; len(b) > 0; {
			num, typ, n := protowire.ConsumeTag(b)
			if n < 0 {
				return nil, protowire.ParseError(n)
			}
			m := protowire.ConsumeFieldValue(num, typ, b[n:])
			if m < 0 {
				return nil, protowire.ParseError(m)
			}
			field := b[:n+m]
			b = b[n+m:]
			switch {
			case num == graphValueInfo:
				continue
			case (num == graphInput || num == graphOutput) && typ == protowire.BytesType:
				v, _ := protowire.ConsumeBytes(field[n:])
				v, err := editField(v, valueType, func(t []byte) ([]byte, error) {
					return editField(t, typeTensor, func(tt []byte) ([]byte, error) {
						return dropField(tt, tensorShape)
					})
				})
				if err != nil {
					return nil, err
				}
				out = protowire.AppendTag(out, num, protowire.BytesType)
				out = protowire.AppendBytes(out, v)
			default:
				out = append(out, field...)
			}
		}
		return out, nil
	})
}

// editField is msg with each length-delimited field num replaced by edit of its value.
func editField(msg []byte, num protowire.Number, edit func([]byte) ([]byte, error)) ([]byte, error) {
	var out []byte
	found := false
	for b := msg; len(b) > 0; {
		f, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return nil, protowire.ParseError(n)
		}
		m := protowire.ConsumeFieldValue(f, typ, b[n:])
		if m < 0 {
			return nil, protowire.ParseError(m)
		}
		if f == num && typ == protowire.BytesType {
			v, _ := protowire.ConsumeBytes(b[n:])
			v, err := edit(v)
			if err != nil {
				return nil, err
			}
			out = protowire.AppendTag(out, f, protowire.BytesType)
			out = protowire.AppendBytes(out, v)
			found = true
		} else {
			out = append(out, b[:n+m]...)
		}
		b = b[n+m:]
	}
	if num == modelGraph && !found {
		return nil, errors.New("onnxrt: not an ONNX model (no graph)")
	}
	return out, nil
}

// dropField is msg without field num.
func dropField(msg []byte, num protowire.Number) ([]byte, error) {
	var out []byte
	for b := msg; len(b) > 0; {
		f, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return nil, protowire.ParseError(n)
		}
		m := protowire.ConsumeFieldValue(f, typ, b[n:])
		if m < 0 {
			return nil, protowire.ParseError(m)
		}
		if f != num {
			out = append(out, b[:n+m]...)
		}
		b = b[n+m:]
	}
	return out, nil
}
