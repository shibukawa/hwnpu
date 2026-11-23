package airpc

import (
	"bytes"
	"context"
	"encoding/gob"
	"fmt"
	"image"
	"image/color"
	"reflect"
	"sync"
)

var (
	ctxType   = reflect.TypeFor[context.Context]()
	errorType = reflect.TypeFor[error]()
	boolType  = reflect.TypeFor[bool]()
)

// The pictures an image.Image may hold, and the colors of a paletted one's palette, are sent as
// their own types: a value of an interface is sent with the name of its concrete type.
func init() {
	for _, v := range []any{&image.RGBA{}, &image.NRGBA{}, &image.RGBA64{}, &image.NRGBA64{}, &image.Gray{}, &image.Gray16{},
		&image.Alpha{}, &image.Alpha16{}, &image.CMYK{}, &image.YCbCr{}, &image.NYCbCrA{}, &image.Paletted{}, &image.Uniform{},
		color.RGBA{}, color.NRGBA{}, color.RGBA64{}, color.NRGBA64{}, color.Gray{}, color.Gray16{}, color.Alpha{}, color.Alpha16{},
		color.CMYK{}, color.YCbCr{}, color.NYCbCrA{}} {
		gob.Register(v)
	}
}

// Register makes the concrete type of v one a value of an interface can be sent as
// (gob.Register): the types a package's methods take or give behind an interface.
func Register(v any) { gob.Register(v) }

// boxes are the structs a value of each type is sent in (struct{ V T }): gob sends a nil
// pointer, a value of an interface and a zero value only as a field.
var boxes sync.Map

func box(t reflect.Type) reflect.Type {
	if b, ok := boxes.Load(t); ok {
		return b.(reflect.Type)
	}
	b := reflect.StructOf([]reflect.StructField{{Name: "V", Type: t}})
	boxes.Store(t, b)
	return b
}

// isSeq reports whether t is an iterator (iter.Seq, iter.Seq2 and the types of their shape:
// flow.Stream): what is sent as a stream, not as a value.
func isSeq(t reflect.Type) bool {
	if t.Kind() != reflect.Func || t.NumIn() != 1 || t.NumOut() != 0 {
		return false
	}
	y := t.In(0)
	return y.Kind() == reflect.Func && y.NumOut() == 1 && y.Out(0) == boolType && y.NumIn() >= 1 && y.NumIn() <= 2
}

// sent reports whether a value of t goes as bytes: not a function, a stream or a channel.
func sent(t reflect.Type) bool { return t.Kind() != reflect.Func && t.Kind() != reflect.Chan }

// encode is v (of type t; the zero value when not valid) as bytes: an error as an *Error's
// parts, the rest with gob. A function or a channel is nil (it goes another way).
func encode(t reflect.Type, v reflect.Value) ([]byte, error) {
	if t == errorType {
		if !v.IsValid() || v.IsNil() {
			return nil, nil
		}
		return encodeError(v.Interface().(error))
	}
	if !sent(t) {
		return nil, nil
	}
	b := reflect.New(box(t)).Elem()
	if v.IsValid() {
		b.Field(0).Set(v)
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).EncodeValue(b); err != nil {
		return nil, fmt.Errorf("airpc: sending a %s: %w", t, err)
	}
	return buf.Bytes(), nil
}

// decode is the value of type t encode made of data (the zero value of t for none).
func decode(t reflect.Type, data []byte) (reflect.Value, error) {
	if len(data) == 0 || !sent(t) {
		return reflect.New(t).Elem(), nil
	}
	if t == errorType {
		e, err := decodeError(data)
		if err != nil {
			return reflect.Value{}, err
		}
		v := reflect.New(t).Elem()
		v.Set(reflect.ValueOf(e))
		return v, nil
	}
	b := reflect.New(box(t))
	if err := gob.NewDecoder(bytes.NewReader(data)).DecodeValue(b); err != nil {
		return reflect.Value{}, fmt.Errorf("airpc: receiving a %s: %w", t, err)
	}
	out := reflect.New(t).Elem()
	out.Set(b.Elem().Field(0))
	return out, nil
}

// funcFields calls fn with the path (the fields' indices) of each function of v that is not nil:
// v itself (an empty path), and the functions of its struct's exported fields, through pointers
// to structs (imagegen.Options' Progress, Checks.Latents). gob leaves them out; they go as
// callbacks.
func funcFields(v reflect.Value, path []int, depth int, fn func(path []int, f reflect.Value)) {
	if depth > 6 || !v.IsValid() {
		return
	}
	switch v.Kind() {
	case reflect.Func:
		if !v.IsNil() {
			fn(append([]int(nil), path...), v)
		}
	case reflect.Pointer:
		if !v.IsNil() && v.Elem().Kind() == reflect.Struct {
			funcFields(v.Elem(), path, depth+1, fn)
		}
	case reflect.Struct:
		t := v.Type()
		for i := range t.NumField() {
			if f := t.Field(i); f.IsExported() {
				funcFields(v.Field(i), append(path, i), depth+1, fn)
			}
		}
	}
}

// at is the field of v at path (funcFields'), making the pointers on the way that are nil.
func at(v reflect.Value, path []int) (reflect.Value, error) {
	for _, i := range path {
		for v.Kind() == reflect.Pointer {
			if v.IsNil() {
				v.Set(reflect.New(v.Type().Elem()))
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct || i >= v.NumField() {
			return reflect.Value{}, fmt.Errorf("airpc: no field %d in a %s", i, v.Type())
		}
		v = v.Field(i)
	}
	if v.Kind() != reflect.Func {
		return reflect.Value{}, fmt.Errorf("airpc: a callback in place of a %s", v.Type())
	}
	return v, nil
}
