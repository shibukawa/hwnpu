package airpc

import (
	"bytes"
	"context"
	"encoding/gob"
	"errors"
	"io"
	"reflect"
	"sync"
)

// The errors a method returns cross the session as their messages, with the sentinels they are
// (errors.Is) and the values of the error types they hold (errors.As) that are registered: the
// caller's errors.Is(err, ai.ErrNotPrepared) and errors.As(err, &licenseErr) hold as they do in
// one process.

var (
	errMu     sync.RWMutex
	sentinels = map[string]error{}
	typed     = map[string]reflect.Type{}
)

func init() {
	RegisterError("context.Canceled", context.Canceled)
	RegisterError("context.DeadlineExceeded", context.DeadlineExceeded)
	RegisterError("io.EOF", io.EOF)
	RegisterError("airpc.ErrClosed", ErrClosed)
}

// RegisterError names a sentinel error (ai.ErrNotPrepared) that errors.Is finds through the
// session; both ends register it under the same name.
func RegisterError(name string, err error) {
	errMu.Lock()
	defer errMu.Unlock()
	sentinels[name] = err
}

// RegisterErrorType names an error type (sample is a value of it, often a nil pointer:
// (*ai.LicenseError)(nil)) whose values errors.As finds through the session; they are sent
// with gob, so its exported fields alone cross.
func RegisterErrorType(name string, sample error) {
	errMu.Lock()
	defer errMu.Unlock()
	typed[name] = reflect.TypeOf(sample)
}

// ErrClosed is the error of a call on a session that has ended (hwnpud's engine exited).
var ErrClosed = errors.New("airpc: the session has ended")

// wireError is an error as it crosses.
type wireError struct {
	Msg string
	Is  []string
	As  []wireTyped
}

type wireTyped struct {
	Name string
	Data []byte
}

// Error is an error that came over a session: its message, with the registered sentinels and
// error types it was.
type Error struct {
	msg string
	is  []error
	as  []reflect.Value
}

func (e *Error) Error() string { return e.msg }

// Is reports whether the error was target on the other end (a registered sentinel).
func (e *Error) Is(target error) bool {
	for _, s := range e.is {
		if s == target {
			return true
		}
	}
	return false
}

// As sets target to the value of its type the error held on the other end (a registered type).
func (e *Error) As(target any) bool {
	p := reflect.ValueOf(target)
	if p.Kind() != reflect.Pointer || p.IsNil() {
		return false
	}
	for _, v := range e.as {
		if v.Type().AssignableTo(p.Elem().Type()) {
			p.Elem().Set(v)
			return true
		}
	}
	return false
}

func encodeError(err error) ([]byte, error) {
	w := wireError{Msg: err.Error()}
	errMu.RLock()
	for name, s := range sentinels {
		if errors.Is(err, s) {
			w.Is = append(w.Is, name)
		}
	}
	for name, t := range typed {
		p := reflect.New(t)
		if errors.As(err, p.Interface()) {
			if data, err := encode(t, p.Elem()); err == nil {
				w.As = append(w.As, wireTyped{Name: name, Data: data})
			}
		}
	}
	errMu.RUnlock()
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(w); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeError(data []byte) (*Error, error) {
	var w wireError
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&w); err != nil {
		return nil, err
	}
	e := &Error{msg: w.Msg}
	errMu.RLock()
	defer errMu.RUnlock()
	for _, name := range w.Is {
		if s, ok := sentinels[name]; ok {
			e.is = append(e.is, s)
		}
	}
	for _, a := range w.As {
		t, ok := typed[a.Name]
		if !ok {
			continue
		}
		if v, err := decode(t, a.Data); err == nil {
			e.as = append(e.as, v)
		}
	}
	return e, nil
}

// errorValue is err as a value of type error for a result (the zero value for nil).
func errorValue(err error) reflect.Value {
	v := reflect.New(errorType).Elem()
	if err != nil {
		v.Set(reflect.ValueOf(err))
	}
	return v
}
