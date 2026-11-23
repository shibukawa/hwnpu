package airpc

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
)

// Handle is the client's end of an object of the server: its id and the type whose methods say
// the types of its methods' arguments and results (Bind).
type Handle struct {
	s        *Session
	id       uint64
	t        reflect.Type
	released atomic.Bool
}

// Object is the handle of the server's object id (0: the session's root). It is bound to a type
// (Bind) before it is called.
func (s *Session) Object(id uint64) *Handle { return &Handle{s: s, id: id} }

// Bind takes the types of the object's methods from the methods of v's type (a value of the
// client's process with the same methods: the model's own type), or from the interface type
// itself when v is a reflect.Type of one (the server's methods). It returns h.
func (h *Handle) Bind(v any) *Handle {
	if t, ok := v.(reflect.Type); ok {
		h.t = t
	} else {
		h.t = reflect.TypeOf(v)
	}
	return h
}

// Session is the session the object is in.
func (h *Handle) Session() *Session { return h.s }

// Release tells the server that the object is not used any more (it forgets it; closing it is
// the server's to decide).
func (h *Handle) Release() error {
	if h.released.Swap(true) {
		return nil
	}
	return h.s.send(&msg{Kind: kRelease, Obj: h.id})
}

// methodType is the type of the method name without its receiver.
func (h *Handle) methodType(name string) (reflect.Type, error) {
	if h.t == nil {
		return nil, fmt.Errorf("airpc: the handle of object %d is bound to no type", h.id)
	}
	m, ok := h.t.MethodByName(name)
	if !ok {
		return nil, fmt.Errorf("airpc: %s has no method %s", h.t, name)
	}
	if h.t.Kind() == reflect.Interface {
		return m.Type, nil
	}
	mt := m.Type
	in := make([]reflect.Type, 0, mt.NumIn()-1)
	for i := 1; i < mt.NumIn(); i++ {
		in = append(in, mt.In(i))
	}
	out := make([]reflect.Type, mt.NumOut())
	for i := range out {
		out[i] = mt.Out(i)
	}
	return reflect.FuncOf(in, out, mt.IsVariadic()), nil
}

// Call calls the object's method with args (after ctx, when the method takes a context) and
// returns its results; the last, when it is an error, is returned as the error. A function
// argument, and a struct argument's function fields, are called back in this process until the
// call returns (or until the streams it returns have been read).
func (h *Handle) Call(ctx context.Context, method string, args ...any) ([]reflect.Value, error) {
	return h.invoke(ctx, method, false, args)
}

// CallKeep is Call whose callbacks stay for the rest of the session: functions the server keeps
// and calls later (a Runtime's log).
func (h *Handle) CallKeep(ctx context.Context, method string, args ...any) ([]reflect.Value, error) {
	return h.invoke(ctx, method, true, args)
}

// Call0 calls a method that returns an error alone.
func Call0(h *Handle, ctx context.Context, method string, args ...any) error {
	_, err := h.Call(ctx, method, args...)
	return err
}

// Call1 calls a method that returns an R and an error (or an R alone).
func Call1[R any](h *Handle, ctx context.Context, method string, args ...any) (R, error) {
	var r R
	out, err := h.Call(ctx, method, args...)
	if len(out) > 0 && out[0].IsValid() {
		r, _ = out[0].Interface().(R)
	}
	return r, err
}

// Call2 calls a method that returns an R1, an R2 and an error (or the two alone).
func Call2[R1, R2 any](h *Handle, ctx context.Context, method string, args ...any) (R1, R2, error) {
	var (
		r1 R1
		r2 R2
	)
	out, err := h.Call(ctx, method, args...)
	if len(out) > 1 {
		r1, _ = out[0].Interface().(R1)
		r2, _ = out[1].Interface().(R2)
	}
	return r1, r2, err
}

// Value calls a method that takes no context and returns an R alone (Model, Sizes): the zero
// value when the session has ended.
func Value[R any](h *Handle, method string, args ...any) R {
	r, _ := Call1[R](h, context.Background(), method, args...)
	return r
}

func (h *Handle) invoke(ctx context.Context, method string, keep bool, args []any) ([]reflect.Value, error) {
	s := h.s
	ft, err := h.methodType(method)
	if err != nil {
		return nil, err
	}
	off := 0
	if ft.NumIn() > 0 && ft.In(0) == ctxType {
		off = 1
	}
	if ft.IsVariadic() || len(args) != ft.NumIn()-off {
		return nil, fmt.Errorf("airpc: %s takes %d arguments, given %d", method, ft.NumIn()-off, len(args))
	}
	id := s.next.Add(1)
	m := &msg{Kind: kCall, ID: id, Obj: h.id, Method: method, Keep: keep}
	var funcs []uint64
	defer func() {
		if funcs != nil {
			s.freeFuncs(funcs)
		}
	}()
	for i, a := range args {
		pt := ft.In(off + i)
		v := reflect.New(pt).Elem()
		if a != nil {
			av := reflect.ValueOf(a)
			switch {
			case av.Type().AssignableTo(pt):
				v.Set(av)
			case av.Type().ConvertibleTo(pt):
				v.Set(av.Convert(pt))
			default:
				return nil, fmt.Errorf("airpc: %s's argument %d is a %s, given a %s", method, i+1, pt, av.Type())
			}
		}
		switch {
		case isSeq(pt):
			if !v.IsNil() {
				m.Funcs = append(m.Funcs, funcRef{Arg: i, ID: s.addSource(v, nil), Stream: true})
			}
			m.Values = append(m.Values, nil)
			continue
		case pt.Kind() == reflect.Func:
			if !v.IsNil() {
				fid := s.addFunc(v)
				funcs = append(funcs, fid)
				m.Funcs = append(m.Funcs, funcRef{Arg: i, ID: fid})
			}
			m.Values = append(m.Values, nil)
			continue
		}
		funcFields(v, nil, 0, func(path []int, f reflect.Value) {
			fid := s.addFunc(f)
			funcs = append(funcs, fid)
			m.Funcs = append(m.Funcs, funcRef{Arg: i, Path: path, ID: fid})
		})
		b, err := encode(pt, v)
		if err != nil {
			return nil, err
		}
		m.Values = append(m.Values, b)
	}
	ch := make(chan *msg, 1)
	s.mu.Lock()
	select {
	case <-s.done:
		s.mu.Unlock()
		return nil, ErrClosed
	default:
	}
	s.replies[id] = ch
	s.mu.Unlock()
	if err := s.send(m); err != nil {
		return nil, err
	}
	var r *msg
	cancelled := false
	ctxDone := ctx.Done()
	for r == nil {
		select {
		case r = <-ch:
		case <-ctxDone:
			if !cancelled {
				cancelled = true
				s.send(&msg{Kind: kCancel, ID: id})
			}
			ctxDone = nil // the server's call returns when it sees it
		case <-s.done:
			return nil, ErrClosed
		}
	}
	if len(r.Err) > 0 {
		e, err := decodeError(r.Err)
		if err != nil {
			return nil, err
		}
		return nil, e
	}
	streams := map[int]uint64{}
	for _, f := range r.Funcs {
		if f.Stream {
			streams[f.Arg] = f.ID
		}
	}
	// the call's callbacks stay while the streams it returned are read
	var left atomic.Int64
	left.Store(int64(len(streams)))
	all := make(chan struct{})
	held := funcs
	if keep {
		held = nil
	}
	ended := func() {
		if left.Add(-1) == 0 {
			close(all)
			s.freeFuncs(held)
		}
	}
	if len(streams) > 0 || keep {
		funcs = nil // freed by ended, or kept
	}
	if len(streams) > 0 {
		go func() {
			select {
			case <-ctx.Done():
				s.send(&msg{Kind: kCancel, ID: id})
			case <-all:
			case <-s.done:
			}
		}()
	}
	var out []reflect.Value
	var callErr error
	for i := range ft.NumOut() {
		ot := ft.Out(i)
		if i == ft.NumOut()-1 && ot == errorType {
			var data []byte
			if i < len(r.Values) {
				data = r.Values[i]
			}
			v, err := decode(ot, data)
			if err != nil {
				return nil, err
			}
			if !v.IsNil() {
				callErr = v.Interface().(error)
			}
			break
		}
		if sid, ok := streams[i]; ok {
			out = append(out, s.sink(ot, sid, ended))
			continue
		}
		var data []byte
		if i < len(r.Values) {
			data = r.Values[i]
		}
		v, err := decode(ot, data)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	if cancelled && callErr != nil && callErr.Error() == context.Canceled.Error() {
		callErr = ctx.Err() // the caller's own: its deadline, when that is what ended it
	}
	return out, callErr
}

func (s *Session) addFunc(v reflect.Value) uint64 {
	id := s.next.Add(1)
	s.mu.Lock()
	s.funcs[id] = v
	s.mu.Unlock()
	return id
}

func (s *Session) freeFuncs(ids []uint64) {
	s.mu.Lock()
	for _, id := range ids {
		delete(s.funcs, id)
	}
	s.mu.Unlock()
}

// callback runs the client's function a server's call called.
func (s *Session) callback(m *msg) {
	reply := &msg{Kind: kCallbackReturn, ID: m.ID}
	s.mu.Lock()
	fv, ok := s.funcs[m.Obj]
	s.mu.Unlock()
	if !ok {
		reply.Err, _ = encodeError(errors.New("airpc: a callback of a call that has returned"))
		s.send(reply)
		return
	}
	ft := fv.Type()
	args := make([]reflect.Value, ft.NumIn())
	for i := range args {
		var data []byte
		if i < len(m.Values) {
			data = m.Values[i]
		}
		v, err := decode(ft.In(i), data)
		if err != nil {
			reply.Err, _ = encodeError(err)
			s.send(reply)
			return
		}
		args[i] = v
	}
	var out []reflect.Value
	if err := protect(func() { out = fv.Call(args) }); err != nil {
		reply.Err, _ = encodeError(err)
		s.send(reply)
		return
	}
	for i, o := range out {
		b, err := encode(ft.Out(i), o)
		if err != nil {
			reply.Err, _ = encodeError(err)
			reply.Values = nil
			break
		}
		reply.Values = append(reply.Values, b)
	}
	s.send(reply)
}

// stub is a function of type ft that calls the client's function fid and returns what it
// returned: the zero values (and the error, when ft returns one) when that fails.
func (s *Session) stub(ft reflect.Type, fid uint64) reflect.Value {
	return reflect.MakeFunc(ft, func(args []reflect.Value) []reflect.Value {
		zero := func(err error) []reflect.Value {
			out := make([]reflect.Value, ft.NumOut())
			for i := range out {
				out[i] = reflect.New(ft.Out(i)).Elem()
			}
			if n := ft.NumOut(); n > 0 && ft.Out(n-1) == errorType {
				out[n-1] = errorValue(err)
			}
			return out
		}
		id := s.next.Add(1)
		m := &msg{Kind: kCallback, ID: id, Obj: fid}
		for i, a := range args {
			b, err := encode(ft.In(i), a)
			if err != nil {
				return zero(err)
			}
			m.Values = append(m.Values, b)
		}
		ch := make(chan *msg, 1)
		s.mu.Lock()
		select {
		case <-s.done:
			s.mu.Unlock()
			return zero(ErrClosed)
		default:
		}
		s.replies[id] = ch
		s.mu.Unlock()
		if err := s.send(m); err != nil {
			return zero(err)
		}
		var r *msg
		select {
		case r = <-ch:
		case <-s.done:
			return zero(ErrClosed)
		}
		if len(r.Err) > 0 {
			e, _ := decodeError(r.Err)
			if e == nil {
				return zero(errors.New("airpc: a callback failed"))
			}
			return zero(e)
		}
		out := make([]reflect.Value, ft.NumOut())
		for i := range out {
			var data []byte
			if i < len(r.Values) {
				data = r.Values[i]
			}
			v, err := decode(ft.Out(i), data)
			if err != nil {
				return zero(err)
			}
			out[i] = v
		}
		return out
	})
}

// serveCall runs a client's call of an object's method.
func (s *Session) serveCall(m *msg) {
	reply := &msg{Kind: kReturn, ID: m.ID}
	fail := func(err error) {
		reply.Err, _ = encodeError(err)
		s.send(reply)
	}
	s.mu.Lock()
	obj, ok := s.objects[m.Obj]
	s.mu.Unlock()
	if !ok {
		fail(fmt.Errorf("airpc: no object %d (released)", m.Obj))
		return
	}
	mv := reflect.ValueOf(obj).MethodByName(m.Method)
	if !mv.IsValid() {
		fail(fmt.Errorf("airpc: %T has no method %s", obj, m.Method))
		return
	}
	ft := mv.Type()
	off := 0
	if ft.NumIn() > 0 && ft.In(0) == ctxType {
		off = 1
	}
	n := ft.NumIn() - off
	if ft.IsVariadic() || len(m.Values) != n {
		fail(fmt.Errorf("airpc: %s takes %d arguments, sent %d (hwnpud and this program are of other builds?)", m.Method, n, len(m.Values)))
		return
	}
	ctx, cancel := context.WithCancel(s.ctx)
	var once sync.Once
	finish := func() {
		once.Do(func() {
			cancel()
			s.mu.Lock()
			delete(s.running, m.ID)
			s.mu.Unlock()
		})
	}
	s.mu.Lock()
	s.running[m.ID] = cancel
	s.mu.Unlock()
	in := make([]reflect.Value, 0, ft.NumIn())
	if off == 1 {
		in = append(in, reflect.ValueOf(ctx))
	}
	args := make([]reflect.Value, n)
	for i := range n {
		v, err := decode(ft.In(off+i), m.Values[i])
		if err != nil {
			finish()
			fail(err)
			return
		}
		args[i] = v
	}
	for _, f := range m.Funcs {
		if f.Arg < 0 || f.Arg >= n {
			continue
		}
		pt := ft.In(off + f.Arg)
		switch {
		case f.Stream:
			args[f.Arg] = s.sink(pt, f.ID, nil)
		case len(f.Path) == 0:
			args[f.Arg] = s.stub(pt, f.ID)
		default:
			fv, err := at(args[f.Arg], f.Path)
			if err != nil {
				finish()
				fail(err)
				return
			}
			fv.Set(s.stub(fv.Type(), f.ID))
		}
	}
	var out []reflect.Value
	if err := protect(func() { out = mv.Call(append(in, args...)) }); err != nil {
		finish()
		fail(err)
		return
	}
	var left atomic.Int64
	for i, o := range out {
		ot := ft.Out(i)
		if isSeq(ot) && !o.IsNil() {
			left.Add(1)
			reply.Funcs = append(reply.Funcs, funcRef{Arg: i, ID: s.addSource(o, func() {
				if left.Add(-1) == 0 {
					finish()
				}
			}), Stream: true})
			reply.Values = append(reply.Values, nil)
			continue
		}
		b, err := encode(ot, o)
		if err != nil {
			finish()
			fail(err)
			return
		}
		reply.Values = append(reply.Values, b)
	}
	if left.Load() == 0 {
		finish()
	}
	s.send(reply)
}
