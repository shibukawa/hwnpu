// Package airpc calls the methods of Go values in another process over one connection: hwnpud's
// session of the ai package (docs/design/hwnpud.md), in which an app's Plan, Runtime and models
// run in hwnpud's engine and the app calls them as if they were its own.
//
// A session is a connection (a Unix domain socket, upgraded from HTTP: Handler, Dial) with a
// gob stream each way, of messages multiplexed by id. The server end holds objects by id (Root,
// Session.Register); the client end calls their methods by name (Handle.Call), with the method's
// types taken from a type of its own that has the same methods (Handle.Bind: the model's type of
// the app's process, an interface of the server's methods). What crosses:
//
//   - the arguments and the results, with gob; an error as its message with the registered
//     sentinels and error types it is (RegisterError, RegisterErrorType); a value of an
//     interface as its concrete type, which is registered (Register; image.Image's are)
//   - a context.Context first argument is not sent: the server's call gets a context of its
//     own, cancelled when the caller's is
//   - a function argument, and the functions of a struct argument's fields (imagegen.Options'
//     Progress), are callbacks: the server's call gets a function that calls the caller's in
//     its process and returns what it returned
//   - an iterator (iter.Seq, iter.Seq2 and their shapes: flow.Stream) is a stream: an argument
//     is read from the caller as the server ranges over it, a result is ranged over by the
//     caller as the server produces it; a caller that stops early stops the producer and waits
//     for it to end; a call's pointer results are sent again with the end of its last stream
//     (a piece's stats, whole when the music ends)
//   - a value of a type registered with RegisterRef stays in the server behind a handle (a
//     picture's encoding), and is the server's own again when it is given back
//   - a variadic method's arguments go as their slice
//
// airpc knows nothing of the ai package; internal/aihost serves the ai package on it.
package airpc

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"
	"sync/atomic"
)

// The kinds of message.
const (
	kCall           uint8 = iota + 1 // client → server: call Obj's Method
	kReturn                          // server → client: the call ID's results
	kCancel                          // client → server: the call ID's context is cancelled
	kCallback                        // server → client: call the function Obj, reply to ID
	kCallbackReturn                  // client → server: the callback ID's results
	kPull                            // the stream ID begins: its producer sends its items
	kItem                            // a value of the stream ID
	kEnd                             // the stream ID has ended
	kStop                            // the stream ID's reader stopped: its producer stops
	kRelease                         // client → server: the object Obj is not used any more
)

// msg is what crosses.
type msg struct {
	Kind   uint8
	ID     uint64
	Obj    uint64
	Method string
	Values [][]byte
	// Funcs are the callbacks and the streams of a call's arguments, or the streams of a
	// return's results
	Funcs []funcRef
	Err   []byte // a wireError
	// Keep keeps a call's callbacks for the rest of the session (Handle.CallKeep)
	Keep bool
}

// funcRef is a callback (a function of the caller's process) or a stream, in place of an
// argument (Arg) or a field of one (Path), or of a result (Arg).
type funcRef struct {
	Arg    int
	Path   []int
	ID     uint64
	Stream bool
	// Ref is an object of the server's (RegisterRef)
	Ref bool
}

// Session is one end of a connection.
type Session struct {
	conn   io.ReadWriteCloser
	r      io.Reader
	server bool

	wmu sync.Mutex
	enc *gob.Encoder

	next atomic.Uint64

	mu       sync.Mutex
	replies  map[uint64]chan *msg // calls (client) and callbacks (server) waiting
	objects  map[uint64]any       // server
	funcs    map[uint64]reflect.Value
	sources  map[uint64]*source // the streams this end produces
	sinks    map[uint64]*queue  // the streams this end reads
	running  map[uint64]context.CancelFunc
	onClosed []func()

	ctx    context.Context // the server's calls'
	cancel context.CancelFunc
	done   chan struct{}
	err    error
}

func newSession(conn io.ReadWriteCloser, r io.Reader, server bool) *Session {
	s := &Session{conn: conn, r: r, server: server, enc: gob.NewEncoder(conn),
		replies: map[uint64]chan *msg{}, objects: map[uint64]any{}, funcs: map[uint64]reflect.Value{},
		sources: map[uint64]*source{}, sinks: map[uint64]*queue{}, running: map[uint64]context.CancelFunc{},
		done: make(chan struct{})}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	return s
}

// NewClient is the client end of a session over conn (read from r, conn when nil).
func NewClient(conn io.ReadWriteCloser, r io.Reader) *Session {
	if r == nil {
		r = conn
	}
	s := newSession(conn, r, false)
	go s.read()
	return s
}

// Serve is the server end of a session over conn (read from r, conn when nil), whose object 0
// is root(s): it serves until the connection ends, then cancels the calls going on and calls
// root's Close when it has one (io.Closer) and the functions given to OnClose.
func Serve(conn io.ReadWriteCloser, r io.Reader, root func(s *Session) any) error {
	if r == nil {
		r = conn
	}
	s := newSession(conn, r, true)
	s.objects[0] = root(s)
	s.read()
	if c, ok := s.objects[0].(io.Closer); ok {
		c.Close()
	}
	if errors.Is(s.err, io.EOF) || errors.Is(s.err, ErrClosed) {
		return nil
	}
	return s.err
}

// Done is closed when the session has ended.
func (s *Session) Done() <-chan struct{} { return s.done }

// Close ends the session.
func (s *Session) Close() error {
	s.fail(ErrClosed)
	return nil
}

// OnClose calls fn when the session ends.
func (s *Session) OnClose(fn func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	select {
	case <-s.done:
		go fn()
	default:
		s.onClosed = append(s.onClosed, fn)
	}
}

func (s *Session) fail(err error) {
	s.mu.Lock()
	select {
	case <-s.done:
		s.mu.Unlock()
		return
	default:
	}
	s.err = err
	close(s.done)
	fns := s.onClosed
	s.onClosed = nil
	for _, q := range s.sinks {
		q.close()
	}
	s.mu.Unlock()
	s.cancel()
	s.conn.Close()
	for _, fn := range fns {
		fn()
	}
}

func (s *Session) send(m *msg) error {
	s.wmu.Lock()
	defer s.wmu.Unlock()
	select {
	case <-s.done:
		return ErrClosed
	default:
	}
	if err := s.enc.Encode(m); err != nil {
		go s.fail(err)
		return err
	}
	return nil
}

// read reads the messages until the connection ends.
func (s *Session) read() {
	dec := gob.NewDecoder(s.r)
	for {
		m := new(msg)
		if err := dec.Decode(m); err != nil {
			s.fail(err)
			return
		}
		switch m.Kind {
		case kCall:
			if s.server {
				go s.serveCall(m)
			}
		case kReturn, kCallbackReturn:
			s.mu.Lock()
			ch := s.replies[m.ID]
			delete(s.replies, m.ID)
			s.mu.Unlock()
			if ch != nil {
				ch <- m
			}
		case kCancel:
			s.mu.Lock()
			cancel := s.running[m.ID]
			s.mu.Unlock()
			if cancel != nil {
				cancel()
			}
		case kCallback:
			go s.callback(m)
		case kPull:
			s.mu.Lock()
			src := s.sources[m.ID]
			s.mu.Unlock()
			if src == nil || src.pulled.Swap(true) {
				e, _ := encodeError(errors.New("airpc: a stream read twice, or not there"))
				go s.send(&msg{Kind: kEnd, ID: m.ID, Err: e})
				continue
			}
			go s.produce(m.ID, src)
		case kStop:
			s.mu.Lock()
			if src := s.sources[m.ID]; src != nil {
				src.stopped.Store(true)
			}
			s.mu.Unlock()
		case kItem, kEnd:
			s.mu.Lock()
			q := s.sinks[m.ID]
			s.mu.Unlock()
			if q != nil {
				q.push(m)
			}
		case kRelease:
			if s.server && m.Obj != 0 {
				s.mu.Lock()
				delete(s.objects, m.Obj)
				s.mu.Unlock()
			}
		}
	}
}

// Register keeps v as an object the client calls (Session.Object on its end) and returns its id.
func (s *Session) Register(v any) uint64 {
	id := s.next.Add(1)
	s.mu.Lock()
	s.objects[id] = v
	s.mu.Unlock()
	return id
}

// Forget drops the object id.
func (s *Session) Forget(id uint64) {
	s.mu.Lock()
	delete(s.objects, id)
	s.mu.Unlock()
}

// source is a stream this end produces: an iterator, and what to call when it has ended (before
// its end is sent: the values it returns go with the end, the call's results again).
type source struct {
	seq     reflect.Value
	pulled  atomic.Bool
	stopped atomic.Bool
	ended   func() [][]byte
}

// addSource keeps seq as a stream for the other end to read and returns its id.
func (s *Session) addSource(seq reflect.Value, ended func() [][]byte) uint64 {
	id := s.next.Add(1)
	s.mu.Lock()
	s.sources[id] = &source{seq: seq, ended: ended}
	s.mu.Unlock()
	return id
}

// produce ranges over the stream id, sending its items, until it ends or its reader stops.
func (s *Session) produce(id uint64, src *source) {
	defer func() {
		s.mu.Lock()
		delete(s.sources, id)
		s.mu.Unlock()
	}()
	yt := src.seq.Type().In(0)
	var failed error
	yield := reflect.MakeFunc(yt, func(args []reflect.Value) []reflect.Value {
		if src.stopped.Load() || failed != nil {
			return []reflect.Value{reflect.ValueOf(false)}
		}
		m := &msg{Kind: kItem, ID: id}
		for i, a := range args {
			b, err := encode(yt.In(i), a)
			if err != nil {
				failed = err
				return []reflect.Value{reflect.ValueOf(false)}
			}
			m.Values = append(m.Values, b)
		}
		if err := s.send(m); err != nil {
			failed = err
			return []reflect.Value{reflect.ValueOf(false)}
		}
		return []reflect.Value{reflect.ValueOf(!src.stopped.Load())}
	})
	err := protect(func() { src.seq.Call([]reflect.Value{yield}) })
	end := &msg{Kind: kEnd, ID: id}
	if err == nil {
		err = failed
	}
	if err != nil {
		end.Err, _ = encodeError(err)
	}
	if src.ended != nil {
		end.Values = src.ended()
	}
	s.send(end)
}

// sink is an iterator of type t that reads the stream id of the other end: it asks for its items
// as it is ranged over (once), and stops the producer when its loop ends early (and waits for it
// to end, as a loop over an iterator of this process does). ended is called when it has been
// read, with its end (nil when the session ended first).
func (s *Session) sink(t reflect.Type, id uint64, ended func(end *msg)) reflect.Value {
	var once atomic.Bool
	return reflect.MakeFunc(t, func(args []reflect.Value) []reflect.Value {
		yield := args[0]
		yt := yield.Type()
		if once.Swap(true) {
			s.yieldErr(yield, errors.New("airpc: a stream of the other process is read once"))
			return nil
		}
		var last *msg
		if ended != nil {
			defer func() { ended(last) }()
		}
		q := newQueue()
		s.mu.Lock()
		select {
		case <-s.done:
			s.mu.Unlock()
			s.yieldErr(yield, ErrClosed)
			return nil
		default:
		}
		s.sinks[id] = q
		s.mu.Unlock()
		defer func() {
			s.mu.Lock()
			delete(s.sinks, id)
			s.mu.Unlock()
		}()
		if err := s.send(&msg{Kind: kPull, ID: id}); err != nil {
			s.yieldErr(yield, err)
			return nil
		}
		stopped := false // the loop ended: the items left are dropped until the end comes
		for {
			m, ok := q.pop()
			if !ok {
				if !stopped {
					s.yieldErr(yield, ErrClosed)
				}
				return nil
			}
			if stopped && m.Kind != kEnd {
				continue
			}
			if m.Kind == kEnd {
				last = m
				if stopped {
					return nil
				}
				if len(m.Err) > 0 {
					e, _ := decodeError(m.Err)
					if e != nil {
						s.yieldErr(yield, e)
					}
				}
				return nil
			}
			vals := make([]reflect.Value, yt.NumIn())
			for i := range vals {
				var data []byte
				if i < len(m.Values) {
					data = m.Values[i]
				}
				v, err := decode(yt.In(i), data)
				if err != nil {
					s.send(&msg{Kind: kStop, ID: id})
					s.yieldErr(yield, err)
					return nil
				}
				vals[i] = v
			}
			if !yield.Call(vals)[0].Bool() {
				s.send(&msg{Kind: kStop, ID: id})
				stopped = true
			}
		}
	})
}

// yieldErr gives a stream's reader err, when the stream's second value is an error
// (flow.Stream's); a stream of one value just ends.
func (s *Session) yieldErr(yield reflect.Value, err error) {
	yt := yield.Type()
	if yt.NumIn() != 2 || yt.In(1) != errorType {
		return
	}
	yield.Call([]reflect.Value{reflect.New(yt.In(0)).Elem(), errorValue(err)})
}

// protect runs fn, turning a panic into an error.
func protect(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("airpc: panic: %v", r)
		}
	}()
	fn()
	return nil
}

// queue is the items of a stream that came and were not read yet: it never blocks the reading
// of the connection.
type queue struct {
	mu     sync.Mutex
	items  []*msg
	more   chan struct{}
	closed bool
}

func newQueue() *queue { return &queue{more: make(chan struct{}, 1)} }

func (q *queue) push(m *msg) {
	q.mu.Lock()
	q.items = append(q.items, m)
	q.mu.Unlock()
	select {
	case q.more <- struct{}{}:
	default:
	}
}

func (q *queue) close() {
	q.mu.Lock()
	q.closed = true
	q.mu.Unlock()
	select {
	case q.more <- struct{}{}:
	default:
	}
}

// pop is the next item; ok is false when the session ended before one came.
func (q *queue) pop() (*msg, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			m := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return m, true
		}
		if q.closed {
			q.mu.Unlock()
			return nil, false
		}
		q.mu.Unlock()
		<-q.more
	}
}
