// Package openvino runs OpenVINO IR (what hwnpu-convert -backend intel writes) with OpenVINO
// itself, through a Python process (serve_ir.py, embedded) that shares the buffers with hwnpu as
// files mapped in memory. It is for checking the Intel path without Intel's NPU: on OpenVINO's
// CPU plugin, hwnpu-chat, hwnpu-agent and the tests run the same IR, by the same argument names,
// as on the NPU through Level Zero. With HWNPU_OV_DEVICE=NPU it runs on OpenVINO's NPU plugin
// instead, to compare with the Level Zero backend on an Intel machine.
//
// The backend lists a device only when asked: HWNPU_BACKEND=openvino, or Enable. HWNPU_OV_PYTHON names a
// Python interpreter with the openvino and numpy packages (default python3, python on Windows), HWNPU_OV_DEVICE the
// OpenVINO device (default CPU), HWNPU_OV_PRECISION the CPU plugin's inference precision (default
// f32; f16 is nearer to what the NPU computes).
package openvino

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

//go:embed serve_ir.py
var serveScript []byte

// Name is the backend's name, and the value of HWNPU_BACKEND that turns it on.
const Name = "openvino"

// Provider is the OpenVINO backend provider.
type Provider struct{}

func (Provider) Name() string { return Name }

func ovDevice() string {
	if d := os.Getenv("HWNPU_OV_DEVICE"); d != "" {
		return d
	}
	return "CPU"
}

var enabled atomic.Bool

// Enable makes the backend list its device without HWNPU_BACKEND=openvino (a command told to run
// on it).
func Enable() { enabled.Store(true) }

func (Provider) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	if !supported || os.Getenv("HWNPU_BACKEND") != Name && !enabled.Load() {
		return nil, nil
	}
	dev := ovDevice()
	v := types.VendorUnknown
	if dev == "NPU" {
		v = types.VendorIntel
	}
	return []types.DeviceInfo{{
		Vendor:  v,
		Backend: Name,
		ID:      dev,
		Name:    "OpenVINO " + dev,
		Caps:    types.CapGraph | types.CapCompile | types.CapHostCoherent,
		Driver:  "OpenVINO through " + python(),
	}}, nil
}

func python() string {
	if p := os.Getenv("HWNPU_OV_PYTHON"); p != "" {
		return p
	}
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func (Provider) Open(ctx context.Context, info types.DeviceInfo, opts backend.Options) (backend.Device, error) {
	if !supported {
		return nil, fmt.Errorf("openvino: %w on this platform", types.ErrUnsupported)
	}
	removeStale()
	dir, err := makeDir()
	if err != nil {
		return nil, err
	}
	script := filepath.Join(dir, "serve_ir.py")
	if err := os.WriteFile(script, serveScript, 0o644); err != nil {
		os.RemoveAll(dir)
		return nil, err
	}
	args := []string{"-I", script, "--device", info.ID}
	if p := os.Getenv("HWNPU_OV_PRECISION"); p != "" {
		args = append(args, "--precision", p)
	}
	cmd := exec.Command(python(), args...)
	cmd.Stderr = os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		os.RemoveAll(dir)
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		os.RemoveAll(dir)
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		os.RemoveAll(dir)
		return nil, fmt.Errorf("openvino: starting %s: %w", python(), err)
	}
	return &device{dir: dir, cmd: cmd, in: in, out: bufio.NewReader(out)}, nil
}

type device struct {
	dir string
	cmd *exec.Cmd

	mu     sync.Mutex // one request at a time
	in     io.WriteCloser
	out    *bufio.Reader
	n      int // buffers made
	closed bool
}

// call sends a request and decodes the reply into res (nil to drop it).
func (d *device) call(req map[string]any, res any) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return errors.New("openvino: the device is closed")
	}
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	if _, err := d.in.Write(append(b, '\n')); err != nil {
		return fmt.Errorf("openvino: the Python process is gone: %w", err)
	}
	line, err := d.out.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("openvino: the Python process is gone: %w (is openvino installed for %s?)", err, python())
	}
	var e struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(line, &e); err != nil {
		return fmt.Errorf("openvino: reply %q: %w", line, err)
	}
	if e.Error != "" {
		return fmt.Errorf("openvino: %s: %s", req["op"], e.Error)
	}
	if res != nil {
		return json.Unmarshal(line, res)
	}
	return nil
}

func (d *device) Close() error {
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil
	}
	d.closed = true
	d.in.Close()
	d.mu.Unlock()
	err := d.cmd.Wait()
	os.RemoveAll(d.dir)
	return err
}

func (d *device) AllocBuffer(size int, kind types.BufferKind) (backend.Buffer, error) {
	d.mu.Lock()
	d.n++
	path := filepath.Join(d.dir, fmt.Sprintf("buf%d", d.n))
	d.mu.Unlock()
	b, err := mapFile(path, size)
	if err != nil {
		return nil, err
	}
	return &buffer{d: d, path: path, b: b}, nil
}

// buffer is a file mapped by hwnpu and the Python process both.
type buffer struct {
	d    *device
	path string
	b    []byte
}

func (b *buffer) Bytes() []byte                   { return b.b }
func (b *buffer) SyncToDevice(off, n int) error   { return nil }
func (b *buffer) SyncFromDevice(off, n int) error { return nil }
func (b *buffer) Close() error {
	if b.b == nil {
		return nil
	}
	b.d.call(map[string]any{"op": "unmap", "path": b.path}, nil)
	err := unmap(b.b)
	b.b = nil
	os.Remove(b.path)
	return err
}

var dtypes = map[string]types.DType{
	"f16": types.F16, "f32": types.F32, "bf16": types.BF16, "i8": types.I8, "u8": types.U8,
	"i32": types.I32, "i64": types.I64,
}

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	path := src.Path
	if !strings.EqualFold(filepath.Ext(path), ".xml") {
		if src.XML == nil {
			return nil, fmt.Errorf("openvino: %s is not OpenVINO IR: %w", path, types.ErrUnsupported)
		}
		d.mu.Lock()
		d.n++
		path = filepath.Join(d.dir, fmt.Sprintf("graph%d.xml", d.n))
		d.mu.Unlock()
		if err := os.WriteFile(path, src.XML, 0o644); err != nil {
			return nil, err
		}
		if err := os.WriteFile(strings.TrimSuffix(path, ".xml")+".bin", src.Bin, 0o644); err != nil {
			return nil, err
		}
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	var res struct {
		ID   int `json:"id"`
		Args []struct {
			Name   string `json:"name"`
			Output bool   `json:"output"`
			Dims   []int  `json:"dims"`
			DType  string `json:"dtype"`
		} `json:"args"`
	}
	if err := d.call(map[string]any{"op": "compile", "xml": abs}, &res); err != nil {
		return nil, &types.BuildError{Log: err.Error(), Err: err}
	}
	g := &graph{d: d, id: res.ID}
	for i, a := range res.Args {
		dt, ok := dtypes[a.DType]
		if !ok {
			g.Close()
			return nil, fmt.Errorf("openvino: argument %q has type %s: %w", a.Name, a.DType, types.ErrUnsupported)
		}
		info := types.ArgInfo{Index: i, Name: a.Name, Dims: a.Dims, DType: dt, ModelDType: dt}
		if a.Output {
			info.Kind = types.ArgOutput
		}
		info.Bytes = dt.Size()
		for _, v := range a.Dims {
			info.Bytes *= v
		}
		g.args = append(g.args, info)
	}
	return g, nil
}

type graph struct {
	d    *device
	id   int
	args []types.ArgInfo
}

func (g *graph) Args() []types.ArgInfo { return g.args }
func (g *graph) Cached() bool          { return false }

// Placement reports the graph as on the NPU only on OpenVINO's NPU plugin, which does not fall
// back to other devices.
func (g *graph) Placement() types.PlacementReport {
	return types.PlacementReport{Verified: ovDevice() == "NPU", Method: "openvino-" + strings.ToLower(ovDevice())}
}

func (g *graph) Close() error {
	if g.id == 0 {
		return nil
	}
	err := g.d.call(map[string]any{"op": "free", "id": g.id}, nil)
	g.id = 0
	return err
}

func (g *graph) NewRequest() (backend.Request, error) {
	return &request{g: g, binds: make([]*buffer, len(g.args))}, nil
}

type request struct {
	g     *graph
	binds []*buffer
}

func (r *request) Bind(index int, b backend.Buffer) error {
	ob, ok := b.(*buffer)
	if !ok || ob.d != r.g.d {
		return errors.New("openvino: buffer was not allocated by this device")
	}
	r.binds[index] = ob
	return nil
}

// Submit runs the graph to the end; the fence it returns is done.
func (r *request) Submit() (backend.Fence, error) {
	binds := make([][2]any, len(r.binds))
	for i, b := range r.binds {
		if b == nil {
			return nil, fmt.Errorf("openvino: argument %d (%s) is not bound", i, r.g.args[i].Name)
		}
		binds[i] = [2]any{b.path, len(b.b)}
	}
	if err := r.g.d.call(map[string]any{"op": "run", "id": r.g.id, "binds": binds}, nil); err != nil {
		return nil, err
	}
	return done{}, nil
}

func (r *request) Close() error { return nil }

type done struct{}

func (done) Wait(ctx context.Context) error { return nil }
func (done) Done() (bool, error)            { return true, nil }
