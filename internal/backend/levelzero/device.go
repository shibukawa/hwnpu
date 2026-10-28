package levelzero

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/ovir"
	"github.com/shibukawa/hwnpu/internal/power"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/ze"
)

const bufferAlignment = 4096

type device struct {
	api   *ze.API
	drv   *driver
	entry *deviceEntry
	ctx   ze.Handle
	opts  backend.Options
	meter *power.RAPL // nil without a readable energy counter
}

// PowerSample implements backend.PowerMeter with the CPU package energy counter.
func (d *device) PowerSample() (types.PowerSample, error) {
	if d.meter == nil {
		return types.PowerSample{}, fmt.Errorf("levelzero: no readable energy counter: %w", types.ErrUnsupported)
	}
	return d.meter.Sample()
}

// newQueue creates a command queue configured for the device's power policy:
//
//	low:         low priority, background workload type
//	balanced:    normal priority
//	performance: high priority, turbo
func (d *device) newQueue() (ze.Handle, error) {
	o := ze.QueueOptions{Ordinal: d.entry.ordinal, Priority: ze.QueuePriorityNormal}
	var keep any
	switch d.opts.Power {
	case types.PowerLow:
		o.Priority = ze.QueuePriorityLow
	case types.PowerPerformance:
		o.Priority = ze.QueuePriorityHigh
		if d.drv.queueExt != nil {
			o.PNext, keep = d.drv.queueExt.Desc(true)
		}
	}
	q, err := d.api.CreateCommandQueue(d.ctx, d.entry.handle, o)
	runtime.KeepAlive(keep)
	if err != nil {
		return 0, err
	}
	if d.opts.Power == types.PowerLow && d.drv.queueExt != nil {
		if err := d.drv.queueExt.SetWorkloadType(q, ze.WorkloadBackground); err != nil {
			d.api.DestroyCommandQueue(q)
			return 0, err
		}
	}
	return q, nil
}

func (d *device) Close() error {
	if d.ctx == 0 {
		return nil
	}
	err := d.api.DestroyContext(d.ctx)
	d.ctx = 0
	return driverErr(err)
}

// buffer is zeMemAllocHost memory: host memory the NPU can address directly.
type buffer struct {
	d *device
	p unsafe.Pointer
	b []byte
}

func (d *device) AllocBuffer(size int, kind types.BufferKind) (backend.Buffer, error) {
	p, err := d.api.AllocHost(d.ctx, size, bufferAlignment)
	if err != nil {
		return nil, driverErr(err)
	}
	return &buffer{d: d, p: p, b: unsafe.Slice((*byte)(p), size)}, nil
}

func (b *buffer) Bytes() []byte                   { return b.b }
func (b *buffer) SyncToDevice(off, n int) error   { return nil }
func (b *buffer) SyncFromDevice(off, n int) error { return nil }
func (b *buffer) Close() error {
	if b.p == nil {
		return nil
	}
	err := b.d.api.Free(b.d.ctx, b.p)
	b.p, b.b = nil, nil
	return driverErr(err)
}

type graph struct {
	d         *device
	h         ze.Handle
	args      []types.ArgInfo
	placement types.PlacementReport
	cached    bool
	mu        sync.Mutex // SetArgument and append must not interleave between requests
}

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	isIR := src.Format == types.GraphFormatOpenVINOIR || strings.EqualFold(filepath.Ext(src.Path), ".xml")
	if isIR {
		return d.compileIR(src)
	}
	if src.Format != types.GraphFormatNative {
		return nil, fmt.Errorf("levelzero: graph format %d: %w", src.Format, types.ErrUnsupported)
	}
	blob := src.Blob
	if blob == nil {
		if src.Path == "" {
			return nil, errors.New("levelzero: GraphSource needs Blob or Path")
		}
		var err error
		if blob, err = os.ReadFile(src.Path); err != nil {
			return nil, err
		}
	}
	h, err := d.drv.graph.Create(d.ctx, d.entry.handle, ze.GraphFormatNative, blob, "", ze.GraphFlagNone)
	if err != nil {
		return nil, buildErr(err)
	}
	// A native blob is code compiled for the NPU, so every operation runs there.
	return d.finishGraph(h, types.PlacementReport{Verified: true, Method: "levelzero-native-blob"})
}

func (d *device) compileIR(src types.GraphSource) (backend.Graph, error) {
	if d.entry.gprops.GraphFormatsSupported&ze.GraphFormatNGraphLite == 0 {
		return nil, fmt.Errorf("levelzero: the driver has no compiler for OpenVINO IR (install the NPU driver compiler): %w", types.ErrUnsupported)
	}
	xmlData, bin := src.XML, src.Bin
	if xmlData == nil {
		var err error
		if xmlData, err = os.ReadFile(src.Path); err != nil {
			return nil, err
		}
		if bin == nil {
			bin, err = os.ReadFile(strings.TrimSuffix(src.Path, filepath.Ext(src.Path)) + ".bin")
			if err != nil && !os.IsNotExist(err) {
				return nil, err
			}
		}
	}
	model, err := ovir.Parse(xmlData)
	if err != nil {
		return nil, err
	}
	injected, err := ovir.InjectRTInfo(xmlData)
	if err != nil {
		return nil, err
	}
	cv := d.entry.gprops.CompilerVersion
	input := ovir.Serialize(cv.Major, cv.Minor, injected, bin)
	flags := model.IOFlags(ovir.UseIndices(cv.Major, cv.Minor))
	if src.BuildFlags != "" {
		flags += " " + src.BuildFlags
	}

	compute := 0
	for _, l := range model.Layers {
		if !ovir.IsStructural(l.Type) {
			compute++
		}
	}
	rep := types.PlacementReport{Verified: true, Method: "levelzero-compiler", NPUOps: compute}

	var key string
	createFlags := uint32(ze.GraphFlagNone)
	if d.opts.CacheDir != "" {
		key = d.cacheKey(input, flags)
		if blobPath, ok := cache.Lookup(d.opts.CacheDir, key, ".blob"); ok {
			if blob, err := os.ReadFile(blobPath); err == nil {
				if h, err := d.drv.graph.Create(d.ctx, d.entry.handle, ze.GraphFormatNative, blob, "", ze.GraphFlagNone); err == nil {
					g, err := d.finishGraph(h, rep)
					if err == nil {
						g.(*graph).cached = true
					}
					return g, err
				}
			}
			// An unreadable or rejected entry falls through to a fresh compilation.
		}
		createFlags = ze.GraphFlagDisableCache // like OpenVINO: our cache replaces the driver's
	}
	h, err := d.drv.graph.Create(d.ctx, d.entry.handle, ze.GraphFormatNGraphLite, input, flags, createFlags)
	if err != nil {
		// Ask the compiler which layers it supports, to tell unsupported operations apart from
		// other build failures. The OpenVINO plugin passes only the --config part here.
		supported, qerr := d.drv.graph.SupportedLayers(d.ctx, d.entry.handle, input, src.BuildFlags)
		if qerr == nil {
			rep := types.PlacementReport{Verified: true, Method: "levelzero-query-network"}
			for _, l := range model.Layers {
				switch {
				case ovir.IsStructural(l.Type):
				case supported[l.Name]:
					rep.NPUOps++
				default:
					rep.Ops = append(rep.Ops, types.OpPlacement{Name: l.Name, Type: l.Type, Device: "unsupported"})
				}
			}
			if len(rep.Ops) > 0 {
				return nil, &types.PlacementError{Report: rep, Cause: buildErr(err)}
			}
		}
		return nil, buildErr(err)
	}
	if key != "" {
		if blob, err := d.drv.graph.NativeBinary(h); err == nil && len(blob) > 0 {
			// A failure to cache is not a failure to compile.
			cache.Store(d.opts.CacheDir, key, ".blob", func(tmp string) error { return os.WriteFile(tmp, blob, 0o644) })
		}
	}
	// The NPU compiler does not fall back to other devices: a successful build runs entirely
	// on the NPU.
	return d.finishGraph(h, rep)
}

// cacheKey covers the compiler input and everything about the driver that changes its output.
func (d *device) cacheKey(input []byte, flags string) string {
	e := d.entry
	return cache.Key(
		[]byte("hwnpu-levelzero-v1"),
		input,
		[]byte(flags),
		[]byte(fmt.Sprintf("driver=%d graph=%s compiler=%s device=%s/%#x",
			d.drv.props.DriverVersion, ze.VersionString(d.drv.graph.Version), e.gprops.CompilerVersion,
			ze.CString(e.props.Name[:]), e.props.DeviceID)),
	)
}

// finishGraph reads the arguments and performs the initialization stages the driver requires.
func (d *device) finishGraph(h ze.Handle, rep types.PlacementReport) (backend.Graph, error) {
	g := &graph{d: d, h: h, placement: rep}
	fail := func(err error) (backend.Graph, error) {
		d.drv.graph.Destroy(h)
		return nil, err
	}
	n, stages, err := d.drv.graph.Properties(h)
	if err != nil {
		return fail(driverErr(err))
	}
	for i := uint32(0); i < n; i++ {
		p, err := d.drv.graph.ArgumentProperties(h, i)
		if err != nil {
			return fail(driverErr(err))
		}
		a, err := argInfo(int(i), p)
		if err != nil {
			return fail(err)
		}
		g.args = append(g.args, a)
	}
	if stages&ze.StageInitialize != 0 {
		if err := d.drv.graph.Initialize(h); err != nil {
			return fail(driverErr(err))
		}
	}
	if stages&ze.StageCommandListInit != 0 {
		if err := d.runOnce(func(list ze.Handle) error { return d.drv.graph.AppendInitialize(list, h) }); err != nil {
			return fail(driverErr(err))
		}
	}
	return g, nil
}

// runOnce records commands into a temporary command list, executes it and waits.
func (d *device) runOnce(record func(list ze.Handle) error) error {
	q, err := d.newQueue()
	if err != nil {
		return err
	}
	defer d.api.DestroyCommandQueue(q)
	l, err := d.api.CreateCommandList(d.ctx, d.entry.handle, d.entry.ordinal)
	if err != nil {
		return err
	}
	defer d.api.DestroyCommandList(l)
	f, err := d.api.CreateFence(q)
	if err != nil {
		return err
	}
	defer d.api.DestroyFence(f)
	if err := record(l); err != nil {
		return err
	}
	if err := d.api.CloseCommandList(l); err != nil {
		return err
	}
	if err := d.api.Execute(q, l, f); err != nil {
		return err
	}
	_, err = d.api.SyncFence(f, ^uint64(0))
	return err
}

var precisions = map[uint32]types.DType{
	ze.PrecisionFP32: types.F32, ze.PrecisionFP16: types.F16, ze.PrecisionBF16: types.BF16,
	ze.PrecisionFP64: types.F64, ze.PrecisionUINT8: types.U8, ze.PrecisionINT8: types.I8,
	ze.PrecisionINT16: types.I16, ze.PrecisionINT32: types.I32, ze.PrecisionINT64: types.I64,
}

func argInfo(index int, p *ze.GraphArgumentProperties3) (types.ArgInfo, error) {
	a := types.ArgInfo{Index: index, Name: ze.CString(p.Name[:])}
	if p.Type == ze.ArgTypeOutput {
		a.Kind = types.ArgOutput
	}
	n := int(p.DimsCount)
	if n > len(p.Dims) {
		n = len(p.Dims)
	}
	elems := 1
	for _, v := range p.Dims[:n] {
		a.Dims = append(a.Dims, int(v))
		elems *= int(v)
	}
	var ok bool
	if a.DType, ok = precisions[p.DevicePrecision]; !ok {
		return a, fmt.Errorf("levelzero: argument %q has device precision %#x: %w", a.Name, p.DevicePrecision, types.ErrUnsupported)
	}
	a.ModelDType = precisions[p.NetworkPrecision]
	a.Bytes = elems * a.DType.Size()
	return a, nil
}

func (g *graph) Args() []types.ArgInfo            { return g.args }
func (g *graph) Cached() bool                     { return g.cached }
func (g *graph) Placement() types.PlacementReport { return g.placement }

func (g *graph) Close() error {
	if g.h == 0 {
		return nil
	}
	err := g.d.drv.graph.Destroy(g.h)
	g.h = 0
	return driverErr(err)
}

func (g *graph) NewRequest() (backend.Request, error) {
	d := g.d
	r := &request{g: g, binds: make([]*buffer, len(g.args))}
	var err error
	if r.queue, err = d.newQueue(); err != nil {
		return nil, driverErr(err)
	}
	if r.list, err = d.api.CreateCommandList(d.ctx, d.entry.handle, d.entry.ordinal); err != nil {
		r.Close()
		return nil, driverErr(err)
	}
	if r.fence, err = d.api.CreateFence(r.queue); err != nil {
		r.Close()
		return nil, driverErr(err)
	}
	return r, nil
}

type request struct {
	g                  *graph
	binds              []*buffer
	queue, list, fence ze.Handle
	submitted          bool
}

func (r *request) Bind(index int, b backend.Buffer) error {
	lb, ok := b.(*buffer)
	if !ok || lb.d != r.g.d {
		return errors.New("levelzero: buffer was not allocated by this device")
	}
	if r.busy() {
		return errors.New("levelzero: cannot bind while a submission is running")
	}
	r.binds[index] = lb
	return nil
}

func (r *request) busy() bool {
	if !r.submitted {
		return false
	}
	done, _ := r.g.d.api.FenceDone(r.fence)
	return !done
}

// Submit records the graph with the bound arguments and executes it. Argument values are
// captured when the graph is appended, so each submission re-records the command list.
func (r *request) Submit() (backend.Fence, error) {
	if r.busy() {
		return nil, errors.New("levelzero: previous submission is still running")
	}
	for i, b := range r.binds {
		if b == nil {
			return nil, fmt.Errorf("levelzero: argument %d (%s) is not bound", i, r.g.args[i].Name)
		}
	}
	api, gx := r.g.d.api, r.g.d.drv.graph
	err := func() error {
		r.g.mu.Lock()
		defer r.g.mu.Unlock()
		if err := api.ResetCommandList(r.list); err != nil {
			return err
		}
		for i, b := range r.binds {
			if err := gx.SetArgument(r.g.h, uint32(i), b.p); err != nil {
				return err
			}
		}
		if err := gx.AppendExecute(r.list, r.g.h); err != nil {
			return err
		}
		return api.CloseCommandList(r.list)
	}()
	if err != nil {
		return nil, driverErr(err)
	}
	if err := api.ResetFence(r.fence); err != nil {
		return nil, driverErr(err)
	}
	if err := api.Execute(r.queue, r.list, r.fence); err != nil {
		return nil, driverErr(err)
	}
	r.submitted = true
	return &fence{api: api, h: r.fence}, nil
}

func (r *request) Close() error {
	api := r.g.d.api
	if r.submitted {
		api.SyncFence(r.fence, ^uint64(0))
	}
	var errs []error
	if r.fence != 0 {
		errs = append(errs, api.DestroyFence(r.fence))
	}
	if r.list != 0 {
		errs = append(errs, api.DestroyCommandList(r.list))
	}
	if r.queue != 0 {
		errs = append(errs, api.DestroyCommandQueue(r.queue))
	}
	r.fence, r.list, r.queue = 0, 0, 0
	return driverErr(errors.Join(errs...))
}

type fence struct {
	api *ze.API
	h   ze.Handle
}

const pollInterval = 2 * time.Millisecond

// Wait polls the fence so that ctx cancellation is honoured.
func (f *fence) Wait(ctx context.Context) error {
	for {
		done, err := f.api.SyncFence(f.h, uint64(pollInterval))
		if err != nil {
			return driverErr(err)
		}
		if done {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
}

func (f *fence) Done() (bool, error) {
	done, err := f.api.FenceDone(f.h)
	return done, driverErr(err)
}

// buildErr wraps a failed graph creation with its build log.
func buildErr(err error) error {
	var zerr *ze.Error
	if errors.As(err, &zerr) {
		return &types.BuildError{Log: zerr.Log, Err: driverErr(&ze.Error{Op: zerr.Op, Result: zerr.Result})}
	}
	return err
}
