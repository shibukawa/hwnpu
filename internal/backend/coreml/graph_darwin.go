package coreml

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/ebitengine/purego/objc"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/types"
)

// MLFeatureType and MLMultiArrayDataType values.
const (
	featureTypeMultiArray = 5

	dataTypeFloat16 = 0x10000 | 16
	dataTypeFloat32 = 0x10000 | 32
	dataTypeFloat64 = 0x10000 | 64
	dataTypeInt32   = 0x20000 | 32
	dataTypeInt8    = 0x20000 | 8
)

func dtypeOf(dt int64) types.DType {
	switch dt {
	case dataTypeFloat16:
		return types.F16
	case dataTypeFloat32:
		return types.F32
	case dataTypeFloat64:
		return types.F64
	case dataTypeInt32:
		return types.I32
	case dataTypeInt8:
		return types.I8
	}
	return types.DTypeUnknown
}

func mlDataType(d types.DType) int64 {
	switch d {
	case types.F16:
		return dataTypeFloat16
	case types.F32:
		return dataTypeFloat32
	case types.F64:
		return dataTypeFloat64
	case types.I32:
		return dataTypeInt32
	case types.I8:
		return dataTypeInt8
	}
	return 0
}

type graph struct {
	model       objc.ID // retained MLModel
	args        []types.ArgInfo
	placement   types.PlacementReport
	compiledDir string // removed on Close when non-empty
	cached      bool
}

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	if src.Format != types.GraphFormatNative || src.Path == "" {
		return nil, fmt.Errorf("coreml: only GraphFormatNative with Path (.mlpackage, .mlmodel or .mlmodelc) is supported: %w", ErrUnsupported)
	}
	if _, err := os.Stat(src.Path); err != nil {
		return nil, fmt.Errorf("coreml: %w", err)
	}
	g := &graph{}
	compiled := src.Path
	if !strings.EqualFold(filepath.Ext(src.Path), ".mlmodelc") {
		var err error
		if compiled, g.cached, g.compiledDir, err = d.compiledModel(src.Path); err != nil {
			return nil, err
		}
	}

	var config objc.ID
	err := withPool(func() error {
		config = send(send(class("MLModelConfiguration"), "alloc"), "init")
		// HWNPU_COREML_UNITS=cpu|gpu loads the model on the CPU or the GPU instead, as the
		// baselines cmd/powerbench compares the Neural Engine with. It is not part of the API.
		units := computeUnitsCPUAndNeuralEngine
		switch os.Getenv("HWNPU_COREML_UNITS") {
		case "cpu":
			units = computeUnitsCPUOnly
		case "gpu":
			units = computeUnitsCPUAndGPU
		}
		send(config, "setComputeUnits:", int64(units))
		var e objc.ID
		model := send(class("MLModel"), "modelWithContentsOfURL:configuration:error:", fileURL(compiled), config, &e)
		if model == 0 {
			return nsError("load "+src.Path, e)
		}
		g.model = retain(model)
		args, err := describeArgs(model)
		g.args = args
		return err
	})
	if err == nil {
		var url objc.ID
		_ = withPool(func() error { url = retain(fileURL(compiled)); return nil })
		g.placement, err = computePlacement(ctx, url, config)
		_ = withPool(func() error { release(url); return nil })
	}
	_ = withPool(func() error { release(config); return nil })
	if err != nil {
		g.Close()
		return nil, err
	}
	return g, nil
}

// compiledModel returns a compiled .mlmodelc for a source model. With a cache directory the
// result is looked up and stored there; otherwise it is a temporary directory owned by the graph.
func (d *device) compiledModel(path string) (compiled string, cached bool, owned string, err error) {
	var key string
	if d.opts.CacheDir != "" {
		digest, err := cache.TreeDigest(path)
		if err != nil {
			return "", false, "", fmt.Errorf("coreml: %w", err)
		}
		var osVersion string
		_ = withPool(func() error {
			osVersion = goString(send(send(class("NSProcessInfo"), "processInfo"), "operatingSystemVersionString"))
			return nil
		})
		key = cache.Key([]byte("hwnpu-coreml-v1"), digest, []byte(osVersion))
		if p, ok := cache.Lookup(d.opts.CacheDir, key, ".mlmodelc"); ok {
			return p, true, "", nil
		}
	}
	var tmp string
	err = withPool(func() error {
		var e objc.ID
		url := send(class("MLModel"), "compileModelAtURL:error:", fileURL(path), &e)
		if url == 0 {
			return nsError("compile "+path, e)
		}
		tmp = goString(send(url, "path"))
		return nil
	})
	if err != nil {
		return "", false, "", err
	}
	if key == "" {
		return tmp, false, tmp, nil
	}
	final, err := cache.Store(d.opts.CacheDir, key, ".mlmodelc", func(dst string) error {
		if os.Rename(tmp, dst) == nil {
			return nil
		}
		return cache.CopyTree(tmp, dst) // the temporary directory may be on another volume
	})
	if err != nil {
		return tmp, false, tmp, nil // caching is best effort
	}
	os.RemoveAll(tmp)
	return final, false, "", nil
}

// describeArgs lists inputs then outputs, each sorted by name, because Core ML model
// descriptions are dictionaries without an order.
func describeArgs(model objc.ID) ([]types.ArgInfo, error) {
	desc := send(model, "modelDescription")
	var args []types.ArgInfo
	for _, part := range []struct {
		sel  string
		kind types.ArgKind
	}{{"inputDescriptionsByName", types.ArgInput}, {"outputDescriptionsByName", types.ArgOutput}} {
		names, descs := dictItems(send(desc, part.sel))
		byName := map[string]objc.ID{}
		for i, n := range names {
			byName[n] = descs[i]
		}
		sort.Strings(names)
		for _, n := range names {
			fd := byName[n]
			if t := sendInt(fd, "type"); t != featureTypeMultiArray {
				return nil, fmt.Errorf("coreml: %s %q has feature type %d; only multi arrays are supported: %w", part.kind, n, t, ErrUnsupported)
			}
			c := send(fd, "multiArrayConstraint")
			dims := intsOfArray(send(c, "shape"))
			dt := dtypeOf(sendInt(c, "dataType"))
			if dt == types.DTypeUnknown {
				return nil, fmt.Errorf("coreml: %s %q has unsupported data type %d: %w", part.kind, n, sendInt(c, "dataType"), ErrUnsupported)
			}
			bytes := dt.Size()
			for _, v := range dims {
				bytes *= v
			}
			args = append(args, types.ArgInfo{Index: len(args), Kind: part.kind, Name: n, Dims: dims, DType: dt, ModelDType: dt, Bytes: bytes})
		}
	}
	return applyBatchPadding(args, creatorMetadata(desc))
}

// Metadata keys written by tools/coreml/ane_pad.py.
const (
	metaLogicalBatch = "hwnpu.logical_batch"
	metaPaddedBatch  = "hwnpu.padded_batch"
)

// creatorMetadata returns the model's user defined metadata (coremltools user_defined_metadata).
func creatorMetadata(desc objc.ID) map[string]string {
	out := map[string]string{}
	if creatorDefinedKey == 0 {
		return out
	}
	meta := send(desc, "metadata")
	if meta == 0 {
		return out
	}
	user := send(meta, "objectForKey:", creatorDefinedKey)
	if user == 0 {
		return out
	}
	keys, values := dictItems(user)
	for i, k := range keys {
		out[k] = goString(values[i])
	}
	return out
}

// applyBatchPadding turns the physical (padded) shapes into the logical shapes callers use.
func applyBatchPadding(args []types.ArgInfo, meta map[string]string) ([]types.ArgInfo, error) {
	ls, ps := meta[metaLogicalBatch], meta[metaPaddedBatch]
	if ls == "" && ps == "" {
		return args, nil
	}
	logical, err1 := strconv.Atoi(ls)
	padded, err2 := strconv.Atoi(ps)
	if err1 != nil || err2 != nil || logical <= 0 || padded < logical {
		return nil, fmt.Errorf("coreml: invalid batch padding metadata %s=%q %s=%q", metaLogicalBatch, ls, metaPaddedBatch, ps)
	}
	for i := range args {
		a := &args[i]
		if len(a.Dims) == 0 || a.Dims[0] != padded {
			return nil, fmt.Errorf("coreml: %s %q has shape %v, but the model says it was padded to batch %d", a.Kind, a.Name, a.Dims, padded)
		}
		a.PaddedDims = a.Dims
		a.Dims = append([]int{logical}, a.Dims[1:]...)
		a.Bytes = a.Bytes / padded * logical
	}
	return args, nil
}

func (g *graph) Args() []types.ArgInfo            { return g.args }
func (g *graph) Cached() bool                     { return g.cached }
func (g *graph) Placement() types.PlacementReport { return g.placement }

func (g *graph) NewRequest() (backend.Request, error) {
	r := &request{g: g, binds: make([]*buffer, len(g.args)), padded: make([]*buffer, len(g.args))}
	for i, a := range g.args {
		if a.PaddedDims == nil {
			continue
		}
		size := a.DType.Size()
		for _, v := range a.PaddedDims {
			size *= v
		}
		b, err := (&device{}).AllocBuffer(size, types.BufferDefault) // zero filled; rows past Dims[0] stay zero
		if err != nil {
			r.Close()
			return nil, err
		}
		r.padded[i] = b.(*buffer)
	}
	return r, nil
}

func (g *graph) Close() error {
	if g.model != 0 {
		_ = withPool(func() error { release(g.model); return nil })
		g.model = 0
	}
	if g.compiledDir != "" {
		err := os.RemoveAll(g.compiledDir)
		g.compiledDir = ""
		return err
	}
	return nil
}
