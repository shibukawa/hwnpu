// Package ovcpu runs an OpenVINO IR with OpenVINO's CPU plugin through tools/ov/run_ir.py, so
// that the IR hwnpu writes for Intel's NPU (internal/ovlower) can be checked on any machine,
// macOS included. It is for tests: HWNPU_OV_PYTHON names a Python interpreter that has the
// openvino and numpy packages, and the tests that need it are skipped without it.
package ovcpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Python is the interpreter from HWNPU_OV_PYTHON, or "" when it is not set.
func Python() string { return os.Getenv("HWNPU_OV_PYTHON") }

func script() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "tools", "ov", "run_ir.py")
}

// Options are the settings of a run.
type Options struct {
	// Precision is the CPU plugin's inference precision: "f32" (the default) or "f16".
	Precision string
}

// Run computes the outputs of the IR at xml (its .bin next to it) from float32 inputs by name,
// and returns them by name as float32.
func Run(python, xml string, inputs map[string][]float32, opts Options) (map[string][]float32, error) {
	dir, err := os.MkdirTemp("", "ovcpu")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	prec := opts.Precision
	if prec == "" {
		prec = "f32"
	}
	args := []string{"-I", script(), "--precision", prec, xml, filepath.Join(dir, "out")}
	for name, v := range inputs {
		p := filepath.Join(dir, "in."+name+".f32")
		if err := writeF32(p, v); err != nil {
			return nil, err
		}
		args = append(args, name+"="+p)
	}
	out, err := exec.Command(python, args...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("ovcpu: %v: %s", err, out)
	}
	files, err := filepath.Glob(filepath.Join(dir, "out", "*.f32"))
	if err != nil {
		return nil, err
	}
	res := map[string][]float32{}
	for _, f := range files {
		v, err := readF32(f)
		if err != nil {
			return nil, err
		}
		res[strings.TrimSuffix(filepath.Base(f), ".f32")] = v
	}
	return res, nil
}

func writeF32(path string, v []float32) error {
	raw := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(x))
	}
	return os.WriteFile(path, raw, 0o644)
}

func readF32(path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	v := make([]float32, len(raw)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return v, nil
}
