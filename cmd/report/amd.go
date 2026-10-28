package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
)

// amdDesign is an optional IRON passthrough design given on the command line.
var amdDesign struct {
	xclbin, insts, kernel string
	bytes                 int
}

func checkKernelDevice(ctx context.Context, dev *hwnpu.Device, runs int) []Check {
	checks := []Check{bufferRoundTrip(dev), instructionBuffer(dev), powerSensor(dev)}
	if amdDesign.xclbin == "" {
		checks = append(checks, Check{Name: "IRON passthrough kernel", Status: Skip,
			Detail: "pass -xclbin and -insts built from mlir-aie programming_examples/basic/passthrough_dmas to run a kernel"})
		return checks
	}
	return append(checks, safeKernel(ctx, dev, runs))
}

func bufferRoundTrip(dev *hwnpu.Device) Check {
	c := Check{Name: "host buffer: allocate, sync to and from the device"}
	const n = 1 << 20
	start := time.Now()
	b, err := dev.AllocBuffer(n)
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	defer b.Close()
	for i := range b.Bytes() {
		b.Bytes()[i] = byte(i * 31)
	}
	if err := b.SyncToDevice(0, n); err != nil {
		c.Status, c.Detail = Fail, "sync to device: "+err.Error()
		return c
	}
	if err := b.SyncFromDevice(0, n); err != nil {
		c.Status, c.Detail = Fail, "sync from device: "+err.Error()
		return c
	}
	for i, v := range b.Bytes() {
		if v != byte(i*31) {
			c.Status, c.Detail = Fail, fmt.Sprintf("byte %d changed", i)
			return c
		}
	}
	c.Status, c.CompileMS = Pass, ms(time.Since(start))
	c.Detail = "1 MiB BO_SHARE buffer; compile ms column shows the elapsed time"
	return c
}

func instructionBuffer(dev *hwnpu.Device) Check {
	c := Check{Name: "instruction buffer in the device heap"}
	b, err := dev.AllocBuffer(4096, hwnpu.WithBufferKind(hwnpu.BufferInstructions))
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	defer b.Close()
	copy(b.Bytes(), "hwnpu")
	if err := b.SyncToDevice(0, 4096); err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	c.Status = Pass
	return c
}

func powerSensor(dev *hwnpu.Device) Check {
	c := Check{Name: "NPU power sensor"}
	s, err := dev.PowerSample()
	if err != nil {
		c.Status, c.Detail = Skip, err.Error()
		return c
	}
	c.Status, c.AverageW, c.EnergyScope = Pass, s.Watts, string(s.Scope)
	c.Detail = fmt.Sprintf("%.3f W from %s", s.Watts, s.Source)
	return c
}

func safeKernel(ctx context.Context, dev *hwnpu.Device, runs int) (c Check) {
	defer func() {
		if r := recover(); r != nil {
			c = Check{Name: "IRON passthrough kernel", Status: Fail, Detail: fmt.Sprint("panic: ", r)}
		}
	}()
	return runPassthrough(ctx, dev, runs)
}

// readInsts reads insts.bin (little endian words) or insts.txt (one hex word per line).
func readInsts(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if !strings.HasSuffix(path, ".txt") {
		return data, nil
	}
	var out []byte
	for _, line := range strings.Fields(string(data)) {
		v, err := strconv.ParseUint(strings.TrimPrefix(line, "0x"), 16, 32)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		out = binary.LittleEndian.AppendUint32(out, uint32(v))
	}
	return out, nil
}

func runPassthrough(ctx context.Context, dev *hwnpu.Device, runs int) Check {
	c := Check{Name: "IRON passthrough kernel (" + amdDesign.xclbin + ")"}
	fail := func(err error) Check { c.Status, c.Detail = Fail, err.Error(); return c }
	kr, err := dev.Kernel()
	if err != nil {
		return fail(err)
	}
	insts, err := readInsts(amdDesign.insts)
	if err != nil {
		return fail(err)
	}
	start := time.Now()
	prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: amdDesign.xclbin})
	c.CompileMS = ms(time.Since(start))
	if err != nil {
		return fail(err)
	}
	defer prog.Close()
	k, err := prog.Kernel(amdDesign.kernel)
	if err != nil {
		return fail(err)
	}
	for _, a := range k.Args() {
		c.Args = append(c.Args, fmt.Sprintf("%d %s %s offset=%#x size=%d buffer=%v", a.Index, a.Name, a.Type, a.Offset, a.Size, a.Buffer))
	}
	n := amdDesign.bytes
	instr, err := dev.AllocBuffer(len(insts), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
	if err != nil {
		return fail(err)
	}
	defer instr.Close()
	bufs := make([]*hwnpu.Buffer, 3)
	for i := range bufs {
		if bufs[i], err = dev.AllocBuffer(n); err != nil {
			return fail(err)
		}
		defer bufs[i].Close()
	}
	copy(instr.Bytes(), insts)
	for i := range bufs[0].Bytes() {
		bufs[0].Bytes()[i] = byte(i*13 + 1)
	}
	instr.SyncToDevice(0, len(insts))
	bufs[0].SyncToDevice(0, n)
	bufs[2].SyncToDevice(0, n)

	run, err := k.NewRun()
	if err != nil {
		return fail(err)
	}
	defer run.Close()
	run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4)).
		SetBuffer(3, bufs[0]).SetBuffer(4, bufs[1]).SetBuffer(5, bufs[2])
	start = time.Now()
	if err := run.Run(ctx); err != nil {
		return fail(err)
	}
	c.FirstRunMS = ms(time.Since(start))
	bufs[2].SyncFromDevice(0, n)
	if !bytes.Equal(bufs[0].Bytes(), bufs[2].Bytes()) {
		i := 0
		for i < n && bufs[0].Bytes()[i] == bufs[2].Bytes()[i] {
			i++
		}
		return fail(fmt.Errorf("output differs from input at byte %d", i))
	}
	times := make([]float64, 0, runs)
	before, meterErr := dev.PowerSample()
	for i := 0; i < runs; i++ {
		start := time.Now()
		if err := run.Run(ctx); err != nil {
			return fail(fmt.Errorf("run %d: %w", i+2, err))
		}
		times = append(times, ms(time.Since(start)))
	}
	if after, err := dev.PowerSample(); meterErr == nil && err == nil && len(times) > 0 {
		c.EnergyPerRunMJ = (after.EnergyJ - before.EnergyJ) / float64(len(times)) * 1000
		c.AverageW = hwnpu.AveragePower(before, after)
		c.EnergyScope = string(after.Scope)
	}
	if len(times) > 0 {
		slices.Sort(times)
		c.Runs, c.MinMS, c.MedianMS = len(times), times[0], times[len(times)/2]
	}
	c.Status = Pass
	return c
}
