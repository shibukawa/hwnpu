package iron

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"syscall"
	"time"
)

// The version checking drafts (tools/amd/iron_verify.py, iron_export_lm.py --verify): a call of
// VerifyRows tokens from start (the next token and the drafts after it) that runs as a prompt
// chunk does, but returns the logits of every row, and runs its linears through the decode
// step's GEMV for all the rows at once. A prompt chunk costs the chunk's time whatever its rows
// (a 2-token call 4 decode steps on Qwen3.5-0.8B); this one costs about a decode step.
//
// The recurrent states (Qwen3.5's Gated DeltaNet) run the decode step's operator a row at a
// time; each layer keeps the state after each row but the last in its "rows_state" buffer, and
// Keep writes back the one after the last row kept.

// rowsState is the suffix of a recurrent layer's buffer of the states after each of a check's
// rows but the last.
const rowsState = ".rows_state"

// VerifyRows is the rows a call checking drafts runs (Verify), 0 for an export without the
// version.
func (m *Model) VerifyRows() int {
	if m.verify == nil || len(m.verify.m.Inputs) == 0 {
		return 0
	}
	return m.verify.m.Inputs[0].Shape[0]
}

// Verify runs ids (VerifyRows of them) from position start and returns the logits after each.
// Every row's keys and values and the recurrent states after the last row are written, as a
// prompt chunk's; Keep then puts back the states after fewer of them. The call must end within
// the prompt chunk it starts in (the recurrent states' slot is the chunk's): start % chunk +
// VerifyRows <= chunk.
func (m *Model) Verify(ctx context.Context, ids []int32, start int) ([][]float32, error) {
	if m.released {
		return nil, errReleased
	}
	v := m.verify
	R := m.VerifyRows()
	if R == 0 {
		return nil, errors.New("iron: the export has no version checking drafts (export it with --verify)")
	}
	if len(ids) != R {
		return nil, fmt.Errorf("iron: %d tokens to check, the version takes %d", len(ids), R)
	}
	C := m.Manifest.Config.PrefillChunk
	if start < 0 || start+R > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: %d tokens from %d do not fit the cache of %d", R, start, m.Manifest.Config.MaxSeqLen)
	}
	if len(m.Manifest.Checkpoints) > 0 && start%C+R > C {
		return nil, fmt.Errorf("iron: a check from %d crosses the chunk boundary at %d", start, (start/C+1)*C)
	}
	values, err := m.complete(v, callValues(ids[R-1], start+R-1, start, R))
	if err != nil {
		return nil, err
	}
	for _, in := range v.m.Inputs {
		if in.DType != "int32" {
			return nil, fmt.Errorf("iron: input %s of %s is not token ids", in.Name, in.DType)
		}
		if err := m.writeInput(v, in.Name, int32Bytes(ids)); err != nil {
			return nil, err
		}
	}
	for _, e := range v.m.Encoders {
		if err := m.encode(v, &e, ids, values); err != nil {
			return nil, err
		}
	}
	if err := m.carry(v, values); err != nil {
		return nil, err
	}
	if err := m.fillHostTables(ids); err != nil {
		return nil, err
	}
	// as before a prompt chunk: an empty command in the decode step's context (chunkBarrier)
	if err := m.chunkBarrier(ctx); err != nil {
		return nil, err
	}
	// The runs kept, as the decode step's, when their control code fits the device heap beside
	// the decode step's (Qwen3.5-0.8B: 26 and 30 MB of ELF, 64 MiB of heap); else made for each
	// call, as a prompt chunk's, the decode step's dropped when even one does not fit.
	if v.runs == nil && !v.perCall {
		for k := range v.m.Kernels {
			run, err := m.newRun(v, k)
			if err != nil {
				v.closeRuns()
				if !errors.Is(err, syscall.ENOSPC) {
					return nil, fmt.Errorf("iron: verify command %d: %w", k, err)
				}
				v.perCall = true
				m.logf("verify: its commands do not stay in the device heap beside the decode step's: made for each call")
				break
			}
			v.runs = append(v.runs, run)
		}
		if len(v.runs) > 0 {
			v.run = v.runs[0]
		}
	}
	if !v.perCall {
		if err := m.runCommands(ctx, v, values); err != nil {
			return nil, err
		}
		return m.rowLogits(v, R)
	}
	timing := os.Getenv("HWNPU_IRON_TIMING") != ""
	for k := range v.m.Kernels {
		t0 := time.Now()
		run, err := m.newRun(v, k)
		if errors.Is(err, syscall.ENOSPC) && m.decode.run != nil {
			m.dropDecode()
			run, err = m.newRun(v, k)
		}
		if err != nil {
			return nil, fmt.Errorf("iron: verify command %d: %w", k, err)
		}
		t1 := time.Now()
		err = writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, values)
		if err == nil {
			err = run.Run(ctx)
		}
		t2 := time.Now()
		run.Close()
		if timing {
			fmt.Fprintf(os.Stderr, "verify command %d: setup %.1f ms, run %.1f ms, close %.1f ms\n", k,
				t1.Sub(t0).Seconds()*1000, t2.Sub(t1).Seconds()*1000, time.Since(t2).Seconds()*1000)
		}
		if err != nil {
			return nil, fmt.Errorf("iron: verify command %d: %w", k, err)
		}
	}
	return m.rowLogits(v, R)
}

// rowLogits reads the version's logits output of rows rows.
func (m *Model) rowLogits(v *version, rows int) ([][]float32, error) {
	if len(v.m.Outputs) != 1 || v.m.Outputs[0].DType != "bfloat16" {
		return nil, errors.New("iron: the version's output is not bfloat16 logits")
	}
	o := v.m.Outputs[0]
	buf := m.buffer(v, o.Argument)
	if err := buf.SyncFromDevice(o.Offset, o.Bytes); err != nil {
		return nil, err
	}
	per := o.Bytes / 2 / rows
	n := min(per, m.Manifest.Config.VocabSize)
	out := make([][]float32, rows)
	for r := range out {
		b := buf.Bytes()[o.Offset+2*r*per:]
		row := make([]float32, n)
		for i := range row {
			row[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		}
		out[r] = row
	}
	return out, nil
}

// Keep makes the recurrent states those after the first kept rows of the check Verify ran
// from start (1 <= kept <= VerifyRows): the states after its last row are written already, so
// kept < VerifyRows copies each layer's state after row kept-1 into the slot the call wrote.
// The keys and values of the rows past kept stay in the caches, unread: the next call writes
// those positions again before it reads them.
func (m *Model) Keep(start, kept int) error {
	if m.released {
		return errReleased
	}
	R := m.VerifyRows()
	if kept < 1 || kept > R {
		return fmt.Errorf("iron: %d rows kept of a check of %d", kept, R)
	}
	if kept == R {
		return nil
	}
	v := m.verify
	for _, c := range m.Manifest.Checkpoints {
		layer, ok := strings.CutSuffix(c.Buffer, ".state.slots")
		if !ok {
			return fmt.Errorf("iron: recurrent state %s: no rows of a check (only Gated DeltaNet's states)", c.Buffer)
		}
		rp, ok := v.m.Buffers[layer+rowsState]
		if !ok {
			if strings.Contains(c.Buffer, ".conv.") {
				continue // a prompt chunk's convolution state (conv_pairs): the check's is in the GDN state
			}
			return fmt.Errorf("iron: no buffer %s%s for the recurrent state %s", layer, rowsState, c.Buffer)
		}
		p := m.decode.m.Buffers[c.Buffer]
		size := p.Bytes / c.Slots
		if rp.Bytes != (R-1)*size {
			return fmt.Errorf("iron: %s%s of %d bytes, %d states of %d", layer, rowsState, rp.Bytes, R-1, size)
		}
		src := m.buffer(v, rp.Argument)
		so := rp.Offset + (kept-1)*size
		if err := src.SyncFromDevice(so, size); err != nil {
			return err
		}
		// the slot the call wrote: its last position's (IRON's Checkpoints.at)
		dst := m.buffer(m.decode, p.Argument)
		do := p.Offset + ((start+R-1)/c.Chunk+1)*size
		copy(dst.Bytes()[do:do+size], src.Bytes()[so:so+size])
		if err := dst.SyncToDevice(do, size); err != nil {
			return err
		}
	}
	return nil
}
