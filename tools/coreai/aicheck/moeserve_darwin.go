package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/backend"
	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
	"github.com/shibukawa/bonsaiai/internal/imagegen/milpart"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
	"github.com/shibukawa/bonsaiai/internal/safetensors"
	"github.com/shibukawa/bonsaiai/internal/types"
)

// moeServe runs a real checkpoint's mixture of experts layers on the Neural Engine for a driver
// that runs the rest of the model (tools/ref/lfm2moe_ane.py): LFM2-MoE's (Lfm2MoeForCausalLM)
// router and experts, every MoE layer's experts held in int8 in Core AI's input arrays (gate|up
// (H, 2I) and down (H, I), a scale a row), the chosen ones bound to one program shared by the
// layers, the scales picked from the layer's table by gather_nd on the Neural Engine.
//
// The protocol on stdin and stdout, little endian: a request is uint32 layer, uint32 rows, then
// rows x hidden float32 (the ffn_norm's output); the answer rows x hidden float32 (the block's
// output), then rows x K int32 (the experts chosen). Layer 0xFFFFFFFF asks for the times on
// stderr. Ready is written to stderr once the experts are held.
//
//	aicheck moeserve CHECKPOINT_DIR
func moeServe(argv []string) error {
	fs := flag.NewFlagSet("moeserve", flag.ExitOnError)
	bits := fs.Int("bits", 8, "the experts' bits: 4 rounds them to int4 first (per -group columns, the clipping searched for the least error) and holds those values in int8, for the 4-bit experts' accuracy")
	group := fs.Int("group", 0, "with -bits 4: columns a scale (0: a row)")
	fs.Parse(argv)
	if fs.NArg() != 1 {
		return fmt.Errorf("moeserve: CHECKPOINT_DIR")
	}
	ckpt := fs.Arg(0)
	var cfg struct {
		Hidden       int      `json:"hidden_size"`
		Inter        int      `json:"moe_intermediate_size"`
		Experts      int      `json:"num_experts"`
		TopK         int      `json:"num_experts_per_tok"`
		Layers       int      `json:"num_hidden_layers"`
		Dense        int      `json:"num_dense_layers"`
		NormTopK     bool     `json:"norm_topk_prob"`
		Scaling      float32  `json:"routed_scaling_factor"`
		ExpertBias   bool     `json:"use_expert_bias"`
		Architecture []string `json:"architectures"`
	}
	raw, err := os.ReadFile(filepath.Join(ckpt, "config.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return err
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	raw, err = os.ReadFile(filepath.Join(ckpt, "model.safetensors.index.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		return err
	}
	files := map[string]*safetensors.File{}
	for _, f := range index.WeightMap {
		if files[f] == nil {
			if files[f], err = safetensors.Open(filepath.Join(ckpt, f)); err != nil {
				return err
			}
		}
	}
	tensor := func(name string) ([]float32, error) {
		f, ok := files[index.WeightMap[name]]
		if !ok {
			return nil, fmt.Errorf("moeserve: no tensor %s", name)
		}
		return f.Float32(name)
	}
	H, I, E, K := cfg.Hidden, cfg.Inter, cfg.Experts, cfg.TopK
	S := 2*I + H
	moeLayers := []int{}
	for l := cfg.Dense; l < cfg.Layers; l++ {
		moeLayers = append(moeLayers, l)
	}
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-moeserve-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	compile := func(name string, body func(g *milpart.Graph)) (backend.Graph, error) {
		path := filepath.Join(dir, name+".mlpackage")
		if _, err := llm.WritePart(path, &milpart.Part{Name: name, Model: "moe", Body: body}, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
			return nil, err
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: path})
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(os.Stderr, "moeserve: %s ane=%v %v\n", name, g.Placement().OnNPU(), g.Placement().Ops)
		return g, nil
	}
	cast := func(g *milpart.Graph, v *mlpackage.Var) *mlpackage.Var {
		return g.B.Op("cast", map[string]any{"x": v, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, v.Shape...)
	}
	mm := func(g *milpart.Graph, x, y *mlpackage.Var, ty bool, shape ...int) *mlpackage.Var {
		return g.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(ty)}, shape...)
	}
	gExp, err := compile("experts", func(g *milpart.Graph) {
		x := g.Input("x", 1, 1, H)
		tbl := g.Input("S", E, S)
		sc := g.Op("gather_nd", map[string]any{"x": tbl, "indices": idx16(g, "i", K, 1), "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, K, S)
		var acc *mlpackage.Var
		for k := 0; k < K; k++ {
			s := g.Reshape(g.Slice(sc, 0, k, k+1), 1, 1, S)
			h := g.Mul(mm(g, x, cast(g, g.InputI8(fmt.Sprintf("w%d", k), 1, H, 2*I)), false, 1, 1, 2*I), g.Slice(s, 2, 0, 2*I))
			a := g.Mul(g.SiLU(g.Slice(h, 2, 0, I)), g.Slice(h, 2, I, 2*I))
			d := g.Mul(mm(g, a, cast(g, g.InputI8(fmt.Sprintf("d%d", k), 1, H, I)), true, 1, 1, H), g.Slice(s, 2, 2*I, S))
			y := g.Mul(d, g.Input(fmt.Sprintf("r%d", k), 1, 1, 1))
			if acc == nil {
				acc = y
			} else {
				acc = g.Add(acc, y)
			}
		}
		g.Output(acc)
	})
	if err != nil {
		return err
	}
	gRouter, err := compile("router", func(g *milpart.Graph) { // the router's weight (H, E) bound a layer
		g.Output(mm(g, g.Input("x", 1, 1, H), g.Input("wr", 1, H, E), false, 1, 1, E))
	})
	if err != nil {
		return err
	}
	argsOf := func(g backend.Graph) map[string]types.ArgInfo {
		m := map[string]types.ArgInfo{}
		for _, a := range g.Args() {
			m[a.Name] = a
		}
		return m
	}
	ea, ra := argsOf(gExp), argsOf(gRouter)
	alloc := func(g backend.Graph, a types.ArgInfo) (backend.Buffer, error) {
		if ia, ok := g.(backend.InputAllocator); ok && a.Kind == types.ArgInput {
			if b, err := ia.AllocInput(a.Index); err != nil || b != nil {
				return b, err
			}
		}
		return dev.AllocBuffer(a.Bytes, types.BufferKind(0))
	}

	// the experts, quantized: int8 a row (absmax / 127), the scale in fp16
	type layer struct {
		gu, dn []backend.Buffer
		tbl    backend.Buffer // (E, S) scales
		wr     backend.Buffer // the router's (H, E)
		bias   []float32
	}
	quant := func(w []float32, rows, cols int, dst []byte, scale []float32) {
		if *bits < 8 {
			w = round4(w, rows, cols, *bits, *group)
		}
		for r := 0; r < rows; r++ {
			row := w[r*cols : (r+1)*cols]
			var m float32
			for _, v := range row {
				m = max(m, float32(math.Abs(float64(v))))
			}
			s := m / 127
			if s == 0 {
				s = 1
			}
			// the scale as the device sees it (fp16), the codes rounded against it
			hb := make([]byte, 2)
			bonsaiai.EncodeF16(hb, []float32{s})
			sv := make([]float32, 1)
			bonsaiai.DecodeF16(sv, hb)
			s = sv[0]
			scale[r] = s
			for c, v := range row {
				q := math.Round(float64(v / s))
				dst[r*cols+c] = byte(int8(max(-127, min(127, q))))
			}
		}
	}
	t := time.Now()
	held := map[int]*layer{}
	var mu sync.Mutex
	var firstErr error
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	for _, l := range moeLayers {
		L := &layer{gu: make([]backend.Buffer, E), dn: make([]backend.Buffer, E)}
		if L.tbl, err = alloc(gExp, ea["S"]); err != nil {
			return err
		}
		if L.wr, err = alloc(gRouter, ra["wr"]); err != nil {
			return err
		}
		pre := fmt.Sprintf("model.layers.%d.feed_forward.", l)
		gate, err := tensor(pre + "gate.weight") // (E, H)
		if err != nil {
			return err
		}
		wt := make([]float32, H*E)
		for e := 0; e < E; e++ {
			for h := 0; h < H; h++ {
				wt[h*E+e] = gate[e*H+h]
			}
		}
		bonsaiai.EncodeF16(L.wr.Bytes(), wt)
		if cfg.ExpertBias {
			if L.bias, err = tensor(pre + "expert_bias"); err != nil {
				return err
			}
		} else {
			L.bias = make([]float32, E)
		}
		tblv := make([]float32, E*S)
		for e := 0; e < E; e++ {
			if L.gu[e], err = alloc(gExp, ea["w0"]); err != nil {
				return err
			}
			if L.dn[e], err = alloc(gExp, ea["d0"]); err != nil {
				return err
			}
			wg.Add(1)
			sem <- struct{}{}
			go func(e int) {
				defer func() { <-sem; wg.Done() }()
				fail := func(err error) {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
				}
				ep := fmt.Sprintf("%sexperts.%d.", pre, e)
				w1, err := tensor(ep + "w1.weight") // (I, H) gate
				if err != nil {
					fail(err)
					return
				}
				w3, err := tensor(ep + "w3.weight") // (I, H) up
				if err != nil {
					fail(err)
					return
				}
				w2, err := tensor(ep + "w2.weight") // (H, I) down
				if err != nil {
					fail(err)
					return
				}
				q := make([]byte, 2*I*H)
				sc := tblv[e*S : (e+1)*S]
				quant(append(w1, w3...), 2*I, H, q, sc[:2*I]) // (2I, H), a row = a column of gu
				gb := L.gu[e].Bytes()                         // (H, 2I)
				for j := 0; j < 2*I; j++ {
					for h := 0; h < H; h++ {
						gb[h*2*I+j] = q[j*H+h]
					}
				}
				quant(w2, H, I, L.dn[e].Bytes(), sc[2*I:]) // (H, I) as it is: down transposed
			}(e)
		}
		wg.Wait()
		if firstErr != nil {
			return firstErr
		}
		bonsaiai.EncodeF16(L.tbl.Bytes(), tblv)
		held[l] = L
		fmt.Fprintf(os.Stderr, "moeserve: layer %d held (%.0fs)\n", l, time.Since(t).Seconds())
	}

	// the programs' own inputs and outputs
	reqE, _ := gExp.NewRequest()
	reqR, _ := gRouter.NewRequest()
	bufE := map[string]backend.Buffer{}
	for _, n := range append([]string{"x", "i", "y"}, func() []string {
		var r []string
		for k := 0; k < K; k++ {
			r = append(r, fmt.Sprintf("r%d", k))
		}
		return r
	}()...) {
		b, err := alloc(gExp, ea[n])
		if err != nil {
			return err
		}
		bufE[n] = b
		reqE.Bind(ea[n].Index, b)
	}
	rx, _ := alloc(gRouter, ra["x"])
	ry, _ := alloc(gRouter, ra["y"])
	reqR.Bind(ra["x"].Index, rx)
	reqR.Bind(ra["y"].Index, ry)
	run := func(r backend.Request) error {
		f, err := r.Submit()
		if err != nil {
			return err
		}
		return f.Wait(ctx)
	}
	fmt.Fprintf(os.Stderr, "moeserve: ready, %d MoE layers held in %.0fs\n", len(moeLayers), time.Since(t).Seconds())

	in := bufio.NewReaderSize(os.Stdin, 1<<20)
	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	var nRows int
	var tRouter, tExperts time.Duration
	logits := make([]float32, E)
	xh := make([]byte, 2*H)
	yv := make([]float32, H)
	for {
		var hdr [2]uint32
		if err := binary.Read(in, binary.LittleEndian, &hdr); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if hdr[0] == 0xFFFFFFFF {
			if nRows > 0 {
				fmt.Fprintf(os.Stderr, "moeserve: %d layer rows, a row: router %.3f ms, experts %.3f ms\n", nRows,
					float64(tRouter.Microseconds())/1e3/float64(nRows), float64(tExperts.Microseconds())/1e3/float64(nRows))
			}
			nRows, tRouter, tExperts = 0, 0, 0
			continue
		}
		L := held[int(hdr[0])]
		if L == nil {
			return fmt.Errorf("moeserve: layer %d is not a MoE layer", hdr[0])
		}
		rows := int(hdr[1])
		x := make([]float32, rows*H)
		if err := binary.Read(in, binary.LittleEndian, x); err != nil {
			return err
		}
		chosenAll := make([]int32, rows*K)
		for r := 0; r < rows; r++ {
			bonsaiai.EncodeF16(xh, x[r*H:(r+1)*H])
			t := time.Now()
			copy(rx.Bytes(), xh)
			reqR.Bind(ra["wr"].Index, L.wr)
			if err := run(reqR); err != nil {
				return err
			}
			bonsaiai.DecodeF16(logits, ry.Bytes())
			// sigmoid, the bias for the choice only, the chosen weights normalized
			sig := make([]float32, E)
			score := make([]float32, E)
			idx := make([]int, E)
			for e := range sig {
				sig[e] = 1 / (1 + float32(math.Exp(float64(-logits[e]))))
				score[e] = sig[e] + L.bias[e]
				idx[e] = e
			}
			sort.SliceStable(idx, func(a, b int) bool { return score[idx[a]] > score[idx[b]] })
			chosen := idx[:K]
			var sum float32
			for _, e := range chosen {
				sum += sig[e]
			}
			fi := make([]float32, K)
			for k, e := range chosen {
				w := sig[e]
				if cfg.NormTopK {
					w /= sum + 1e-6
				}
				if cfg.Scaling != 0 {
					w *= cfg.Scaling
				}
				bonsaiai.EncodeF16(bufE[fmt.Sprintf("r%d", k)].Bytes(), []float32{w})
				reqE.Bind(ea[fmt.Sprintf("w%d", k)].Index, L.gu[e])
				reqE.Bind(ea[fmt.Sprintf("d%d", k)].Index, L.dn[e])
				fi[k] = float32(e)
				chosenAll[r*K+k] = int32(e)
			}
			reqE.Bind(ea["S"].Index, L.tbl)
			bonsaiai.EncodeF16(bufE["i"].Bytes(), fi)
			copy(bufE["x"].Bytes(), xh)
			t2 := time.Now()
			tRouter += t2.Sub(t)
			if err := run(reqE); err != nil {
				return err
			}
			bonsaiai.DecodeF16(yv, bufE["y"].Bytes())
			tExperts += time.Since(t2)
			copy(x[r*H:(r+1)*H], yv)
			nRows++
		}
		if err := binary.Write(out, binary.LittleEndian, x); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, chosenAll); err != nil {
			return err
		}
		if err := out.Flush(); err != nil {
			return err
		}
	}
}

// round4 rounds w (rows, cols) to signed codes of bits bits, a scale every group columns of a
// row (0: the row), the scale's clipping searched among 0.5 to 1 of the largest magnitude for the
// least squared error, and returns the values the codes stand for.
func round4(w []float32, rows, cols, bits, group int) []float32 {
	if group <= 0 || group > cols {
		group = cols
	}
	q := float64(int(1)<<(bits-1) - 1) // 7 for 4 bits: -7..7
	out := make([]float32, len(w))
	for r := 0; r < rows; r++ {
		for g0 := 0; g0 < cols; g0 += group {
			seg := w[r*cols+g0 : r*cols+min(g0+group, cols)]
			var m float64
			for _, v := range seg {
				m = max(m, math.Abs(float64(v)))
			}
			if m == 0 {
				continue
			}
			best, bestErr := m/q, math.Inf(1)
			for f := 1.0; f >= 0.5; f -= 0.025 {
				s := f * m / q
				var e float64
				for _, v := range seg {
					c := max(-q, min(q, math.Round(float64(v)/s)))
					d := float64(v) - c*s
					e += d * d
				}
				if e < bestErr {
					best, bestErr = s, e
				}
			}
			for i, v := range seg {
				out[r*cols+g0+i] = float32(max(-q, min(q, math.Round(float64(v)/best))) * best)
			}
		}
	}
	return out
}
