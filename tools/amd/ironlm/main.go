// Command ironlm runs a language model exported by tools/amd/iron_export_lm.py on an AMD NPU
// through hwnpu, and checks it against the logits the export recorded (check/).
//
//	go run ./tools/amd/ironlm DIR
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

type check struct {
	Prompt []int32 `json:"prompt"`
	Calls  []struct {
		Tokens int    `json:"tokens"`
		Logits string `json:"logits"`
		Top    int32  `json:"top"`
	} `json:"calls"`
}

func readF32(path string) []float32 {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func argmax(x []float32) int32 {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return int32(best)
}

// compare reports the cosine similarity and the largest difference relative to the
// reference's largest magnitude.
func compare(got, want []float32) (cos, rel float64) {
	var dot, ng, nw, worst, scale float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		dot += g * w
		ng += g * g
		nw += w * w
		worst = max(worst, math.Abs(g-w))
		scale = max(scale, math.Abs(w))
	}
	return dot / math.Sqrt(ng*nw), worst / scale
}

// seqNum is the step number in a buffer's name (gemv947 is 947), or a large number.
func seqNum(name string) int {
	i := len(name)
	for i > 0 && name[i-1] >= '0' && name[i-1] <= '9' {
		i--
	}
	n, err := strconv.Atoi(name[i:])
	if err != nil {
		return 1 << 30
	}
	return n
}

func main() {
	steps := flag.Int("steps", 16, "decode steps timed after the check")
	stepsOnly := flag.Bool("decode-only", false, "run the prompt one decode step at a time")
	nan := flag.Bool("nan", false, "after one decode step, list the buffers holding NaN")
	prompt := flag.String("prompt", "", "generate greedily after this user message (Gemma 4's chat format)")
	tokPath := flag.String("tokenizer", "", "tokenizer.json for -prompt")
	maxTokens := flag.Int("max", 200, "tokens to generate for -prompt")
	verify := flag.Bool("verify", false, "check the version checking drafts (--verify) against decode steps, Keep after fewer rows, and time it")
	bench := flag.Bool("bench", false, "time a decode step against prompt calls of a few tokens (what checking drafts costs)")
	depth := flag.Bool("depth", false, "time a whole prompt chunk and a decode step at each chunk boundary up to the context (an amdxdna command is cut off after 2 s)")
	forced := flag.String("forced", "", "a text file: read its first -split tokens as a prompt, then feed the rest one decode step at a time, writing the logits after each to -logits (float32 rows)")
	split := flag.Int("split", 256, "tokens of -forced read as the prompt")
	logitsOut := flag.String("logits", "", "where -forced writes its logits")
	bos := flag.Int("bos", 2, "the token -forced puts before its text (Gemma 4's <bos> 2; PLaMo 3's 1)")
	record := flag.String("record", "", "write a check (check.json and the logits) into this directory from this export's own logits: a prompt of -record-tokens tokens of -text (tokenized with -tokenizer), then -record-calls greedy steps; another export's run on it compares the two")
	text := flag.String("text", "", "the text file -record reads its prompt from")
	recordTokens := flag.Int("record-tokens", 2600, "the prompt's tokens for -record")
	recordCalls := flag.Int("record-calls", 4, "the calls -record writes: the prompt's, then greedy decode steps")
	flag.Parse()
	dir := flag.Arg(0)
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	t0 := time.Now()
	m, err := iron.Open(ctx, dev, dir, log.Printf)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	log.Printf("loaded in %.1f s", time.Since(t0).Seconds())
	if *verify {
		verifyCheck(ctx, m)
		return
	}
	if *bench {
		benchCalls(ctx, m)
		return
	}
	if *depth {
		benchDepth(ctx, m)
		return
	}
	if *record != "" {
		recordCheck(ctx, m, *record, *text, *tokPath, *recordTokens, *recordCalls)
		return
	}
	if *prompt != "" {
		generate(ctx, m, *prompt, *tokPath, *maxTokens)
		return
	}
	if *forced != "" {
		teacher(ctx, m, *forced, *tokPath, int32(*bos), *split, *maxTokens, *logitsOut)
		return
	}
	if *nan {
		name := "model.layers.26.per_layer_embedding"
		img, _ := os.ReadFile(filepath.Join(dir, "bank_scratch_1.bin"))
		cmp := func(when string) {
			b, p, err := m.Buffer("decode", name)
			if err != nil {
				log.Fatal(err)
			}
			want := img[p.Offset : p.Offset+p.Bytes]
			first, last, n := -1, -1, 0
			for i := range b {
				if b[i] != want[i] {
					if first < 0 {
						first = i
					}
					last = i
					n++
				}
			}
			fmt.Printf("%s: %s differs from the image in %d bytes, [%d, %d] of %d\n", when, name, n, first, last, p.Bytes)
		}
		cmp("before the step")
		if _, err := m.Step(ctx, 7, 0); err != nil {
			log.Fatal(err)
		}
		cmp("after the step")
		type hit struct {
			name    string
			n, size int
			off     int
			arg     string
		}
		var hits []hit
		for name, p := range m.Buffers("decode") {
			if p.DType != "bfloat16" {
				continue
			}
			b, _, err := m.Buffer("decode", name)
			if err != nil {
				log.Fatal(err)
			}
			n := 0
			for i := 0; i+1 < len(b); i += 2 {
				h := binary.LittleEndian.Uint16(b[i:])
				if h&0x7f80 == 0x7f80 && h&0x7f != 0 {
					n++
				}
			}
			if n > 0 {
				hits = append(hits, hit{name, n, len(b) / 2, p.Offset, p.Argument})
			}
		}
		sort.Slice(hits, func(i, j int) bool { return seqNum(hits[i].name) < seqNum(hits[j].name) })
		for i, h := range hits {
			if i < 25 {
				fmt.Printf("%s: %d of %d NaN (%s +%d)\n", h.name, h.n, h.size, h.arg, h.off)
			}
		}
		fmt.Printf("%d buffers hold NaN\n", len(hits))
		return
	}
	var c check
	b, err := os.ReadFile(filepath.Join(dir, "check", "check.json"))
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, &c); err != nil {
		log.Fatal(err)
	}
	ok := true
	history := append([]int32(nil), c.Prompt...)
	var logits []float32
	for k, call := range c.Calls {
		t := time.Now()
		if k == 0 && *stepsOnly {
			for i, tok := range history {
				if logits, err = m.Step(ctx, tok, i); err != nil {
					break
				}
				if i < 3 {
					top := argmax(logits)
					fmt.Printf("  step %d: top %d logit %v\n", i, top, logits[top])
				}
			}
		} else if k == 0 {
			logits, err = m.Prompt(ctx, history, 0)
		} else {
			logits, err = m.Step(ctx, history[len(history)-1], len(history)-1)
		}
		if err != nil {
			log.Fatalf("call %d: %v", k, err)
		}
		ms := time.Since(t).Seconds() * 1000
		want := readF32(filepath.Join(dir, "check", call.Logits))
		cos, rel := compare(logits, want)
		top := argmax(logits)
		fmt.Printf("call %d (%d tokens, %.1f ms): top %d (reference %d), cosine %.5f, worst difference %.3f of the largest logit\n", k, len(history), ms, top, call.Top, cos, rel)
		if !(cos >= 0.99) {
			ok = false
		}
		history = append(history, call.Top) // teacher-forced, as the reference
	}
	// Decode speed.
	t := time.Now()
	for i := 0; i < *steps; i++ {
		tok := argmax(logits)
		if logits, err = m.Step(ctx, tok, len(history)-1+i); err != nil {
			log.Fatal(err)
		}
	}
	el := time.Since(t).Seconds()
	fmt.Printf("decode: %d steps in %.2f s, %.1f tokens/s\n", *steps, el, float64(*steps)/el)
	if !ok {
		fmt.Println("FAIL")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func generate(ctx context.Context, m *iron.Model, text, tokPath string, maxTokens int) {
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Fatal(err)
	}
	p, _, err := chat.Gemma4.Render([]chat.Message{{Role: "user", Content: chat.Content{Text: text}}}, nil, chat.Options{NoThink: true})
	if err != nil {
		log.Fatal(err)
	}
	ids := tok.Encode(p)
	stops := map[int32]bool{}
	for _, s := range chat.Gemma4.Stops() {
		if id, ok := tok.ID(s); ok {
			stops[id] = true
		}
	}
	t := time.Now()
	logits, err := m.Prompt(ctx, ids, 0)
	if err != nil {
		log.Fatal(err)
	}
	ttft := time.Since(t)
	var out []int32
	dev := m.Device()
	p0, perr := dev.PowerSample()
	t = time.Now()
	for len(out) < maxTokens {
		next := argmax(logits)
		if stops[next] {
			break
		}
		out = append(out, next)
		fmt.Print(tok.Decode([]int32{next}, false))
		if logits, err = m.Step(ctx, next, len(ids)+len(out)-1); err != nil {
			log.Fatal(err)
		}
	}
	el := time.Since(t).Seconds()
	if p1, err := dev.PowerSample(); perr == nil && err == nil && len(out) > 0 {
		fmt.Printf("\n\ngeneration: %.2f W on average (%s), %.2f J per token", hwnpu.AveragePower(p0, p1), p1.Scope, (p1.EnergyJ-p0.EnergyJ)/float64(len(out)))
	}
	fmt.Printf("\n\n%d prompt tokens in %.2f s (%.0f tokens/s); %d tokens in %.2f s (%.1f tokens/s)\n",
		len(ids), ttft.Seconds(), float64(len(ids))/ttft.Seconds(), len(out), el, float64(len(out))/el)
}

// benchCalls times the decode step and prompt calls of k tokens at the same position, after a
// prompt of 500 tokens: a call checking k-1 drafts costs what a prompt call of k costs.
func benchCalls(ctx context.Context, m *iron.Model) {
	toks := make([]int32, 600)
	for i := range toks {
		toks[i] = int32(1000 + i)
	}
	if _, err := m.Prompt(ctx, toks[:500], 0); err != nil {
		log.Fatal(err)
	}
	median := func(f func() error) float64 {
		var ts []float64
		for i := 0; i < 9; i++ {
			t := time.Now()
			if err := f(); err != nil {
				log.Fatal(err)
			}
			ts = append(ts, time.Since(t).Seconds()*1000)
		}
		sort.Float64s(ts)
		return ts[len(ts)/2]
	}
	step := median(func() error { _, err := m.Step(ctx, toks[500], 500); return err })
	fmt.Printf("decode step: %.1f ms\n", step)
	for _, k := range []int{2, 4, 8, m.Manifest.Config.PrefillChunk} {
		if k > m.Manifest.Config.PrefillChunk || k > 64 {
			continue
		}
		ms := median(func() error { _, err := m.Prompt(ctx, toks[500:500+k], 500); return err })
		fmt.Printf("prompt call of %d tokens: %.1f ms (%.2f decode steps)\n", k, ms, ms/step)
	}
}

// verifyCheck compares the version checking drafts with decode steps on the same tokens after a
// prompt of 500 tokens: each row's logits, then the step after keeping fewer rows (Keep), then
// the times of a check and of a decode step.
func verifyCheck(ctx context.Context, m *iron.Model) {
	R := m.VerifyRows()
	if R == 0 {
		log.Fatal("the export has no version checking drafts")
	}
	toks := make([]int32, 600)
	for i := range toks {
		toks[i] = int32(1000 + 7*i)
	}
	const at = 500
	if _, err := m.Prompt(ctx, toks[:at], 0); err != nil {
		log.Fatal(err)
	}
	saved, err := m.StateAt(at)
	if err != nil {
		log.Fatal(err)
	}
	steps := func(ts []int32) [][]float32 {
		if err := m.SetStateAt(at, saved); err != nil {
			log.Fatal(err)
		}
		var out [][]float32
		for i, t := range ts {
			l, err := m.Step(ctx, t, at+i)
			if err != nil {
				log.Fatal(err)
			}
			out = append(out, l)
		}
		return out
	}
	check := toks[at : at+R]
	want := steps(check)
	if err := m.SetStateAt(at, saved); err != nil {
		log.Fatal(err)
	}
	got, err := m.Verify(ctx, check, at)
	if err != nil {
		log.Fatal(err)
	}
	bad := 0
	for r := range got {
		cos, rel := compare(got[r], want[r])
		g, w := argmax(got[r]), argmax(want[r])
		fmt.Printf("row %d: cosine %.6f, largest error %.4f of the largest logit, top %d (decode %d)\n", r, cos, rel, g, w)
		if cos < 0.99 {
			bad++
		}
	}
	// keep k rows, then a decode step on another token: as k rows then that token by decode steps
	for k := 1; k < R; k++ {
		if err := m.SetStateAt(at, saved); err != nil {
			log.Fatal(err)
		}
		if _, err := m.Verify(ctx, check, at); err != nil {
			log.Fatal(err)
		}
		if err := m.Keep(at, k); err != nil {
			log.Fatal(err)
		}
		next := int32(4242 + k)
		l, err := m.Step(ctx, next, at+k)
		if err != nil {
			log.Fatal(err)
		}
		ref := steps(append(slices.Clone(check[:k]), next))
		cos, rel := compare(l, ref[k])
		fmt.Printf("keep %d, then a step: cosine %.6f, largest error %.4f, top %d (decode %d)\n", k, cos, rel, argmax(l), argmax(ref[k]))
		if cos < 0.99 {
			bad++
		}
	}
	median := func(f func() error) float64 {
		var ts []float64
		for i := 0; i < 15; i++ {
			if err := m.SetStateAt(at, saved); err != nil {
				log.Fatal(err)
			}
			t := time.Now()
			if err := f(); err != nil {
				log.Fatal(err)
			}
			ts = append(ts, time.Since(t).Seconds()*1000)
		}
		sort.Float64s(ts)
		return ts[len(ts)/2]
	}
	step := median(func() error { _, err := m.Step(ctx, check[0], at); return err })
	ver := median(func() error { _, err := m.Verify(ctx, check, at); return err })
	keep := median(func() error { return m.Keep(at, 1) })
	fmt.Printf("decode step %.1f ms; check of %d rows %.1f ms (%.2f steps); keep 1 row %.2f ms\n", step, R, ver, ver/step, keep)
	if bad > 0 {
		fmt.Println("FAIL")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

// benchDepth reads a prompt of whole chunks up to the context, timing each chunk, and a decode
// step after each: the time grows with the position the attention reaches.
func benchDepth(ctx context.Context, m *iron.Model) {
	C, L := m.Manifest.Config.PrefillChunk, m.Manifest.Config.MaxSeqLen
	toks := make([]int32, L)
	for i := range toks {
		toks[i] = int32(1000 + (i*7919)%20000)
	}
	dev := m.Device()
	for start := 0; start < L-1; start += C {
		n := min(C, L-1-start) // the last chunk leaves a position for the decode step
		p0, perr := dev.PowerSample()
		t := time.Now()
		if _, err := m.Prompt(ctx, toks[start:start+n], start); err != nil {
			log.Fatalf("chunk at %d: %v", start, err)
		}
		chunk := time.Since(t).Seconds()
		if p1, err := dev.PowerSample(); perr == nil && err == nil {
			fmt.Printf("chunk of %d from %d: %.1f J, %.2f W on average (%s)\n", n, start, p1.EnergyJ-p0.EnergyJ, hwnpu.AveragePower(p0, p1), p1.Scope)
		}
		t = time.Now()
		if _, err := m.Step(ctx, toks[start+n], start+n); err != nil {
			log.Fatalf("step at %d: %v", start+n, err)
		}
		fmt.Printf("chunk of %d from %d: %.2f s (%.0f tokens/s); decode step at %d: %.1f ms\n", n, start, chunk, float64(n)/chunk, start+n, time.Since(t).Seconds()*1000)
	}
}

// teacher reads the first split tokens of a text as a prompt, then feeds the next ones (up to
// max) one decode step at a time and writes the logits before each fed token (the prompt's
// last, then each step's) as float32 rows: two exports' rows compare position by position.
func teacher(ctx context.Context, m *iron.Model, path, tokPath string, bos int32, split, max int, out string) {
	text, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Fatal(err)
	}
	ids := append([]int32{bos}, tok.Encode(string(text))...)
	if len(ids) < split+2 {
		log.Fatalf("%d tokens, fewer than -split %d and one to feed", len(ids), split)
	}
	ids = ids[:min(len(ids), split+max)]
	f, err := os.Create(out)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	write := func(l []float32) {
		if err := binary.Write(f, binary.LittleEndian, l); err != nil {
			log.Fatal(err)
		}
	}
	t0 := time.Now()
	logits, err := m.Prompt(ctx, ids[:split], 0)
	if err != nil {
		log.Fatal(err)
	}
	write(logits)
	var steps time.Duration // the decode steps alone (not the prompt, not the logits' writing)
	for p := split; p < len(ids)-1; p++ {
		t := time.Now()
		if logits, err = m.Step(ctx, ids[p], p); err != nil {
			log.Fatal(err)
		}
		steps += time.Since(t)
		write(logits)
	}
	fed := len(ids) - 1 - split
	log.Printf("%d tokens: %d prompt, %d fed, %d rows of %d logits in %.1f s (a decode step %.2f ms)", len(ids), split, fed, len(ids)-split, len(logits), time.Since(t0).Seconds(), steps.Seconds()*1000/float64(fed))
	ids32 := make([]byte, 4*len(ids))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(ids32[4*i:], uint32(id))
	}
	if err := os.WriteFile(out+".ids", ids32, 0o644); err != nil {
		log.Fatal(err)
	}
}

// recordCheck writes a check from this export's own logits: the first n tokens of the text
// (repeated past its end), then calls-1 greedy decode steps. Run on another export of the same
// model (its check/), ironlm compares the two exports call by call.
func recordCheck(ctx context.Context, m *iron.Model, dir, textPath, tokPath string, n, calls int) {
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		log.Fatal(err)
	}
	b, err := os.ReadFile(textPath)
	if err != nil {
		log.Fatal(err)
	}
	var ids []int32
	for len(ids) < n {
		ids = append(ids, tok.Encode(string(b))...)
	}
	ids = ids[:n]
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Fatal(err)
	}
	c := check{Prompt: ids}
	history := append([]int32(nil), ids...)
	var logits []float32
	for k := 0; k < calls; k++ {
		t := time.Now()
		if k == 0 {
			logits, err = m.Prompt(ctx, history, 0)
		} else {
			logits, err = m.Step(ctx, history[len(history)-1], len(history)-1)
		}
		if err != nil {
			log.Fatalf("call %d: %v", k, err)
		}
		name := fmt.Sprintf("logits_%d.bin", k)
		out := make([]byte, 4*len(logits))
		for i, v := range logits {
			binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
		}
		if err := os.WriteFile(filepath.Join(dir, name), out, 0o644); err != nil {
			log.Fatal(err)
		}
		top := argmax(logits)
		c.Calls = append(c.Calls, struct {
			Tokens int    `json:"tokens"`
			Logits string `json:"logits"`
			Top    int32  `json:"top"`
		}{len(history), name, top})
		fmt.Printf("call %d (%d tokens, %.1f ms): top %d\n", k, len(history), time.Since(t).Seconds()*1000, top)
		history = append(history, top)
	}
	j, err := json.Marshal(c)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "check.json"), append(j, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
}
