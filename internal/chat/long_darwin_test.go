package chat

import (
	"context"
	"io"
	"math"
	"os"
	"runtime/pprof"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// Measures the two-level attention (decode.Sparse) of a model split at its attention on the
// Neural Engine over a long context: it reads HWNPU_LONG_TOKENS tokens of the text
// HWNPU_LONG_DOC (repeated if short) as a prompt, with the whole cache or with the two-level
// attention while reading (HWNPU_LONG_PREFILL dense or sparse), then, for each setting of
// HWNPU_LONG_CONFIGS, goes back to the prompt's end and reads the next HWNPU_LONG_SCORE tokens
// of the text one per call as generation would, keeping each call's logits. It logs the time of
// the prompt and of a call, and for each setting the loss of the text's next tokens and the KL
// and top-1 agreement against the first setting (dense: the whole cache; rN: the two-level
// attention choosing its buckets every N calls; eN: N chunks of keys read exactly). Needs
// HWNPU_LONG_DIR, the directory of a model hwnpu-chat converted (its programs, config.json,
// tokenizer.json), and HWNPU_LONG_REPO, its repository. HWNPU_LONG_PROFILE writes a CPU profile
// of the settings' calls.
func TestLongContext(t *testing.T) {
	dir, docPath := os.Getenv("HWNPU_LONG_DIR"), os.Getenv("HWNPU_LONG_DOC")
	if dir == "" || docPath == "" {
		t.Skip("HWNPU_LONG_DIR and HWNPU_LONG_DOC not set")
	}
	num := func(name string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
			return v
		}
		return def
	}
	n, m := num("HWNPU_LONG_TOKENS", 131072), num("HWNPU_LONG_SCORE", 256)
	configs := strings.Split(cmpOr(os.Getenv("HWNPU_LONG_CONFIGS"), "dense,r1,r4,r16"), ",")
	ctx := context.Background()
	c, err := Prepare(ctx, Setup{Dir: dir, Repo: os.Getenv("HWNPU_LONG_REPO"), Draft: 8, Backend: "coreml", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	c.MTP = nil
	e, err := Load(ctx, c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ds, ok := e.sess.(*decode.Session)
	if !ok || ds.Paged() == 0 {
		t.Fatal("not a session split at its attention")
	}
	raw, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatal(err)
	}
	doc := e.Tok.Encode(string(raw))
	for len(doc) < n+m+1 {
		doc = append(doc, doc...)
	}
	doc = doc[:n+m+1]

	sparse := func(name string) decode.Sparse {
		sp := decode.DefaultSparse()
		switch {
		case name == "dense":
			return decode.Sparse{}
		case strings.HasPrefix(name, "r"):
			sp.Refresh, _ = strconv.Atoi(name[1:])
		case strings.HasPrefix(name, "e"):
			sp.Exact, _ = strconv.Atoi(name[1:])
		default:
			t.Fatalf("setting %q", name)
		}
		return sp
	}
	pre := sparse("r1")
	if os.Getenv("HWNPU_LONG_PREFILL") != "sparse" {
		pre = decode.Sparse{}
	}
	if err := ds.SetSparse(pre); err != nil {
		t.Fatal(err)
	}
	q := ds.Qs()[len(ds.Qs())-1]
	t0 := time.Now()
	for pos := 0; pos < n; pos += q {
		rows, err := e.rows(doc[pos:min(pos+q, n)])
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ds.Step(rows); err != nil {
			t.Fatal(err)
		}
		if pos/q%64 == 0 {
			t.Logf("read %d of %d tokens, %.0f s", pos+q, n, time.Since(t0).Seconds())
		}
	}
	t.Logf("prompt of %d tokens (%s two-level attention) read in %.1f s, %.0f tokens/s", n, map[bool]string{true: "with", false: "without"}[pre.Bucket > 0], time.Since(t0).Seconds(), float64(n)/time.Since(t0).Seconds())
	mark := ds.Mark()

	if p := os.Getenv("HWNPU_LONG_PROFILE"); p != "" {
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			t.Fatal(err)
		}
		defer pprof.StopCPUProfile()
	}
	var ref [][]float32 // the first setting's log-probabilities
	for ci, name := range configs {
		ds.Rewind(mark)
		if err := ds.SetSparse(sparse(name)); err != nil {
			t.Fatal(err)
		}
		var nll, kl float64
		var klMax float64
		agree := 0
		t1 := time.Now()
		for i := 0; i < m; i++ {
			rows, err := e.rows(doc[n+i : n+i+1])
			if err != nil {
				t.Fatal(err)
			}
			logits, err := ds.Step(rows)
			if err != nil {
				t.Fatal(err)
			}
			lp := logSoftmax(logits)
			nll -= float64(lp[doc[n+i+1]])
			if ci == 0 {
				ref = append(ref, lp)
				continue
			}
			var d float64
			for j, r := range ref[i] {
				if r > -30 {
					d += math.Exp(float64(r)) * float64(r-lp[j])
				}
			}
			kl += d
			klMax = math.Max(klMax, d)
			if argmax(lp) == argmax(ref[i]) {
				agree++
			}
		}
		ms := float64(time.Since(t1).Microseconds()) / float64(m) / 1000
		if ci == 0 {
			t.Logf("%-6s %.1f ms a call, loss %.4f nats a token", name, ms, nll/float64(m))
			continue
		}
		t.Logf("%-6s %.1f ms a call, loss %.4f nats a token, KL against %s %.5f (largest %.4f), top-1 the same %d/%d", name, ms, nll/float64(m), configs[0], kl/float64(m), klMax, agree, m)
	}
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// logSoftmax is log softmax of v in float32.
func logSoftmax(v []float32) []float32 {
	mx := v[0]
	for _, x := range v {
		mx = max(mx, x)
	}
	var s float64
	for _, x := range v {
		s += math.Exp(float64(x - mx))
	}
	ls := float32(math.Log(s)) + mx
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x - ls
	}
	return out
}
