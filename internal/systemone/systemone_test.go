package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// Prompt writes what the systemone template llama.cpp writes for LFM3 d1 renders. The expected
// prompts in testdata/prompts.json were rendered with jinja2 by tools/ref/systemone_golden.py.
func TestPromptMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/prompts.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Name    string          `json:"name"`
			Request json.RawMessage `json:"request"`
			Labels  []string        `json:"labels"`
			Control []string        `json:"control"`
			Want    string          `json:"want"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases {
		t.Run(c.Name, func(t *testing.T) {
			r, err := ParseRequest(c.Request)
			if err != nil {
				t.Fatal(err)
			}
			if got := Prompt(r.State, r.Questions[0], c.Labels, Escaper(c.Control)); got != c.Want {
				t.Errorf("prompt\n%q\nwant\n%q", got, c.Want)
			}
		})
	}
}

// fakeEncoder has one token for each single character, for yes, Yes, no and No, and for a space
// followed by a capital letter; any other text is a token per byte.
type fakeEncoder map[string]int32

func newFakeEncoder() fakeEncoder {
	e := fakeEncoder{}
	add := func(s string) { e[s] = int32(len(e)) }
	for c := 0; c < 128; c++ {
		add(string(rune(c)))
	}
	for _, s := range []string{"yes", "Yes", "no", "No"} {
		add(s)
	}
	for c := 'A'; c <= 'Z'; c++ {
		add(" " + string(c))
	}
	return e
}

func (e fakeEncoder) Encode(text string) []int32 {
	if id, ok := e[text]; ok {
		return []int32{id}
	}
	var out []int32
	for i := 0; i < len(text); i++ {
		out = append(out, e[text[i:i+1]]%128)
	}
	return out
}

func question(t *testing.T, js string) Question {
	t.Helper()
	r, err := ParseRequest([]byte(`{"state": null, "questions": {"q": ` + js + `}}`))
	if err != nil {
		t.Fatal(err)
	}
	return r.Questions[0]
}

func TestLabels(t *testing.T) {
	e := newFakeEncoder()
	tok := func(s string) int32 { return e[s] }
	keys := func(n int) string {
		var b strings.Builder
		for i := range n {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(`"opt` + string(rune('a'+i%26)) + string(rune('a'+i/26)) + `": null`)
		}
		return b.String()
	}
	for _, c := range []struct {
		name, q string
		codes   []string
		groups  [][]int32
	}{
		{"letter keys", `{"type": "choice", "instructions": "?", "criteria": {"x": null, "B": null}}`,
			[]string{"x", "B"}, [][]int32{{tok("x")}, {tok("B"), tok(" B")}}},
		{"other keys", `{"type": "choice", "instructions": "?", "criteria": {` + keys(3) + `}}`,
			[]string{"A", "B", "C"}, [][]int32{{tok("A"), tok(" A")}, {tok("B"), tok(" B")}, {tok("C"), tok(" C")}}},
		{"noul", `{"type": "noul", "instructions": "?"}`,
			[]string{"true", "false"}, [][]int32{{tok("yes"), tok("Yes")}, {tok("no"), tok("No")}}},
		{"score", `{"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]}`,
			[]string{"0", "1", "2"}, [][]int32{{tok("0")}, {tok("1")}, {tok("2")}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			codes, groups, err := Labels(question(t, c.q), e.Encode)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(codes, c.codes) || len(groups) != len(c.groups) {
				t.Fatalf("codes %v, groups %v", codes, groups)
			}
			for i := range groups {
				if !slices.Equal(groups[i], c.groups[i]) {
					t.Errorf("group %d %v, want %v", i, groups[i], c.groups[i])
				}
			}
		})
	}
	// past Z the codes AA, AB, ... are not single tokens here: they come from the pool (the
	// letters are taken, two digit numbers are not single tokens, so a, b, ...)
	codes, _, err := Labels(question(t, `{"type": "choice", "instructions": "?", "criteria": {`+keys(28)+`}}`), e.Encode)
	if err != nil {
		t.Fatal(err)
	}
	if codes[25] != "Z" || codes[26] != "a" || codes[27] != "b" {
		t.Errorf("codes %v", codes)
	}
}

func TestDecide(t *testing.T) {
	choice := question(t, `{"type": "choice", "instructions": "?", "criteria": {"x": null, "y": null, "z": null}}`)
	a := Decide(choice, []float64{1, 3, 2}, 2)
	z := math.Exp(-1) + 1 + math.Exp(-0.5)
	if a.Choice != "y" || math.Abs(a.Probabilities[1]-1/z) > 1e-12 || a.Confidence != a.Probabilities[1] {
		t.Errorf("choice %+v", a)
	}
	score := question(t, `{"type": "score", "instructions": "?", "criteria": ["a", "b", "c", "d"]}`)
	// probabilities 0.3, 0.15, 0.15, 0.4: the median is level 2, the mode 3
	a = Decide(score, []float64{math.Log(0.3), math.Log(0.15), math.Log(0.15), math.Log(0.4)}, 1)
	if a.Score != 2 || math.Abs(a.Confidence-0.4) > 1e-12 {
		t.Errorf("score %+v", a)
	}
	noul := question(t, `{"type": "noul", "instructions": "?"}`)
	if a = Decide(noul, []float64{0, math.Log(3)}, 1); math.Abs(a.Noul-0.25) > 1e-12 {
		t.Errorf("noul %+v", a)
	}
	b, _ := json.Marshal(&Response{Model: "m", IDs: []string{"s", "n"}, Answers: []Answer{
		Decide(score, []float64{0, 0, 0, 0}, 1), Decide(noul, []float64{0, 0}, 1)}, InputTokens: 7})
	want := `{"model":"m","answers":{"s":{"type":"score","score":1,"legend":{"0":"a","1":"b","2":"c","3":"d"},` +
		`"probabilities":{"0":0.25,"1":0.25,"2":0.25,"3":0.25},"confidence":0.25},"n":{"type":"noul","noul":0.5}},` +
		`"usage":{"input_tokens":7,"output_tokens":0}}`
	if string(b) != want {
		t.Errorf("response\n%s\nwant\n%s", b, want)
	}
}

func TestParseRequest(t *testing.T) {
	for _, c := range []struct {
		name, body  string
		unsupported bool
	}{
		{"no state", `{"questions": {"q": {"type": "noul", "instructions": "?"}}}`, false},
		{"no questions", `{"state": "s", "questions": {}}`, false},
		{"no instructions", `{"state": "s", "questions": {"q": {"type": "noul"}}}`, false},
		{"bad type", `{"state": "s", "questions": {"q": {"type": "rank", "instructions": "?"}}}`, false},
		{"one level", `{"state": "s", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["a"]}}}`, false},
		{"empty choice", `{"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}}`, false},
		{"spoken", `{"state": "s", "questions": {"q": {"type": "noul", "instructions": {"type": "input_audio"}}}}`, false},
		{"images", `{"state": "s", "images": ["data:image/png;base64,AA=="], "questions": {"q": {"type": "noul", "instructions": "?"}}}`, true},
		{"image part", `{"state": [{"role": "user", "content": [{"type": "image_url"}]}], "questions": {"q": {"type": "noul", "instructions": "?"}}}`, true},
	} {
		_, err := ParseRequest([]byte(c.body))
		var badReq *BadRequest
		var unsup *Unsupported
		if c.unsupported && !errors.As(err, &unsup) || !c.unsupported && !errors.As(err, &badReq) {
			t.Errorf("%s: %v", c.name, err)
		}
	}
	r, err := ParseRequest([]byte(`{"state": {"b": 1, "a": 2}, "questions": {"z": {"type": "noul", "instructions": "?"}, "a": {"type": "noul", "instructions": "?"}}}`))
	if err != nil || r.Questions[0].ID != "z" || r.State.String() != `{"b": 1, "a": 2}` {
		t.Errorf("order not kept: %+v, %v", r, err)
	}
}

// A tiny d1 model answers on the CPU interpreter with the probabilities of its label logits at
// the prompt's last token, read from a program longer than the prompt (padded); a prompt longer
// than every program is a bad request.
func TestModel(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyD1))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	dir := t.TempDir()
	open := func(seq int) program {
		path := filepath.Join(dir, fmt.Sprintf("d1-%d.mlpackage", seq))
		if _, err := llm.WritePart(path, cfg, w, llm.Options{Seq: seq, Format: "fp16"}, nil); err != nil {
			t.Fatal(err)
		}
		p, err := openCPU(path, cfg.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { p.Close() })
		return p
	}
	enc := vocabEncoder(cfg.VocabSize)
	m := &Model{Name: "tiny", tok: enc, cfg: cfg, w: w, temperature: 0.7, escape: func(s string) string { return s },
		emb: map[int32][]float32{}, heads: map[int32][]float32{}}
	req, err := ParseRequest([]byte(`{"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"x": null, "y": null, "z": null}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	ids, _, err := m.Tokens(req.State, req.Questions[0])
	if err != nil {
		t.Fatal(err)
	}
	n := len(ids)
	m.progs = []program{open(n + 3), open(n + 20)}
	got, err := m.Decide(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if got.InputTokens != n || got.Answers[0].Type != "choice" {
		t.Fatalf("response %+v", got)
	}
	// the expectation from the whole output of a program of the prompt's length: the hidden
	// state at the last token times the rows of the labels' tokens in the tied embedding table
	path := filepath.Join(dir, "whole.mlpackage")
	if _, err := llm.WritePart(path, cfg, w, llm.Options{Seq: n, Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	prog, f, err := milcpu.OpenPackage(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	x, err := llm.Embed(w, cfg, ids)
	if err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	res, err := prog.Run(map[string]milcpu.Tensor{"x": {Shape: []int{1, n, cfg.HiddenSize}, Data: x}, "mask": {Shape: []int{1, n}, Data: ones}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	last := res["y"].Data[(n-1)*cfg.HiddenSize:]
	_, groups, err := Labels(req.Questions[0], enc.Encode)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float64, len(groups))
	for i, g := range groups {
		if len(g) != 2 {
			t.Fatalf("label group %v, want the code and the code after a space", g)
		}
		logits[i] = math.Inf(-1)
		for _, tok := range g {
			row, _ := w.Rows(cfg.Embedding(), int(tok), int(tok)+1)
			var dot float64
			for j, v := range row {
				dot += float64(v) * float64(last[j])
			}
			logits[i] = math.Max(logits[i], dot)
		}
	}
	want := Decide(req.Questions[0], logits, m.temperature)
	for i, p := range want.Probabilities {
		if math.Abs(got.Answers[0].Probabilities[i]-p) > 1e-4 {
			t.Errorf("probabilities %v, want %v", got.Answers[0].Probabilities, want.Probabilities)
			break
		}
	}
	if slices.Max(want.Probabilities) > 0.999 {
		t.Errorf("degenerate probabilities %v", want.Probabilities)
	}
	m.progs = []program{open(n - 1)}
	var badReq *BadRequest
	if _, err := m.Decide(context.Background(), req); !errors.As(err, &badReq) {
		t.Errorf("a prompt longer than the program: %v", err)
	}
}

// vocabEncoder maps each byte to a token below vocab, and a space followed by a letter to one
// token of its own.
type vocabEncoder int

func (v vocabEncoder) Encode(text string) []int32 {
	if len(text) == 2 && text[0] == ' ' {
		return []int32{(int32(text[1]) + 7) % int32(v)}
	}
	var out []int32
	for i := 0; i < len(text); i++ {
		out = append(out, int32(text[i])%int32(v))
	}
	return out
}

const tinyD1 = `{"architectures": ["D1Model"], "model_type": "lfm3_vl", "tie_word_embeddings": true,
	"d1_codes": 2, "d1_attention": "set",
	"text_config": {"model_type": "lfm3", "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
		"num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 8, "vocab_size": 48, "norm_eps": 1e-5,
		"conv_L_cache": 3, "conv_bias": false, "sliding_window": 3, "use_sandwich_norm": true,
		"layer_types": ["full_attention", "conv", "sliding_attention", "full_attention"],
		"rope_parameters": {"full_attention": {"rope_theta": 10000, "partial_rotary_factor": 0.5, "rope_type": "default"},
			"sliding_attention": {"rope_theta": 1000, "rope_type": "default"}}}}`
