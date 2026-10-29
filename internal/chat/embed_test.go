package chat

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/plamo"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

const tinyPlamo = `{"architectures": ["PlamoBiModel"], "hidden_size": 32, "hidden_size_per_head": 8,
 "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 1, "num_hidden_layers": 2,
 "vocab_size": 300, "max_position_embeddings": 256, "rms_norm_eps": 1e-6}`

func pbVarint(b []byte, field int, v uint64) []byte {
	return binary.AppendUvarint(binary.AppendUvarint(b, uint64(field<<3)), v)
}

func pbBytes(b []byte, field int, data string) []byte {
	b = binary.AppendUvarint(b, uint64(field<<3|2))
	return append(binary.AppendUvarint(b, uint64(len(data))), data...)
}

func pbFloat(b []byte, field int, x float32) []byte {
	return binary.LittleEndian.AppendUint32(binary.AppendUvarint(b, uint64(field<<3|5)), math.Float32bits(x))
}

// tinyTokenizer is a SentencePiece BPE of PLaMo's kind: <|unknown|>, the BOS <|startoftext|>, the
// 256 bytes and a few pieces, so that "ab" is one token and other characters are their bytes.
func tinyTokenizer(t *testing.T) *tokenizer.SentencePiece {
	t.Helper()
	const normal, unknown, control, byteType = 1, 2, 3, 6
	type piece struct {
		s     string
		score float32
		typ   uint64
	}
	pieces := []piece{{"<|unknown|>", 0, unknown}, {"<|startoftext|>", 0, control}}
	for c := 0; c < 256; c++ {
		pieces = append(pieces, piece{fmt.Sprintf("<0x%02X>", c), 0, byteType})
	}
	pieces = append(pieces, piece{"▁", 0, normal}, piece{"a", 0, normal}, piece{"b", 0, normal}, piece{"ab", -1, normal})
	var b []byte
	for _, p := range pieces {
		b = pbBytes(b, 1, string(pbVarint(pbFloat(pbBytes(nil, 1, p.s), 2, p.score), 3, p.typ)))
	}
	trainer := string(pbVarint(pbVarint(nil, 3, 2), 35, 1)) // BPE, byte fallback
	normalizer := string(pbVarint(pbVarint(pbBytes(nil, 1, "identity"), 3, 1), 4, 0))
	path := filepath.Join(t.TempDir(), "tokenizer.model")
	if err := os.WriteFile(path, pbBytes(pbBytes(b, 2, trainer), 3, normalizer), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadSentencePiece(path)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// tinyEmbedder is an Embedder of a tiny PLaMo encoder with programs of 8 and 128 tokens (the
// instruction of a query takes 81) on the CPU interpreter.
func tinyEmbedder(t *testing.T) (*Embedder, *plamo.Config, llm.Weights) {
	t.Helper()
	cfg, err := plamo.ParseConfig([]byte(tinyPlamo))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	e, err := newEmbedder(tinyTokenizer(t), cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.w, e.Name = w, "tiny-embed"
	for _, seq := range []int{128, 8} {
		if err := e.add(cpuPackage(t, cfg, w, llm.Options{Seq: seq, Format: "fp16"})); err != nil {
			t.Fatal(err)
		}
	}
	return e, cfg, w
}

// The embedding of a text padded to a longer program is the mean of its own tokens' last hidden
// state from skip on, as the program of its exact length gives it, scaled to length 1.
func TestEmbedderPadding(t *testing.T) {
	e, cfg, w := tinyEmbedder(t)
	if e.MaxTokens() != 128 || e.Dim() != 32 {
		t.Fatalf("max tokens %d, dimensions %d; want 128 and 32", e.MaxTokens(), e.Dim())
	}
	d := cfg.HiddenSize
	for _, c := range []struct{ n, skip int }{{5, 1}, {8, 1}, {11, 3}, {128, 1}} {
		ids := make([]int32, c.n)
		for i := range ids {
			ids[i] = int32(1 + (i*37)%250)
		}
		got, err := e.run(ids, c.skip)
		if err != nil {
			t.Fatal(err)
		}
		p := cpuPackage(t, cfg, w, llm.Options{Seq: c.n, Format: "fp16"})
		x, _ := llm.Embed(w, cfg, ids)
		mask := make([]float32, c.n)
		for i := range mask {
			mask[i] = 1
		}
		hwnpu.EncodeF16(p.In["x"], x)
		hwnpu.EncodeF16(p.In["mask"], mask)
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		h := make([]float32, c.n*d)
		hwnpu.DecodeF16(h, p.Out["y"])
		want := make([]float64, d)
		var norm float64
		for j := range want {
			for r := c.skip; r < c.n; r++ {
				want[j] += float64(h[r*d+j])
			}
			norm += want[j] * want[j]
		}
		var num, gotNorm float64
		for j := range want {
			want[j] /= math.Sqrt(norm)
			num += (float64(got[j]) - want[j]) * (float64(got[j]) - want[j])
			gotNorm += float64(got[j]) * float64(got[j])
		}
		if err := math.Sqrt(num); err > 2e-3 {
			t.Errorf("%d tokens from %d: error %g against the unpadded program", c.n, c.skip, err)
		}
		if math.Abs(gotNorm-1) > 1e-5 {
			t.Errorf("%d tokens: squared length %g, want 1", c.n, gotNorm)
		}
	}
	if _, err := e.run(make([]int32, 129), 1); !errors.Is(err, ErrTooLong) {
		t.Errorf("129 tokens: %v, want ErrTooLong", err)
	}
}

// Embed puts the BOS (and for a query the instruction) before the text, leaves them out of the
// mean, and cuts or rejects a text that is too long.
func TestEmbed(t *testing.T) {
	e, _, _ := tinyEmbedder(t)
	ctx := context.Background()
	ids, skip := e.Tokens("ab ab", false)
	if want := []int32{1, e.Tok.Encode("ab ab")[0]}; !slices.Equal(ids[:2], want) || skip != 1 {
		t.Errorf("document: tokens %v from %d, want %v... from 1", ids, skip, want)
	}
	qids, qskip := e.Tokens("ab ab", true)
	// The ▁ a text starts with is the instruction's last token in a query: its trailing space.
	if qskip != 1+len(e.Tok.Encode(QueryPrefix)) || !slices.Equal(qids[qskip:], ids[2:]) {
		t.Errorf("query: tokens %v from %d; want the document's tokens after the BOS, the ▁ and the instruction", qids, qskip)
	}
	doc, err := e.Embed(ctx, "ab ab", false, false)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Tokens != len(ids) || doc.Truncated {
		t.Errorf("document: %d tokens (truncated %v), want %d", doc.Tokens, doc.Truncated, len(ids))
	}
	if _, err := e.Embed(ctx, "", false, false); !errors.Is(err, ErrEmpty) {
		t.Errorf("empty text: %v, want ErrEmpty", err)
	}
	long := strings.Repeat("ab ", 100)
	if _, err := e.Embed(ctx, long, false, false); !errors.Is(err, ErrTooLong) {
		t.Errorf("long text: %v, want ErrTooLong", err)
	}
	cut, err := e.Embed(ctx, long, false, true)
	if err != nil || !cut.Truncated || cut.Tokens != e.MaxTokens() {
		t.Errorf("long text truncated: %d tokens, truncated %v, %v; want %d, true", cut.Tokens, cut.Truncated, err, e.MaxTokens())
	}
}

// The server serves the embedding model on /v1/embeddings and lists it, and says so when a
// request goes to a model that is not loaded.
func TestEmbeddingsAPI(t *testing.T) {
	e, _, _ := tinyEmbedder(t)
	srv := httptest.NewServer((&Server{Emb: e}).Handler())
	defer srv.Close()
	post := func(path, body string) (int, map[string]any) {
		resp, err := http.Post(srv.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		json.NewDecoder(resp.Body).Decode(&out)
		return resp.StatusCode, out
	}
	vectors := func(out map[string]any) [][]float32 {
		var vs [][]float32
		for _, d := range out["data"].([]any) {
			switch v := d.(map[string]any)["embedding"].(type) {
			case string:
				b, _ := base64.StdEncoding.DecodeString(v)
				f := make([]float32, len(b)/4)
				for i := range f {
					f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
				}
				vs = append(vs, f)
			case []any:
				f := make([]float32, len(v))
				for i, x := range v {
					f[i] = float32(x.(float64))
				}
				vs = append(vs, f)
			}
		}
		return vs
	}

	code, out := post("/v1/embeddings", `{"model": "tiny-embed", "input": ["ab", "ab ab b"]}`)
	if code != http.StatusOK {
		t.Fatalf("status %d: %v", code, out)
	}
	docs := vectors(out)
	if len(docs) != 2 || len(docs[0]) != e.Dim() {
		t.Fatalf("got %d embeddings, want 2 of %d", len(docs), e.Dim())
	}
	want, _ := e.Embed(context.Background(), "ab ab b", false, false)
	if !slices.Equal(docs[1], want.Vector) {
		t.Error("the second embedding is not the text's")
	}
	if u := out["usage"].(map[string]any); u["prompt_tokens"].(float64) == 0 {
		t.Errorf("usage %v", u)
	}

	_, out = post("/v1/embeddings", `{"input": "ab ab b", "encoding_format": "base64"}`)
	if b := vectors(out); len(b) != 1 || !slices.Equal(b[0], docs[1]) {
		t.Error("base64 differs from float")
	}
	_, out = post("/v1/embeddings", `{"input": "ab ab b", "input_type": "query"}`)
	if q := vectors(out); len(q) != 1 || slices.Equal(q[0], docs[1]) {
		t.Error("a query's embedding is the document's")
	}

	for _, c := range []struct {
		path, body string
		code       int
	}{
		{"/v1/embeddings", `{"input": [[1, 2, 3]]}`, http.StatusBadRequest},
		{"/v1/embeddings", `{"input": []}`, http.StatusBadRequest},
		{"/v1/embeddings", `{"input": ""}`, http.StatusBadRequest},
		{"/v1/embeddings", `{"input": "ab", "dimensions": 8}`, http.StatusBadRequest},
		{"/v1/embeddings", `{"input": "ab", "input_type": "image"}`, http.StatusBadRequest},
		{"/v1/embeddings", `{"input": "` + strings.Repeat("ab ", 100) + `"}`, http.StatusBadRequest},
		{"/v1/embeddings", `{"input": "` + strings.Repeat("ab ", 100) + `", "truncate": true}`, http.StatusOK},
		{"/v1/chat/completions", `{"messages": [{"role": "user", "content": "hi"}]}`, http.StatusNotFound},
	} {
		if code, out := post(c.path, c.body); code != c.code {
			t.Errorf("%s %s: status %d (%v), want %d", c.path, c.body, code, out, c.code)
		}
	}

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var models struct {
		Data []struct {
			ID         string `json:"id"`
			Dimensions int    `json:"dimensions"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&models)
	if len(models.Data) != 1 || models.Data[0].ID != "tiny-embed" || models.Data[0].Dimensions != e.Dim() {
		t.Errorf("models %+v, want the embedding model alone", models.Data)
	}
}

// With the real tokenizer (HWNPU_PLAMO_DIR holds tokenizer.model), a query's tokens are the BOS
// and the tokenizer's tokens of the instruction and the text together, and the mean starts
// exactly at the text.
func TestPlamoQueryTokens(t *testing.T) {
	dir := os.Getenv("HWNPU_PLAMO_DIR")
	if dir == "" {
		t.Skip("HWNPU_PLAMO_DIR is not set")
	}
	tok, err := tokenizer.LoadSentencePiece(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := plamo.LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	e, err := newEmbedder(tok, cfg)
	if err != nil {
		t.Fatal(err)
	}
	const text = "東京の天気は晴れです。"
	ids, skip := e.Tokens(text, true)
	// the ids of the Hugging Face tokenizer, internal/tokenizer/testdata/plamo-embedding-1b.json
	want := []int32{1, 35913, 36338, 36000, 40364, 36003, 36179, 36524, 36001, 36604, 36034, 40364, 36006, 36639, 36524, 36166,
		35953, 35913, 36330, 36000, 38598, 36007, 40639, 36028, 36002}
	if !slices.Equal(ids, want) {
		t.Errorf("tokens %v, want %v", ids, want)
	}
	if got := tok.Decode(ids[skip:], false); got != text {
		t.Errorf("the mean starts at %q, want the text %q", got, text)
	}
	if !strings.HasPrefix(tok.Decode(ids[1:skip], false), strings.TrimSpace(QueryPrefix)) {
		t.Errorf("the tokens left out are %q, want the instruction", tok.Decode(ids[1:skip], false))
	}
}
