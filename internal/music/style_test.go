package music

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

func TestLegacyNormal(t *testing.T) {
	// np.random.RandomState(0).randn(5)
	want := []float64{1.764052345967664, 0.4001572083672233, 0.9787379841057392, 2.240893199201458, 1.8675579901499675}
	got := LegacyNormal(0, 5)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

// mccRef prepares MusicCoCa's text side in the model's directory (HWNPU_MRT_DIR) and gives
// mcc_ref.py's prefix (HWNPU_MUSICCOCA_REF).
func mccRef(t *testing.T) (Paths, string) {
	dir, ref := os.Getenv("HWNPU_MRT_DIR"), os.Getenv("HWNPU_MUSICCOCA_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_MRT_DIR and HWNPU_MUSICCOCA_REF are not set")
	}
	p, err := Prepare(context.Background(), Setup{Dir: dir, Repo: "google/magenta-realtime-2", Text: true, AudioStyle: os.Getenv("HWNPU_MUSICCOCA_AUDIO_REF") != ""})
	if err != nil {
		t.Fatal(err)
	}
	return p, ref
}

// The text's ids and, on the CPU interpreter, its codes are the reference's.
func TestStyle(t *testing.T) {
	p, ref := mccRef(t)
	var meta struct {
		IDs    map[string][]int32 `json:"ids"`
		Tokens []int              `json:"tokens"`
	}
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	sp, err := tokenizer.LoadSentencePieceUnigram(p.SPM)
	if err != nil {
		t.Fatal(err)
	}
	for text, want := range meta.IDs {
		if got := sp.Encode(strings.ToLower(text)); !slices.Equal(got, want) {
			t.Errorf("%q: ids %v, want %v", text, got, want)
		}
	}
	s, err := OpenStyler(context.Background(), nil, p)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	emb, err := s.Text("disco funk")
	if err != nil {
		t.Fatal(err)
	}
	codes, err := s.Tokens(emb)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(codes, meta.Tokens) {
		t.Errorf("codes %v, want %v", codes, meta.Tokens)
	}
}
