package iron

import (
	"bytes"
	"context"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
)

// An export with towers loads only the start of its bank that the decode step and the prompt
// chunk use; a tower grows it to its whole size, keeping what the device wrote into the text's
// part (the caches of the sequence) and reading the towers' weights from the image, with zeros
// past the image's end.
func TestLazyTowerBanks(t *testing.T) {
	const text, whole, image = 3 << 12, 7 << 12, 6<<12 + 100 // the image lacks the bank's trailing zeros
	man := func(towers bool) *Manifest {
		m := &Manifest{
			Banks: map[string]Bank{"scratch": {Bytes: whole, Image: "bank_scratch.bin"}},
			Versions: map[string]*Version{
				"decode": {Arguments: []Argument{{Index: 2, Kind: "scratch", Bytes: text - 4096}},
					Buffers: map[string]Placement{"l0.cache.k": {Argument: "scratch", Offset: 4096, Bytes: 4096}}},
				"prompt": {Arguments: []Argument{{Index: 2, Kind: "scratch", Bytes: text}}},
			},
		}
		if towers {
			m.Versions["vision"] = &Version{Arguments: []Argument{{Index: 2, Kind: "scratch", Bytes: whole}}}
		}
		return m
	}
	if got := man(false).textBanks(); got != nil {
		t.Errorf("an export without towers leaves out %v", got)
	}
	if got := man(true).textBanks(); got["scratch"] != text {
		t.Fatalf("the text's part of the bank: %v, want %d", got, text)
	}
	past := man(true)
	past.Versions["decode"].Buffers["x"] = Placement{Argument: "scratch", Offset: text, Bytes: 16}
	if got := past.textBanks(); got != nil {
		t.Errorf("a text buffer past its argument's bytes still leaves out %v", got)
	}

	dir := t.TempDir()
	img := make([]byte, image)
	rng := rand.New(rand.NewPCG(3, 4))
	for i := range img {
		img[i] = byte(rng.Uint32()) | 1
	}
	if err := os.WriteFile(filepath.Join(dir, "bank_scratch.bin"), img, 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend(simBackend))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	m := &Model{Manifest: man(true), dir: dir, dev: dev, banks: map[string]*hwnpu.Buffer{}, logf: t.Logf}
	m.decode = &version{m: m.Manifest.Versions["decode"], args: map[string]*hwnpu.Buffer{}}
	defer m.Release()
	if err := m.loadBanks(); err != nil {
		t.Fatal(err)
	}
	b := m.banks["scratch"]
	if b.Size() != text || !bytes.Equal(b.Bytes(), img[:text]) {
		t.Fatalf("loaded %d bytes, want the image's first %d", b.Size(), text)
	}
	if _, _, err := m.Buffer("decode", "l0.cache.k"); err != nil {
		t.Fatal(err)
	}
	// the sequence so far: rows the device wrote into a cache
	cache := bytes.Repeat([]byte{0xca}, 4096)
	copy(b.Bytes()[4096:], cache)
	if err := b.SyncToDevice(4096, 4096); err != nil {
		t.Fatal(err)
	}
	if err := m.growBanks(); err != nil {
		t.Fatal(err)
	}
	g := m.banks["scratch"].Bytes()
	switch {
	case len(g) != whole:
		t.Fatalf("grown to %d bytes, want %d", len(g), whole)
	case !bytes.Equal(g[4096:8192], cache):
		t.Error("the cache's rows were lost")
	case !bytes.Equal(g[:4096], img[:4096]) || !bytes.Equal(g[8192:text], img[8192:text]):
		t.Error("the text's weights changed")
	case !bytes.Equal(g[text:image], img[text:]):
		t.Error("the towers' weights are not the image's")
	case bytes.Count(g[image:], []byte{0}) != whole-image:
		t.Error("past the image's end the bank is not zero")
	}
	if err := m.growBanks(); err != nil || &m.banks["scratch"].Bytes()[0] != &g[0] {
		t.Errorf("a whole bank was grown again (%v)", err)
	}
}
