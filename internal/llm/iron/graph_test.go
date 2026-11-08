package iron

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
)

// With SwapVariants a graph keeps one set of banks: Use reads the variant's image into the
// bank's start (the weights) and leaves the rest (the work area) as it is, reads nothing when
// the variant is already there, and after a failed read reads the next variant whole again.
func TestGraphSwapVariants(t *testing.T) {
	const bank, image = 4 << 12, 3<<12 + 100
	dir := t.TempDir()
	imgs := [][]byte{bytes.Repeat([]byte{1}, image), bytes.Repeat([]byte{2}, image), bytes.Repeat([]byte{3}, image)}
	names := []string{"bank_scratch.bin", "bank_g1_scratch.bin", "bank_g2_scratch.bin"}
	for i, n := range names {
		if err := os.WriteFile(filepath.Join(dir, n), imgs[i], 0o644); err != nil {
			t.Fatal(err)
		}
	}
	man := &GraphManifest{
		Name:  "blocks",
		Banks: map[string]Bank{"scratch": {Bytes: bank, Image: names[0]}},
	}
	for _, n := range names {
		man.Variants = append(man.Variants, map[string]string{"scratch": n})
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend(simBackend))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	buf, err := dev.AllocBuffer(bank)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	copy(buf.Bytes(), imgs[0])
	banks := map[string]*hwnpu.Buffer{"scratch": buf}
	g := &Graph{Manifest: man, dev: dev, dir: dir, banks: banks, variants: []map[string]*hwnpu.Buffer{banks}, swapped: true}

	if g.Variants() != 3 {
		t.Fatalf("%d variants, want 3", g.Variants())
	}
	work := bytes.Repeat([]byte{0xaa}, bank-image)
	copy(buf.Bytes()[image:], work)
	for _, i := range []int{2, 1, 1, 0} {
		if err := g.Use(i); err != nil {
			t.Fatal(err)
		}
		b := buf.Bytes()
		if !bytes.Equal(b[:image], imgs[i]) {
			t.Fatalf("variant %d: the bank does not hold its image", i)
		}
		if !bytes.Equal(b[image:], work) {
			t.Fatalf("variant %d: the work area past the image changed", i)
		}
	}
	if err := g.Use(3); err == nil {
		t.Error("variant 3 of 3 was taken")
	}

	// a read that fails leaves the bank unknown: the variant there before is read again
	if err := os.Rename(filepath.Join(dir, names[1]), filepath.Join(dir, "gone")); err != nil {
		t.Fatal(err)
	}
	if err := g.Use(1); err == nil {
		t.Fatal("a missing image was read")
	}
	clear(buf.Bytes()[:image]) // what a part-read would leave
	if err := g.Use(0); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf.Bytes()[:image], imgs[0]) {
		t.Error("after a failed read, variant 0 was not read again")
	}
}
