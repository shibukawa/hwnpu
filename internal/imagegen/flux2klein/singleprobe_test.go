package flux2klein

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// TestWriteSingleProbe writes one group of the DiT's single blocks as a Core AI program of one
// function, to compile it offline (coreai-build compile --architecture ...) and find which
// configuration a chip's compiler leaves off the Neural Engine. HWNPU_TEST_KLEIN_PROBE_OUT is the
// .aimodel, HWNPU_TEST_KLEIN_DIR the checkpoint, HWNPU_TEST_KLEIN_PROBE_REFS the edit's references
// (side divisors, "2,2,2": three of half the sides; empty: no edit), _SIZE "W,H" (1024,1024),
// _GROUP "first,count" (0,5), _KIND "double" for the double blocks' group in place, _PIECES milpart.PieceKeys for this program, _FORMAT (lut8),
// _REF a Core ML package of the same part too, for the CPU interpreter to check the program's
// output on a device against (go run ./tools/coreai/aicheck run -cpu -ref REF OUT#FUNCTION).
func TestWriteSingleProbe(t *testing.T) {
	out, root := os.Getenv("HWNPU_TEST_KLEIN_PROBE_OUT"), os.Getenv("HWNPU_TEST_KLEIN_DIR")
	if out == "" || root == "" {
		t.Skip("HWNPU_TEST_KLEIN_PROBE_OUT, HWNPU_TEST_KLEIN_DIR")
	}
	w, h, first, count, format := 1024, 1024, 0, 5, "lut8"
	if s := os.Getenv("HWNPU_TEST_KLEIN_PROBE_SIZE"); s != "" {
		fmt.Sscanf(s, "%d,%d", &w, &h)
	}
	if s := os.Getenv("HWNPU_TEST_KLEIN_PROBE_GROUP"); s != "" {
		fmt.Sscanf(s, "%d,%d", &first, &count)
	}
	if s := os.Getenv("HWNPU_TEST_KLEIN_PROBE_FORMAT"); s != "" {
		format = s
	}
	if s := os.Getenv("HWNPU_TEST_KLEIN_PROBE_PIECES"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		defer func(k int) { milpart.PieceKeys = k }(milpart.PieceKeys)
		milpart.PieceKeys = n
	}
	c, err := mil.ReadDit(root, w, h)
	if err != nil {
		t.Fatal(err)
	}
	if s := os.Getenv("HWNPU_TEST_KLEIN_PROBE_REFS"); s != "" {
		for _, f := range strings.Split(s, ",") {
			d, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			c.Refs = append(c.Refs, d)
		}
		if err := c.CheckRefs(); err != nil {
			t.Fatal(err)
		}
	}
	wt, err := mil.OpenWeights(root, c)
	if err != nil {
		t.Fatal(err)
	}
	defer wt.Close()
	fn := SizeFunction(w, h)
	if len(c.Refs) > 0 {
		fn = EditFunction(w, h, c.Refs)
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		t.Fatal(err)
	}
	opts := llm.Options{Backend: llm.BackendCoreAI, Format: format}
	part := func() *mil.Part {
		if os.Getenv("HWNPU_TEST_KLEIN_PROBE_KIND") == "double" {
			return mil.Doubles(c, first, count)
		}
		return mil.Singles(c, first, count)
	}
	if _, err := llm.WriteAIFunctions(out, wt, []llm.AIFunction{{Name: fn, Model: part(), Options: opts}}); err != nil {
		t.Fatal(err)
	}
	if ref := os.Getenv("HWNPU_TEST_KLEIN_PROBE_REF"); ref != "" {
		if _, err := llm.WritePart(ref, part(), wt, llm.Options{Backend: llm.BackendCoreML, Format: format}, nil); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%s: %s, %d rows, PieceKeys %d", out, fn, c.Rows(), milpart.PieceKeys)
}
