package torchzip

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// testdata/small.pth is torch.save of {"state_dict": OrderedDict(a, t, h, b, p, s), "metadata":
// {"kwargs": {...}}} (torch 2.15): a float32 tensor, a transposed view of it, float16 and
// bfloat16 tensors, a Parameter and a slice at an offset into a's storage.
func TestSmall(t *testing.T) {
	f, err := Open(filepath.Join("testdata", "small.pth"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	want := []string{"state_dict.a", "state_dict.b", "state_dict.h", "state_dict.p", "state_dict.s", "state_dict.t"}
	if got := f.Names(); !slices.Equal(got, want) {
		t.Fatalf("names %v, want %v", got, want)
	}
	a, err := f.Float32("state_dict.a")
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != float32(i) {
			t.Fatalf("a[%d] = %v", i, v)
		}
	}
	tt, _ := f.Lookup("state_dict.t")
	if !slices.Equal(tt.Shape, []int{4, 3, 2}) {
		t.Fatalf("t shape %v", tt.Shape)
	}
	tv, err := f.Float32("state_dict.t")
	if err != nil {
		t.Fatal(err)
	}
	// t[i][j][k] = a[k][j][i] = k*12 + j*4 + i
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 2; k++ {
				if got := tv[i*6+j*2+k]; got != float32(k*12+j*4+i) {
					t.Fatalf("t[%d][%d][%d] = %v", i, j, k, got)
				}
			}
		}
	}
	check := func(name string, want ...float32) {
		t.Helper()
		got, err := f.Float32(name)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
	check("state_dict.h", 1.5, -2.25, 65504)
	check("state_dict.b", 3, -0.5)
	check("state_dict.p", 1, 2, 3, 4)
	check("state_dict.s", 16, 17, 18, 19, 20, 21, 22, 23)
	meta := f.Root.(map[any]any)["metadata"].(map[any]any)["kwargs"].(map[any]any)
	if r, _ := ints(meta["rates"]); !slices.Equal(r, []int{2, 3}) || meta["x"] != 1.5 || meta["on"] != true {
		t.Fatalf("metadata %v", meta)
	}
}

// TestMossVAE reads the DAC VAE of OpenMOSS-Team/MOSS-SoundEffect-v2.0 when it has been
// downloaded (HWNPU_MOSS_VAE, the vae_128d_48k.pth).
func TestMossVAE(t *testing.T) {
	p := os.Getenv("HWNPU_MOSS_VAE")
	if p == "" {
		t.Skip("HWNPU_MOSS_VAE is not set")
	}
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	x, ok := f.Lookup("state_dict.decoder.model.1.block.1.weight_v")
	if !ok || !slices.Equal(x.Shape, []int{2048, 1024, 16}) {
		t.Fatalf("weight_v %v", x)
	}
	v, err := f.Float32("state_dict.decoder.model.7.weight_v")
	if err != nil || len(v) != 64*7 {
		t.Fatal(len(v), err)
	}
	kw := f.Root.(map[any]any)["metadata"].(map[any]any)["kwargs"].(map[any]any)
	if r, _ := ints(kw["decoder_rates"]); !slices.Equal(r, []int{8, 5, 4, 3, 2}) {
		t.Fatalf("decoder_rates %v", kw["decoder_rates"])
	}
	t.Logf("%d tensors", len(f.Names()))
}
