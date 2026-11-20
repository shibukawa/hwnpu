package gptoss

import (
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// gpt-oss-20b's config.json, 4 layers and a small vocabulary.
const cfg20b = `{"architectures":["GptOssForCausalLM"],"attention_bias":true,"experts_per_token":4,"head_dim":64,
"hidden_size":2880,"intermediate_size":2880,"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
"max_position_embeddings":131072,"model_type":"gpt_oss","num_attention_heads":64,"num_experts_per_tok":4,"num_hidden_layers":4,
"num_key_value_heads":8,"num_local_experts":32,"quantization_config":{"quant_method":"mxfp4"},"rms_norm_eps":1e-05,
"rope_scaling":{"beta_fast":32.0,"beta_slow":1.0,"factor":32.0,"original_max_position_embeddings":4096,"rope_type":"yarn","truncate":false},
"rope_theta":150000,"sliding_window":128,"swiglu_limit":7.0,"tie_word_embeddings":false,"vocab_size":1024}`

func TestConfig(t *testing.T) {
	c, err := ParseConfig([]byte(cfg20b))
	if err != nil {
		t.Fatal(err)
	}
	if c.TopK != 4 || c.SwigluAlpha != 1.702 || c.Yarn.Factor != 32 || c.Yarn.Truncate {
		t.Errorf("config %+v", c)
	}
	if !slices.Equal(c.PagedLayers(), []int{1, 3}) {
		t.Errorf("paged layers %v", c.PagedLayers())
	}
	// transformers' _compute_yarn_parameters for gpt-oss-20b: inv_freq[0] = 1 (extrapolated),
	// the last interpolated (1/(32·theta^(62/64))), attention factor 0.1·ln 32 + 1
	inv, f := c.RopeInvFreq()
	if math.Abs(inv[0]-1) > 1e-12 || math.Abs(f-(0.1*math.Log(32)+1)) > 1e-12 {
		t.Errorf("inv_freq[0] %g, factor %g", inv[0], f)
	}
	last := 1 / (32 * math.Pow(150000, 62.0/64))
	if math.Abs(inv[31]-last)/last > 1e-9 {
		t.Errorf("inv_freq[31] %g, want %g", inv[31], last)
	}
}

// Every layer ends with its experts: a part is one layer's step, and a full attention layer gets
// a part of no layers from the host's attention to its router.
func TestPlanAndBuild(t *testing.T) {
	c, err := ParseConfig([]byte(cfg20b))
	if err != nil {
		t.Fatal(err)
	}
	for _, opts := range []llm.Options{{Format: "lut8", Seq: 8}, {Format: "lut8", Seq: 8, Paged: 1024, Cache: 4096}} {
		plan, err := llm.PlanRouted(c, opts, 1<<32)
		if err != nil {
			t.Fatal(err)
		}
		want := [][2]int{{0, llm.NoLayers}, {0, 1}, {1, 1}, {2, 1}, {3, 1}}
		if opts.Paged > 0 {
			want = [][2]int{{0, llm.NoLayers}, {0, 1}, {1, llm.NoLayers}, {1, 1}, {2, 1}, {3, llm.NoLayers}, {3, 1}}
		}
		var got [][2]int
		for _, p := range plan {
			got = append(got, [2]int{p.Start, p.Layers})
		}
		if !slices.Equal(got, want) {
			t.Fatalf("paged %d: plan %v, want %v", opts.Paged, got, want)
		}
		w := llm.NewSynthetic(c)
		for _, p := range plan {
			o := opts
			o.Start, o.Layers = p.Start, p.Layers
			o.LastRow = opts.Cache > 0
			blobs, err := mlpackage.NewBlobWriter(filepath.Join(t.TempDir(), "w.bin"))
			if err != nil {
				t.Fatal(err)
			}
			ins, outs, err := c.BuildCached(mlpackage.NewBuilder("CoreML8", blobs), w, o)
			blobs.Close()
			if err != nil {
				t.Fatalf("part %v: %v", p, err)
			}
			var in, out []string
			for _, f := range ins {
				in = append(in, f.Name)
			}
			for _, f := range outs {
				out = append(out, f.Name)
			}
			t.Logf("paged %d part %d+%d: in %s; out %s", opts.Paged, p.Start, p.Layers, strings.Join(in, " "), strings.Join(out, " "))
		}
	}
}
