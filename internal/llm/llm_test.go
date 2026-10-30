package llm

import "testing"

// FormatOf: KeepFP16 over every rule, then the last FormatFor rule that matches, then Format; the
// size estimate follows it.
func TestFormatOf(t *testing.T) {
	o := Options{Format: "lut6", FormatFor: []string{"mlp=lut8", "mlp.down=lut4"}, KeepFP16: []string{"embed_tokens"}}
	for name, want := range map[string]string{
		"model.layers.0.mlp.up_proj.weight":      "lut8",
		"model.layers.0.mlp.down_proj.weight":    "lut4",
		"model.layers.0.self_attn.q_proj.weight": "lut6",
		"model.embed_tokens.weight":              "fp16",
	} {
		if got := o.FormatOf(name); got != want {
			t.Errorf("%s: %s, want %s", name, got, want)
		}
	}
	shape := []int{32, 64}
	for name, want := range map[string]int64{
		"mlp.up_proj":  32*64 + 2*256*2,    // 8 bits, a 256 entry table per 16 rows
		"mlp.down":     32*64/2 + 2*16*2,   // 4 bits
		"q_proj":       32*64*6/8 + 2*64*2, // 6 bits
		"embed_tokens": 32 * 64 * 2,        // fp16
	} {
		if got := tensorBytes(name, shape, o); got != want {
			t.Errorf("%s: %d bytes, want %d", name, got, want)
		}
	}
}
