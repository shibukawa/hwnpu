package supertonic

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
)

// Style is a voice: the vectors the text side (style_ttl, (1, 50, 256)) and the duration
// predictor (style_dp, (1, 8, 16)) read, from a voice_styles/*.json.
type Style struct {
	TTL, DP []float32
}

// Shapes of a style's vectors.
var (
	TTLShape = []int{1, 50, 256}
	DPShape  = []int{1, 8, 16}
)

// LoadStyle reads a voice style.
func LoadStyle(path string) (Style, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Style{}, err
	}
	var f struct {
		TTL, DP struct {
			Data json.RawMessage `json:"data"`
			Dims []int           `json:"dims"`
		}
	}
	// the files name them style_ttl and style_dp
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return Style{}, fmt.Errorf("supertonic: %s: %w", path, err)
	}
	json.Unmarshal(m["style_ttl"], &f.TTL)
	json.Unmarshal(m["style_dp"], &f.DP)
	var s Style
	for _, v := range []struct {
		data json.RawMessage
		dims []int
		want []int
		out  *[]float32
	}{{f.TTL.Data, f.TTL.Dims, TTLShape, &s.TTL}, {f.DP.Data, f.DP.Dims, DPShape, &s.DP}} {
		if !slices.Equal(v.dims, v.want) {
			return Style{}, fmt.Errorf("supertonic: %s: a style of %v, want %v", path, v.dims, v.want)
		}
		var x any
		if err := json.Unmarshal(v.data, &x); err != nil {
			return Style{}, fmt.Errorf("supertonic: %s: %w", path, err)
		}
		*v.out = flatten(x, nil)
		if n := v.want[0] * v.want[1] * v.want[2]; len(*v.out) != n {
			return Style{}, fmt.Errorf("supertonic: %s: %d values, want %d", path, len(*v.out), n)
		}
	}
	return s, nil
}

// flatten appends the numbers of nested arrays in their order.
func flatten(x any, out []float32) []float32 {
	switch v := x.(type) {
	case float64:
		return append(out, float32(v))
	case []any:
		for _, e := range v {
			out = flatten(e, out)
		}
	}
	return out
}
