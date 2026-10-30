package lfm3

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// klDiv is KL(p || q) of the softmax distributions of two logit vectors.
func klDiv(p, q []float32) float64 {
	lp, lq := logSoftmax(p), logSoftmax(q)
	var kl float64
	for i := range lp {
		kl += math.Exp(lp[i]) * (lp[i] - lq[i])
	}
	return kl
}

func logSoftmax(v []float32) []float64 {
	m := float64(v[argmax(v)])
	var z float64
	for _, x := range v {
		z += math.Exp(float64(x) - m)
	}
	lz := math.Log(z) + m
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x) - lz
	}
	return out
}

func readF32(t *testing.T, path string) []float32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func readI32(t *testing.T, path string) []int32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}
