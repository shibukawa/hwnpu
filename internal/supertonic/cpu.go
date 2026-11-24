package supertonic

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// Graphs are the model's four ONNX files, in its onnx/ directory.
var Graphs = struct{ Duration, Encoder, Estimator, Vocoder string }{
	"duration_predictor.onnx", "text_encoder.onnx", "vector_estimator.onnx", "vocoder.onnx",
}

// cpuStages run the graphs with ONNX Runtime on the CPU.
type cpuStages struct {
	dp, enc, est, voc *onnxrt.Session
}

// OpenCPU opens the graphs of dir (the model's onnx/) with ONNX Runtime, each on threads
// threads (0: one).
func OpenCPU(dir string, threads int) (Stages, error) {
	rt, err := onnxrt.Load()
	if err != nil {
		return nil, err
	}
	s := &cpuStages{}
	for _, g := range []struct {
		file string
		out  **onnxrt.Session
	}{{Graphs.Duration, &s.dp}, {Graphs.Encoder, &s.enc}, {Graphs.Estimator, &s.est}, {Graphs.Vocoder, &s.voc}} {
		raw, err := onnxrt.ReadModel(filepath.Join(dir, g.file))
		if err != nil {
			s.Close()
			return nil, err
		}
		*g.out, err = rt.Open(raw.Bytes(), onnxrt.Options{Threads: threads})
		raw.Close()
		if err != nil {
			s.Close()
			return nil, fmt.Errorf("supertonic: %s: %w", g.file, err)
		}
	}
	return s, nil
}

func ones(n int) []float32 {
	m := make([]float32, n)
	for i := range m {
		m[i] = 1
	}
	return m
}

func (s *cpuStages) Duration(ctx context.Context, ids []int64, style []float32) (float32, error) {
	T := int64(len(ids))
	out, err := s.dp.Run(map[string]onnxrt.Tensor{
		"text_ids":  {Shape: []int64{1, T}, Int64: ids},
		"style_dp":  {Shape: []int64{1, 8, 16}, Data: style},
		"text_mask": {Shape: []int64{1, 1, T}, Data: ones(int(T))},
	})
	if err != nil {
		return 0, err
	}
	return out["duration"].Data[0], nil
}

func (s *cpuStages) Encode(ctx context.Context, ids []int64, style []float32) ([]float32, error) {
	T := int64(len(ids))
	out, err := s.enc.Run(map[string]onnxrt.Tensor{
		"text_ids":  {Shape: []int64{1, T}, Int64: ids},
		"style_ttl": {Shape: []int64{1, 50, 256}, Data: style},
		"text_mask": {Shape: []int64{1, 1, T}, Data: ones(int(T))},
	})
	if err != nil {
		return nil, err
	}
	return out["text_emb"].Data, nil
}

func (s *cpuStages) Estimate(ctx context.Context, x []float32, L int, emb []float32, T int, style []float32, step, total int) ([]float32, error) {
	out, err := s.est.Run(map[string]onnxrt.Tensor{
		"noisy_latent": {Shape: []int64{1, Latent, int64(L)}, Data: x},
		"text_emb":     {Shape: []int64{1, TextDim, int64(T)}, Data: emb},
		"style_ttl":    {Shape: []int64{1, 50, 256}, Data: style},
		"latent_mask":  {Shape: []int64{1, 1, int64(L)}, Data: ones(L)},
		"text_mask":    {Shape: []int64{1, 1, int64(T)}, Data: ones(T)},
		"current_step": {Shape: []int64{1}, Data: []float32{float32(step)}},
		"total_step":   {Shape: []int64{1}, Data: []float32{float32(total)}},
	})
	if err != nil {
		return nil, err
	}
	return out["denoised_latent"].Data, nil
}

func (s *cpuStages) Vocode(ctx context.Context, x []float32, L int) ([]float32, error) {
	out, err := s.voc.Run(map[string]onnxrt.Tensor{"latent": {Shape: []int64{1, Latent, int64(L)}, Data: x}})
	if err != nil {
		return nil, err
	}
	return out["wav_tts"].Data, nil
}

func (s *cpuStages) Close() error {
	for _, x := range []*onnxrt.Session{s.dp, s.enc, s.est, s.voc} {
		if x != nil {
			x.Close()
		}
	}
	return nil
}
