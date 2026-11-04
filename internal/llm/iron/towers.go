package iron

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
)

// ImageSize returns the size the vision tower takes an image of w x h pixels at (Gemma 4's
// processor, VisionConfig.size): whole blocks of pool x patch pixels at the image's aspect
// ratio, as many as the soft-token budget holds, and the soft tokens it gives.
func (m *Model) ImageSize(w, h int) (ow, oh, tokens int, err error) {
	mm := m.Manifest.Multimodal
	if mm == nil || m.vision == nil {
		return 0, 0, 0, errors.New("iron: the model has no vision tower")
	}
	v := mm.Vision
	if mm.Kind == "lfm3vl" {
		// one image size: every image is resized (stretched) to it, as hwnpu's Core ML programs
		return v.Width, v.Height, v.ImageTokens, nil
	}
	side := v.PoolingKernelSize * v.PatchSize
	maxPatches := v.ImageTokens * v.PoolingKernelSize * v.PoolingKernelSize
	target := float64(maxPatches * v.PatchSize * v.PatchSize)
	factor := math.Sqrt(target / float64(h*w))
	oh = int(math.Floor(factor*float64(h)/float64(side))) * side
	ow = int(math.Floor(factor*float64(w)/float64(side))) * side
	if oh == 0 && ow == 0 {
		return 0, 0, 0, fmt.Errorf("iron: a %dx%d image rounds to no %d-pixel block", w, h, side)
	}
	maxSide := v.ImageTokens * side
	switch {
	case oh == 0:
		oh, ow = side, min(int(math.Floor(float64(w)/float64(h)))*side, maxSide)
	case ow == 0:
		ow, oh = side, min(int(math.Floor(float64(h)/float64(w)))*side, maxSide)
	}
	if float64(oh*ow) > target {
		return 0, 0, 0, fmt.Errorf("iron: resizing %dx%d to %dx%d exceeds %d patches", w, h, ow, oh, maxPatches)
	}
	return ow, oh, oh * ow / (side * side), nil
}

// RunVision runs the vision tower on img: resized on the host to ImageSize (antialiased bicubic,
// as the processor's), so the tower's own resampling is 1:1, and its soft tokens left where
// a placed prompt chunk reads them. It returns the soft tokens' count.
func (m *Model) RunVision(ctx context.Context, img image.Image) (int, error) {
	if mm := m.Manifest.Multimodal; mm != nil && mm.Kind == "lfm3vl" {
		return m.runLfm3VlVision(ctx, img)
	}
	b := img.Bounds()
	ow, oh, tokens, err := m.ImageSize(b.Dx(), b.Dy())
	if err != nil {
		return 0, err
	}
	mm, v := m.Manifest.Multimodal, m.vision
	vc := mm.Vision
	g := v.m.Grid
	if g == nil || g.Side == 0 {
		return 0, errors.New("iron: the vision tower has no grid of words")
	}
	hk, wk := oh/g.Side, ow/g.Side
	row := -1
	for i, p := range g.Pairs {
		if p[0] == hk && p[1] == wk {
			row = i
			break
		}
	}
	if row < 0 {
		return 0, fmt.Errorf("iron: no words for an image of %dx%d blocks", wk, hk)
	}
	if n := oh / vc.PatchSize * (ow / vc.PatchSize); n > vc.Patches {
		return 0, fmt.Errorf("iron: %d patches past the tower's %d", n, vc.Patches)
	}
	line := (3*ow + vc.Chunk - 1) / vc.Chunk * vc.Chunk
	if oh*line > vc.ImageChunks*vc.Chunk {
		return 0, fmt.Errorf("iron: a %dx%d image does not fit the tower's input", ow, oh)
	}
	rgb := lfm3vl.Resize(img, ow, oh, "bicubic")
	buf := make([]byte, oh*line)
	for y := 0; y < oh; y++ {
		copy(buf[y*line:], rgb[y*3*ow:(y+1)*3*ow])
	}
	input := towerInput(mm, mm.ImageToken)
	own := map[string]int64{}
	walked := 0
	for _, gr := range vc.Groups {
		walked += max(1, min(hk-gr[0], gr[1]))
	}
	for k, x := range map[string]int{"n": oh / vc.PatchSize * (ow / vc.PatchSize), "hk": hk, "wk": wk,
		"walked": vc.PoolingKernelSize * vc.PoolingKernelSize * wk * walked,
		"height": oh, "width": ow, "out_height": oh, "out_width": ow} {
		own[input+"_"+k] = int64(x)
	}
	if err := m.runTower(ctx, v, input, buf, own, row); err != nil {
		return 0, err
	}
	return tokens, nil
}

// AudioTokens returns the valid feature frames and the soft tokens of a clip of n samples (at
// the tower's sample rate): Gemma 4's AudioConfig.frames and tokens.
func (m *Model) AudioTokens(n int) (frames, tokens int, err error) {
	mm := m.Manifest.Multimodal
	if mm == nil || m.audio == nil {
		return 0, 0, errors.New("iron: the model has no audio tower")
	}
	a := mm.Audio
	if mm.Kind == "lfm3audio" {
		// lfm3audio.Features' frames: the clip padded on the right to whole hops past a frame
		padded := lfm3audioFrame
		if n > lfm3audioFrame {
			padded = n + (lfm3audioHop-(n-lfm3audioFrame)%lfm3audioHop)%lfm3audioHop
		}
		frames = (padded-lfm3audioFrame)/lfm3audioHop + 1
		if frames > a.MaxFrames {
			return 0, 0, fmt.Errorf("iron: a clip of %.1f s is past the encoder's %.1f s", float64(n)/16000, float64(a.MaxFrames)/100)
		}
		return frames, (frames + 7) / 8, nil
	}
	audio := min(n, a.MaxSamples) - (a.Frame - a.Frame/2)
	frames = max(0, (audio+a.Hop-1)/a.Hop)
	tokens = ((frames+1)/2 + 1) / 2
	if tokens < 1 || tokens > a.MaxTokens {
		return 0, 0, fmt.Errorf("iron: %d samples give %d soft tokens, not 1 to %d", n, tokens, a.MaxTokens)
	}
	return frames, tokens, nil
}

// RunAudio runs the audio tower on a mono clip at the tower's sample rate (16 kHz; past 30 s it
// is cut), its soft tokens left where a placed prompt chunk reads them. It returns their count.
func (m *Model) RunAudio(ctx context.Context, wave []float32) (int, error) {
	if mm := m.Manifest.Multimodal; mm != nil && mm.Kind == "lfm3audio" {
		return m.runLfm3Audio(ctx, wave)
	}
	frames, tokens, err := m.AudioTokens(len(wave))
	if err != nil {
		return 0, err
	}
	mm := m.Manifest.Multimodal
	a, v := mm.Audio, m.audio
	g := v.m.Grid
	if g == nil || frames-1 >= len(g.Frames) || g.Frames[frames-1] != frames {
		return 0, fmt.Errorf("iron: no words for %d frames", frames)
	}
	buf := make([]byte, 4*(4*a.MaxTokens+1)*a.Hop)
	held := min(len(wave), a.MaxSamples, len(buf)/4-a.Hop)
	for i := 0; i < held; i++ {
		binary.LittleEndian.PutUint32(buf[4*(a.Hop+i):], math.Float32bits(wave[i]))
	}
	input := towerInput(mm, mm.AudioToken)
	own := map[string]int64{input + "_n": int64((frames + 1) / 2), input + "_frames": int64(frames), input + "_tokens": int64(tokens)}
	if err := m.runTower(ctx, v, input, buf, own, frames-1); err != nil {
		return 0, err
	}
	return tokens, nil
}

// towerInput is the name of a tower's input (its values are named after it: rgb_n).
func towerInput(mm *Multimodal, token int32) string {
	vs := mm.TowerValues[fmt.Sprint(token)]
	if len(vs) > 0 {
		for i := 0; i < len(vs[0]); i++ {
			if vs[0][i] == '_' {
				return vs[0][:i]
			}
		}
	}
	if token == mm.ImageToken {
		return "rgb"
	}
	return "wave"
}

// HasVision and HasAudio report whether the export holds the towers.
func (m *Model) HasVision() bool { return m.vision != nil && m.placed != nil }
func (m *Model) HasAudio() bool  { return m.audio != nil && m.placed != nil }

// runLfm3VlVision runs LFM3-VL's vision tower on img: resized to the tower's size and cut into
// patches normalized by the checkpoint's image processor (lfm3vl.Processor, as the Core ML
// encoder's input), in bf16.
func (m *Model) runLfm3VlVision(ctx context.Context, img image.Image) (int, error) {
	mm := m.Manifest.Multimodal
	v := mm.Vision
	if m.vlProc == nil {
		raw, err := os.ReadFile(filepath.Join(m.dir, "config.json"))
		if err != nil {
			return 0, err
		}
		cfg, err := lfm3vl.ParseConfig(raw)
		if err != nil {
			return 0, err
		}
		if m.vlProc, err = lfm3vl.LoadProcessor(m.dir, cfg); err != nil {
			return 0, err
		}
	}
	patches := m.vlProc.Patches(img, v.Width, v.Height)
	if len(patches) != v.Patches*v.PatchDim {
		return 0, fmt.Errorf("iron: %d patch values for %d patches of %d", len(patches), v.Patches, v.PatchDim)
	}
	buf := make([]byte, 2*len(patches))
	for i, x := range patches {
		binary.LittleEndian.PutUint16(buf[2*i:], bf16(x))
	}
	if err := m.runTower(ctx, m.vision, towerInput(mm, mm.ImageToken), buf, map[string]int64{}, 0); err != nil {
		return 0, err
	}
	return v.ImageTokens, nil
}

// bf16 rounds x to bfloat16 (to nearest, ties to even).
func bf16(x float32) uint16 {
	b := math.Float32bits(x)
	if b&0x7fffffff > 0x7f800000 { // NaN
		return uint16(b>>16) | 0x40
	}
	return uint16((b + 0x7fff + (b>>16)&1) >> 16)
}

// lfm3audio.Features' framing: 400 samples a frame every 160.
const (
	lfm3audioFrame = 400
	lfm3audioHop   = 160
)

// runLfm3Audio runs LFM3-Audio's audio encoder on a mono clip at 16 kHz: its log-mel features
// (lfm3audio.Features, as the Core ML encoder's input) after a zero row, zeros past them, in
// bf16, with the valid rows of its three stride-2 stages.
func (m *Model) runLfm3Audio(ctx context.Context, wave []float32) (int, error) {
	mm := m.Manifest.Multimodal
	a := mm.Audio
	frames, tokens, err := m.AudioTokens(len(wave))
	if err != nil {
		return 0, err
	}
	mel, n := lfm3audio.Features(wave, a.Mels)
	if n != frames {
		return 0, fmt.Errorf("iron: %d feature frames, %d counted", n, frames)
	}
	buf := make([]byte, 2*(a.MaxFrames+2)*a.Mels)
	for i, x := range mel {
		binary.LittleEndian.PutUint16(buf[2*(a.Mels+i):], bf16(x))
	}
	n1 := (frames + 1) / 2
	n2 := (n1 + 1) / 2
	input := towerInput(mm, mm.AudioToken)
	own := map[string]int64{input + "_n1": int64(n1), input + "_n2": int64(n2), input + "_tokens": int64(tokens)}
	// its words of rows bounded by the clip are looked up by frames, as Gemma 4's
	if g := m.audio.m.Grid; g == nil || frames-1 >= len(g.Frames) || g.Frames[frames-1] != frames {
		return 0, fmt.Errorf("iron: no words for %d frames", frames)
	}
	if err := m.runTower(ctx, m.audio, input, buf, own, frames-1); err != nil {
		return 0, err
	}
	return tokens, nil
}
