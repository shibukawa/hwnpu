// Package pitch estimates the fundamental frequency (F0) of a voice or an instrument with
// SwiftF0 (lars76/swift-f0, MIT): a pitch and a voicing confidence every 16 ms of 16 kHz audio.
//
// SwiftF0 ships as one ONNX graph. Its front end is three short-time Fourier transforms (windows
// of 1024, 2048 and 512 samples, hops of 256) whose magnitudes a filter bank folds into 128 bands
// each and a log takes; the host computes these here (Features). Its body is a residual CNN of 12
// channels over (bands, frames), with a head of 95 log-spaced pitch logits and one of the voicing
// logit; that is the program the NPU runs (Program). The pitch is the softmax-weighted mean of
// the log frequencies of the best logit and its neighbours, as the graph's tail computes (decode).
package pitch

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
)

// The model's constants (swift_f0/core.py).
const (
	SampleRate = 16000
	Hop        = 256
	FMin       = 46.875
	FMax       = 2093.75
	// Bins are the pitch candidates; Bands the filter bank's bands of each transform.
	Bins  = 95
	Bands = 128
	// LeftFrames and LookaheadFrames are the receptive field around a frame: the windows of
	// WindowFrames that Detect runs each take these many frames of context on either side.
	LeftFrames      = 11
	LookaheadFrames = 10
	WindowFrames    = 1875
	// ProgramFrames are the frames of one call of the program: a window and its context.
	ProgramFrames = WindowFrames + LeftFrames + LookaheadFrames
	// silencePeak: frames whose audio peaks below it get confidence 0 (digital silence makes
	// the model answer with random voiced frames).
	silencePeak = 1e-3
)

// ModelSHA256 is the model.onnx of swift-f0 0.3.0 (commit d1ba77f), whose initializers the
// names below are of.
const ModelSHA256 = "6385e8c2ebc3872e82c9ff5946228de44cd3be77a750ea53698b7dbfe94b0a22"

// transform is one of the three spectrograms: the window's length and the names of its window,
// and of the indices and values of its filter bank, scattered into a (N/2+1, 128) matrix.
type transform struct {
	n                      int
	window, index, weights string
}

// transforms in the order of the CNN's input channels.
var transforms = []transform{
	{1024, "v41", "v123", "v124"},
	{2048, "v161", "v241", "v242"},
	{512, "v262", "v342", "v343"},
}

// conv is one convolution of the CNN, in the graph's order: its weight and bias, the kernel's
// dilation over (bands, frames) and its pads (top, bottom, left, right in MIL's order).
type conv struct {
	weight, bias string
	dil          [2]int
	pad          [4]int
}

// body: the stem, then six residual blocks of two convolutions each (y = relu(x + c2(relu(c1(x))))).
var (
	stem = conv{"v353", "v354", [2]int{1, 1}, [4]int{1, 1, 1, 1}}
	body = [][2]conv{
		{{"v357", "v358", [2]int{1, 1}, [4]int{3, 3, 1, 1}}, {"v361", "v362", [2]int{1, 1}, [4]int{3, 3, 1, 1}}},
		{{"v366", "v367", [2]int{17, 1}, [4]int{51, 51, 0, 0}}, {"v370", "v371", [2]int{17, 1}, [4]int{51, 51, 0, 0}}},
		{{"v375", "v376", [2]int{27, 1}, [4]int{54, 54, 0, 0}}, {"v379", "v380", [2]int{27, 1}, [4]int{54, 54, 0, 0}}},
		{{"v384", "v385", [2]int{40, 1}, [4]int{40, 40, 0, 0}}, {"v388", "v389", [2]int{40, 1}, [4]int{40, 40, 0, 0}}},
		{{"v393", "v394", [2]int{48, 1}, [4]int{48, 48, 0, 0}}, {"v397", "v398", [2]int{48, 1}, [4]int{48, 48, 0, 0}}},
		{{"v402", "v403", [2]int{2, 1}, [4]int{6, 6, 1, 1}}, {"v406", "v407", [2]int{2, 1}, [4]int{6, 6, 1, 1}}},
	}
	// the head over the lowest 96 bands: to 95 rows of 4 channels, then their pitch logits (1x1)
	// and the voicing branch's 4 channels
	head1     = conv{"v413", "v414", [2]int{1, 1}, [4]int{1, 0, 1, 1}}
	head2     = conv{"v417", "v418", [2]int{1, 1}, [4]int{2, 2, 0, 0}}
	pitchConv = conv{"v425", "v426", [2]int{1, 1}, [4]int{0, 0, 0, 0}}
	// the voicing logit from the 4 channels' maxima over the bins and the input's mean, by frame
	voice1 = conv{"v430", "v431", [2]int{1, 1}, [4]int{0, 0, 1, 1}}
	voice2 = conv{"v434", "v435", [2]int{1, 1}, [4]int{0, 0, 0, 0}}
	// headBands are the bands the head reads.
	headBands = 96
	// binHz are the candidates' frequencies, binLog their logs (float64 in the graph).
	binHz, binLog = "v438", "v455"
)

// Model holds SwiftF0's weights and tables.
type Model struct {
	t      map[string]tensor
	banks  [][]float64 // per transform, (N/2+1) x 128
	sparse [][][]tap   // per transform and band, the bins of the bank that are not 0
	hz     []float64
	logHz  []float64
	source string
}

// Load reads model.onnx.
func Load(path string) (*Model, error) {
	t, err := readInitializers(path)
	if err != nil {
		return nil, err
	}
	m := &Model{t: t, source: path}
	for _, tr := range transforms {
		w, idx, vals := t[tr.window], t[tr.index], t[tr.weights]
		bins := tr.n/2 + 1
		if len(w.vals) != tr.n || len(idx.vals) != len(vals.vals) {
			return nil, fmt.Errorf("pitch: %s: no window of %d and filter bank (%d, %d)", path, tr.n, len(idx.vals), len(vals.vals))
		}
		bank := make([]float64, bins*Bands)
		for i, k := range idx.vals {
			if k < 0 || int(k) >= len(bank) {
				return nil, fmt.Errorf("pitch: %s: filter bank index %v", path, k)
			}
			bank[int(k)] = vals.vals[i]
		}
		m.banks = append(m.banks, bank)
		bands := make([][]tap, Bands)
		for k := range bins {
			for b := range Bands {
				if v := bank[k*Bands+b]; v != 0 {
					bands[b] = append(bands[b], tap{k, v})
				}
			}
		}
		m.sparse = append(m.sparse, bands)
	}
	m.hz, m.logHz = t[binHz].vals, t[binLog].vals
	if len(m.hz) != Bins || len(m.logHz) != Bins {
		return nil, fmt.Errorf("pitch: %s: %d and %d pitch bins, want %d", path, len(m.hz), len(m.logHz), Bins)
	}
	for _, c := range append([]conv{stem, head1, head2, pitchConv, voice1, voice2}, flat(body)...) {
		if _, ok := t[c.weight]; !ok {
			return nil, fmt.Errorf("pitch: %s has no %s: not swift-f0 0.3.0's model", path, c.weight)
		}
		if _, ok := t[c.bias]; !ok {
			return nil, fmt.Errorf("pitch: %s has no %s", path, c.bias)
		}
	}
	return m, nil
}

// tap is a bin of a band of a filter bank and its weight.
type tap struct {
	bin int
	w   float64
}

func flat(b [][2]conv) []conv {
	var out []conv
	for _, p := range b {
		out = append(out, p[0], p[1])
	}
	return out
}

// Frames is how many frames the graph gives a clip of n samples: one per hop, at least one.
func Frames(n int) int { return max(1, n/Hop) }

// Features computes the CNN's input of a clip of 16 kHz samples, (3, 128, Frames) row major, in
// float64: for each transform, the clip padded with zeros to at least a hop and to whole hops,
// and by half a window on both sides; frame t is the window from sample t*Hop of that, and its
// band is log(|STFT| · bank + 1e-8).
func (m *Model) Features(x []float32) []float64 {
	frames := Frames(len(x))
	n := max(len(x), Hop)
	n = (n + Hop - 1) / Hop * Hop
	out := make([]float64, len(transforms)*Bands*frames)
	for c, tr := range transforms {
		w := m.t[tr.window].vals
		half := tr.n / 2
		padded := make([]float64, half+n+half)
		for i, v := range x {
			padded[half+i] = float64(v)
		}
		bins := half + 1
		bands := m.sparse[c]
		tw := twiddles(tr.n)
		buf := make([]complex128, tr.n)
		mag := make([]float64, bins)
		for t := range frames {
			for i := range buf {
				buf[i] = complex(padded[t*Hop+i]*w[i], 0)
			}
			fftWith(buf, tw)
			for k := range mag {
				mag[k] = cmplx.Abs(buf[k])
			}
			for b, taps := range bands {
				var s float64
				for _, p := range taps {
					s += mag[p.bin] * p.w
				}
				out[(c*Bands+b)*frames+t] = math.Log(s + 1e-8)
			}
		}
	}
	return out
}

// twiddles are e^(-2πik/n) for k < n/2.
func twiddles(n int) []complex128 {
	tw := make([]complex128, n/2)
	for k := range tw {
		a := -2 * math.Pi * float64(k) / float64(n)
		tw[k] = complex(math.Cos(a), math.Sin(a))
	}
	return tw
}

// fft is an in-place radix-2 FFT (the length a power of two).
func fft(a []complex128) { fftWith(a, twiddles(len(a))) }

func fftWith(a []complex128, tw []complex128) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		stride := n / size
		for start := 0; start < n; start += size {
			for k := range size / 2 {
				u, v := a[start+k], a[start+k+size/2]*tw[k*stride]
				a[start+k], a[start+k+size/2] = u+v, u-v
			}
		}
	}
}

// Result is the pitch of each frame (16 ms).
type Result struct {
	// PitchHz is the F0 estimate; Confidence the voicing score (voiced at 0.5 and over);
	// LoudnessDB the level of the 32 ms centered on the frame.
	PitchHz, Confidence, LoudnessDB []float64
}

// decode turns the program's output of a window, y (96, frames) as the program lays it out
// (row 95 the voicing logits), into the pitch and confidence of frames [from, to), with the band
// [fmin, fmax], as the graph's tail does: the best candidate within the band and its two
// neighbours, softmaxed, weight the logs of their frequencies; the confidence is the voicing
// logit's sigmoid if the best candidate of all is within the band, else 0.
func (m *Model) decode(y []float32, frames, from, to int, fmin, fmax float64) (pitch, conf []float64) {
	in := make([]bool, Bins)
	for i, hz := range m.hz {
		in[i] = hz >= fmin && hz <= fmax
	}
	for t := from; t < to; t++ {
		best, all := -1, 0
		for i := range Bins {
			v := y[i*frames+t]
			if in[i] && (best < 0 || v > y[best*frames+t]) {
				best = i
			}
			if v > y[all*frames+t] {
				all = i
			}
		}
		var logits [3]float64
		hi := math.Inf(-1)
		for j := range 3 {
			i := best - 1 + j
			logits[j] = math.Inf(-1)
			if i >= 0 && i < Bins && in[i] {
				logits[j] = float64(y[i*frames+t])
			}
			hi = math.Max(hi, logits[j])
		}
		var z, s float64
		for j := range 3 {
			e := math.Exp(logits[j] - hi)
			z += e
			s += e * m.logHz[min(max(best-1+j, 0), Bins-1)]
		}
		pitch = append(pitch, math.Exp(s/z))
		c := 0.0
		if in[all] {
			c = float64(float32(1 / (1 + math.Exp(-float64(y[(Bins)*frames+t])))))
		}
		conf = append(conf, c)
	}
	return pitch, conf
}

// loudness and silence: the level of each frame of a clip from the power of its hop and the one
// before (0 before the first), and whether its hop peaks under silencePeak. A clip shorter than a
// hop is one frame.
func levels(x []float32, frames int) (db []float64, silent []bool) {
	power := make([]float64, frames)
	silent = make([]bool, frames)
	for t := range frames {
		hop := x[t*Hop : min((t+1)*Hop, len(x))]
		if len(x) < Hop {
			hop = x
		}
		peak := 0.0
		for _, v := range hop {
			power[t] += float64(v) * float64(v)
			peak = math.Max(peak, math.Abs(float64(v)))
		}
		silent[t] = peak < silencePeak
	}
	for t := range frames {
		prev := 0.0
		if t > 0 {
			prev = power[t-1]
		}
		db = append(db, 20*math.Log10(math.Max(math.Sqrt((prev+power[t])/512), 1e-7)))
	}
	return db, silent
}

// Runner runs the CNN: x (1, 3, 128, ProgramFrames) and mask (1, 1, 1, ProgramFrames) in, y (1,
// 96, ProgramFrames) out.
type Runner func(x, mask []float32) ([]float32, error)

// Detect estimates the pitch of 16 kHz samples, with candidates between fmin and fmax (0 for the
// model's range), as SwiftF0.detect does: windows of WindowFrames frames, each run with
// LeftFrames frames before and LookaheadFrames after it, so that every frame is the whole clip's.
func (m *Model) Detect(x []float32, fmin, fmax float64, run Runner) (Result, error) {
	if fmin <= 0 {
		fmin = FMin
	}
	if fmax <= 0 {
		fmax = FMax
	}
	fmin, fmax = math.Max(fmin, FMin), math.Min(fmax, FMax)
	if !(fmin < fmax) || fmax < fmin*math.Pow(FMax/FMin, 1.0/(Bins-1)) {
		return Result{}, fmt.Errorf("pitch: the band %g to %g Hz holds no candidate between %g and %g", fmin, fmax, FMin, FMax)
	}
	if len(x) == 0 {
		return Result{}, fmt.Errorf("pitch: no audio")
	}
	n := Frames(len(x))
	var r Result
	in := make([]float32, len(transforms)*Bands*ProgramFrames)
	mask := make([]float32, ProgramFrames)
	for start := 0; start < n; start += WindowFrames {
		end := min(start+WindowFrames, n)
		left := max(0, start-LeftFrames)
		win := x[left*Hop:]
		if end < n {
			win = x[left*Hop : (end+LookaheadFrames)*Hop]
		}
		frames := Frames(len(win))
		f := m.Features(win)
		clear(in)
		clear(mask)
		for c := range len(transforms) * Bands {
			for t := range frames {
				in[c*ProgramFrames+t] = float32(f[c*frames+t])
			}
		}
		for t := range frames {
			mask[t] = 1
		}
		y, err := run(in, mask)
		if err != nil {
			return Result{}, err
		}
		p, c := m.decode(y, ProgramFrames, start-left, end-left, fmin, fmax)
		db, silent := levels(win, frames)
		for i := range p {
			if silent[start-left+i] {
				c[i] = 0
			}
		}
		r.PitchHz = append(r.PitchHz, p...)
		r.Confidence = append(r.Confidence, c...)
		r.LoudnessDB = append(r.LoudnessDB, db[start-left:end-left]...)
	}
	return r, nil
}

// Voiced reports the frames whose confidence is at least threshold.
func (r Result) Voiced(threshold float64) []bool {
	return slices.Collect(func(yield func(bool) bool) {
		for _, c := range r.Confidence {
			if !yield(c >= threshold) {
				return
			}
		}
	})
}
