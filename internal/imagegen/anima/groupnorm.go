package anima

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// HostNorm is a GroupNorm of a VAE stage whose small arithmetic the host does between two
// commands (tools/amd/iron_groupnorm.py, a stage's “norms“ in its meta): the device writes each
// line's sums and sums of squares of every channel into the state Stats (float32: lines of 2
// Channels), and reads each channel's factor and term from the state Row (bf16: the factors'
// high limbs, their low limbs, the terms' high limbs, their low limbs).
type HostNorm struct {
	Name  string `json:"name"`
	Stats string `json:"stats"`
	// StatsBytes is how much of the state Stats this norm's sums take (the stage's norms use
	// one state in turn; 0: all of it)
	StatsBytes int       `json:"stats_bytes"`
	Row        string    `json:"row"`
	Channels   int       `json:"channels"`
	Groups     int       `json:"groups"`
	Pixels     int       `json:"pixels"`
	Eps        float64   `json:"eps"`
	Gamma      []float64 `json:"gamma"`
	Beta       []float64 `json:"beta"`
}

// normPending is what a stats state holds until the device writes it: a NaN no sum gives.
const normPending = 0x7fc0dead

// Coefficients computes the norm's factors and terms from the lines' statistics (stats: lines of the
// channels' sums, then their sums of squares, float32) into row: y = x a + b with a = gamma /
// sqrt(var + eps) and b = beta - mean a, the mean and the variance a group's, over the image's
// pixels (the pads of a grid are zeros and add nothing).
func (n HostNorm) Coefficients(stats, row []byte) error {
	C, G := n.Channels, n.Groups
	if C <= 0 || G <= 0 || C%G != 0 || len(n.Gamma) != C || len(n.Beta) != C || len(stats)%(8*C) != 0 || len(row) < 8*C {
		return fmt.Errorf("anima: norm %s: %d channels in %d groups, %d bytes of statistics, a row of %d", n.Name, C, G, len(stats), len(row))
	}
	sum, sq := make([]float64, C), make([]float64, C)
	for o := 0; o < len(stats); o += 8 * C {
		line := stats[o : o+8*C]
		for c := 0; c < C; c++ {
			sum[c] += float64(math.Float32frombits(binary.LittleEndian.Uint32(line[4*c:])))
			sq[c] += float64(math.Float32frombits(binary.LittleEndian.Uint32(line[4*(C+c):])))
		}
	}
	cg := C / G
	count := float64(n.Pixels * cg)
	limbs := func(at int, v float64) {
		hi := bf16(float32(v))
		lo := bf16(float32(v - float64(fromBF16(hi))))
		binary.LittleEndian.PutUint16(row[2*at:], hi)
		binary.LittleEndian.PutUint16(row[2*(C+at):], lo)
	}
	for g := 0; g < G; g++ {
		var s, q float64
		for c := g * cg; c < (g+1)*cg; c++ {
			s += sum[c]
			q += sq[c]
		}
		mean := s / count
		inv := 1 / math.Sqrt(q/count-mean*mean+n.Eps)
		for c := g * cg; c < (g+1)*cg; c++ {
			a := n.Gamma[c] * inv
			limbs(c, a)
			limbs(2*C+c, n.Beta[c]-mean*a)
		}
	}
	return nil
}

// hostNorms sets g up to run norms' host steps during Run: each stats state is marked pending,
// and after each command the next norm whose statistics the device has written gets its row.
// The returned function reports the norms the Run left without their statistics.
func hostNorms(g *iron.Graph, norms []HostNorm) (func() error, error) {
	if len(norms) == 0 {
		return func() error { return nil }, nil
	}
	// mark marks norm i's sums as not written yet (the device writes them whole in one command)
	mark := func(i int) error {
		b, _, to, err := g.Named(norms[i].Stats)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(b, normPending)
		return to()
	}
	for i := range norms {
		if err := mark(i); err != nil {
			return nil, err
		}
	}
	next := 0
	g.After = func(int) error {
		for next < len(norms) {
			n := norms[next]
			stats, from, _, err := g.Named(n.Stats)
			if err != nil {
				return err
			}
			if err := from(); err != nil {
				return err
			}
			if binary.LittleEndian.Uint32(stats) == normPending {
				return nil
			}
			row, _, to, err := g.Named(n.Row)
			if err != nil {
				return err
			}
			if n.StatsBytes > 0 && n.StatsBytes <= len(stats) {
				stats = stats[:n.StatsBytes]
			}
			if err := n.Coefficients(stats, row); err != nil {
				return err
			}
			if err := to(); err != nil {
				return err
			}
			next++
			if next < len(norms) { // the next norm's sums may come into the same state
				if err := mark(next); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return func() error {
		if next != len(norms) {
			return fmt.Errorf("anima: norm %s: the device never wrote its statistics", norms[next].Name)
		}
		return nil
	}, nil
}
