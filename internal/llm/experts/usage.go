package experts

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
)

// Usage counts how many rows ran each expert of each layer. Counts are integers: they add
// exactly, a file merges with any order of runs, and when one would pass usageCap every count is
// halved (the old runs weigh half; the order of the experts, all that choosing them needs, stays).
type Usage struct {
	mu      sync.Mutex
	Layers  int
	Experts int
	Counts  []uint32 // Counts[l*Experts+e]
}

// usageCap is the count past which a file's counts are halved.
const usageCap = 1 << 30

// usageMagic heads a usage file: magic, version 1, layers, experts (uint32 little endian each),
// then layers·experts uint32 counts.
var usageMagic = [4]byte{'H', 'W', 'E', 'U'}

// NewUsage is an empty count of layers × experts.
func NewUsage(layers, experts int) *Usage {
	return &Usage{Layers: layers, Experts: experts, Counts: make([]uint32, layers*experts)}
}

// Add counts the experts idx (a row's each) of layer.
func (u *Usage) Add(layer int, idx [][]int) {
	if layer < 0 || layer >= u.Layers {
		return
	}
	u.mu.Lock()
	c := u.Counts[layer*u.Experts : (layer+1)*u.Experts]
	for _, row := range idx {
		for _, e := range row {
			if e >= 0 && e < u.Experts {
				c[e]++
			}
		}
	}
	u.mu.Unlock()
}

// AddIDs counts the experts ids (any rows' choices, flat) of layer.
func (u *Usage) AddIDs(layer int, ids []int32) {
	if layer < 0 || layer >= u.Layers {
		return
	}
	u.mu.Lock()
	c := u.Counts[layer*u.Experts : (layer+1)*u.Experts]
	for _, e := range ids {
		if e >= 0 && int(e) < u.Experts {
			c[e]++
		}
	}
	u.mu.Unlock()
}

// Merge adds v's counts to u's, halving all of them while one would pass usageCap.
func (u *Usage) Merge(v *Usage) error {
	if v.Layers != u.Layers || v.Experts != u.Experts {
		return fmt.Errorf("experts: usage of %d×%d against %d×%d", v.Layers, v.Experts, u.Layers, u.Experts)
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	sum := make([]uint64, len(u.Counts))
	var top uint64
	for i := range sum {
		sum[i] = uint64(u.Counts[i]) + uint64(v.Counts[i])
		top = max(top, sum[i])
	}
	shift := 0
	for top>>shift > usageCap {
		shift++
	}
	for i := range sum {
		u.Counts[i] = uint32(sum[i] >> shift)
	}
	return nil
}

// ReadUsage reads a usage file.
func ReadUsage(path string) (*Usage, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 16 || [4]byte(b[:4]) != usageMagic || binary.LittleEndian.Uint32(b[4:]) != 1 {
		return nil, fmt.Errorf("experts: %s is not a usage file", path)
	}
	l, e := int(binary.LittleEndian.Uint32(b[8:])), int(binary.LittleEndian.Uint32(b[12:]))
	if l <= 0 || e <= 0 || len(b) != 16+4*l*e {
		return nil, fmt.Errorf("experts: %s: %d×%d counts in %d bytes", path, l, e, len(b))
	}
	u := NewUsage(l, e)
	for i := range u.Counts {
		u.Counts[i] = binary.LittleEndian.Uint32(b[16+4*i:])
	}
	return u, nil
}

// Write writes u to path, through a file beside it renamed over it.
func (u *Usage) Write(path string) error {
	u.mu.Lock()
	b := make([]byte, 16+4*len(u.Counts))
	copy(b, usageMagic[:])
	binary.LittleEndian.PutUint32(b[4:], 1)
	binary.LittleEndian.PutUint32(b[8:], uint32(u.Layers))
	binary.LittleEndian.PutUint32(b[12:], uint32(u.Experts))
	for i, c := range u.Counts {
		binary.LittleEndian.PutUint32(b[16+4*i:], c)
	}
	u.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := fmt.Sprintf("%s.%d.tmp", path, os.Getpid())
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// AddTo adds u's counts to the file at path (made if missing; a file of other sizes is
// replaced).
func (u *Usage) AddTo(path string) error {
	total := NewUsage(u.Layers, u.Experts)
	if old, err := ReadUsage(path); err == nil && old.Layers == u.Layers && old.Experts == u.Experts {
		total = old
	} else if err != nil && !errors.Is(err, fs.ErrNotExist) {
		fmt.Fprintf(os.Stderr, "experts: usage: %v (starting again)\n", err)
	}
	if err := total.Merge(u); err != nil {
		return err
	}
	return total.Write(path)
}
