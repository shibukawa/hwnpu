// Package iron runs a language model compiled by amd/IRON on an AMD NPU (XDNA2): the full
// ELFs and the arena that tools/amd/iron_export_lm.py writes. IRON is the compiler; this
// package is the runtime, in Go, without IRON or XRT. See docs/amd-npu-models.md.
package iron

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ManifestVersion is the manifest format this package reads.
const ManifestVersion = 1

// Manifest is manifest.json.
type Manifest struct {
	Version int    `json:"version"`
	Model   string `json:"model"`
	Config  struct {
		VocabSize       int `json:"vocab_size"`
		HiddenSize      int `json:"hidden_size"`
		MaxSeqLen       int `json:"max_seq_len"`
		PrefillChunk    int `json:"prefill_chunk"`
		NumHiddenLayers int `json:"num_hidden_layers"`
	} `json:"config"`
	RopeStart string              `json:"rope_start"` // "start": a chunk's RoPE row is its start
	Banks     map[string]Bank     `json:"banks"`
	Versions  map[string]*Version `json:"versions"`
}

// Bank is one arena bank: a buffer every version shares, holding weights and states.
type Bank struct {
	Bytes int    `json:"bytes"`
	Image string `json:"image"` // the initial contents, or "" for zeros
}

// Version is one compiled call shape: the decode step or a prompt chunk.
type Version struct {
	ELF       string               `json:"elf"`
	Kernels   []Kernel             `json:"kernels"` // one per command, run in order
	Buffers   map[string]Placement `json:"buffers"`
	Arguments []Argument           `json:"arguments"`
	Values    []string             `json:"values"`
	Carried   *Carried             `json:"carried"`
	Outputs   []Handle             `json:"outputs"`
	Inputs    []Handle             `json:"inputs"`
	Encoders  []Encoder            `json:"encoders"`
	Plan      string               `json:"plan"`
}

// Placement is where a buffer of the graph lives.
type Placement struct {
	Argument string `json:"argument"`
	Offset   int    `json:"offset"`
	Bytes    int    `json:"bytes"`
	DType    string `json:"dtype"`
}

// Kernel is one command: the ELF's kernel instance and the scratchpad words it reads.
type Kernel struct {
	Name       string  `json:"name"`
	Parameters []Param `json:"parameters"`
}

// Argument is one argument of every command of a version, by index.
type Argument struct {
	Index int    `json:"index"`
	Kind  string `json:"kind"` // input, output, feedback, or an arena bank
	Bytes int    `json:"bytes"`
}

// Handle is where a named buffer lives: in an argument, at a byte offset.
type Handle struct {
	Name     string `json:"name"`
	Shape    []int  `json:"shape"`
	DType    string `json:"dtype"`
	Argument string `json:"argument"`
	Offset   int    `json:"offset"`
	Bytes    int    `json:"bytes"`
}

// Carried says where a call's carried values (the token and the position) go.
type Carried struct {
	Carry *struct {
		Argument string   `json:"argument"`
		Offset   int      `json:"offset"`
		Names    []string `json:"names"`
	} `json:"carry"`
}

// Encoder turns an input of token ids into the control words of an embedding gather
// (IRON's Gather.control_words): per row, a DMA descriptor's address of the table's row.
type Encoder struct {
	Input         string  `json:"input"`
	StartWord     int     `json:"start_word"`
	Words         int     `json:"words"`
	Table         string  `json:"table"`
	TableArgument string  `json:"table_argument"`
	TableOffset   int     `json:"table_offset"`
	Rows          int     `json:"rows"`
	TableRows     int     `json:"table_rows"`
	Stride        int     `json:"stride"`
	PadBytes      int     `json:"pad_bytes"`
	Template      string  `json:"template"`
	Slots         []int   `json:"slots"`
	Bound         *Affine `json:"bound"`

	template []uint32
}

// ReadManifest reads dir/manifest.json and the encoders' templates.
func ReadManifest(dir string) (*Manifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("iron: manifest: %w", err)
	}
	if m.Version != ManifestVersion {
		return nil, fmt.Errorf("iron: manifest version %d, this reads %d", m.Version, ManifestVersion)
	}
	if m.RopeStart != "start" {
		return nil, fmt.Errorf("iron: rope_start %q is not supported", m.RopeStart)
	}
	for name, v := range m.Versions {
		for i := range v.Encoders {
			e := &v.Encoders[i]
			raw, err := os.ReadFile(filepath.Join(dir, e.Template))
			if err != nil {
				return nil, err
			}
			if len(raw) != 4*e.Words {
				return nil, fmt.Errorf("iron: %s: gather template %s has %d bytes, want %d", name, e.Template, len(raw), 4*e.Words)
			}
			e.template = make([]uint32, e.Words)
			for j := range e.template {
				e.template[j] = uint32(raw[4*j]) | uint32(raw[4*j+1])<<8 | uint32(raw[4*j+2])<<16 | uint32(raw[4*j+3])<<24
			}
			for _, s := range e.Slots {
				if s < 1 || s+2 >= e.Words {
					return nil, fmt.Errorf("iron: %s: gather slot %d outside %d words", name, s, e.Words)
				}
			}
			if len(e.Slots) != e.Rows {
				return nil, fmt.Errorf("iron: %s: %d gather slots for %d rows", name, len(e.Slots), e.Rows)
			}
		}
	}
	return &m, nil
}
