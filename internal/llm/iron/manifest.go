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
	RopeStart string `json:"rope_start"` // "start": a chunk's RoPE row is its start
	// Restartable says the state is the KV caches written by position (no rings, no recurrent
	// layers): a sequence goes back by its position, and its rows are all of it. Exports
	// without it (older ones) are taken as not restartable.
	Restartable bool `json:"restartable"`
	// Checkpoints are the states of the recurrent layers (IRON's Checkpoints: LFM3's short
	// convolutions), and Rings the ring caches, of a model that is not Restartable. A state
	// has a slot per prompt-chunk boundary, slot k the state after the positions up to
	// k * chunk run so far; a call reads the slot of its start and writes the one of its end,
	// and ends within the chunk it starts in. Copying the slot a position reads (StateAt) is
	// what a sequence needs besides its rows to go back to that position; no copy brings a
	// ring back.
	Checkpoints []Checkpoint        `json:"checkpoints"`
	Rings       int                 `json:"rings"`
	Banks       map[string]Bank     `json:"banks"`
	Versions    map[string]*Version `json:"versions"`
	// Multimodal describes the towers of a model with them (Gemma4Multimodal), or nil.
	Multimodal *Multimodal `json:"multimodal"`
}

// Checkpoint is a recurrent layer's state: the buffer of its slots.
type Checkpoint struct {
	Buffer string `json:"buffer"`
	Slots  int    `json:"slots"`
	Chunk  int    `json:"chunk"`
}

// Multimodal is what the host needs to run a model's towers: the tokens around and in an
// image's or a clip's run of soft tokens, and the towers' shapes.
type Multimodal struct {
	// Kind is "" for Gemma 4's towers (Gemma4Multimodal), "lfm3vl" for LFM3-VL's vision tower
	// of one image size, whose input is the image's patches (Vision.Width by Height), and
	// "lfm3audio" for LFM3-Audio's audio encoder, whose input is the clip's log-mel features.
	// A marker token below zero is none: a run without it.
	Kind         string  `json:"kind"`
	ImageToken   int32   `json:"image_token_id"`
	AudioToken   int32   `json:"audio_token_id"`
	BOI          int32   `json:"boi_token_id"`
	EOI          int32   `json:"eoi_token_id"`
	BOA          int32   `json:"boa_token_id"`
	EOA          int32   `json:"eoa_token_id"`
	Placeholders []int32 `json:"placeholders"` // the order of the placed chunk's before
	Vision       struct {
		PoolingKernelSize int     `json:"pooling_kernel_size"`
		PatchSize         int     `json:"patch_size"`
		ImageTokens       int     `json:"image_tokens"` // the soft tokens an image may give
		Patches           int     `json:"patches"`
		Largest           [2]int  `json:"largest"`
		Chunk             int     `json:"chunk"`
		ImageChunks       int     `json:"image_chunks"`
		Groups            [][]int `json:"groups"`
		// of LFM3-VL's: the image size the tower takes, and a patch's values
		Width    int `json:"width"`
		Height   int `json:"height"`
		PatchDim int `json:"patch_dim"`
	} `json:"vision"`
	Audio struct {
		// of LFM3-Audio's (Kind "lfm3audio"): the longest clip's log-mel frames, and their width
		MaxFrames  int `json:"max_frames"`
		Mels       int `json:"mels"`
		SampleRate int `json:"sample_rate"`
		Frame      int `json:"frame"`
		Hop        int `json:"hop"`
		MaxSamples int `json:"max_samples"`
		MaxTokens  int `json:"max_tokens"`
	} `json:"audio"`
	TowerValues map[string][]string `json:"tower_values"`
}

// Address is a per-call value computed from a buffer's device address: the low or the high
// bits of a gather table's address (IRON's GatherWords.address_words).
type Address struct {
	Name     string `json:"name"`
	Buffer   string `json:"buffer"`
	Argument string `json:"argument"`
	Offset   int    `json:"offset"`
	LowBits  uint   `json:"low_bits"`
	Member   string `json:"member"` // base_lo or base_hi
}

// Grid holds a tower's words for every call it can make: the vision tower's by the image's
// blocks (Pairs, hk by wk), the audio tower's by the clip's valid frames (Frames).
type Grid struct {
	Side   int                `json:"side"`
	Pairs  [][2]int           `json:"pairs"`
	Frames []int              `json:"frames"`
	Values map[string][]int64 `json:"values"`
}

// Bank is one arena bank: a buffer every version shares, holding weights and states.
type Bank struct {
	Bytes int    `json:"bytes"`
	Image string `json:"image"` // the initial contents, or "" for zeros
}

// Version is one compiled call shape: the decode step or a prompt chunk. It is one full ELF
// (NPU2: ELF, Kernels) or an xclbin per step (NPU1, which has no full ELF: Steps, Words).
type Version struct {
	ELF       string               `json:"elf"`
	Kernels   []Kernel             `json:"kernels"` // one per command, run in order
	Steps     []Step               `json:"steps"`
	Words     map[string]*Param    `json:"words"` // the steps' dispatch-time scalars, by name
	Insts     string               `json:"insts"` // the static steps' instruction streams
	Buffers   map[string]Placement `json:"buffers"`
	Arguments []Argument           `json:"arguments"`
	Values    []string             `json:"values"`
	Carried   *Carried             `json:"carried"`
	Outputs   []Handle             `json:"outputs"`
	Inputs    []Handle             `json:"inputs"`
	Encoders  []Encoder            `json:"encoders"`
	Addresses []Address            `json:"addresses"`
	Grid      *Grid                `json:"grid"`
	Plan      string               `json:"plan"`
}

// Step is one dispatch of an xclbin per step: a kernel of an xclbin, its instruction stream
// (static: Insts, a word offset and a count in the version's Insts file; or built per call by
// Program from the dispatch-time scalars named by Scalars), and its buffers, in the kernel's
// order after the opcode, the stream and its length.
type Step struct {
	Xclbin  string       `json:"xclbin"`
	Kernel  string       `json:"kernel"`
	Insts   []int        `json:"insts"`
	Program string       `json:"program"`
	Scalars []string     `json:"scalars"`
	Buffers []StepBuffer `json:"buffers"`
}

// StepBuffer is a part of an argument a step takes as one buffer.
type StepBuffer struct {
	Argument string `json:"argument"`
	Offset   int    `json:"offset"`
	Bytes    int    `json:"bytes"`
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
		if (v.ELF == "") == (len(v.Steps) == 0) {
			return nil, fmt.Errorf("iron: version %s is neither one full ELF nor an xclbin per step", name)
		}
		for i, st := range v.Steps {
			if st.Program == "" && len(st.Insts) != 2 {
				return nil, fmt.Errorf("iron: %s: step %d has no instruction stream", name, i)
			}
			for _, sc := range st.Scalars {
				if v.Words[sc] == nil {
					return nil, fmt.Errorf("iron: %s: step %d reads the scalar %s, which has no word", name, i, sc)
				}
			}
		}
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
