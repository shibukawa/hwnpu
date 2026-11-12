// Command qlut palettizes the matrices of one Qwen-Image-2.1 program with hwnpu's own quantizer
// (internal/palette) and writes them to a safetensors file, for the Core AI experiment
// (tools/coreai/qwenimage_dit.py builds the same programs as internal/imagegen/qwenimage/mil from
// them). Each matrix becomes NAME.idx (uint8, rows x cols, one index per weight, not packed) and
// NAME.lut (fp16, rows/group x 1 x 2^bits x 1, coreai_torch's lut_to_dense layout), NAME as
// mil.Weights calls it ("dit." or "text." first). The matrices are transformed as mil does: to_q and
// to_k's rows in headPerm's order, img_in and txt_in's out_layer scaled by 1/DitScale, the text
// encoder's o_proj and down_proj by 1/TextScale.
//
//	-part dit       mil.Blocks (the image's rows; the first group with img_in, -final with proj_out)
//	-part dittext   mil.TextBlocks (the same blocks' matrices)
//	-part text      mil.TextGroup (Qwen3-VL-8B's layers; -final with txt_in)
//
//	go run ./tools/coreai/qlut -root ~/Library/Caches/hwnpu/models/Qwen-Image/Qwen-Image-2.1-Turbo -first 0 -count 4 -out g0.safetensors
package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/palette"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

func main() {
	root := flag.String("root", "", "the Qwen-Image-2.1 repository (transformer/)")
	text := flag.String("text", "", "the repository holding Qwen3-VL-8B (text_encoder/), for -part text")
	part := flag.String("part", "dit", "dit, dittext or text")
	first := flag.Int("first", 0, "the group's first block")
	count := flag.Int("count", 4, "the group's blocks")
	final := flag.Bool("final", false, "the last group: add proj_out (dit) or txt_in (text)")
	bits := flag.Int("bits", 4, "4 or 8 bit indices")
	group := flag.Int("group", 16, "rows per table")
	quant := flag.String("quant", "kmeans-minmax", "uniform, kmeans or kmeans-minmax")
	out := flag.String("out", "", "the safetensors file to write")
	flag.Parse()
	if *root == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	mode := map[string]palette.Mode{"uniform": palette.Uniform, "kmeans": palette.KMeans, "kmeans-minmax": palette.KMeansMinMax}[*quant]
	if *bits != 4 && *bits != 8 {
		log.Fatal("qlut: 4 or 8 bits")
	}
	w, err := mil.OpenWeights(*root, *text)
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()
	c, err := mil.ReadDit(*root, 512, 512)
	if err != nil {
		log.Fatal(err)
	}
	type mat struct {
		name  string
		perm  bool
		scale float32
	}
	var mats []mat
	switch *part {
	case "dit", "dittext":
		if *part == "dit" && *first == 0 {
			mats = append(mats, mat{"dit.img_in.weight", false, 1.0 / mil.DitScale})
		}
		for b := *first; b < *first+*count; b++ {
			p := fmt.Sprintf("dit.transformer_blocks.%d.", b)
			mats = append(mats,
				mat{p + "attn.to_q.weight", true, 1}, mat{p + "attn.to_k.weight", true, 1}, mat{p + "attn.to_v.weight", false, 1},
				mat{p + "attn.to_out.0.weight", false, 1},
				mat{p + "img_mlp.gate_layer.weight", false, 1}, mat{p + "img_mlp.proj.weight", false, 1}, mat{p + "img_mlp.out.weight", false, 1})
		}
		if *part == "dit" && *final {
			mats = append(mats, mat{"dit.proj_out.weight", false, 1})
		}
	case "text":
		for i := *first; i < *first+*count; i++ {
			p := fmt.Sprintf("text.layers.%d.", i)
			mats = append(mats,
				mat{p + "self_attn.q_proj.weight", false, 1}, mat{p + "self_attn.k_proj.weight", false, 1}, mat{p + "self_attn.v_proj.weight", false, 1},
				mat{p + "self_attn.o_proj.weight", false, 1.0 / mil.TextScale},
				mat{p + "mlp.gate_proj.weight", false, 1}, mat{p + "mlp.up_proj.weight", false, 1}, mat{p + "mlp.down_proj.weight", false, 1.0 / mil.TextScale})
		}
		if *final {
			mats = append(mats, mat{"dit.txt_in.in_layer.weight", false, 1}, mat{"dit.txt_in.out_layer.weight", false, 1.0 / mil.DitScale})
		}
	default:
		log.Fatalf("qlut: unknown part %q", *part)
	}
	var ts []safetensors.Out
	t0 := time.Now()
	for _, m := range mats {
		v, shape, err := w.Get(m.name)
		if err != nil {
			log.Fatal(err)
		}
		rows, cols := shape[0], shape[1]
		if m.perm {
			v = permRows(v, c.Heads, c.HeadDim, cols)
		}
		if m.scale != 1 {
			for i := range v {
				v[i] *= m.scale
			}
		}
		q := palette.QuantizeBits(v, rows, cols, *group, 0, *bits, 8, mode)
		idx := unpack(q.Indices, rows*cols, *bits)
		var lut bytes.Buffer
		binary.Write(&lut, binary.LittleEndian, q.LUT)
		ts = append(ts,
			safetensors.Out{Name: m.name + ".idx", DType: "U8", Shape: []int{rows, cols}, Size: int64(len(idx)), Data: bytes.NewReader(idx)},
			safetensors.Out{Name: m.name + ".lut", DType: "F16", Shape: []int{rows / *group, 1, 1 << *bits, 1}, Size: int64(lut.Len()), Data: bytes.NewReader(lut.Bytes())})
	}
	meta := map[string]string{"bits": fmt.Sprint(*bits), "group": fmt.Sprint(*group), "quant": *quant}
	if err := safetensors.Write(*out, ts, meta); err != nil {
		log.Fatal(err)
	}
	log.Printf("qlut: %d matrices in %.0f s to %s", len(mats), time.Since(t0).Seconds(), *out)
}

// unpack is the n indices of packed (bits 4: two a byte, the low nibble first; bits 8: one a
// byte), one a byte.
func unpack(packed []byte, n, bits int) []byte {
	if bits == 8 {
		return packed[:n]
	}
	out := make([]byte, n)
	for i := range out {
		b := packed[i/2]
		if i%2 == 0 {
			out[i] = b & 15
		} else {
			out[i] = b >> 4
		}
	}
	return out
}

// permRows is mil's: each head's rows (heads x D, in) in the order that puts Qwen-Image's RoPE
// pair (2i, 2i + 1) at (i, i + D/2).
func permRows(w []float32, heads, D, in int) []float32 {
	p := make([]int, 0, D)
	for i := 0; i < D; i += 2 {
		p = append(p, i)
	}
	for i := 1; i < D; i += 2 {
		p = append(p, i)
	}
	out := make([]float32, len(w))
	for h := 0; h < heads; h++ {
		for i, src := range p {
			copy(out[(h*D+i)*in:(h*D+i+1)*in], w[(h*D+src)*in:(h*D+src+1)*in])
		}
	}
	return out
}
