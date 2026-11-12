// Command bcrecode reads an MLIR bytecode file and writes it again with mlirbc's writer, the
// attributes, types and properties copied as their bytes: whether the writer's container (the
// sections, the IR, the strings) is what Core AI reads.
//
//	go run ./tools/coreai/bcrecode IN.aimodel OUT.aimodel
package main

import (
	"crypto/sha256"
	"log"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

func main() {
	in, out := os.Args[1], os.Args[2]
	b, err := os.ReadFile(filepath.Join(in, "main.mlirb"))
	if err != nil {
		log.Fatal(err)
	}
	f, err := mlirbc.Read(b)
	if err != nil {
		log.Fatal(err)
	}
	w := mlirbc.NewWriter(f.Producer, f.Versions)
	w.RawStrings(f.Strings)
	for _, a := range f.Attrs {
		w.RawAttr(a)
	}
	for _, t := range f.Types {
		w.RawType(t)
	}
	props := map[string]int{}
	for _, rs := range f.Resources {
		for _, r := range rs {
			w.Blob("aicode", r.Key, int(r.Align), r.Data)
		}
	}
	var block func(src *mlirbc.Block) *mlirbc.BlockW
	block = func(src *mlirbc.Block) *mlirbc.BlockW {
		args := make([]int, len(src.Args))
		for i, t := range src.Args {
			args[i] = int(t)
		}
		dst, _ := w.NewBlock(args...)
		locs := make([]int, len(src.ArgLocs))
		for i, l := range src.ArgLocs {
			locs[i] = int(l)
		}
		dst.SetArgLocs(locs)
		for _, op := range src.Ops {
			var regs []*mlirbc.BlockW
			for _, r := range op.Regions {
				regs = append(regs, block(r.Blocks[0]))
			}
			p := -1
			if op.Props != nil {
				k := string(op.Props)
				if i, ok := props[k]; ok {
					p = i
				} else {
					p = w.RawProps(op.Props)
					props[k] = p
				}
			}
			var ops []mlirbc.Value
			for _, v := range op.Operands {
				ops = append(ops, dst.ValueAt(int(v)))
			}
			res := make([]int, len(op.Results))
			for i, t := range op.Results {
				res[i] = int(t)
			}
			dst.OpRaw(op.Name, int(op.Loc), p, ops, res, regs...)
		}
		return dst
	}
	top := block(f.Top)
	ob := w.Bytes(top)
	os.MkdirAll(out, 0o755)
	sum := sha256.Sum256(ob)
	os.WriteFile(filepath.Join(out, "main.mlirb"), ob, 0o644)
	os.WriteFile(filepath.Join(out, "main.hash"), sum[:], 0o644)
	md, _ := os.ReadFile(filepath.Join(in, "metadata.json"))
	os.WriteFile(filepath.Join(out, "metadata.json"), md, 0o644)
	log.Printf("%d -> %d bytes", len(b), len(ob))
}
