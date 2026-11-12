// Command bcdump prints an MLIR bytecode file's tables and IR (mlirbc.Read): for comparing what
// hwnpu writes with what coreai-core writes.
//
//	go run ./tools/coreai/bcdump model.aimodel/main.mlirb
package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

func main() {
	b, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	f, err := mlirbc.Read(b)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("version %d producer %q\n", f.Version, f.Producer)
	fmt.Printf("sections %q\n", f.Sections)
	fmt.Printf("dialects %q\n", f.Dialects)
	for d, v := range f.Versions {
		fmt.Printf("version of %s: %x\n", d, v)
	}
	fmt.Printf("ops %q\n", f.OpNames)
	for i, a := range f.Attrs {
		fmt.Printf("attr %d [%s] %s\n", i, a.Dialect, trim(a.String()))
	}
	for i, t := range f.Types {
		fmt.Printf("type %d [%s] %s\n", i, t.Dialect, trim(t.String()))
	}
	for g, rs := range f.Resources {
		for _, r := range rs {
			fmt.Printf("resource %s %s kind %d align %d %d bytes\n", g, r.Key, r.Kind, r.Align, len(r.Data))
		}
	}
	dump(f.Top, 0)
}

func trim(s string) string {
	if len(s) > 160 {
		return s[:160] + "..."
	}
	return s
}

func dump(b *mlirbc.Block, depth int) {
	ind := strings.Repeat("  ", depth)
	if len(b.Args) > 0 {
		fmt.Printf("%s^bb args types %v\n", ind, b.Args)
	}
	for _, op := range b.Ops {
		fmt.Printf("%s%s loc %d attrs %d props %x/%d results %v operands %v\n", ind, op.Name, op.Loc, op.Attrs, op.Props, op.PropsAttr, op.Results, op.Operands)
		for _, r := range op.Regions {
			fmt.Printf("%s  region values %d iso %v\n", ind, r.NumValues, op.Isolated)
			for _, blk := range r.Blocks {
				dump(blk, depth+2)
			}
		}
	}
}
