// Command lutscan lists the blobs of a Core AI program that look like lookup tables (a whole
// number of 2^bits fp16 entries), with the leading run of identical tables: coreai-build's ANE
// compiler infers how many rows share a table from that run and fails ("not content-uniform",
// "leading run length doesn't divide total_cout") when it is longer than one table and the
// tables after it differ.
//
//	go run ./tools/coreai/lutscan [-bits 8] [-min 2] X.aimodel/main.mlirb
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/shibukawa/bonsaiai/internal/coreai/mlirbc"
)

func main() {
	bits := flag.Int("bits", 8, "table entries are 2^bits halves")
	min := flag.Int("min", 2, "print blobs whose leading run is at least this many tables")
	tile := flag.Int("tile", 0, "also check the leading run of each run of this many tables (the compiler's cout tiles: 16 tables of 16 rows for 256)")
	flag.Parse()
	b, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	f, err := mlirbc.Read(b)
	if err != nil {
		log.Fatal(err)
	}
	size := 2 << *bits // bytes of a table
	for group, rs := range f.Resources {
		for _, r := range rs {
			if r.Kind != 0 || len(r.Data) < 2*size || len(r.Data)%size != 0 {
				continue
			}
			n := len(r.Data) / size
			t := func(i int) []byte { return r.Data[i*size : (i+1)*size] }
			run := 1
			for run < n && bytes.Equal(t(run), t(0)) {
				run++
			}
			distinct := map[string]bool{}
			for i := 0; i < n; i++ {
				distinct[string(t(i))] = true
			}
			if *tile > 0 {
				for s := 0; s < n; s += *tile {
					e := s + *tile
					if e > n {
						e = n
					}
					k := 1
					for s+k < e && bytes.Equal(t(s+k), t(s)) {
						k++
					}
					if k > 1 && k < e-s {
						fmt.Printf("%s %s: tables %d..%d (of %d): leading run %d\n", group, r.Key, s, e, n, k)
					}
				}
			}
			if run >= *min {
				fmt.Printf("%s %s: %d tables, leading run %d, %d distinct; first table head % x\n", group, r.Key, n, run, len(distinct), t(0)[:16])
			}
		}
	}
}
