package main

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/coreai/mlirbc"
)

// lintProgram finds in an .aimodel's bytecode what ailower.LintPackage finds in a package: the
// slices of an fp16 tensor's last axis starting at 1..14 mod 32, where the Neural Engine takes
// values above 4094 to Inf (docs/design/apple-neural-engine.md). The exports keep the lowered programs only.
func lintProgram(dir string) ([]string, error) {
	b, err := os.ReadFile(filepath.Join(dir, "main.mlirb"))
	if err != nil {
		return nil, err
	}
	f, err := mlirbc.Read(b)
	if err != nil {
		return nil, err
	}
	var notes []string
	var walk func(blk *mlirbc.Block, types map[uint64]uint64, consts map[uint64]uint64, next *uint64)
	walk = func(blk *mlirbc.Block, types, consts map[uint64]uint64, next *uint64) {
		for _, t := range blk.Args {
			types[*next] = t
			*next++
		}
		for _, op := range blk.Ops {
			first := *next
			for _, t := range op.Results {
				types[*next] = t
				*next++
			}
			switch op.Name {
			case "aicode.primitive.constant_v1":
				if a, ok := propAttr(op.Props); ok {
					consts[first] = a
				}
			case "aicode.primitive.slice_v1":
				if n := sliceNote(f, op, types, consts); n != "" {
					notes = append(notes, n)
				}
			}
			for _, r := range op.Regions {
				for _, rb := range r.Blocks {
					if op.Isolated {
						var n uint64
						walk(rb, map[uint64]uint64{}, map[uint64]uint64{}, &n)
					} else {
						walk(rb, types, consts, next)
					}
				}
			}
		}
	}
	var n uint64
	walk(f.Top, map[uint64]uint64{}, map[uint64]uint64{}, &n)
	return notes, nil
}

// propAttr is the attribute index a constant's properties hold (an MLIR varint).
func propAttr(p []byte) (uint64, bool) {
	if len(p) == 0 || p[0] == 0 {
		return 0, false
	}
	extra := 0
	for p[0]>>extra&1 == 0 {
		extra++
	}
	if len(p) < extra+1 {
		return 0, false
	}
	var buf [8]byte
	copy(buf[:], p[:extra+1])
	return binary.LittleEndian.Uint64(buf[:]) >> (extra + 1), true
}

var tensorType = regexp.MustCompile(`tensor_v1<\[([0-9, ]*)\] : !aicode\.(\w+)>`)
var denseAttr = regexp.MustCompile(`dense_elements<\[splat = (true|false)\] "([^"]*)">`)

func shapeElem(f *mlirbc.File, t uint64) ([]int, string) {
	if int(t) >= len(f.Types) {
		return nil, ""
	}
	m := tensorType.FindStringSubmatch(f.Types[t].String())
	if m == nil {
		return nil, ""
	}
	var shape []int
	for _, s := range strings.Split(m[1], ",") {
		if s = strings.TrimSpace(s); s != "" {
			v, _ := strconv.Atoi(s)
			shape = append(shape, v)
		}
	}
	return shape, m[2]
}

// ints decodes a constant of int32 values, n of them (a splat repeated).
func ints(f *mlirbc.File, attr uint64, n int) []int {
	if int(attr) >= len(f.Attrs) {
		return nil
	}
	m := denseAttr.FindStringSubmatch(f.Attrs[attr].String())
	if m == nil {
		return nil
	}
	raw, err := base64.StdEncoding.DecodeString(m[2])
	if err != nil || len(raw) < 4 {
		return nil
	}
	out := make([]int, n)
	for i := range out {
		j := i
		if m[1] == "true" {
			j = 0
		}
		if 4*j+4 > len(raw) {
			return nil
		}
		out[i] = int(int32(binary.LittleEndian.Uint32(raw[4*j:])))
	}
	return out
}

func sliceNote(f *mlirbc.File, op *mlirbc.Op, types, consts map[uint64]uint64) string {
	if len(op.Operands) < 4 {
		return ""
	}
	shape, elem := shapeElem(f, types[op.Operands[0]])
	if elem != "float16_v1" || len(shape) == 0 {
		return ""
	}
	r := len(shape)
	ba, ok1 := consts[op.Operands[1]]
	sa, ok3 := consts[op.Operands[3]]
	if !ok1 || !ok3 {
		return "" // given at run time
	}
	begin, stride := ints(f, ba, r), ints(f, sa, r)
	if begin == nil || stride == nil {
		return ""
	}
	if os.Getenv("LINT_ALL") != "" {
		fmt.Printf("  slice %v begin %v stride %v\n", shape, begin, stride)
	}
	if b := begin[r-1]; b%32 >= 1 && b%32 <= 14 && stride[r-1] == 1 {
		return fmt.Sprintf("slice of the last axis (%v) from %d", shape, b)
	}
	return ""
}
