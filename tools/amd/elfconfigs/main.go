// Command elfconfigs counts the reconfigurations of the array in IRON's full ELFs: each
// load_pdi in a command's control code starts a new configuration (the cores' program memory,
// their data memory's initial words, the locks, DMAs and switches), written by the control code
// itself. A change of operator inside a command costs about 100 µs whatever the configuration's
// size (docs/design/amd.md), so their number is what to compare.
//
//	go run ./tools/amd/elfconfigs [-seq] [-tokens N] ELF...
//
// For each ELF: the commands (its runtime sequences), the reconfigurations, the configurations
// that differ, their bytes, the cores' program memory and the blocks of zeros (the kernels'
// zero-initialized data, .bss) among them, and how many loads a reorder
// inside a command could remove at most (the loads less the configurations that differ in it,
// ignoring the dependencies: meaningful for a command of one layer, not for a whole step).
// A configuration is named by the bytes of its writes, so one operator with per-layer runtime
// parameters is a configuration a layer. -seq prints each command's configurations in order.
package main

import (
	"crypto/sha1"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/aieelf"
)

func main() {
	seq := flag.Bool("seq", false, "print each command's configurations in order")
	flag.Parse()
	for _, p := range flag.Args() {
		if err := report(p, *seq); err != nil {
			log.Fatalf("%s: %v", p, err)
		}
	}
}

// config is one configuration: the writes from a load_pdi up to the first operation that is not
// a write (the operator's run: DMA tasks, polls).
type config struct {
	ops, bytes, prog, zero int
}

func report(path string, seq bool) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	f, err := aieelf.Parse(data)
	if err != nil {
		return err
	}
	var insts []*aieelf.Instance
	for _, in := range f.Instances {
		if in.Kernel == "main" && len(in.Ctrl) > 16 {
			insts = append(insts, in)
		}
	}
	sort.Slice(insts, func(i, j int) bool { return seqIndex(insts[i].Name) < seqIndex(insts[j].Name) })
	ids := map[[20]byte]int{}
	var cfgs []config
	var loads, ctrl, cfgBytes, progBytes, zeroBytes, reorder int
	for _, in := range insts {
		_, ops, err := aieelf.DecodeTxn(in.Ctrl)
		if err != nil {
			return fmt.Errorf("%s: %w", in.FullName(), err)
		}
		ctrl += len(in.Ctrl)
		var order []string
		inCmd := map[int]bool{}
		for i, o := range ops {
			if o.Code != aieelf.OpLoadPDI {
				continue
			}
			loads++
			h := sha1.New()
			var c config
			for _, w := range ops[i+1:] {
				if w.Code != aieelf.OpWrite && w.Code != aieelf.OpBlockWrite && w.Code != aieelf.OpMaskWrite &&
					w.Code != aieelf.OpBlockSet {
					break
				}
				h.Write(in.Ctrl[w.Offset : w.Offset+w.Size])
				c.ops++
				c.bytes += w.Size
				if w.Code == aieelf.OpBlockWrite && programMemory(w.RegOff) {
					c.prog += w.Size
				}
				if w.Code == aieelf.OpBlockWrite && allZero(in.Ctrl[w.Offset+blockWriteHeader:w.Offset+w.Size]) {
					c.zero += w.Size
				}
			}
			var k [20]byte
			copy(k[:], h.Sum(nil))
			id, ok := ids[k]
			if !ok {
				id = len(cfgs)
				ids[k] = id
				cfgs = append(cfgs, c)
			}
			cfgBytes += c.bytes
			progBytes += c.prog
			zeroBytes += c.zero
			inCmd[id] = true
			order = append(order, strconv.Itoa(id))
		}
		reorder += len(order) - len(inCmd)
		if seq {
			fmt.Printf("  %s: %s\n", in.Name, strings.Join(order, " "))
		}
	}
	fmt.Printf("%s: %d commands, %d reconfigurations (%d configurations), control code %d KB, configuration %d KB (program memory %d KB, blocks of zeros %d KB); a reorder inside the commands (ignoring their dependencies) removes at most %d\n",
		path, len(insts), loads, len(cfgs), ctrl>>10, cfgBytes>>10, progBytes>>10, zeroBytes>>10, reorder)
	if seq {
		for id, c := range cfgs {
			fmt.Printf("  configuration %d: %d writes, %d B (program memory %d B, blocks of zeros %d B)\n", id, c.ops, c.bytes, c.prog, c.zero)
		}
	}
	return nil
}

// blockWriteHeader is the bytes of a block write before its data (aieelf.DecodeTxn).
const blockWriteHeader = 16

// allZero says whether b holds only zeros (a core's .bss, written at every configuration).
func allZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return len(b) > 0
}

// programMemory says whether a register address is in a core tile's program memory (rows 2 and
// up, 0x20000 to 0x30000 in the tile; row 1 is the memory tile, whose data memory spans it).
func programMemory(reg uint64) bool {
	row, off := (reg>>20)&0x1f, reg&0xfffff
	return row >= 2 && off >= 0x20000 && off < 0x30000
}

// seqIndex orders "sequence", "sequence_1", "sequence_2", ... by their number.
func seqIndex(name string) int {
	_, n, ok := strings.Cut(name, "_")
	if !ok {
		return 0
	}
	i, err := strconv.Atoi(n)
	if err != nil {
		return 1 << 30
	}
	return i
}
