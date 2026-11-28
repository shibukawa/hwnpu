// Command elfredundant measures, in IRON's full ELFs, how much of what a reconfiguration of the
// array writes is already there: it follows the control code of every command in order, keeps
// what each write left in the tiles' memories and registers, and counts the bytes of the writes
// that put back what was there already (as if a load_pdi cleared nothing), by region.
//
//	go run ./tools/amd/elfredundant [-passes 2] [-dump N] ELF...
//
// It is a static count: whether a write may really be left out depends on what load_pdi's reset
// clears on the device (docs/design/amd.md).
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/aieelf"
)

type region int

const (
	coreProg region = iota
	coreData
	coreReg
	memData
	memReg
	shimReg
	nRegions
)

var names = [...]string{"core program", "core data memory", "core registers", "memtile data", "memtile registers", "shim registers"}

func classify(reg uint64) region {
	row, off := (reg>>20)&0x1f, reg&0xfffff
	switch {
	case row == 0:
		return shimReg
	case row == 1:
		if off < 0x80000 {
			return memData
		}
		return memReg
	case off < 0x10000:
		return coreData
	case off >= 0x20000 && off < 0x30000:
		return coreProg
	}
	return coreReg
}

func main() {
	passes := flag.Int("passes", 2, "times the commands run in order; the last is counted (a decode step after another)")
	dump := flag.Int("dump", -1, "print the operations of this reconfiguration (its index in the last pass)")
	hist := flag.Bool("ops", false, "count the commands' operations by kind instead")
	flag.Parse()
	for _, p := range flag.Args() {
		if *hist {
			if err := histogram(p); err != nil {
				log.Fatalf("%s: %v", p, err)
			}
			continue
		}
		if err := report(p, *passes, *dump); err != nil {
			log.Fatalf("%s: %v", p, err)
		}
	}
}

func report(path string, passes, dump int) error {
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
	mem := map[uint64]uint32{} // word address -> what the last write left
	var total, same [nRegions]int
	var loads, others, otherBytes int
	le := binary.LittleEndian
	for pass := 0; pass < passes; pass++ {
		count := pass == passes-1
		n := 0
		for _, in := range insts {
			_, ops, err := aieelf.DecodeTxn(in.Ctrl)
			if err != nil {
				return fmt.Errorf("%s: %w", in.FullName(), err)
			}
			for _, o := range ops {
				show := count && dump >= 0 && n-1 == dump
				switch o.Code {
				case aieelf.OpLoadPDI:
					if count {
						loads++
					}
					n++
					if count && dump >= 0 && n-1 == dump {
						fmt.Printf("reconfiguration %d in %s:\n", dump, in.Name)
					}
				case aieelf.OpBlockWrite:
					b := in.Ctrl[o.Offset+16 : o.Offset+o.Size]
					r := classify(o.RegOff)
					eq := 0
					for i := 0; i+4 <= len(b); i += 4 {
						a, v := o.RegOff+uint64(i), le.Uint32(b[i:])
						if old, ok := mem[a]; ok && old == v {
							eq += 4
						}
						mem[a] = v
					}
					if count {
						total[r] += len(b)
						same[r] += eq
					}
					if show {
						fmt.Printf("  blockwrite col %d row %d off %#x: %d B (%s), %d B as before\n", o.RegOff>>25, (o.RegOff>>20)&0x1f, o.RegOff&0xfffff, len(b), names[r], eq)
					}
				case aieelf.OpWrite, aieelf.OpMaskWrite:
					r := classify(o.RegOff)
					eq := 0
					if old, ok := mem[o.RegOff]; ok && old == o.Value && o.Code == aieelf.OpWrite {
						eq = 4
					}
					if o.Code == aieelf.OpWrite {
						mem[o.RegOff] = o.Value
					} else {
						delete(mem, o.RegOff)
					}
					if count {
						total[r] += o.Size
						if eq > 0 {
							same[r] += o.Size
						}
					}
					if show {
						fmt.Printf("  write%d col %d row %d off %#x = %#x (%s)\n", o.Code, o.RegOff>>25, (o.RegOff>>20)&0x1f, o.RegOff&0xfffff, o.Value, names[r])
					}
				default:
					if count {
						others++
						otherBytes += o.Size
					}
					if show {
						fmt.Printf("  op %d (%d B)\n", o.Code, o.Size)
					}
				}
			}
		}
	}
	var t, s int
	fmt.Printf("%s: %d commands, %d reconfigurations a pass\n", path, len(insts), loads)
	for r := region(0); r < nRegions; r++ {
		fmt.Printf("  %-18s %8d KB written, %8d KB (%4.1f %%) as it was\n", names[r], total[r]>>10, same[r]>>10, pct(same[r], total[r]))
		t += total[r]
		s += same[r]
	}
	fmt.Printf("  %-18s %8d KB written, %8d KB (%4.1f %%) as it was; other operations %d (%d KB)\n", "all", t>>10, s>>10, pct(s, t), others, otherBytes>>10)
	return nil
}

var opNames = map[uint8]string{0: "write32", 1: "blockwrite", 2: "blockset", 3: "maskwrite", 4: "maskpoll", 5: "noop", 6: "preempt",
	7: "maskpoll busy", 8: "load_pdi", 9: "load_pm_start", 10: "create_scratchpad", 11: "update_state_table", 12: "update_reg",
	13: "update_scratch", 0x80: "tct (wait for a task's token)", 0x81: "ddr patch", 0x82: "read regs", 0x83: "rec timer", 0x84: "merge sync"}

// histogram prints, for an ELF's commands together, the operations of the control code by kind:
// their count and bytes, and for the writes the region they go to.
func histogram(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	f, err := aieelf.Parse(data)
	if err != nil {
		return err
	}
	type key struct {
		code uint8
		r    region
	}
	n, b := map[key]int{}, map[key]int{}
	ctrl, ctrlData := 0, 0
	for _, in := range f.Instances {
		if in.Kernel != "main" || len(in.Ctrl) <= 16 {
			continue
		}
		_, ops, err := aieelf.DecodeTxn(in.Ctrl)
		if err != nil {
			return fmt.Errorf("%s: %w", in.FullName(), err)
		}
		ctrl += len(in.Ctrl)
		ctrlData += len(in.CtrlData)
		for _, o := range ops {
			k := key{o.Code, nRegions}
			if o.Code <= aieelf.OpMaskPoll || o.Code == aieelf.OpMaskPollBusy {
				k.r = classify(o.RegOff)
			}
			n[k]++
			b[k] += o.Size
		}
	}
	keys := make([]key, 0, len(n))
	for k := range n {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return b[keys[i]] > b[keys[j]] })
	fmt.Printf("%s: control code %d KB, control packets %d KB\n", path, ctrl>>10, ctrlData>>10)
	for _, k := range keys {
		where := ""
		if k.r < nRegions {
			where = " to " + names[k.r]
		}
		fmt.Printf("  %7d x %-30s %8d B\n", n[k], opNames[k.code]+where, b[k])
	}
	return nil
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

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
