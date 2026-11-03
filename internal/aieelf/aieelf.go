// Package aieelf reads the "full ELF" that mlir-aie's aiecc writes for AIE2P NPUs (Strix
// Point and later): the control code of each runtime sequence, the PDIs it loads, and the
// relocations that put buffer addresses into the control code. XRT calls it a group ELF and
// loads it with xrt::elf and xrt::module (xrt_module.cpp, module_elf_aie2p).
//
// The layout, from XRT 2.21 and an IRON build:
//
//   - FUNC symbols in .symtab are kernels, their names mangled: _Z4mainPcPcPc is main(char*,
//     char*, char*), three arguments.
//   - OBJECT symbols in .symtab are the kernels' instances (runtime sequences); st_shndx is
//     the index of the kernel's symbol. XRT names one "kernel:instance", "main:sequence".
//   - Each SHT_GROUP section is one instance: sh_info is the instance's symbol, the members
//     its .ctrltext.N (and .ctrldata.N, the preemption save and restore code).
//   - .dynsym holds what the relocations refer to: an argument index as a decimal string
//     ("0", "1"), a PDI by section name (".pdi.1"), or a special buffer (scratch-pad-ctrl).
//     st_shndx is the section the relocation patches.
//   - .rela.dyn: r_offset is a byte offset in that section.
//   - .note.xrt.configuration: note type 6, the partition size in columns.
package aieelf

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// OSABI is the ELF OS/ABI byte of an AIE2P control code ELF.
const OSABI = 0x45

// Relocation schemes (XRT's patcher::symbol_type).
const (
	SchemeScalar32         = 3 // w0 = w0&^mask | v&mask, mask = the symbol's size
	SchemeControlPacket48  = 4 // a control packet's address (not supported yet)
	SchemeShimDMA48        = 5 // a shim DMA buffer descriptor's 48-bit address, + the DDR aperture
	SchemeAddress64        = 8 // a 64-bit address, words 0 and 1
	DDRApertureOffset      = 0x80000000
	partitionSizeNoteType  = 6
	symbolScratchpadCtrl   = "scratch-pad-ctrl"
	symbolControlPacket    = "control-packet"
	sectionCtrlText        = ".ctrltext"
	sectionCtrlData        = ".ctrldata"
	sectionPreemptSave     = ".preempt_save"
	sectionPreemptRestore  = ".preempt_restore"
	sectionPDI             = ".pdi"
	sectionConfigNote      = ".note.xrt.configuration"
	elfABIVersionSchemeLow = 1 // ABI version 1 keeps the scheme in the addend's low 4 bits
)

// Target is what a relocation writes.
type Target int

const (
	TargetArg        Target = iota // a kernel argument's buffer address (Arg)
	TargetPDI                      // the address of a PDI (PDI names the section)
	TargetScratchpad               // the ctrl scratchpad (scratch-pad-ctrl)
)

// Reloc is one place in the control code to patch.
type Reloc struct {
	Offset int // byte offset in the control code
	Scheme int
	Addend uint64
	Target Target
	Arg    int    // TargetArg
	PDI    string // TargetPDI: the section name
	Size   uint32 // the symbol's size: the mask of SchemeScalar32
}

// Instance is one runtime sequence: the control code a command runs.
type Instance struct {
	Kernel string // "main"
	Name   string // "sequence"
	Args   int    // the kernel's arguments
	Ctrl   []byte // .ctrltext
	Relocs []Reloc
	// Scratchpad is the size of the ctrl scratchpad the code reads, 0 without one.
	Scratchpad int
}

// FullName is the name XRT and IRON use: "main:sequence".
func (i *Instance) FullName() string { return i.Kernel + ":" + i.Name }

// PDIs lists the PDI sections the instance's relocations refer to, in order of first use.
func (i *Instance) PDIs() []string {
	var names []string
	seen := map[string]bool{}
	for _, r := range i.Relocs {
		if r.Target == TargetPDI && !seen[r.PDI] {
			seen[r.PDI] = true
			names = append(names, r.PDI)
		}
	}
	return names
}

// File is a parsed full ELF.
type File struct {
	Columns   int // the partition size
	Instances []*Instance
	PDIs      map[string][]byte // by section name
}

// Instance finds an instance by its full name, "main:sequence".
func (f *File) Instance(name string) (*Instance, bool) {
	for _, in := range f.Instances {
		if in.FullName() == name {
			return in, true
		}
	}
	return nil, false
}

// IsELF reports whether data starts like an ELF file.
func IsELF(data []byte) bool { return len(data) >= 4 && bytes.Equal(data[:4], []byte(elf.ELFMAG)) }

// Parse reads a full ELF.
func Parse(data []byte) (*File, error) {
	ef, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("aieelf: %w", err)
	}
	if ef.Class != elf.ELFCLASS32 || ef.Data != elf.ELFDATA2LSB {
		return nil, errors.New("aieelf: not a 32-bit little-endian ELF")
	}
	if ef.OSABI != OSABI {
		return nil, fmt.Errorf("aieelf: OS/ABI %#x is not an AIE2P control code ELF (%#x)", byte(ef.OSABI), OSABI)
	}
	abiVersion := ef.ABIVersion
	f := &File{PDIs: map[string][]byte{}}
	for _, s := range ef.Sections {
		switch {
		case s.Name == sectionConfigNote:
			b, err := s.Data()
			if err != nil {
				return nil, err
			}
			if f.Columns, err = partitionSize(b); err != nil {
				return nil, err
			}
		case s.Name == sectionPDI || strings.HasPrefix(s.Name, sectionPDI+"."):
			b, err := s.Data()
			if err != nil {
				return nil, err
			}
			f.PDIs[s.Name] = b
		}
	}
	if f.Columns == 0 {
		return nil, errors.New("aieelf: no partition size (.note.xrt.configuration): not a full ELF")
	}
	syms, err := ef.Symbols()
	if err != nil {
		return nil, fmt.Errorf("aieelf: .symtab: %w", err)
	}
	// debug/elf drops the null symbol 0: .symtab index i is syms[i-1].
	sym := func(i uint32) (elf.Symbol, bool) {
		if i == 0 || int(i) > len(syms) {
			return elf.Symbol{}, false
		}
		return syms[i-1], true
	}
	bySection := map[int]*Instance{}
	for si, s := range ef.Sections {
		if s.Type != elf.SHT_GROUP {
			continue
		}
		inst, ok := sym(s.Info)
		if !ok || elf.ST_TYPE(inst.Info) != elf.STT_OBJECT {
			return nil, fmt.Errorf("aieelf: group %s: signature %d is no instance symbol", s.Name, s.Info)
		}
		kern, ok := sym(uint32(inst.Section))
		if !ok || elf.ST_TYPE(kern.Info) != elf.STT_FUNC {
			return nil, fmt.Errorf("aieelf: instance %s: %d is no kernel symbol", inst.Name, inst.Section)
		}
		name, args, err := demangle(kern.Name)
		if err != nil {
			return nil, err
		}
		in := &Instance{Kernel: name, Name: inst.Name, Args: args}
		b, err := s.Data()
		if err != nil {
			return nil, err
		}
		if len(b) < 4 || len(b)%4 != 0 {
			return nil, fmt.Errorf("aieelf: group %s: bad size %d", s.Name, len(b))
		}
		ctrl := -1
		for j := 4; j < len(b); j += 4 { // word 0 is the GRP_COMDAT flag
			m := int(binary.LittleEndian.Uint32(b[j:]))
			if m <= 0 || m >= len(ef.Sections) {
				return nil, fmt.Errorf("aieelf: group %s: member %d", s.Name, m)
			}
			ms := ef.Sections[m]
			switch {
			case strings.HasPrefix(ms.Name, sectionCtrlText):
				if in.Ctrl, err = ms.Data(); err != nil {
					return nil, err
				}
				ctrl = m
			case strings.HasPrefix(ms.Name, sectionCtrlData), strings.HasPrefix(ms.Name, sectionPreemptSave),
				strings.HasPrefix(ms.Name, sectionPreemptRestore):
				return nil, fmt.Errorf("aieelf: %s: %s is not supported yet", in.FullName(), ms.Name)
			}
		}
		if ctrl < 0 {
			return nil, fmt.Errorf("aieelf: %s has no control code", in.FullName())
		}
		bySection[ctrl] = in
		_ = si
		f.Instances = append(f.Instances, in)
	}
	if len(f.Instances) == 0 {
		return nil, errors.New("aieelf: no kernel instances (groups)")
	}
	if err := readRelocs(ef, abiVersion, bySection, f); err != nil {
		return nil, err
	}
	return f, nil
}

func readRelocs(ef *elf.File, abiVersion uint8, bySection map[int]*Instance, f *File) error {
	rela := ef.Section(".rela.dyn")
	if rela == nil {
		return nil
	}
	dsyms, err := ef.DynamicSymbols()
	if err != nil {
		return fmt.Errorf("aieelf: .dynsym: %w", err)
	}
	b, err := rela.Data()
	if err != nil {
		return err
	}
	if len(b)%12 != 0 {
		return fmt.Errorf("aieelf: .rela.dyn size %d", len(b))
	}
	for o := 0; o < len(b); o += 12 {
		off := binary.LittleEndian.Uint32(b[o:])
		info := binary.LittleEndian.Uint32(b[o+4:])
		// r_addend is a byte offset into the argument's buffer, unsigned: IRON's arena banks
		// are 3.5 GiB, and an offset past 2 GiB read as signed pointed below the buffer (a
		// decode step of Gemma 4 E2B wrote into its own weights).
		addend := uint64(binary.LittleEndian.Uint32(b[o+8:]))
		si := elf.R_SYM32(info)
		if si == 0 || int(si) > len(dsyms) {
			return fmt.Errorf("aieelf: relocation at %#x: symbol %d", off, si)
		}
		s := dsyms[si-1]
		in, ok := bySection[int(s.Section)]
		if !ok {
			continue // the control code of an instance with no group, never run
		}
		scheme := int(elf.R_TYPE32(info))
		if abiVersion == elfABIVersionSchemeLow {
			scheme, addend = int(addend&0xf), addend>>4
		}
		r := Reloc{Offset: int(off), Scheme: scheme, Addend: addend, Size: uint32(s.Size)}
		switch {
		case strings.Contains(s.Name, "pdi"):
			if _, ok := f.PDIs[s.Name]; !ok {
				return fmt.Errorf("aieelf: %s: no section %s", in.FullName(), s.Name)
			}
			r.Target, r.PDI = TargetPDI, s.Name
		case s.Name == symbolScratchpadCtrl:
			r.Target = TargetScratchpad
			in.Scratchpad = max(in.Scratchpad, int(s.Size))
		case s.Name == symbolControlPacket || strings.HasPrefix(s.Name, "ctrlpkt") || strings.HasPrefix(s.Name, "scratch-pad"):
			return fmt.Errorf("aieelf: %s: %s is not supported yet", in.FullName(), s.Name)
		default:
			n, err := strconv.Atoi(s.Name)
			if err != nil || n < 0 || n >= in.Args {
				return fmt.Errorf("aieelf: %s: relocation symbol %q is no argument of %d", in.FullName(), s.Name, in.Args)
			}
			r.Target, r.Arg = TargetArg, n
		}
		switch r.Scheme {
		case SchemeShimDMA48:
			if r.Offset+16 > len(in.Ctrl) {
				return fmt.Errorf("aieelf: %s: relocation at %#x is outside the control code", in.FullName(), off)
			}
		case SchemeAddress64:
			if r.Offset+8 > len(in.Ctrl) {
				return fmt.Errorf("aieelf: %s: relocation at %#x is outside the control code", in.FullName(), off)
			}
		case SchemeScalar32:
			if r.Offset+4 > len(in.Ctrl) {
				return fmt.Errorf("aieelf: %s: relocation at %#x is outside the control code", in.FullName(), off)
			}
		default:
			return fmt.Errorf("aieelf: %s: relocation scheme %d at %#x is not supported", in.FullName(), r.Scheme, off)
		}
		in.Relocs = append(in.Relocs, r)
	}
	return nil
}

// partitionSize reads note 6 of .note.xrt.configuration: a u32, the partition's columns.
func partitionSize(b []byte) (int, error) {
	for len(b) >= 12 {
		namesz := int(binary.LittleEndian.Uint32(b))
		descsz := int(binary.LittleEndian.Uint32(b[4:]))
		typ := binary.LittleEndian.Uint32(b[8:])
		name := (namesz + 3) &^ 3
		desc := (descsz + 3) &^ 3
		if 12+name+desc > len(b) {
			break
		}
		if typ == partitionSizeNoteType && descsz >= 4 {
			return int(binary.LittleEndian.Uint32(b[12+name:])), nil
		}
		b = b[12+name+desc:]
	}
	return 0, errors.New("aieelf: no partition size note")
}

// demangle reads an Itanium-mangled kernel name of pointer arguments: _Z4mainPcPcPc is
// main(char*, char*, char*). A plain name has no arguments.
func demangle(s string) (string, int, error) {
	if !strings.HasPrefix(s, "_Z") {
		return s, 0, nil
	}
	rest := s[2:]
	i := 0
	for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
		i++
	}
	n, err := strconv.Atoi(rest[:i])
	if err != nil || i+n > len(rest) {
		return "", 0, fmt.Errorf("aieelf: kernel symbol %q", s)
	}
	name, params := rest[i:i+n], rest[i+n:]
	args := 0
	for len(params) > 0 {
		switch {
		case strings.HasPrefix(params, "Pc"), strings.HasPrefix(params, "Pv"), strings.HasPrefix(params, "Ph"):
			params = params[2:]
		case params == "v":
			params = ""
			continue
		default:
			return "", 0, fmt.Errorf("aieelf: kernel %s: argument types %q are not pointers", name, params)
		}
		args++
	}
	return name, args, nil
}

// Patch writes v at r in code (the instance's control code, or a copy of it). For the
// relative schemes it adds to the words of pristine, the unpatched code, so patching again
// with another address gives the same result as patching once.
func Patch(code, pristine []byte, r Reloc, v uint64) {
	le := binary.LittleEndian
	o := r.Offset
	switch r.Scheme {
	case SchemeShimDMA48:
		w1, w2 := le.Uint32(pristine[o+4:]), le.Uint32(pristine[o+8:])
		b := (uint64(w2&0xffff)<<32 | uint64(w1)) + v + r.Addend + DDRApertureOffset
		le.PutUint32(code[o+4:], uint32(b)&0xfffffffc)
		le.PutUint32(code[o+8:], w2&0xffff0000|uint32(b>>32))
	case SchemeAddress64:
		le.PutUint64(code[o:], v)
	case SchemeScalar32:
		w := le.Uint32(pristine[o:])
		le.PutUint32(code[o:], w&^r.Size|uint32(v)&r.Size)
	}
}
