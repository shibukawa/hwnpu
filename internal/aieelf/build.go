package aieelf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sort"
	"strconv"
)

// BuildInstance describes one instance for Build.
type BuildInstance struct {
	Kernel string // a kernel of Args char* arguments
	Name   string
	Args   int
	Ctrl   []byte
	Relocs []Reloc // Target and Arg / PDI name the symbol
}

// BuildSpec describes a full ELF for Build.
type BuildSpec struct {
	Columns   int
	Instances []BuildInstance
	PDIs      map[string][]byte // ".pdi.1" -> image
}

// Build writes a full ELF in the layout aiecc uses, for tests: the sections, symbols, groups
// and relocations Parse reads.
func Build(spec BuildSpec) []byte {
	type section struct {
		name                   string
		typ, flags, link, info uint32
		align, entsize         uint32
		data                   []byte
	}
	var secs []section
	add := func(s section) int { secs = append(secs, s); return len(secs) } // index after the null section
	le := binary.LittleEndian
	strtab := []byte{0}
	str := func(tab *[]byte, s string) uint32 {
		o := len(*tab)
		*tab = append(append(*tab, s...), 0)
		return uint32(o)
	}
	symEntry := func(name uint32, value, size uint32, info byte, shndx uint16) []byte {
		b := make([]byte, 16)
		le.PutUint32(b, name)
		le.PutUint32(b[4:], value)
		le.PutUint32(b[8:], size)
		b[12] = info
		le.PutUint16(b[14:], shndx)
		return b
	}
	const (
		shtProgbits = 1
		shtSymtab   = 2
		shtStrtab   = 3
		shtRela     = 4
		shtNote     = 7
		shtDynsym   = 11
		shtGroup    = 17
		stbGlobal   = 1
		stbWeak     = 2
		sttObject   = 1
		sttFunc     = 2
	)
	// Section indexes: 1 .shstrtab, 2 .strtab, 3 .symtab, then the rest.
	shstrIdx := add(section{name: ".shstrtab", typ: shtStrtab, align: 1})
	strIdx := add(section{name: ".strtab", typ: shtStrtab})
	symIdx := add(section{name: ".symtab", typ: shtSymtab, link: uint32(strIdx), info: 1, align: 4, entsize: 16})
	symtab := symEntry(0, 0, 0, 0, 0)
	nsym := uint32(1)
	pdiNames := make([]string, 0, len(spec.PDIs))
	for n := range spec.PDIs {
		pdiNames = append(pdiNames, n)
	}
	sort.Strings(pdiNames)
	pdiIdx := map[string]int{}
	for _, n := range pdiNames {
		pdiIdx[n] = add(section{name: n, typ: shtProgbits, flags: 6, align: 16, data: spec.PDIs[n]})
	}
	ctrlIdx := make([]int, len(spec.Instances))
	for i, in := range spec.Instances {
		kern := nsym
		mangled := "_Z" + strconv.Itoa(len(in.Kernel)) + in.Kernel
		for a := 0; a < in.Args; a++ {
			mangled += "Pc"
		}
		symtab = append(symtab, symEntry(str(&strtab, mangled), 0, 0, stbWeak<<4|sttFunc, 0)...)
		nsym++
		inst := nsym
		symtab = append(symtab, symEntry(str(&strtab, in.Name), 0, 0, stbWeak<<4|sttObject, uint16(kern))...)
		nsym++
		ctrlIdx[i] = add(section{name: fmt.Sprintf(".ctrltext.%d", i), typ: shtProgbits, flags: 6, align: 16, data: in.Ctrl})
		g := le.AppendUint32(nil, 1) // GRP_COMDAT
		g = le.AppendUint32(g, uint32(ctrlIdx[i]))
		add(section{name: fmt.Sprintf(".group.%d", i), typ: shtGroup, link: uint32(symIdx), info: inst, align: 4, entsize: 4, data: g})
	}
	dynstr := []byte{0}
	dynsym := symEntry(0, 0, 0, 0, 0)
	ndyn := uint32(1)
	var rela []byte
	dynKey := map[string]uint32{}
	for i, in := range spec.Instances {
		for _, r := range in.Relocs {
			name, size := "", r.Size
			switch r.Target {
			case TargetArg:
				name = strconv.Itoa(r.Arg)
			case TargetPDI:
				name = r.PDI
			case TargetScratchpad:
				name = symbolScratchpadCtrl
			}
			key := fmt.Sprintf("%d/%s", ctrlIdx[i], name)
			si, ok := dynKey[key]
			if !ok {
				dynsym = append(dynsym, symEntry(str(&dynstr, name), 0, size, stbGlobal<<4|sttObject, uint16(ctrlIdx[i]))...)
				si = ndyn
				dynKey[key] = si
				ndyn++
			}
			rela = le.AppendUint32(rela, uint32(r.Offset))
			rela = le.AppendUint32(rela, si<<8|uint32(r.Scheme))
			rela = le.AppendUint32(rela, uint32(int32(r.Addend)))
		}
	}
	dynstrIdx := add(section{name: ".dynstr", typ: shtStrtab, data: dynstr})
	dynsymIdx := add(section{name: ".dynsym", typ: shtDynsym, link: uint32(dynstrIdx), info: 1, align: 8, entsize: 16, data: dynsym})
	add(section{name: ".rela.dyn", typ: shtRela, link: uint32(dynsymIdx), align: 8, entsize: 12, data: rela})
	note := le.AppendUint32(nil, 4)
	note = le.AppendUint32(note, 4)
	note = le.AppendUint32(note, partitionSizeNoteType)
	note = append(note, 'X', 'R', 'T', 0)
	note = le.AppendUint32(note, uint32(spec.Columns))
	add(section{name: sectionConfigNote, typ: shtNote, align: 1, data: note})
	secs[strIdx-1].data = strtab
	secs[symIdx-1].data = symtab
	shstr := []byte{0}
	names := make([]uint32, len(secs))
	for i := range secs {
		names[i] = str(&shstr, secs[i].name)
	}
	secs[shstrIdx-1].data = shstr

	var out bytes.Buffer
	hdr := make([]byte, 52)
	copy(hdr, "\x7fELF")
	hdr[4], hdr[5], hdr[6], hdr[7], hdr[8] = 1, 1, 1, OSABI, 0x21
	le.PutUint16(hdr[16:], 2)  // ET_EXEC
	le.PutUint16(hdr[18:], 10) // a machine debug/elf accepts (aiecc writes 1)
	le.PutUint32(hdr[20:], 1)
	le.PutUint16(hdr[40:], 52)
	le.PutUint16(hdr[46:], 40)
	le.PutUint16(hdr[48:], uint16(len(secs)+1))
	le.PutUint16(hdr[50:], uint16(shstrIdx))
	out.Write(hdr)
	offs := make([]uint32, len(secs))
	for i := range secs {
		for out.Len()%16 != 0 {
			out.WriteByte(0)
		}
		offs[i] = uint32(out.Len())
		out.Write(secs[i].data)
	}
	for out.Len()%4 != 0 {
		out.WriteByte(0)
	}
	shoff := out.Len()
	out.Write(make([]byte, 40)) // the null section
	for i, s := range secs {
		b := make([]byte, 40)
		le.PutUint32(b, names[i])
		le.PutUint32(b[4:], s.typ)
		le.PutUint32(b[8:], s.flags)
		le.PutUint32(b[16:], offs[i])
		le.PutUint32(b[20:], uint32(len(s.data)))
		le.PutUint32(b[24:], s.link)
		le.PutUint32(b[28:], s.info)
		le.PutUint32(b[32:], s.align)
		le.PutUint32(b[36:], s.entsize)
		out.Write(b)
	}
	b := out.Bytes()
	le.PutUint32(b[32:], uint32(shoff))
	return b
}
