// Package mlirbc reads and writes MLIR bytecode (version 6, mlir/Bytecode/Encoding.h), the
// program format of a Core AI model (.aimodel's main.mlirb). The reader decodes the container
// (strings, dialects, attribute and type tables, properties, resources, the IR's structure)
// without knowing any dialect: attributes and types with a custom encoding stay raw bytes. It is
// for checking what the writer produces against what coreai-core writes.
package mlirbc

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// Section ids.
const (
	secString = iota
	secDialect
	secAttrType
	secAttrTypeOffset
	secIR
	secResource
	secResourceOffset
	secDialectVersions
	secProperties
	numSections
)

// Op encoding mask bits.
const (
	opHasAttrs         = 1 << 0
	opHasResults       = 1 << 1
	opHasOperands      = 1 << 2
	opHasSuccessors    = 1 << 3
	opHasInlineRegions = 1 << 4
	opHasUseListOrders = 1 << 5
	opHasProperties    = 1 << 6
)

// Version is the bytecode version read and written.
const Version = 6

var magic = []byte("ML\xefR")

type reader struct {
	b   []byte
	off int
}

func (r *reader) empty() bool { return r.off >= len(r.b) }

func (r *reader) byte() (byte, error) {
	if r.off >= len(r.b) {
		return 0, errors.New("mlirbc: unexpected end")
	}
	c := r.b[r.off]
	r.off++
	return c, nil
}

func (r *reader) bytes(n uint64) ([]byte, error) {
	if uint64(len(r.b)-r.off) < n {
		return nil, fmt.Errorf("mlirbc: %d bytes past the end", n)
	}
	out := r.b[r.off : r.off+int(n)]
	r.off += int(n)
	return out, nil
}

func (r *reader) varint() (uint64, error) {
	c, err := r.byte()
	if err != nil {
		return 0, err
	}
	if c&1 != 0 {
		return uint64(c >> 1), nil
	}
	if c == 0 {
		b, err := r.bytes(8)
		if err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(b), nil
	}
	extra := 0
	for c>>extra&1 == 0 {
		extra++
	}
	var buf [8]byte
	buf[0] = c
	b, err := r.bytes(uint64(extra))
	if err != nil {
		return 0, err
	}
	copy(buf[1:], b)
	return binary.LittleEndian.Uint64(buf[:]) >> (extra + 1), nil
}

func (r *reader) flagged() (uint64, bool, error) {
	v, err := r.varint()
	return v >> 1, v&1 != 0, err
}

func (r *reader) cstring() (string, error) {
	i := bytes.IndexByte(r.b[r.off:], 0)
	if i < 0 {
		return "", errors.New("mlirbc: unterminated string")
	}
	s := string(r.b[r.off : r.off+i])
	r.off += i + 1
	return s, nil
}

// section reads a section header and its data (alignment padding skipped).
func (r *reader) section() (id byte, data []byte, align uint64, err error) {
	c, err := r.byte()
	if err != nil {
		return 0, nil, 0, err
	}
	id = c & 0x7f
	n, err := r.varint()
	if err != nil {
		return 0, nil, 0, err
	}
	align = 1
	if c&0x80 != 0 {
		if align, err = r.varint(); err != nil {
			return 0, nil, 0, err
		}
		for r.off%int(align) != 0 {
			if _, err := r.byte(); err != nil {
				return 0, nil, 0, err
			}
		}
	}
	data, err = r.bytes(n)
	return id, data, align, err
}

// Entry is an attribute or a type of the file's tables.
type Entry struct {
	Dialect string
	Custom  bool   // Data is the dialect's own encoding
	Data    []byte // the textual form (without the NUL) when not Custom
}

func (e Entry) String() string {
	if e.Custom {
		return fmt.Sprintf("<%s custom %x>", e.Dialect, e.Data)
	}
	return string(e.Data)
}

// Resource is one entry of a resource group.
type Resource struct {
	Key   string
	Kind  byte // 0 blob, 1 bool, 2 string
	Align uint64
	Data  []byte // the blob's bytes
}

// Op is one operation of the IR.
type Op struct {
	Name       string // dialect.op
	Registered bool
	Loc        uint64
	Attrs      int64 // attribute index of the dictionary, -1 none
	Props      []byte
	PropsAttr  int64    // the properties' attribute index of an op written unregistered, -1 none
	Results    []uint64 // type indices
	Operands   []uint64 // value indices in the region's scope
	Regions    []*Region
	Isolated   bool
}

// Region is a list of blocks.
type Region struct {
	NumValues uint64
	Blocks    []*Block
}

// Block is a list of operations with typed arguments.
type Block struct {
	Args    []uint64 // type indices
	ArgLocs []int64  // location attribute indices, -1 none
	Ops     []*Op
}

// File is a decoded bytecode file.
type File struct {
	Version   uint64
	Producer  string
	Strings   []string
	Dialects  []string
	Versions  map[string][]byte // dialect version sections
	OpNames   []string
	OpReg     []bool // whether each op name was registered when written
	Attrs     []Entry
	Types     []Entry
	Props     [][]byte
	Resources map[string][]Resource // by dialect or external provider
	Top       *Block
	Sections  []string // id, alignment and length of each top-level section, in file order
}

// Read decodes b.
func Read(b []byte) (*File, error) {
	if !bytes.HasPrefix(b, magic) {
		return nil, errors.New("mlirbc: not MLIR bytecode")
	}
	r := &reader{b: b, off: len(magic)}
	f := &File{Versions: map[string][]byte{}, Resources: map[string][]Resource{}}
	var err error
	if f.Version, err = r.varint(); err != nil {
		return nil, err
	}
	if f.Version != Version {
		return nil, fmt.Errorf("mlirbc: version %d, not %d", f.Version, Version)
	}
	if f.Producer, err = r.cstring(); err != nil {
		return nil, err
	}
	var secs [numSections][]byte
	var have [numSections]bool
	for !r.empty() {
		id, data, align, err := r.section()
		if err != nil {
			return nil, err
		}
		if int(id) >= numSections || have[id] {
			return nil, fmt.Errorf("mlirbc: section %d unknown or twice", id)
		}
		f.Sections = append(f.Sections, fmt.Sprintf("%d/align %d/%d bytes", id, align, len(data)))
		secs[id], have[id] = data, true
	}
	if err := f.readStrings(secs[secString]); err != nil {
		return nil, err
	}
	if err := f.readDialects(secs[secDialect]); err != nil {
		return nil, err
	}
	if err := f.readAttrTypes(secs[secAttrType], secs[secAttrTypeOffset]); err != nil {
		return nil, err
	}
	if have[secProperties] {
		if err := f.readProps(secs[secProperties]); err != nil {
			return nil, err
		}
	}
	if have[secResource] {
		if err := f.readResources(secs[secResource], secs[secResourceOffset]); err != nil {
			return nil, err
		}
	}
	ir := &reader{b: secs[secIR]}
	f.Top = &Block{}
	if err := f.readBlock(ir, f.Top, true); err != nil {
		return nil, fmt.Errorf("mlirbc: IR: %w", err)
	}
	return f, nil
}

func (f *File) readStrings(b []byte) error {
	r := &reader{b: b}
	n, err := r.varint()
	if err != nil {
		return err
	}
	lens := make([]uint64, n)
	for i := int(n) - 1; i >= 0; i-- {
		if lens[i], err = r.varint(); err != nil {
			return err
		}
	}
	for _, l := range lens {
		s, err := r.bytes(l)
		if err != nil {
			return err
		}
		f.Strings = append(f.Strings, string(bytes.TrimSuffix(s, []byte{0}))) // each written with its NUL
	}
	return nil
}

func (f *File) str(i uint64) (string, error) {
	if i >= uint64(len(f.Strings)) {
		return "", fmt.Errorf("mlirbc: string %d of %d", i, len(f.Strings))
	}
	return f.Strings[i], nil
}

func (f *File) readDialects(b []byte) error {
	r := &reader{b: b}
	n, err := r.varint()
	if err != nil {
		return err
	}
	for range n {
		idx, ver, err := r.flagged()
		if err != nil {
			return err
		}
		name, err := f.str(idx)
		if err != nil {
			return err
		}
		f.Dialects = append(f.Dialects, name)
		if ver {
			id, data, _, err := r.section()
			if err != nil {
				return err
			}
			if id != secDialectVersions {
				return fmt.Errorf("mlirbc: dialect %s: section %d, not a version", name, id)
			}
			f.Versions[name] = data
		}
	}
	if _, err := r.varint(); err != nil { // the number of op names
		return err
	}
	for !r.empty() {
		d, err := r.varint()
		if err != nil {
			return err
		}
		k, err := r.varint()
		if err != nil {
			return err
		}
		for range k {
			idx, reg, err := r.flagged()
			if err != nil {
				return err
			}
			s, err := f.str(idx)
			if err != nil {
				return err
			}
			f.OpNames = append(f.OpNames, f.Dialects[d]+"."+s)
			f.OpReg = append(f.OpReg, reg)
		}
	}
	return nil
}

func (f *File) readAttrTypes(data, offs []byte) error {
	r := &reader{b: offs}
	na, err := r.varint()
	if err != nil {
		return err
	}
	nt, err := r.varint()
	if err != nil {
		return err
	}
	off := uint64(0)
	read := func(n uint64) ([]Entry, error) {
		var out []Entry
		for uint64(len(out)) < n {
			d, err := r.varint()
			if err != nil {
				return nil, err
			}
			k, err := r.varint()
			if err != nil {
				return nil, err
			}
			for range k {
				size, custom, err := r.flagged()
				if err != nil {
					return nil, err
				}
				if off+size > uint64(len(data)) {
					return nil, errors.New("mlirbc: attribute or type past the section")
				}
				e := Entry{Dialect: f.Dialects[d], Custom: custom, Data: data[off : off+size]}
				if !custom {
					e.Data = bytes.TrimSuffix(e.Data, []byte{0})
				}
				out = append(out, e)
				off += size
			}
		}
		return out, nil
	}
	if f.Attrs, err = read(na); err != nil {
		return err
	}
	f.Types, err = read(nt)
	return err
}

func (f *File) readProps(b []byte) error {
	r := &reader{b: b}
	n, err := r.varint()
	if err != nil {
		return err
	}
	for range n {
		size, err := r.varint()
		if err != nil {
			return err
		}
		p, err := r.bytes(size)
		if err != nil {
			return err
		}
		f.Props = append(f.Props, p)
	}
	return nil
}

func (f *File) readResources(data, offs []byte) error {
	r := &reader{b: offs}
	res := &reader{b: data}
	ext, err := r.varint()
	if err != nil {
		return err
	}
	group := func(name string) error {
		n, err := r.varint()
		if err != nil {
			return err
		}
		for range n {
			ki, err := r.varint()
			if err != nil {
				return err
			}
			key, err := f.str(ki)
			if err != nil {
				return err
			}
			size, err := r.varint()
			if err != nil {
				return err
			}
			kind, err := r.byte()
			if err != nil {
				return err
			}
			start := res.off
			raw, err := res.bytes(size)
			if err != nil {
				return err
			}
			e := Resource{Key: key, Kind: kind}
			if kind == 0 && len(raw) > 0 {
				er := &reader{b: data, off: start}
				align, err := er.varint()
				if err != nil {
					return err
				}
				n, err := er.varint()
				if err != nil {
					return err
				}
				for er.off%int(align) != 0 {
					er.off++
				}
				e.Align, e.Data = align, data[er.off:er.off+int(n)]
			} else {
				e.Data = raw
			}
			f.Resources[name] = append(f.Resources[name], e)
		}
		return nil
	}
	for range ext {
		ki, err := r.varint()
		if err != nil {
			return err
		}
		key, err := f.str(ki)
		if err != nil {
			return err
		}
		if err := group(key); err != nil {
			return err
		}
	}
	for !r.empty() {
		d, err := r.varint()
		if err != nil {
			return err
		}
		if err := group(f.Dialects[d]); err != nil {
			return err
		}
	}
	return nil
}

func (f *File) readBlock(r *reader, b *Block, top bool) error {
	nops, hasArgs, err := r.flagged()
	if err != nil {
		return err
	}
	if hasArgs {
		n, err := r.varint()
		if err != nil {
			return err
		}
		for range n {
			t, hasLoc, err := r.flagged()
			if err != nil {
				return err
			}
			b.Args = append(b.Args, t)
			loc := int64(-1)
			if hasLoc {
				l, err := r.varint()
				if err != nil {
					return err
				}
				loc = int64(l)
			}
			b.ArgLocs = append(b.ArgLocs, loc)
		}
		uses, err := r.byte()
		if err != nil {
			return err
		}
		if uses != 0 {
			return errors.New("block argument use-list orders are not supported")
		}
	}
	for range nops {
		op, err := f.readOp(r)
		if err != nil {
			return err
		}
		b.Ops = append(b.Ops, op)
	}
	return nil
}

func (f *File) readOp(r *reader) (*Op, error) {
	ni, err := r.varint()
	if err != nil {
		return nil, err
	}
	if ni >= uint64(len(f.OpNames)) {
		return nil, fmt.Errorf("op name %d of %d", ni, len(f.OpNames))
	}
	op := &Op{Name: f.OpNames[ni], Registered: f.OpReg[ni], Attrs: -1, PropsAttr: -1}
	mask, err := r.byte()
	if err != nil {
		return nil, err
	}
	if op.Loc, err = r.varint(); err != nil {
		return nil, err
	}
	if mask&opHasAttrs != 0 {
		a, err := r.varint()
		if err != nil {
			return nil, err
		}
		op.Attrs = int64(a)
	}
	if mask&opHasProperties != 0 {
		p, err := r.varint()
		if err != nil {
			return nil, err
		}
		if !op.Registered { // the properties are one attribute
			op.PropsAttr = int64(p)
		} else {
			if p >= uint64(len(f.Props)) {
				return nil, fmt.Errorf("properties %d of %d", p, len(f.Props))
			}
			op.Props = f.Props[p]
		}
	}
	if mask&opHasResults != 0 {
		n, err := r.varint()
		if err != nil {
			return nil, err
		}
		for range n {
			t, err := r.varint()
			if err != nil {
				return nil, err
			}
			op.Results = append(op.Results, t)
		}
	}
	if mask&opHasOperands != 0 {
		n, err := r.varint()
		if err != nil {
			return nil, err
		}
		for range n {
			v, err := r.varint()
			if err != nil {
				return nil, err
			}
			op.Operands = append(op.Operands, v)
		}
	}
	if mask&opHasSuccessors != 0 {
		return nil, errors.New("successors are not supported")
	}
	if mask&opHasUseListOrders != 0 {
		return nil, errors.New("use-list orders are not supported")
	}
	if mask&opHasInlineRegions != 0 {
		n, iso, err := r.flagged()
		if err != nil {
			return nil, err
		}
		op.Isolated = iso
		rr := r
		if iso { // all the regions of an op isolated from above are one section of their own
			id, data, _, err := r.section()
			if err != nil {
				return nil, err
			}
			if id != secIR {
				return nil, fmt.Errorf("isolated regions in section %d", id)
			}
			rr = &reader{b: data}
		}
		for range n {
			reg, err := f.readRegion(rr)
			if err != nil {
				return nil, err
			}
			op.Regions = append(op.Regions, reg)
		}
	}
	return op, nil
}

func (f *File) readRegion(r *reader) (*Region, error) {
	nb, err := r.varint()
	if err != nil {
		return nil, err
	}
	reg := &Region{}
	if nb == 0 {
		return reg, nil
	}
	if reg.NumValues, err = r.varint(); err != nil {
		return nil, err
	}
	for range nb {
		b := &Block{}
		if err := f.readBlock(r, b, false); err != nil {
			return nil, err
		}
		reg.Blocks = append(reg.Blocks, b)
	}
	return reg, nil
}
