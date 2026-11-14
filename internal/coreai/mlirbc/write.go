package mlirbc

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

// Writer builds a bytecode file. Every attribute and type is written in its textual form (no
// dialect encoding: the reader parses it with the dialect's own parser). An operation's
// properties are the dialect's native encoding (ODS readProperties): each property in the op's
// order, a required one as its attribute's index, an optional one as (index << 1 | 1), or 0 when
// absent. The order is the op's (Prop lists it), learned from what coreai-core writes.
type Writer struct {
	Producer string
	// Trace, when set, is called with each operation's location id (its op_id), name and types.
	Trace func(id int, name string, operands, results []int)

	strings  []string
	strIndex map[string]int

	dialects     []string
	dialectIndex map[string]int
	versions     map[string][]byte

	opNames  []opName
	opIndex  map[string]int
	attrs    []entry
	attrIdx  map[string]int
	types    []entry
	typeIdx  map[string]int
	res      []resource
	props    [][]byte
	locs     int // the locations made
	resByKey map[string]bool

	// spill, when set (Spill), holds the blobs' bytes from Blob to WriteTo: a weight is held in
	// memory only while it is added, not until the file is written
	spill    Spill
	spillEnd int64
	keep     int64 // the blobs' bytes held in memory before they go to spill
	kept     int64
}

// Spill is a file the blobs' bytes are written to until the bytecode is (an *os.File).
type Spill interface {
	io.Writer
	io.ReaderAt
}

// Spill makes Blob hold the blobs' bytes in memory up to keep bytes in all, and write those past
// it to f (empty), which WriteTo copies them from: a small program does not touch the disk, a
// large one holds keep bytes at most.
func (w *Writer) Spill(f Spill, keep int64) { w.spill, w.spillEnd, w.keep = f, 0, keep }

type opName struct {
	dialect int
	name    int // string index of the name after the dialect
}

type entry struct {
	dialect int
	text    string
	custom  bool // text is the dialect's own encoding (RawAttr, RawType)
}

type resource struct {
	dialect int
	key     int
	align   int
	data    []byte // nil when spilled: size bytes at off in the spill file
	off     int64
	size    int64
}

// NewWriter starts a file; the dialects' version sections (raw bytes, as the dialect's
// readVersion takes them) are versions.
func NewWriter(producer string, versions map[string][]byte) *Writer {
	w := &Writer{Producer: producer, strIndex: map[string]int{}, dialectIndex: map[string]int{}, versions: versions,
		opIndex: map[string]int{}, attrIdx: map[string]int{}, typeIdx: map[string]int{}, resByKey: map[string]bool{}}
	return w
}

// Loc is a new location attribute: Core AI's own (aicode.debuginfo.location_v1, as coreai-core
// writes them), its op_id the next number; Core AI does not take builtin locations.
func (w *Writer) Loc() int {
	id := w.locs
	w.locs++
	return w.Attr("aicode", fmt.Sprintf(`#aicode.debuginfo.location_v1<src = <file = <filename = "-", directory = "", sha256Sum = "">, startLine = 0, startColumn = 0, endLine = 0, endColumn = 0>, scope = #aicode.unit_v1, metadata = <"op_id": <value = #aicode.serialization.dictionary<{<"type" = [#aicode.serialization.string<"coreai">]>, <"value" = [#aicode.serialization.integer<(64bit unsigned) %d>]>}>>>>`, id))
}

func (w *Writer) str(s string) int {
	if i, ok := w.strIndex[s]; ok {
		return i
	}
	w.strings = append(w.strings, s)
	w.strIndex[s] = len(w.strings) - 1
	return len(w.strings) - 1
}

func (w *Writer) dialect(d string) int {
	if i, ok := w.dialectIndex[d]; ok {
		return i
	}
	w.str(d)
	w.dialects = append(w.dialects, d)
	w.dialectIndex[d] = len(w.dialects) - 1
	return len(w.dialects) - 1
}

// Attr is the index of the attribute of dialect d with the textual form text.
func (w *Writer) Attr(d, text string) int {
	if i, ok := w.attrIdx[text]; ok {
		return i
	}
	w.attrs = append(w.attrs, entry{dialect: w.dialect(d), text: text})
	w.attrIdx[text] = len(w.attrs) - 1
	return len(w.attrs) - 1
}

// RawAttr appends an attribute as the bytes e holds (its dialect's encoding or its text),
// without merging it with an equal one: for copying a file's tables (tools/coreai/bcrecode).
func (w *Writer) RawAttr(e Entry) int {
	w.attrs = append(w.attrs, entry{w.dialect(e.Dialect), string(e.Data), e.Custom})
	return len(w.attrs) - 1
}

// RawType is RawAttr for a type.
func (w *Writer) RawType(e Entry) int {
	w.types = append(w.types, entry{w.dialect(e.Dialect), string(e.Data), e.Custom})
	return len(w.types) - 1
}

// RawStrings interns strings in their order first (custom encodings refer to their indices).
func (w *Writer) RawStrings(ss []string) {
	for _, s := range ss {
		w.str(s)
	}
}

// RawProps adds properties as their encoded bytes; OpRaw uses the index.
func (w *Writer) RawProps(b []byte) int {
	w.props = append(w.props, b)
	return len(w.props) - 1
}

// OpRaw is Op with the location attribute and the properties index given (-1 none).
func (b *BlockW) OpRaw(name string, loc, props int, operands []Value, results []int, regions ...*BlockW) []Value {
	o := opW{loc: loc, name: b.w.op(name), props: props, results: results, regions: regions}
	for _, v := range operands {
		o.operands = append(o.operands, v.id)
	}
	out := make([]Value, len(results))
	for i, t := range results {
		out[i] = Value{b.values, t}
		b.values++
	}
	b.ops = append(b.ops, o)
	return out
}

// ValueAt is the block's value of index id (for copying operands by their number).
func (b *BlockW) ValueAt(id int) Value { return Value{id: id} }

// TypeText is the textual form of type index i.
func (w *Writer) TypeText(i int) string { return w.types[i].text }

// Type is the index of the type of dialect d with the textual form text.
func (w *Writer) Type(d, text string) int {
	if i, ok := w.typeIdx[text]; ok {
		return i
	}
	w.types = append(w.types, entry{dialect: w.dialect(d), text: text})
	w.typeIdx[text] = len(w.types) - 1
	return len(w.types) - 1
}

// Blob adds a resource blob of dialect d named key (its dense_resource handle).
func (w *Writer) Blob(d, key string, align int, data []byte) {
	if w.resByKey[d+"\x00"+key] {
		panic("mlirbc: resource " + key + " twice")
	}
	w.resByKey[d+"\x00"+key] = true
	r := resource{dialect: w.dialect(d), key: w.str(key), align: align, data: data, size: int64(len(data))}
	if w.spill != nil && w.kept+r.size <= w.keep {
		w.kept += r.size
	} else if w.spill != nil {
		if _, err := w.spill.Write(data); err != nil {
			panic(spillError{err})
		}
		r.data, r.off = nil, w.spillEnd
		w.spillEnd += r.size
	}
	w.res = append(w.res, r)
}

// spillError is the panic of Blob when its write to the spill file fails (SpillError).
type spillError struct{ err error }

// SpillError is the error of a panic of Blob's failed write to the spill file, nil for any other
// value: what recovers the panics of a lowering returns it.
func SpillError(r any) error {
	if e, ok := r.(spillError); ok {
		return fmt.Errorf("mlirbc: writing a blob: %w", e.err)
	}
	return nil
}

func (w *Writer) op(full string) int {
	if i, ok := w.opIndex[full]; ok {
		return i
	}
	d, name := splitOp(full)
	w.opNames = append(w.opNames, opName{w.dialect(d), w.str(name)})
	w.opIndex[full] = len(w.opNames) - 1
	return len(w.opNames) - 1
}

func splitOp(full string) (string, string) {
	for i := 0; i < len(full); i++ {
		if full[i] == '.' {
			return full[:i], full[i+1:]
		}
	}
	panic("mlirbc: op name without a dialect: " + full)
}

// Value is a value of a region: its index in the region's numbering and its type.
type Value struct {
	id  int
	typ int
}

// BlockW is the one block of a region under construction (regions here have one block and are
// isolated from above: their values are numbered from 0).
type BlockW struct {
	w       *Writer
	args    []int
	argLocs []int
	ops     []opW
	values  int
}

// Prop is one property of an operation: the attribute index (-1 for an absent optional one).
type Prop struct {
	Attr     int
	Optional bool
}

type opW struct {
	loc      int
	name     int
	props    int // index into the properties section, -1 none
	results  []int
	operands []int
	regions  []*BlockW
}

// NewBlock is an empty block with arguments of the given types.
func (w *Writer) NewBlock(argTypes ...int) (*BlockW, []Value) {
	b := &BlockW{w: w, args: argTypes}
	vs := make([]Value, len(argTypes))
	for i, t := range argTypes {
		vs[i] = Value{b.values, t}
		b.values++
		b.argLocs = append(b.argLocs, w.Loc()) // Core AI needs every block argument's location
	}
	return b, vs
}

// SetArgLocs replaces the arguments' locations (attribute indices; for copying a file).
func (b *BlockW) SetArgLocs(locs []int) { b.argLocs = locs }

// Op appends the operation name with its properties (none: nil), operands, result types and
// regions (each one isolated block), and returns its results.
func (b *BlockW) Op(name string, props []Prop, operands []Value, results []int, regions ...*BlockW) []Value {
	id := b.w.locs
	o := opW{loc: b.w.Loc(), name: b.w.op(name), props: -1, results: results, regions: regions}
	if b.w.Trace != nil {
		var ts []int
		for _, v := range operands {
			ts = append(ts, v.typ)
		}
		b.w.Trace(id, name, ts, results)
	}
	if props != nil {
		e := &enc{}
		for _, p := range props {
			switch {
			case !p.Optional:
				e.varint(uint64(p.Attr))
			case p.Attr < 0:
				e.varint(0)
			default:
				e.flagged(uint64(p.Attr), true)
			}
		}
		b.w.props = append(b.w.props, e.b)
		o.props = len(b.w.props) - 1
	}
	for _, v := range operands {
		o.operands = append(o.operands, v.id)
	}
	out := make([]Value, len(results))
	for i, t := range results {
		out[i] = Value{b.values, t}
		b.values++
	}
	b.ops = append(b.ops, o)
	return out
}

// Type is a value's type index.
func (v Value) Type() int { return v.typ }

type enc struct{ b []byte }

func (e *enc) byte(c byte) { e.b = append(e.b, c) }

func (e *enc) varint(v uint64) {
	switch {
	case v < 1<<7:
		e.b = append(e.b, byte(v<<1|1))
	case v >= 1<<56:
		e.b = append(e.b, 0)
		e.b = binary.LittleEndian.AppendUint64(e.b, v)
	default:
		n := 1 // extra bytes
		for v >= 1<<(7*(n+1)) {
			n++
		}
		x := (v<<1 | 1) << n
		for i := 0; i <= n; i++ {
			e.b = append(e.b, byte(x>>(8*i)))
		}
	}
}

func (e *enc) flagged(v uint64, f bool) {
	x := v << 1
	if f {
		x |= 1
	}
	e.varint(x)
}

// section appends section id with data; align > 1 pads its data to that alignment of the
// absolute position (base is the position of e's first byte in the file).
func (e *enc) section(id byte, data []byte, align int, base int) {
	if align > 1 {
		e.byte(id | 0x80)
	} else {
		e.byte(id)
	}
	e.varint(uint64(len(data)))
	if align > 1 {
		e.varint(uint64(align))
		for (base+len(e.b))%align != 0 {
			e.byte(0xcb)
		}
	}
	e.b = append(e.b, data...)
}

func (w *Writer) block(e *enc, b *BlockW) {
	e.flagged(uint64(len(b.ops)), len(b.args) > 0)
	if len(b.args) > 0 {
		e.varint(uint64(len(b.args)))
		for i, t := range b.args {
			e.flagged(uint64(t), true)
			e.varint(uint64(b.argLocs[i]))
		}
		e.byte(0) // no use-list orders
	}
	for _, o := range b.ops {
		e.varint(uint64(o.name))
		var mask byte
		if o.props >= 0 {
			mask |= opHasProperties
		}
		if len(o.results) > 0 {
			mask |= opHasResults
		}
		if len(o.operands) > 0 {
			mask |= opHasOperands
		}
		if len(o.regions) > 0 {
			mask |= opHasInlineRegions
		}
		e.byte(mask)
		e.varint(uint64(o.loc))
		if o.props >= 0 {
			e.varint(uint64(o.props))
		}
		if len(o.results) > 0 {
			e.varint(uint64(len(o.results)))
			for _, t := range o.results {
				e.varint(uint64(t))
			}
		}
		if len(o.operands) > 0 {
			e.varint(uint64(len(o.operands)))
			for _, v := range o.operands {
				e.varint(uint64(v))
			}
		}
		if len(o.regions) > 0 {
			e.flagged(uint64(len(o.regions)), true)
			sub := &enc{}
			for _, r := range o.regions {
				sub.varint(1) // one block
				sub.varint(uint64(r.values))
				w.block(sub, r)
			}
			e.section(secIR, sub.b, 1, 0)
		}
	}
}

// Bytes encodes the file with top as the top level block (the module's body).
func (w *Writer) Bytes(top *BlockW) []byte {
	var b bytes.Buffer
	if _, err := w.WriteTo(&b, top); err != nil {
		panic(err) // only the spill file's reads fail
	}
	return b.Bytes()
}

// WriteTo writes the file Bytes encodes to dst, the blobs copied from the spill file (Spill) in
// pieces, so that the file is never whole in memory. It returns the bytes written.
func (w *Writer) WriteTo(dst io.Writer, top *BlockW) (int64, error) {
	// the ops of every block first: they intern their names
	ir := &enc{}
	w.block(ir, top)

	out := &enc{b: append([]byte{}, magic...)}
	out.varint(Version)
	out.b = append(out.b, w.Producer...)
	out.byte(0)

	// attributes and types, grouped by runs of one dialect
	at, off := &enc{}, &enc{}
	off.varint(uint64(len(w.attrs)))
	off.varint(uint64(len(w.types)))
	group := func(es []entry) {
		for i := 0; i < len(es); {
			j := i
			for j < len(es) && es[j].dialect == es[i].dialect {
				j++
			}
			off.varint(uint64(es[i].dialect))
			off.varint(uint64(j - i))
			for _, e := range es[i:j] {
				at.b = append(at.b, e.text...)
				if e.custom {
					off.flagged(uint64(len(e.text)), true)
					continue
				}
				at.byte(0)
				off.flagged(uint64(len(e.text)+1), false)
			}
			i = j
		}
	}
	group(w.attrs)
	group(w.types)

	// dialects and op names (the dialect names are interned already)
	dl := &enc{}
	dl.varint(uint64(len(w.dialects)))
	for _, d := range w.dialects {
		v, ok := w.versions[d]
		dl.flagged(uint64(w.strIndex[d]), ok)
		if ok {
			dl.section(secDialectVersions, v, 1, 0)
		}
	}
	dl.varint(uint64(len(w.opNames)))
	for i := 0; i < len(w.opNames); {
		j := i
		for j < len(w.opNames) && w.opNames[j].dialect == w.opNames[i].dialect {
			j++
		}
		dl.varint(uint64(w.opNames[i].dialect))
		dl.varint(uint64(j - i))
		for _, o := range w.opNames[i:j] {
			dl.flagged(uint64(o.name), true) // registered
		}
		i = j
	}

	// strings, last: everything above has interned its strings
	st := &enc{}
	st.varint(uint64(len(w.strings)))
	for i := len(w.strings) - 1; i >= 0; i-- {
		st.varint(uint64(len(w.strings[i]) + 1))
	}
	for _, s := range w.strings {
		st.b = append(st.b, s...)
		st.byte(0)
	}

	out.section(secString, st.b, 1, 0)
	out.section(secDialect, dl.b, 1, 0)
	out.section(secAttrType, at.b, 1, 0)
	out.section(secAttrTypeOffset, off.b, 1, 0)
	out.section(secIR, ir.b, 1, 0)
	pr := &enc{}
	pr.varint(uint64(len(w.props)))
	for _, p := range w.props {
		pr.varint(uint64(len(p)))
		pr.b = append(pr.b, p...)
	}
	out.section(secProperties, pr.b, 1, 0)
	if len(w.res) == 0 {
		n, err := dst.Write(out.b)
		return int64(n), err
	}
	const secAlign = 16
	// the resource section: its blobs' headers and padding (relative to the section's data,
	// which starts aligned to secAlign, at least every blob's alignment) laid out first, so that
	// its length is known before its data is written
	heads := make([][]byte, len(w.res))
	ro := &enc{}
	ro.varint(0) // no external resource groups
	var rdLen int64
	for i := 0; i < len(w.res); {
		j := i
		for j < len(w.res) && w.res[j].dialect == w.res[i].dialect {
			j++
		}
		ro.varint(uint64(w.res[i].dialect))
		ro.varint(uint64(j - i))
		for k := i; k < j; k++ {
			r := w.res[k]
			if r.align > secAlign {
				panic(fmt.Sprintf("mlirbc: blob alignment %d over %d", r.align, secAlign))
			}
			h := &enc{}
			h.varint(uint64(r.align))
			h.varint(uint64(r.size))
			for (rdLen+int64(len(h.b)))%int64(r.align) != 0 {
				h.byte(0xcb)
			}
			heads[k] = h.b
			n := int64(len(h.b)) + r.size
			rdLen += n
			ro.varint(uint64(r.key))
			ro.varint(uint64(n))
			ro.byte(0) // a blob
		}
		i = j
	}
	// the section's header, as enc.section writes it
	out.byte(secResource | 0x80)
	out.varint(uint64(rdLen))
	out.varint(secAlign)
	for len(out.b)%secAlign != 0 {
		out.byte(0xcb)
	}
	cw := &countWriter{w: dst}
	cw.Write(out.b)
	for k, r := range w.res {
		cw.Write(heads[k])
		if r.data != nil {
			cw.Write(r.data)
		} else if cw.err == nil {
			if _, err := io.Copy(cw, io.NewSectionReader(w.spill, r.off, r.size)); err != nil && cw.err == nil {
				cw.err = err
			}
		}
	}
	tail := &enc{}
	tail.section(secResourceOffset, ro.b, 1, 0)
	cw.Write(tail.b)
	return cw.n, cw.err
}

// countWriter counts the bytes written and keeps the first error, after which it writes nothing.
type countWriter struct {
	w   io.Writer
	n   int64
	err error
}

func (c *countWriter) Write(p []byte) (int, error) {
	if c.err != nil {
		return 0, c.err
	}
	n, err := c.w.Write(p)
	c.n += int64(n)
	c.err = err
	return n, err
}
