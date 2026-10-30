package ovir

import (
	"fmt"
	"html"
	"io"
	"strconv"
	"strings"
)

// Builder writes an OpenVINO IR (version 11): the xml graph and the bin file of constants.
type Builder struct {
	name   string
	layers []string
	edges  []string
	bin    []byte
	// binW, when set, receives the constants instead of bin: binLen bytes so far, err the first
	// write error.
	binW   io.Writer
	binLen int
	err    error
	rtInfo []string
}

// Value is the output port of a layer.
type Value struct {
	layer, port int
	Elem        string // element type, e.g. "f16", "i8", "i4", "i64"
	Shape       []int
}

// NewBuilder starts a model. sizeHint reserves the bin buffer (bytes of constants).
func NewBuilder(name string, sizeHint int) *Builder {
	return &Builder{name: name, bin: make([]byte, 0, sizeHint)}
}

// NewStreamBuilder starts a model whose constants are written to w as they are added, for models
// too large to hold in memory; Bytes then returns no bin, and Err the first write error.
func NewStreamBuilder(name string, w io.Writer) *Builder {
	return &Builder{name: name, binW: w}
}

// Err is the first error writing the constants of a NewStreamBuilder.
func (b *Builder) Err() error { return b.err }

// RTInfo adds an entry to the model's rt_info (Model::get_rt_info in OpenVINO). The key must be
// an XML name.
func (b *Builder) RTInfo(key, value string) {
	b.rtInfo = append(b.rtInfo, fmt.Sprintf(`<%s value="%s"/>`, key, html.EscapeString(value)))
}

func portXML(id int, v Value, names string) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, `<port id="%d" precision="%s"`, id, precisionOf(v.Elem))
	if names != "" {
		fmt.Fprintf(&sb, ` names="%s"`, html.EscapeString(names))
	}
	sb.WriteString(">")
	for _, d := range v.Shape {
		fmt.Fprintf(&sb, "<dim>%d</dim>", d)
	}
	sb.WriteString("</port>")
	return sb.String()
}

func precisionOf(elem string) string {
	if p, ok := elementTypes[elem]; ok {
		return p
	}
	return strings.ToUpper(elem)
}

func shapeAttr(shape []int) string {
	s := make([]string, len(shape))
	for i, d := range shape {
		s[i] = strconv.Itoa(d)
	}
	return strings.Join(s, ",")
}

// layer appends a layer and returns its outputs. attrs are name/value pairs of the data element.
func (b *Builder) layer(typ, version, name string, attrs []string, inputs []Value, outs []Value, outNames []string) []Value {
	id := len(b.layers)
	if name == "" {
		name = fmt.Sprintf("%s_%d", strings.ToLower(typ), id)
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, `<layer id="%d" name="%s" type="%s" version="%s">`, id, html.EscapeString(name), typ, version)
	if len(attrs) > 0 {
		sb.WriteString("<data")
		for i := 0; i+1 < len(attrs); i += 2 {
			fmt.Fprintf(&sb, ` %s="%s"`, attrs[i], html.EscapeString(attrs[i+1]))
		}
		sb.WriteString("/>")
	}
	if len(inputs) > 0 {
		sb.WriteString("<input>")
		for i, in := range inputs {
			sb.WriteString(portXML(i, in, ""))
			b.edges = append(b.edges, fmt.Sprintf(`<edge from-layer="%d" from-port="%d" to-layer="%d" to-port="%d"/>`, in.layer, in.port, id, i))
		}
		sb.WriteString("</input>")
	}
	res := make([]Value, len(outs))
	if len(outs) > 0 {
		sb.WriteString("<output>")
		for i, o := range outs {
			o.layer, o.port = id, len(inputs)+i
			n := ""
			if i < len(outNames) {
				n = outNames[i]
			}
			sb.WriteString(portXML(o.port, o, n))
			res[i] = o
		}
		sb.WriteString("</output>")
	}
	sb.WriteString("</layer>")
	b.layers = append(b.layers, sb.String())
	return res
}

// Parameter adds a model input.
func (b *Builder) Parameter(name, elem string, shape ...int) Value {
	v := Value{Elem: elem, Shape: shape}
	return b.layer("Parameter", "opset1", name, []string{"shape", shapeAttr(shape), "element_type", elem}, nil, []Value{v}, []string{name})[0]
}

// Const adds a constant whose bytes are appended to the bin file (i4 and u4: two per byte, low
// nibble first).
func (b *Builder) Const(elem string, shape []int, data []byte) Value {
	var off int
	if b.binW != nil {
		if pad := (64 - b.binLen%64) % 64; pad > 0 {
			b.write(make([]byte, pad))
		}
		off = b.binLen
		b.write(data)
	} else {
		for len(b.bin)%64 != 0 {
			b.bin = append(b.bin, 0)
		}
		off = len(b.bin)
		b.bin = append(b.bin, data...)
	}
	v := Value{Elem: elem, Shape: shape}
	attrs := []string{"element_type", elem, "shape", shapeAttr(shape), "offset", strconv.Itoa(off), "size", strconv.Itoa(len(data))}
	return b.layer("Const", "opset1", "", attrs, nil, []Value{v}, nil)[0]
}

func (b *Builder) write(p []byte) {
	if b.err != nil {
		return
	}
	n, err := b.binW.Write(p)
	b.binLen += n
	b.err = err
}

// Op adds an operation with one output of the given element type and shape.
func (b *Builder) Op(typ, version string, attrs []string, elem string, shape []int, inputs ...Value) Value {
	return b.layer(typ, version, "", attrs, inputs, []Value{{Elem: elem, Shape: shape}}, nil)[0]
}

// NamedOp is Op with the layer's name given ("" for an automatic one). The name of the layer
// that produces a Result is the name the NPU driver reports for that output.
func (b *Builder) NamedOp(typ, version, name string, attrs []string, elem string, shape []int, inputs ...Value) Value {
	return b.layer(typ, version, name, attrs, inputs, []Value{{Elem: elem, Shape: shape}}, nil)[0]
}

// Result marks v as a model output with the given tensor name.
func (b *Builder) Result(name string, v Value) {
	// The producing layer keeps its name; the tensor name is what the runtime reports.
	b.layers[v.layer] = strings.Replace(b.layers[v.layer], fmt.Sprintf(`<port id="%d" precision="%s">`, v.port, precisionOf(v.Elem)),
		fmt.Sprintf(`<port id="%d" precision="%s" names="%s">`, v.port, precisionOf(v.Elem), html.EscapeString(name)), 1)
	b.layer("Result", "opset1", name+"/sink_port_0", nil, []Value{v}, nil, nil)
}

// Bytes returns the xml and the bin file (nil for a NewStreamBuilder, which has written it).
func (b *Builder) Bytes() (xmlData, bin []byte) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "<?xml version=\"1.0\"?>\n<net name=\"%s\" version=\"11\">\n<layers>\n", html.EscapeString(b.name))
	for _, l := range b.layers {
		sb.WriteString(l)
		sb.WriteString("\n")
	}
	sb.WriteString("</layers>\n<edges>\n")
	for _, e := range b.edges {
		sb.WriteString(e)
		sb.WriteString("\n")
	}
	sb.WriteString("</edges>\n")
	if len(b.rtInfo) == 0 {
		sb.WriteString("<rt_info/>\n")
	} else {
		sb.WriteString("<rt_info>\n")
		for _, r := range b.rtInfo {
			sb.WriteString(r)
			sb.WriteString("\n")
		}
		sb.WriteString("</rt_info>\n")
	}
	sb.WriteString("</net>\n")
	return []byte(sb.String()), b.bin
}
