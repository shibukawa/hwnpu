// Package ovir reads the parts of an OpenVINO IR (xml + bin) that the Intel NPU driver compiler
// needs, and builds the ZE_GRAPH_FORMAT_NGRAPH_LITE input and build flags the same way as the
// OpenVINO NPU plugin (src/plugins/intel_npu/src/compiler_adapter/src/model_serializer.cpp).
package ovir

import (
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Port is a model input (Parameter) or output (Result).
type Port struct {
	Index     int
	Name      string // friendly name used by the compiler when indices are not used
	Precision string // legacy precision string, e.g. "FP16"
	Dims      []int
}

// Layer is one operation of the IR.
type Layer struct {
	ID   string
	Name string
	Type string
}

// Model is the parsed IR summary.
type Model struct {
	Version int
	Inputs  []Port
	Outputs []Port
	Layers  []Layer
}

// IsStructural reports whether a layer type is bookkeeping rather than computation.
func IsStructural(typ string) bool {
	switch typ {
	case "Parameter", "Result", "Const":
		return true
	}
	return false
}

type xmlNet struct {
	Version string     `xml:"version,attr"`
	Layers  []xmlLayer `xml:"layers>layer"`
	Edges   []xmlEdge  `xml:"edges>edge"`
}

type xmlLayer struct {
	ID     string    `xml:"id,attr"`
	Name   string    `xml:"name,attr"`
	Type   string    `xml:"type,attr"`
	Data   xmlData   `xml:"data"`
	Input  []xmlPort `xml:"input>port"`
	Output []xmlPort `xml:"output>port"`
}

type xmlData struct {
	ElementType string `xml:"element_type,attr"`
	Shape       string `xml:"shape,attr"`
}

type xmlPort struct {
	ID        string   `xml:"id,attr"`
	Precision string   `xml:"precision,attr"`
	Dims      []string `xml:"dim"`
}

type xmlEdge struct {
	FromLayer string `xml:"from-layer,attr"`
	ToLayer   string `xml:"to-layer,attr"`
}

// Parse reads an IR xml.
func Parse(src []byte) (*Model, error) {
	var n xmlNet
	if err := xml.Unmarshal(src, &n); err != nil {
		return nil, fmt.Errorf("ovir: %w", err)
	}
	if len(n.Layers) == 0 {
		return nil, errors.New("ovir: no layers")
	}
	m := &Model{}
	m.Version, _ = strconv.Atoi(n.Version)
	names := map[string]string{}
	for _, l := range n.Layers {
		names[l.ID] = l.Name
		m.Layers = append(m.Layers, Layer{ID: l.ID, Name: l.Name, Type: l.Type})
	}
	producer := map[string]string{}
	for _, e := range n.Edges {
		producer[e.ToLayer] = e.FromLayer
	}
	for _, l := range n.Layers {
		switch l.Type {
		case "Parameter":
			p := Port{Index: len(m.Inputs), Name: l.Name}
			prec, err := precisionFromElementType(l.Data.ElementType)
			if err != nil {
				return nil, fmt.Errorf("ovir: input %q: %w", l.Name, err)
			}
			p.Precision = prec
			if len(l.Output) > 0 {
				p.Dims, err = dims(l.Output[0].Dims)
			} else {
				p.Dims, err = shapeDims(l.Data.Shape)
			}
			if err != nil {
				return nil, fmt.Errorf("ovir: input %q: %w", l.Name, err)
			}
			m.Inputs = append(m.Inputs, p)
		case "Result":
			if len(l.Input) == 0 {
				return nil, fmt.Errorf("ovir: result %q has no input port", l.Name)
			}
			p := Port{Index: len(m.Outputs), Name: names[producer[l.ID]], Precision: normalizePrecision(l.Input[0].Precision)}
			var err error
			if p.Dims, err = dims(l.Input[0].Dims); err != nil {
				return nil, fmt.Errorf("ovir: result %q: %w", l.Name, err)
			}
			m.Outputs = append(m.Outputs, p)
		}
	}
	if len(m.Inputs) == 0 || len(m.Outputs) == 0 {
		return nil, errors.New("ovir: the model needs at least one Parameter and one Result")
	}
	return m, nil
}

func dims(ss []string) ([]int, error) {
	out := make([]int, len(ss))
	for i, s := range ss {
		v, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || v < 0 {
			return nil, fmt.Errorf("dynamic or invalid dimension %q; the NPU needs static shapes", s)
		}
		out[i] = v
	}
	return out, nil
}

func shapeDims(shape string) ([]int, error) {
	if strings.TrimSpace(shape) == "" {
		return nil, nil
	}
	return dims(strings.Split(shape, ","))
}

var elementTypes = map[string]string{
	"f16": "FP16", "f32": "FP32", "f64": "FP64", "bf16": "BF16",
	"i4": "I4", "i8": "I8", "i16": "I16", "i32": "I32", "i64": "I64",
	"u4": "U4", "u8": "U8", "u16": "U16", "u32": "U32", "u64": "U64",
	"u1": "BIN", "boolean": "BOOL",
}

func precisionFromElementType(et string) (string, error) {
	if p, ok := elementTypes[strings.ToLower(et)]; ok {
		return p, nil
	}
	return "", fmt.Errorf("unsupported element_type %q", et)
}

// normalizePrecision maps port precisions ("FP32", "f32", "I64", "BOOL") to legacy strings.
func normalizePrecision(p string) string {
	if v, ok := elementTypes[strings.ToLower(p)]; ok {
		return v
	}
	return strings.ToUpper(p)
}

// legacyLayout is rankToLegacyLayoutString of the OpenVINO NPU plugin.
func legacyLayout(rank int) string {
	switch rank {
	case 0:
		return "**SCALAR**"
	case 1:
		return "C"
	case 2:
		return "NC"
	case 3:
		return "CHW"
	case 4:
		return "NCHW"
	case 5:
		return "NCDHW"
	}
	return "BLOCKED"
}

// UseIndices reports whether a compiler identifies I/O by index (compiler 5.9 and later).
func UseIndices(major, minor uint16) bool { return major > 5 || (major == 5 && minor >= 9) }

// IOFlags returns the --inputs/--outputs precision and layout flags (serializeIOInfo).
func (m *Model) IOFlags(useIndices bool) string {
	part := func(key string, ports []Port, value func(Port) string) string {
		var b strings.Builder
		b.WriteString(key + `="`)
		for i, p := range ports {
			if i > 0 {
				b.WriteByte(' ')
			}
			if useIndices {
				b.WriteString(strconv.Itoa(p.Index))
			} else {
				b.WriteString(p.Name)
			}
			b.WriteString(":" + value(p))
		}
		b.WriteByte('"')
		return b.String()
	}
	prec := func(p Port) string { return p.Precision }
	layout := func(p Port) string { return legacyLayout(len(p.Dims)) }
	return strings.Join([]string{
		part("--inputs_precisions", m.Inputs, prec),
		part("--inputs_layouts", m.Inputs, layout),
		part("--outputs_precisions", m.Outputs, prec),
		part("--outputs_layouts", m.Outputs, layout),
	}, " ")
}

// BuildFlags combines the I/O flags with a --config section. config entries are KEY=VALUE pairs
// rendered as KEY="VALUE"; an empty map adds no --config.
func (m *Model) BuildFlags(useIndices bool, config map[string]string, keys []string) string {
	flags := m.IOFlags(useIndices)
	if len(keys) == 0 {
		return flags
	}
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%q", k, config[k]))
	}
	return flags + " --config " + strings.Join(parts, " ")
}

// Serialize builds the ALL_WEIGHTS_COPY layout:
// compiler version (u16 major, u16 minor) | u32 2 | u64 len(xml) | xml | u64 len(bin) | bin.
func Serialize(major, minor uint16, xmlData, bin []byte) []byte {
	out := make([]byte, 0, 4+4+8+len(xmlData)+8+len(bin))
	out = binary.LittleEndian.AppendUint16(out, major)
	out = binary.LittleEndian.AppendUint16(out, minor)
	out = binary.LittleEndian.AppendUint32(out, 2)
	out = binary.LittleEndian.AppendUint64(out, uint64(len(xmlData)))
	out = append(out, xmlData...)
	out = binary.LittleEndian.AppendUint64(out, uint64(len(bin)))
	return append(out, bin...)
}

// RequiredRTInfo are the model level runtime attributes the OpenVINO NPU plugin always sets
// before handing a model to the driver compiler.
var RequiredRTInfo = []string{"is_new_api", "use_indices_for_io_metadata"}

// InjectRTInfo adds `<key value="true"/>` for each RequiredRTInfo key to the net level
// <rt_info>, creating it when absent. Layer level rt_info elements are left alone.
func InjectRTInfo(src []byte) ([]byte, error) {
	dec := xml.NewDecoder(bytes.NewReader(src))
	depth := 0
	rtStart, rtEnd, netEnd := -1, -1, -1
	selfClosing := false
	existing := map[string]bool{}
	inNetRT := false
	for {
		before := int(dec.InputOffset())
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("ovir: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			depth++
			if depth == 2 && t.Name.Local == "rt_info" {
				rtStart = int(dec.InputOffset())
				selfClosing = bytes.HasSuffix(bytes.TrimSpace(src[before:rtStart]), []byte("/>"))
				if selfClosing {
					rtStart = before
					rtEnd = rtStart
				}
				inNetRT = true
			} else if inNetRT && depth == 3 {
				existing[t.Name.Local] = true
			}
		case xml.EndElement:
			if depth == 2 && t.Name.Local == "rt_info" {
				if !selfClosing {
					rtEnd = before
				}
				inNetRT = false
			}
			if depth == 1 && t.Name.Local == "net" {
				netEnd = before
			}
			depth--
		}
	}
	var add strings.Builder
	for _, k := range RequiredRTInfo {
		if !existing[k] {
			fmt.Fprintf(&add, "\t\t<%s value=\"true\" />\n", k)
		}
	}
	if add.Len() == 0 {
		return src, nil
	}
	var out bytes.Buffer
	switch {
	case rtStart >= 0 && selfClosing:
		end := rtStart + bytes.Index(src[rtStart:], []byte("/>")) + 2
		out.Write(src[:rtStart])
		out.WriteString("<rt_info>\n" + add.String() + "\t</rt_info>")
		out.Write(src[end:])
	case rtStart >= 0:
		out.Write(src[:rtEnd])
		out.WriteString(add.String())
		out.Write(src[rtEnd:])
	case netEnd >= 0:
		out.Write(src[:netEnd])
		out.WriteString("\t<rt_info>\n" + add.String() + "\t</rt_info>\n")
		out.Write(src[netEnd:])
	default:
		return nil, errors.New("ovir: no <net> element")
	}
	return out.Bytes(), nil
}
