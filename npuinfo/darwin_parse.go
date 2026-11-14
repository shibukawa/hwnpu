package npuinfo

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/xml"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// The decoders here have no build tag so that their tests run on every OS.

// parsePlist decodes an XML property list (ioreg -a, powermetrics -f plist) into Go values:
// map[string]any, []any, string, int64 (uint64 when it does not fit), float64, bool, []byte,
// time.Time.
func parsePlist(r io.Reader) (any, error) {
	d := xml.NewDecoder(r)
	d.Strict = false
	for {
		tok, err := d.Token()
		if err != nil {
			return nil, err
		}
		if se, ok := tok.(xml.StartElement); ok {
			if se.Name.Local == "plist" {
				continue
			}
			return plistValue(d, se)
		}
	}
}

func plistValue(d *xml.Decoder, se xml.StartElement) (any, error) {
	switch se.Name.Local {
	case "dict":
		m := map[string]any{}
		var key string
		haveKey := false
		for {
			tok, err := d.Token()
			if err != nil {
				return nil, err
			}
			switch t := tok.(type) {
			case xml.StartElement:
				if t.Name.Local == "key" {
					s, err := plistText(d)
					if err != nil {
						return nil, err
					}
					key, haveKey = s, true
					continue
				}
				v, err := plistValue(d, t)
				if err != nil {
					return nil, err
				}
				if haveKey {
					m[key] = v
					haveKey = false
				}
			case xml.EndElement:
				return m, nil
			}
		}
	case "array":
		a := []any{}
		for {
			tok, err := d.Token()
			if err != nil {
				return nil, err
			}
			switch t := tok.(type) {
			case xml.StartElement:
				v, err := plistValue(d, t)
				if err != nil {
					return nil, err
				}
				a = append(a, v)
			case xml.EndElement:
				return a, nil
			}
		}
	case "true", "false":
		if err := d.Skip(); err != nil {
			return nil, err
		}
		return se.Name.Local == "true", nil
	}
	s, err := plistText(d)
	if err != nil {
		return nil, err
	}
	switch se.Name.Local {
	case "string":
		return s, nil
	case "integer":
		s = strings.TrimSpace(s)
		if v, err := strconv.ParseInt(s, 0, 64); err == nil {
			return v, nil
		}
		if v, err := strconv.ParseUint(s, 0, 64); err == nil {
			return v, nil
		}
		return s, nil
	case "real":
		// NaN and infinities stay strings: encoding/json cannot write them.
		if v, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil && !math.IsNaN(v) && !math.IsInf(v, 0) {
			return v, nil
		}
		return s, nil
	case "data":
		clean := strings.Map(func(r rune) rune {
			if r == ' ' || r == '\n' || r == '\t' || r == '\r' {
				return -1
			}
			return r
		}, s)
		if b, err := base64.StdEncoding.DecodeString(clean); err == nil {
			return b, nil
		}
		return s, nil
	case "date":
		if t, err := time.Parse(time.RFC3339, strings.TrimSpace(s)); err == nil {
			return t, nil
		}
		return s, nil
	}
	return s, nil
}

// plistText reads the character data up to the end of the current element.
func plistText(d *xml.Decoder) (string, error) {
	var sb strings.Builder
	for {
		tok, err := d.Token()
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.CharData:
			sb.Write(t)
		case xml.StartElement:
			if err := d.Skip(); err != nil {
				return "", err
			}
		case xml.EndElement:
			return sb.String(), nil
		}
	}
}

var ioregClass = regexp.MustCompile(`<class ([^,>]+)`)

// aneClasses finds, in ioreg's text output, the classes whose name contains "ANE".
func aneClasses(text []byte) []string {
	seen := map[string]bool{}
	var out []string
	sc := bufio.NewScanner(bytes.NewReader(text))
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		m := ioregClass.FindSubmatch(sc.Bytes())
		if m == nil {
			continue
		}
		c := string(m[1])
		if strings.Contains(c, "ANE") && !seen[c] {
			seen[c] = true
			out = append(out, c)
		}
	}
	return out
}

// ioregNodes turns ioreg -a output (an array of objects, each with IORegistryEntryChildren) into
// DarwinIONodes.
func ioregNodes(v any) []DarwinIONode {
	var out []DarwinIONode
	switch t := v.(type) {
	case []any:
		for _, e := range t {
			out = append(out, ioregNodes(e)...)
		}
	case map[string]any:
		out = append(out, ioregNode(t))
	}
	return out
}

func ioregNode(m map[string]any) DarwinIONode {
	n := DarwinIONode{Properties: map[string]any{}}
	for k, v := range m {
		switch k {
		case "IOObjectClass":
			n.Class, _ = v.(string)
		case "IORegistryEntryName":
			n.Name, _ = v.(string)
		case "IORegistryEntryChildren":
			n.Children = ioregNodes(v)
		default:
			n.Properties[k] = v
		}
	}
	if len(n.Properties) == 0 {
		n.Properties = nil
	}
	return n
}

var creatorRe = regexp.MustCompile(`^pid (\d+), (.*)$`)

// aneClients finds the user clients (IOUserClientCreator) in the trees.
func aneClients(nodes []DarwinIONode) []DarwinANEClient {
	var out []DarwinANEClient
	var walk func(n DarwinIONode)
	walk = func(n DarwinIONode) {
		if s, ok := n.Properties["IOUserClientCreator"].(string); ok {
			c := DarwinANEClient{Class: n.Class, Creator: s}
			if m := creatorRe.FindStringSubmatch(s); m != nil {
				c.PID, _ = strconv.Atoi(m[1])
				c.Process = m[2]
			}
			out = append(out, c)
		}
		for _, ch := range n.Children {
			walk(ch)
		}
	}
	for _, n := range nodes {
		walk(n)
	}
	return out
}

// nodeClasses adds every class in the trees to seen.
func nodeClasses(nodes []DarwinIONode, seen map[string]bool) {
	for _, n := range nodes {
		seen[n.Class] = true
		nodeClasses(n.Children, seen)
	}
}
