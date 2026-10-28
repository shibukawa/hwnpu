package main

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// powermetrics -f plist writes one XML property list per sample, separated by NUL bytes. Only
// the numbers are needed, so each document is flattened into "processor.ane_power" style keys.
// Arrays (per cluster and per CPU details) are skipped.

// parseSamples splits the powermetrics output and flattens every sample.
func parseSamples(out []byte) ([]map[string]float64, error) {
	var samples []map[string]float64
	for _, doc := range bytes.Split(out, []byte{0}) {
		if len(bytes.TrimSpace(doc)) == 0 {
			continue
		}
		s, err := flattenPlist(doc)
		if err != nil {
			return samples, fmt.Errorf("sample %d: %w", len(samples)+1, err)
		}
		samples = append(samples, s)
	}
	return samples, nil
}

func flattenPlist(doc []byte) (map[string]float64, error) {
	d := xml.NewDecoder(bytes.NewReader(doc))
	d.Strict = false
	out := map[string]float64{}
	for {
		tok, err := d.Token()
		if err == io.EOF {
			return nil, fmt.Errorf("no <dict> in the property list")
		}
		if err != nil {
			return nil, err
		}
		if se, ok := tok.(xml.StartElement); ok && se.Name.Local == "dict" {
			return out, readDict(d, "", out)
		}
	}
}

// readDict reads up to the matching </dict>.
func readDict(d *xml.Decoder, prefix string, out map[string]float64) error {
	key := ""
	for {
		tok, err := d.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case xml.EndElement:
			if t.Name.Local == "dict" {
				return nil
			}
		case xml.StartElement:
			if t.Name.Local == "key" {
				s, err := text(d)
				if err != nil {
					return err
				}
				key = prefix + s
				continue
			}
			if err := readValue(d, t, key, out); err != nil {
				return err
			}
		}
	}
}

func readValue(d *xml.Decoder, se xml.StartElement, key string, out map[string]float64) error {
	switch se.Name.Local {
	case "dict":
		return readDict(d, key+".", out)
	case "real", "integer":
		s, err := text(d)
		if err != nil {
			return err
		}
		if v, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil && key != "" {
			out[key] = v
		}
		return nil
	case "true", "false":
		if key != "" {
			out[key] = map[bool]float64{true: 1, false: 0}[se.Name.Local == "true"]
		}
		return d.Skip()
	default: // array, string, date, data
		return d.Skip()
	}
}

func text(d *xml.Decoder) (string, error) {
	var b strings.Builder
	for {
		tok, err := d.Token()
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.CharData:
			b.Write(t)
		case xml.EndElement:
			return b.String(), nil
		}
	}
}
