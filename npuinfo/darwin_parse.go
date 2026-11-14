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

// isANEChannel picks the ANE's IOReport channels out of the groups subscribed to.
func isANEChannel(group, subgroup, name string) bool {
	n := strings.TrimSpace(name)
	switch group {
	case "Interrupt Statistics (by index)":
		return subgroup == "ane 0" || strings.HasPrefix(subgroup, "ane ") || strings.HasPrefix(subgroup, "dart-ane")
	case "Energy Model", "PMP", "AMC Stats", "SoC Stats":
		return strings.Contains(n, "ANE")
	}
	return false
}

// energyJoules converts an IOReport energy value to joules; ok is false for other units.
func energyJoules(v int64, unit string) (float64, bool) {
	switch strings.TrimSpace(unit) {
	case "J":
		return float64(v), true
	case "mJ":
		return float64(v) / 1e3, true
	case "uJ", "µJ":
		return float64(v) / 1e6, true
	case "nJ":
		return float64(v) / 1e9, true
	}
	return 0, false
}

// summarize fills the ANE fields from the channels.
func (r *DarwinIOReport) summarize() {
	if r.Seconds <= 0 {
		return
	}
	var energyEM, energyPMP float64
	var haveEM bool
	for _, c := range r.Channels {
		if c.Value == nil {
			continue
		}
		v := *c.Value
		n := strings.TrimSpace(c.Name)
		switch {
		case c.Group == "Energy Model" && strings.HasPrefix(n, "ANE"):
			// Later chips have ANE (and sometimes per-core ANE0…); the total is the
			// largest, or the sum of the parts when there is no total.
			if j, ok := energyJoules(v, c.Unit); ok {
				if n == "ANE" {
					energyEM, haveEM = j, true
				} else if !haveEM {
					energyEM += j
				}
			}
		case c.Group == "PMP" && c.SubGroup == "Energy Counters" && n == "ANE":
			if j, ok := energyJoules(v, c.Unit); ok {
				energyPMP = j
			}
		case c.Group == "AMC Stats" && n == "ANE RD":
			r.ANEReadBytesPerSec = float64(v) / r.Seconds
		case c.Group == "AMC Stats" && n == "ANE WR":
			r.ANEWriteBytesPerSec = float64(v) / r.Seconds
		case c.Group == "Interrupt Statistics (by index)" && c.SubGroup == "ane 0" && n == "First Level Interrupt Handler Count":
			r.ANEInterruptsPerSec = float64(v) / r.Seconds
		}
	}
	e := energyPMP
	if energyEM > 0 {
		e = energyEM
	}
	r.ANEPowerW = e / r.Seconds
}

// aneDevice finds the node with the ANE driver's DeviceProperties.
func aneDevice(nodes []DarwinIONode) *DarwinANEDevice {
	var found *DarwinANEDevice
	var walk func(n DarwinIONode)
	walk = func(n DarwinIONode) {
		if found != nil {
			return
		}
		if dp, ok := n.Properties["DeviceProperties"].(map[string]any); ok {
			if _, ok := dp["ANEDevicePropertyNumANECores"]; ok {
				d := &DarwinANEDevice{Class: n.Class}
				d.Architecture, _ = dp["ANEDevicePropertyTypeANEArchitectureTypeStr"].(string)
				d.Cores = plistInt(dp["ANEDevicePropertyNumANECores"])
				d.Version = plistInt(dp["ANEDevicePropertyANEVersion"])
				d.MinorVersion = plistInt(dp["ANEDevicePropertyANEMinorVersion"])
				d.BoardType = plistInt(dp["ANEDevicePropertyANEHWBoardType"])
				d.BoardSubType = plistInt(dp["ANEDevicePropertyANEHWBoardSubType"])
				d.CPUSubType = plistInt(dp["ANEDevicePropertyANECPUSubType"])
				d.FirmwareLoaded, _ = n.Properties["FirmwareLoaded"].(bool)
				d.Compatible, _ = n.Properties["IONameMatched"].(string)
				if pm, ok := n.Properties["IOPowerManagement"].(map[string]any); ok {
					d.PowerState = plistInt(pm["CurrentPowerState"])
					d.MaxPowerState = plistInt(pm["MaxPowerState"])
				}
				found = d
				return
			}
		}
		for _, ch := range n.Children {
			walk(ch)
		}
	}
	for _, n := range nodes {
		walk(n)
	}
	return found
}

func plistInt(v any) int {
	switch t := v.(type) {
	case int64:
		return int(t)
	case uint64:
		return int(t)
	case float64:
		return int(t)
	case bool:
		if t {
			return 1
		}
	}
	return 0
}

// isSoCChannel picks the channels that put the ANE beside the rest of the chip: the energy of
// every power domain and the DRAM traffic of every client.
func isSoCChannel(group, subgroup, name string) bool {
	switch {
	case group == "PMP" && subgroup == "Energy Counters":
		return true
	case group == "Energy Model":
		return !strings.Contains(name, "DTL") // per-core details
	case group == "AMC Stats" && subgroup == "Perf Counters":
		return true
	}
	return false
}

// addSoC adds a SoC channel's change to SoCPowerW (as W) or DRAMBytesPerSec. Energy Model
// keys are prefixed, since its domains overlap PMP's.
func (r *DarwinIOReport) addSoC(group, name, unit string, v int64) {
	if r.Seconds <= 0 {
		return
	}
	name = strings.TrimSpace(name)
	switch group {
	case "PMP", "Energy Model":
		j, ok := energyJoules(v, unit)
		if !ok {
			return
		}
		if group == "Energy Model" {
			if v == 0 {
				return // M1 has the channels but leaves them at 0
			}
			name = "Energy Model/" + name
		}
		if r.SoCPowerW == nil {
			r.SoCPowerW = map[string]float64{}
		}
		r.SoCPowerW[name] = j / r.Seconds
	case "AMC Stats":
		if strings.TrimSpace(unit) != "B" {
			return
		}
		if r.DRAMBytesPerSec == nil {
			r.DRAMBytesPerSec = map[string]float64{}
		}
		r.DRAMBytesPerSec[name] = float64(v) / r.Seconds
	}
}

// thermalPressureNames are the levels of com.apple.system.thermalpressurelevel
// (OSThermalNotification.h).
var thermalPressureNames = map[uint64]string{0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}

// aneKextVersion reads a kext's Info.plist.
func aneKextVersion(r io.Reader) (id, version string, ok bool) {
	v, err := parsePlist(r)
	if err != nil {
		return "", "", false
	}
	m, _ := v.(map[string]any)
	id, _ = m["CFBundleIdentifier"].(string)
	version, _ = m["CFBundleVersion"].(string)
	return id, version, id != ""
}
