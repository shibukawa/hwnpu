package npuinfo

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf16"
)

// The decoders here have no build tag so that their tests run on every OS.

var (
	instPID     = regexp.MustCompile(`(?:^|_)pid_(\d+)(?:_|$)`)
	instLUID    = regexp.MustCompile(`(?i)(?:^|_)luid_(0x[0-9a-f]+_0x[0-9a-f]+)(?:_|$)`)
	instPhys    = regexp.MustCompile(`(?:^|_)phys_(\d+)(?:_|$)`)
	instEng     = regexp.MustCompile(`(?:^|_)eng_(\d+)(?:_|$)`)
	instEngType = regexp.MustCompile(`(?:^|_)engtype_(.*)$`)
)

// parseGPUInstance fills the parts of a GPU-style performance counter instance name
// ("pid_1234_luid_0x00000000_0x0000D1A2_phys_0_eng_3_engtype_Compute").
func parseGPUInstance(c *WindowsCounter) {
	s := c.Instance
	if m := instPID.FindStringSubmatch(s); m != nil {
		c.PID, _ = strconv.Atoi(m[1])
	}
	if m := instLUID.FindStringSubmatch(s); m != nil {
		c.LUID = normLUID(m[1])
	}
	if m := instPhys.FindStringSubmatch(s); m != nil {
		v, _ := strconv.Atoi(m[1])
		c.Phys = &v
	}
	if m := instEng.FindStringSubmatch(s); m != nil {
		v, _ := strconv.Atoi(m[1])
		c.Engine = &v
	}
	if m := instEngType.FindStringSubmatch(s); m != nil {
		c.EngType = m[1]
	}
}

// normLUID writes a LUID as 0xHHHHHHHH_0xLLLLLLLL (upper-case hex), so that counters and
// adapters compare equal.
func normLUID(s string) string {
	hi, lo, ok := strings.Cut(strings.ToLower(s), "_")
	if !ok {
		return s
	}
	h, err1 := strconv.ParseUint(strings.TrimPrefix(hi, "0x"), 16, 32)
	l, err2 := strconv.ParseUint(strings.TrimPrefix(lo, "0x"), 16, 32)
	if err1 != nil || err2 != nil {
		return s
	}
	return formatLUID(uint32(h), uint32(l))
}

func formatLUID(high, low uint32) string { return fmt.Sprintf("0x%08X_0x%08X", high, low) }

// multiSZ splits a double-NUL-terminated UTF-16 string list.
func multiSZ(u []uint16) []string {
	var out []string
	start := 0
	for i, c := range u {
		if c != 0 {
			continue
		}
		if i == start {
			break
		}
		out = append(out, string(utf16.Decode(u[start:i])))
		start = i + 1
	}
	return out
}

// utf16z decodes a NUL-terminated UTF-16 string.
func utf16z(u []uint16) string {
	for i, c := range u {
		if c == 0 {
			return string(utf16.Decode(u[:i]))
		}
	}
	return string(utf16.Decode(u))
}

var adapterTypeBits = []string{"RenderSupported", "DisplaySupported", "SoftwareDevice", "PostDevice",
	"HybridDiscrete", "HybridIntegrated", "IndirectDisplayDevice", "Paravirtualized", "ACGSupported",
	"SupportSetTimingsFromVidPn", "Detachable", "ComputeOnly", "Prototype", "RuntimePowerManagement"}

func adapterTypeNames(f uint32) []string {
	var out []string
	for i, n := range adapterTypeBits {
		if f&(1<<i) != 0 {
			out = append(out, n)
		}
	}
	return out
}

var engineTypes = []string{"Other", "3D", "VideoDecode", "VideoEncode", "VideoProcessing",
	"SceneAssembly", "Copy", "Overlay", "Crypto", "VideoCodec"}

func engineTypeName(t uint32) string {
	if int(t) < len(engineTypes) {
		return engineTypes[t]
	}
	return strconv.Itoa(int(t))
}
