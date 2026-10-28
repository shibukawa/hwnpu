// Package xclbin reads the parts of an AXLF (xclbin) container that the amdxdna driver needs:
// kernel metadata (EMBEDDED_METADATA XML), compute units (IP_LAYOUT) and the AIE partition
// with its PDIs (AIE_PARTITION). Layouts follow xrt/detail/xclbin.h; offsets inside a section
// are relative to the section start, as in XRT's xclbin_parser.cpp.
package xclbin

import (
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Section kinds (enum axlf_section_kind).
const (
	SectionEmbeddedMetadata = 2
	SectionIPLayout         = 8
	SectionAIEPartition     = 32
)

// IP types (enum IP_TYPE).
const (
	IPKernel   = 1
	IPPSKernel = 7
)

// Struct sizes and offsets of xclbin.h (64-bit).
const (
	axlfHeaderOffset   = 304
	axlfSectionsOffset = 456
	axlfHeaderSize     = 152
	sectionHeaderSize  = 40
	ipLayoutDataOffset = 8
	ipDataSize         = 80
	aiePartitionSize   = 184
	aiePDISize         = 96
	cdoGroupSize       = 96
)

// Magic is the first 8 bytes of an xclbin.
var Magic = []byte("xclbin2\x00")

// Arg is one kernel argument from the metadata.
type Arg struct {
	Index            int
	Name             string
	Type             string
	Offset           int // byte offset in the register map
	Size             int // byte size in the register map
	AddressQualifier int // 0 scalar, 1 global buffer
}

// IsBuffer reports whether the argument is a device buffer address.
func (a Arg) IsBuffer() bool { return a.AddressQualifier == 1 }

// Kernel is a kernel described by the metadata.
type Kernel struct {
	Name       string
	Type       string // "dpu" for NPU kernels
	Args       []Arg
	Functional uint32 // from <extended-data functional="...">
	KernelID   uint64 // from <extended-data dpu_kernel_id="...">
}

// CU is a compute unit instance from IP_LAYOUT.
type CU struct {
	Name       string // "kernel:instance"
	Kernel     string
	Type       uint32
	Functional uint32 // PS kernel bit field, used when the metadata has no extended-data
	KernelID   uint32
}

// CDOGroup selects the kernels a PDI serves.
type CDOGroup struct {
	Name      string
	Type      uint8
	PDIID     uint64
	KernelIDs []uint64
}

// PDI is an AIE configuration image.
type PDI struct {
	UUID   [16]byte
	Image  []byte
	Groups []CDOGroup
}

// Partition is the AIE_PARTITION section.
type Partition struct {
	Columns      int
	StartColumns []uint16
	OpsPerCycle  uint32
	PDIs         []PDI
}

// File is a parsed xclbin.
type File struct {
	UUID      [16]byte
	Kernels   []Kernel
	CUs       []CU
	Partition *Partition
}

// Kernel returns the kernel with the given name.
func (f *File) Kernel(name string) (*Kernel, bool) {
	for i := range f.Kernels {
		if f.Kernels[i].Name == name {
			return &f.Kernels[i], true
		}
	}
	return nil, false
}

// PDIForKernel returns the PDI whose CDO groups list the kernel id, as the XRT shim does.
func (f *File) PDIForKernel(id uint64) ([]byte, bool) {
	if f.Partition == nil {
		return nil, false
	}
	for _, p := range f.Partition.PDIs {
		for _, g := range p.Groups {
			for _, k := range g.KernelIDs {
				if k == id {
					return p.Image, true
				}
			}
		}
	}
	return nil, false
}

// Parse reads an xclbin.
func Parse(data []byte) (*File, error) {
	if len(data) < axlfSectionsOffset || !bytes.Equal(data[:8], Magic) {
		return nil, errors.New("xclbin: not an xclbin (bad magic)")
	}
	le := binary.LittleEndian
	h := data[axlfHeaderOffset:]
	if length := le.Uint64(h[0:]); length > uint64(len(data)) {
		return nil, fmt.Errorf("xclbin: header length %d exceeds file size %d", length, len(data))
	}
	f := &File{}
	copy(f.UUID[:], h[112:128])
	n := int(le.Uint32(h[144:]))
	if axlfSectionsOffset+n*sectionHeaderSize > len(data) {
		return nil, errors.New("xclbin: section table out of range")
	}
	var xmlData []byte
	for i := 0; i < n; i++ {
		s := data[axlfSectionsOffset+i*sectionHeaderSize:]
		kind := le.Uint32(s[0:])
		off, size := le.Uint64(s[24:]), le.Uint64(s[32:])
		if off+size > uint64(len(data)) || off+size < off {
			return nil, fmt.Errorf("xclbin: section %d (kind %d) out of range", i, kind)
		}
		sec := data[off : off+size]
		var err error
		switch kind {
		case SectionEmbeddedMetadata:
			xmlData = sec
		case SectionIPLayout:
			f.CUs, err = parseIPLayout(sec)
		case SectionAIEPartition:
			f.Partition, err = parsePartition(sec)
		}
		if err != nil {
			return nil, err
		}
	}
	if xmlData == nil {
		return nil, errors.New("xclbin: no EMBEDDED_METADATA section")
	}
	var err error
	if f.Kernels, err = parseMetadata(xmlData); err != nil {
		return nil, err
	}
	return f, nil
}

func cstring(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		return string(b[:i])
	}
	return string(b)
}

func parseIPLayout(sec []byte) ([]CU, error) {
	le := binary.LittleEndian
	if len(sec) < ipLayoutDataOffset {
		return nil, errors.New("xclbin: IP_LAYOUT too small")
	}
	count := int(int32(le.Uint32(sec)))
	if count < 0 || ipLayoutDataOffset+count*ipDataSize > len(sec) {
		return nil, errors.New("xclbin: IP_LAYOUT count out of range")
	}
	var cus []CU
	for i := 0; i < count; i++ {
		d := sec[ipLayoutDataOffset+i*ipDataSize:]
		typ := le.Uint32(d[0:])
		if typ != IPPSKernel && typ != IPKernel {
			continue
		}
		props := le.Uint32(d[4:])
		name := cstring(d[16:80])
		kernel, _, _ := strings.Cut(name, ":")
		cus = append(cus, CU{
			Name: name, Kernel: kernel, Type: typ,
			Functional: (props >> 4) & 0x3,    // ps_kernel.m_functional
			KernelID:   (props >> 16) & 0xfff, // ps_kernel.m_kernel_id
		})
	}
	return cus, nil
}

// span returns sec[off:off+n*elem] after bounds checks.
func span(sec []byte, off, n, elem uint32, what string) ([]byte, error) {
	end := uint64(off) + uint64(n)*uint64(elem)
	if end > uint64(len(sec)) {
		return nil, fmt.Errorf("xclbin: AIE_PARTITION %s out of range", what)
	}
	return sec[off:end], nil
}

func sectionString(sec []byte, off uint32) string {
	if uint64(off) >= uint64(len(sec)) {
		return ""
	}
	return cstring(sec[off:])
}

func parsePartition(sec []byte) (*Partition, error) {
	le := binary.LittleEndian
	if len(sec) < aiePartitionSize {
		return nil, errors.New("xclbin: AIE_PARTITION too small")
	}
	p := &Partition{OpsPerCycle: le.Uint32(sec[8:]), Columns: int(le.Uint16(sec[32:]))}
	sc, err := span(sec, le.Uint32(sec[44:]), le.Uint32(sec[40:]), 2, "start_columns")
	if err != nil {
		return nil, err
	}
	for i := 0; i+1 < len(sc); i += 2 {
		p.StartColumns = append(p.StartColumns, le.Uint16(sc[i:]))
	}
	pdis, err := span(sec, le.Uint32(sec[124:]), le.Uint32(sec[120:]), aiePDISize, "aie_pdi")
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(pdis); i += aiePDISize {
		d := pdis[i:]
		var pdi PDI
		copy(pdi.UUID[:], d[0:16])
		img, err := span(sec, le.Uint32(d[20:]), le.Uint32(d[16:]), 1, "pdi_image")
		if err != nil {
			return nil, err
		}
		pdi.Image = img
		groups, err := span(sec, le.Uint32(d[28:]), le.Uint32(d[24:]), cdoGroupSize, "cdo_groups")
		if err != nil {
			return nil, err
		}
		for j := 0; j < len(groups); j += cdoGroupSize {
			g := groups[j:]
			cg := CDOGroup{Name: sectionString(sec, le.Uint32(g[0:])), Type: g[4], PDIID: le.Uint64(g[8:])}
			ids, err := span(sec, le.Uint32(g[20:]), le.Uint32(g[16:]), 8, "dpu_kernel_ids")
			if err != nil {
				return nil, err
			}
			for k := 0; k < len(ids); k += 8 {
				cg.KernelIDs = append(cg.KernelIDs, le.Uint64(ids[k:]))
			}
			pdi.Groups = append(pdi.Groups, cg)
		}
		p.PDIs = append(p.PDIs, pdi)
	}
	return p, nil
}

type xmlProject struct {
	Kernels []xmlKernel `xml:"platform>device>core>kernel"`
}

type xmlKernel struct {
	Name     string   `xml:"name,attr"`
	Type     string   `xml:"type,attr"`
	Args     []xmlArg `xml:"arg"`
	Extended []xmlExt `xml:"extended-data"`
}

type xmlArg struct {
	Name             string `xml:"name,attr"`
	ID               string `xml:"id,attr"`
	Offset           string `xml:"offset,attr"`
	Size             string `xml:"size,attr"`
	Type             string `xml:"type,attr"`
	AddressQualifier string `xml:"addressQualifier,attr"`
}

type xmlExt struct {
	Functional string `xml:"functional,attr"`
	KernelID   string `xml:"dpu_kernel_id,attr"`
}

// convert parses decimal or 0x-prefixed hexadecimal like XRT's convert().
func convert(s string) (uint64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, nil
	}
	return strconv.ParseUint(s, 0, 64)
}

func parseMetadata(data []byte) ([]Kernel, error) {
	var p xmlProject
	if err := xml.Unmarshal(bytes.TrimRight(data, "\x00"), &p); err != nil {
		return nil, fmt.Errorf("xclbin: EMBEDDED_METADATA: %w", err)
	}
	var ks []Kernel
	for _, xk := range p.Kernels {
		k := Kernel{Name: xk.Name, Type: xk.Type}
		if k.Type == "" {
			k.Type = "pl"
		}
		for _, xa := range xk.Args {
			var vals [4]uint64
			for i, s := range []string{xa.ID, xa.Offset, xa.Size, xa.AddressQualifier} {
				v, err := convert(s)
				if err != nil {
					return nil, fmt.Errorf("xclbin: kernel %s arg %s: %w", xk.Name, xa.Name, err)
				}
				vals[i] = v
			}
			k.Args = append(k.Args, Arg{Index: int(vals[0]), Name: xa.Name, Type: xa.Type,
				Offset: int(vals[1]), Size: int(vals[2]), AddressQualifier: int(vals[3])})
		}
		sort.SliceStable(k.Args, func(i, j int) bool { return k.Args[i].Index < k.Args[j].Index })
		if len(xk.Extended) > 0 {
			fn, err1 := convert(xk.Extended[0].Functional)
			id, err2 := convert(xk.Extended[0].KernelID)
			if err := errors.Join(err1, err2); err != nil {
				return nil, fmt.Errorf("xclbin: kernel %s extended-data: %w", xk.Name, err)
			}
			k.Functional, k.KernelID = uint32(fn), id
		}
		ks = append(ks, k)
	}
	return ks, nil
}
