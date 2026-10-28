package xclbin

import (
	"encoding/binary"
	"fmt"
	"strings"
)

// Spec describes a synthetic xclbin for tests. Build lays it out exactly like xclbinutil:
// header, section table, then EMBEDDED_METADATA, IP_LAYOUT and AIE_PARTITION.
type Spec struct {
	UUID        [16]byte
	Kernels     []Kernel // metadata; Functional and KernelID go to <extended-data>
	CUs         []CU     // IP_LAYOUT entries (type IP_PS_KERNEL)
	Columns     int
	StartCols   []uint16
	OpsPerCycle uint32
	PDIs        []PDI
}

// Build serializes a Spec.
func Build(s Spec) []byte {
	le := binary.LittleEndian
	type section struct {
		kind uint32
		name string
		data []byte
	}
	sections := []section{
		{SectionEmbeddedMetadata, "", []byte(metadataXML(s.Kernels))},
		{SectionIPLayout, "", ipLayout(s.CUs)},
		{SectionAIEPartition, "", aiePartition(s)},
	}
	off := uint64(axlfSectionsOffset + len(sections)*sectionHeaderSize)
	total := off
	for _, sec := range sections {
		total = align8(total) + uint64(len(sec.data))
	}
	out := make([]byte, total)
	copy(out, Magic)
	le.PutUint32(out[8:], 0xffffffff) // no signature
	h := out[axlfHeaderOffset:]
	le.PutUint64(h[0:], total)
	h[26], h[27] = 2, 1
	copy(h[112:128], s.UUID[:])
	le.PutUint32(h[144:], uint32(len(sections)))
	for i, sec := range sections {
		off = align8(off)
		sh := out[axlfSectionsOffset+i*sectionHeaderSize:]
		le.PutUint32(sh[0:], sec.kind)
		le.PutUint64(sh[24:], off)
		le.PutUint64(sh[32:], uint64(len(sec.data)))
		copy(out[off:], sec.data)
		off += uint64(len(sec.data))
	}
	return out
}

func align8(v uint64) uint64 { return (v + 7) &^ 7 }

func metadataXML(ks []Kernel) string {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n<project name=\"test\"><platform><device name=\"npu\"><core name=\"OCL_REGION_0\">\n")
	for _, k := range ks {
		fmt.Fprintf(&b, "<kernel name=%q language=\"c\" type=%q>\n", k.Name, k.Type)
		fmt.Fprintf(&b, "<extended-data subtype=\"DPU\" functional=\"%d\" dpu_kernel_id=\"0x%x\"/>\n", k.Functional, k.KernelID)
		for _, a := range k.Args {
			fmt.Fprintf(&b, "<arg name=%q addressQualifier=\"%d\" id=\"%d\" size=\"0x%x\" offset=\"0x%x\" hostOffset=\"0x0\" hostSize=\"0x%x\" type=%q/>\n",
				a.Name, a.AddressQualifier, a.Index, a.Size, a.Offset, a.Size, a.Type)
		}
		b.WriteString("</kernel>\n")
	}
	b.WriteString("</core></device></platform></project>\n")
	return b.String()
}

func ipLayout(cus []CU) []byte {
	le := binary.LittleEndian
	out := make([]byte, ipLayoutDataOffset+len(cus)*ipDataSize)
	le.PutUint32(out, uint32(len(cus)))
	for i, cu := range cus {
		d := out[ipLayoutDataOffset+i*ipDataSize:]
		le.PutUint32(d[0:], IPPSKernel)
		le.PutUint32(d[4:], 1|(cu.Functional&3)<<4|(cu.KernelID&0xfff)<<16) // subtype DPU
		copy(d[16:79], cu.Name)
	}
	return out
}

func aiePartition(s Spec) []byte {
	le := binary.LittleEndian
	buf := make([]byte, aiePartitionSize)
	put := func(data []byte) uint32 { // appends 8-byte aligned data, returns its offset
		for len(buf)%8 != 0 {
			buf = append(buf, 0)
		}
		off := uint32(len(buf))
		buf = append(buf, data...)
		return off
	}
	le.PutUint32(buf[8:], s.OpsPerCycle)
	le.PutUint16(buf[32:], uint16(s.Columns))
	cols := make([]byte, 2*len(s.StartCols))
	for i, c := range s.StartCols {
		le.PutUint16(cols[2*i:], c)
	}
	colOff := put(cols)
	le.PutUint32(buf[40:], uint32(len(s.StartCols)))
	le.PutUint32(buf[44:], colOff)

	pdiTable := put(make([]byte, len(s.PDIs)*aiePDISize))
	le.PutUint32(buf[120:], uint32(len(s.PDIs)))
	le.PutUint32(buf[124:], pdiTable)
	for i, p := range s.PDIs {
		img := put(p.Image)
		groupTable := put(make([]byte, len(p.Groups)*cdoGroupSize))
		for j, g := range p.Groups {
			name := put(append([]byte(g.Name), 0))
			ids := make([]byte, 8*len(g.KernelIDs))
			for k, id := range g.KernelIDs {
				le.PutUint64(ids[8*k:], id)
			}
			idOff := put(ids)
			gd := buf[groupTable+uint32(j*cdoGroupSize):]
			le.PutUint32(gd[0:], name)
			gd[4] = g.Type
			le.PutUint64(gd[8:], g.PDIID)
			le.PutUint32(gd[16:], uint32(len(g.KernelIDs)))
			le.PutUint32(gd[20:], idOff)
		}
		d := buf[pdiTable+uint32(i*aiePDISize):]
		copy(d[0:16], p.UUID[:])
		le.PutUint32(d[16:], uint32(len(p.Image)))
		le.PutUint32(d[20:], img)
		le.PutUint32(d[24:], uint32(len(p.Groups)))
		le.PutUint32(d[28:], groupTable)
	}
	return buf
}
