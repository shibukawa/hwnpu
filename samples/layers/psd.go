package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"unicode/utf16"
)

// Layer is a layer of a PSD: a name and a picture of the document's size with its transparency.
type Layer struct {
	Name string
	Img  *image.NRGBA
}

// WritePSD writes a Photoshop document: the layers from the bottom one to the top one, and the
// picture a program that reads no layers shows (merged, opaque). The document is 8 bits a
// channel RGB; the layers are of the document's size, their channels packed with PackBits.
func WritePSD(w io.Writer, merged *image.RGBA, layers []Layer) error {
	width, height := merged.Rect.Dx(), merged.Rect.Dy()
	if width > 30000 || height > 30000 {
		return fmt.Errorf("psd: %d x %d is over a PSD's 30000 pixels", width, height)
	}
	for _, l := range layers {
		if l.Img.Rect.Dx() != width || l.Img.Rect.Dy() != height {
			return fmt.Errorf("psd: the layer %q is %d x %d, the document %d x %d", l.Name, l.Img.Rect.Dx(), l.Img.Rect.Dy(), width, height)
		}
	}
	be := binary.BigEndian
	var out bytes.Buffer
	put := func(b *bytes.Buffer, vs ...any) {
		for _, v := range vs {
			binary.Write(b, be, v)
		}
	}

	// the header: the signature, version 1, 6 reserved bytes, 3 channels, the size, 8 bits, RGB
	out.WriteString("8BPS")
	put(&out, uint16(1), [6]byte{}, uint16(3), uint32(height), uint32(width), uint16(8), uint16(3))
	put(&out, uint32(0)) // no colour mode data
	put(&out, uint32(0)) // no image resources

	// the layers: their records, then their channels' pixels in the records' order
	var records, pixels bytes.Buffer
	for _, l := range layers {
		// a layer's channels: its transparency (-1), then red, green and blue
		chans := make([][]byte, 4)
		for k, at := range []int{3, 0, 1, 2} {
			chans[k] = packChannel(l.Img.Pix[at:], 4, width, height, l.Img.Stride)
		}
		put(&records, int32(0), int32(0), int32(height), int32(width), uint16(4))
		for k, id := range []int16{-1, 0, 1, 2} {
			put(&records, id, uint32(2+len(chans[k]))) // the compression's 2 bytes are counted
		}
		records.WriteString("8BIM")
		records.WriteString("norm")
		put(&records, uint8(255), uint8(0), uint8(0), uint8(0)) // opaque, not clipped, shown, a filler

		var extra bytes.Buffer
		put(&extra, uint32(0), uint32(0)) // no mask, no blending ranges
		// the name as a Pascal string padded to 4 bytes (Latin; others as ?), then in Unicode
		var name []byte
		for _, r := range l.Name {
			if r >= 0x80 {
				r = '?'
			}
			name = append(name, byte(r))
		}
		name = name[:min(len(name), 255)]
		extra.WriteByte(uint8(len(name)))
		extra.Write(name)
		extra.Write(make([]byte, (4-(1+len(name))%4)%4))
		u := utf16.Encode([]rune(l.Name))
		extra.WriteString("8BIM")
		extra.WriteString("luni")
		size := 4 + 2*len(u)
		put(&extra, uint32(size+size%2), uint32(len(u)), u)
		if size%2 != 0 {
			extra.WriteByte(0)
		}
		put(&records, uint32(extra.Len()))
		records.Write(extra.Bytes())

		for _, c := range chans {
			put(&pixels, uint16(1)) // PackBits
			pixels.Write(c)
		}
	}
	var info bytes.Buffer
	put(&info, int16(len(layers)))
	info.Write(records.Bytes())
	info.Write(pixels.Bytes())
	if info.Len()%2 != 0 {
		info.WriteByte(0)
	}
	// the section of layers and masks: the layers' length and the layers, then no global mask
	put(&out, uint32(4+info.Len()+4), uint32(info.Len()))
	out.Write(info.Bytes())
	put(&out, uint32(0))

	// the merged picture: PackBits, the rows' lengths of every channel first, then their bytes
	put(&out, uint16(1))
	var counts, rows bytes.Buffer
	for k := 0; k < 3; k++ {
		c := packChannel(merged.Pix[k:], 4, width, height, merged.Stride)
		counts.Write(c[:2*height])
		rows.Write(c[2*height:])
	}
	out.Write(counts.Bytes())
	out.Write(rows.Bytes())
	_, err := w.Write(out.Bytes())
	return err
}

// packChannel is a channel of a picture packed row by row with PackBits: the rows' packed
// lengths (2 bytes each), then the rows. The channel's bytes are every step-th of pix, in rows
// of stride bytes.
func packChannel(pix []byte, step, width, height, stride int) []byte {
	out := make([]byte, 2*height, 2*height+width*height/4)
	row := make([]byte, width)
	for y := 0; y < height; y++ {
		for x := range row {
			row[x] = pix[y*stride+x*step]
		}
		n := len(out)
		out = packBits(out, row)
		binary.BigEndian.PutUint16(out[2*y:], uint16(len(out)-n))
	}
	return out
}

// packBits appends row packed with PackBits to dst: a count byte n and then n + 1 bytes as they
// are (n from 0 to 127), or a count byte 257 - n and then one byte to repeat n times (n from 2
// to 128).
func packBits(dst, row []byte) []byte {
	for i := 0; i < len(row); {
		// a run of the same byte
		run := 1
		for i+run < len(row) && run < 128 && row[i+run] == row[i] {
			run++
		}
		if run >= 2 {
			dst = append(dst, byte(257-run), row[i])
			i += run
			continue
		}
		// bytes as they are, up to the next run of three (a run of two is not worth leaving for)
		start := i
		i++
		for i < len(row) && i-start < 128 {
			if i+2 < len(row) && row[i] == row[i+1] && row[i] == row[i+2] {
				break
			}
			i++
		}
		dst = append(dst, byte(i-start-1))
		dst = append(dst, row[start:i]...)
	}
	return dst
}
