package hfcmd

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// SizesCmd is hwnpu-dev hf sizes: what each model downloads and keeps on the disk, by the NPU's
// vendor, from the conversions the layers hold (models.yaml's bytes and disk).
type SizesCmd struct {
	Sel   []string `arg:"" optional:"" name:"N:MODEL" help:"Models (every model when left out), the layer they come from before :."`
	Write string   `placeholder:"FILE" help:"Set models.yaml's disk in FILE (the source tree's: models.yaml) to the sizes found, by vendor."`
}

// vendorOf is the vendor of the NPU a group runs on (models.yaml's tables of bytes and disk).
var vendorOf = map[string]string{groupAMD: "amd", groupApple: "apple"}

func (s SizesCmd) Run(c *Cmd) error {
	if err := load(c); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	ls := layers()
	var sels []selector
	for _, a := range s.Sel {
		sel, err := parseSelector(a)
		if err != nil {
			return err
		}
		sels = append(sels, sel)
	}
	if len(sels) == 0 {
		for _, m := range chat.Catalog {
			sels = append(sels, selector{layer: -1, model: m})
		}
	}
	var src []byte
	if s.Write != "" {
		b, err := os.ReadFile(s.Write)
		if err != nil {
			return err
		}
		src = b
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "model\tvendor\tdownload GB\tdisk GB\tin models.yaml (bytes, disk)\tnote")
	written := 0
	for _, sel := range sels {
		m := sel.model
		disk := map[string]float64{}
		var lack []string
		for _, r := range rows(ctx, ls, sel.layer, m, true) {
			v := vendorOf[r.group]
			if len(r.entries) == 0 {
				lack = append(lack, v)
				continue
			}
			// AMD: the export of the model as it is, not its variants' (as bytes)
			ents := r.entries
			if r.group == groupAMD {
				if ws, err := amdWants(ctx, m); err == nil && len(ws) > 0 {
					ents = slices.DeleteFunc(slices.Clone(ents), func(f found) bool { return f.want.Path != ws[0].Path })
				}
			}
			var d, dl int64
			for _, f := range ents {
				d += f.got.Bytes()
				if f.got.Pack != nil {
					dl += f.got.Pack.Size
				} else {
					dl += f.got.Bytes()
				}
			}
			disk[v] = float64(d)
			note := ""
			if r.group == groupApple {
				note = "bytes: the checkpoint, converted here"
			}
			fmt.Fprintf(tw, "%s\t%s\t%.2f\t%.2f\t%s, %s\t%s\n", m.Name, v, float64(dl)/1e9, float64(d)/1e9,
				gb(m.BytesFor(v)), gb(m.DiskFor(v)), note)
		}
		if len(disk) == 0 {
			continue
		}
		if src == nil {
			continue
		}
		// a vendor the table leaves out is not known (DiskFor): the others are written
		if len(lack) > 0 {
			fmt.Fprintf(os.Stderr, "%s: no size for %s: none of its conversions in the layers (Core AI's: at its version)\n", m.Name, strings.Join(lack, ", "))
		}
		var parts []string
		for _, v := range slices.Sorted(mapKeys(disk)) {
			parts = append(parts, fmt.Sprintf("%s: %.2fe9", v, disk[v]/1e9))
		}
		out, err := fieldYAML(src, m.Name, "disk", "{"+strings.Join(parts, ", ")+"}", "bytes")
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: not written: %v\n", m.Name, err)
			continue
		}
		src = out
		written++
	}
	tw.Flush()
	if s.Write != "" {
		if err := os.WriteFile(s.Write, src, 0o644); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "disk written for %d models in %s\n", written, s.Write)
	}
	return nil
}

func gb(n float64) string {
	if n <= 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", n/1e9)
}
