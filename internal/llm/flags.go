package llm

import (
	"flag"
	"strings"
)

// BindFlags defines flags for the weight format options and for every registered architecture's
// parameters (an architecture ignores the others'). Call the returned function after fs.Parse
// to copy the values into o. An empty defaultFormat leaves Format empty unless -format is given,
// for the caller to take the architecture's (Arch.DefaultFormat).
func BindFlags(fs *flag.FlagSet, o *Options, defaultFormat string) func() {
	usage := "weight format: lut4 (4 bit palettized; lut6 and lut8 have 64 and 256 entry tables), int4 or int8 (symmetric, a scale per row; int4-g128 per 128 columns; the formats of Intel's NPU) or fp16"
	if defaultFormat == "" {
		usage += " (default: the architecture's, lut4 but for Gemma 4's lut6 and the lut8 of LFM3's multimodal and d1 checkpoints, BERT and PLaMo-Embedding; for Intel's NPU int8 (its 4 bit formats lose far more than lut4))"
	}
	format := fs.String("format", defaultFormat, usage)
	group := fs.Int("group", 16, "rows per lookup table for lut4")
	colBlock := fs.Int("colblock", 0, "columns per lookup table for lut4 (0: whole rows)")
	keep := fs.String("keep-fp16", "", "comma separated substrings of tensor names to keep in fp16, e.g. embed_tokens")
	formatFor := fs.String("format-for", "", "comma separated SUBSTRING=FORMAT: another weight format for the tensors whose names contain SUBSTRING, e.g. mlp=lut8 (the last match wins)")
	quant := fs.String("quant", "", "lut4 table fitting: uniform (the default), kmeans or kmeans-minmax; intN scales: mse (the default: the clipping with the smallest squared error) or uniform (the block's largest weight)")
	params := map[string]*string{}
	for _, a := range Archs() {
		for _, p := range a.Params {
			if _, ok := params[p.Name]; !ok {
				params[p.Name] = fs.String(p.Name, p.Default, a.Title+": "+p.Usage)
			}
		}
	}
	return func() {
		o.Format, o.Group, o.ColBlock, o.Quant = *format, *group, *colBlock, *quant
		o.KeepFP16 = nil
		if *keep != "" {
			o.KeepFP16 = strings.Split(*keep, ",")
		}
		o.FormatFor = nil
		if *formatFor != "" {
			o.FormatFor = strings.Split(*formatFor, ",")
		}
		o.Params = map[string]string{}
		for name, v := range params {
			o.Params[name] = *v
		}
	}
}
