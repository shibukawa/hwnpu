package llm

import (
	"flag"
	"strings"
)

// BindFlags defines flags for the weight format options and for every registered architecture's
// parameters (an architecture ignores the others'). Call the returned function after fs.Parse
// to copy the values into o.
func BindFlags(fs *flag.FlagSet, o *Options, defaultFormat string) func() {
	format := fs.String("format", defaultFormat, "weight format: lut4 (4 bit palettized) or fp16")
	group := fs.Int("group", 16, "rows per lookup table for lut4")
	colBlock := fs.Int("colblock", 0, "columns per lookup table for lut4 (0: whole rows)")
	keep := fs.String("keep-fp16", "", "comma separated substrings of tensor names to keep in fp16, e.g. embed_tokens")
	quant := fs.String("quant", "uniform", "lut4 table fitting: uniform, kmeans or kmeans-minmax")
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
		o.Params = map[string]string{}
		for name, v := range params {
			o.Params[name] = *v
		}
	}
}
