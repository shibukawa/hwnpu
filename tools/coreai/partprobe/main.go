// Command partprobe writes parts of a chat model as the functions of one Core AI program, each
// with its own options: for telling on a device which part of a program a prebuilt program
// fails on (one program is finished once, whatever its functions).
//
//	go run ./tools/coreai/partprobe -arch qwen35 -dir CHECKPOINT -o OUT.aimodel NAME=start:layers:seq:paged[:key=value,...] ...
//
// layers -1 is no layer (llm.NoLayers); paged 0 a part with its cache inside (cache 8192). The
// options are the chat's (lut8 by k-means for Qwen3.5); key=value are the architecture's
// parameters (state-rows=1) or last=1 (the logits of the last row alone).
package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
)

func main() {
	arch := flag.String("arch", "qwen35", "the architecture")
	dir := flag.String("dir", "", "the checkpoint's folder (config.json, model.safetensors.index.json)")
	weights := flag.String("weights", "model.safetensors.index.json", "the weights' file in -dir")
	out := flag.String("o", "", "the program (.aimodel)")
	cache := flag.Int("cache", 262144, "the cache of a part split at its attention")
	flag.Parse()
	a, model, err := llm.LoadConfig(filepath.Join(*dir, "config.json"), *arch)
	if err != nil {
		log.Fatal(err)
	}
	w, err := llm.OpenWeights(filepath.Join(*dir, *weights), a)
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()
	var fns []llm.AIFunction
	for _, spec := range flag.Args() {
		name, rest, ok := strings.Cut(spec, "=")
		f := strings.SplitN(rest, ":", 5)
		if !ok || len(f) < 4 {
			log.Fatalf("%s: NAME=start:layers:seq:paged[:key=value,...]", spec)
		}
		n := make([]int, 4)
		for i := range n {
			if n[i], err = strconv.Atoi(f[i]); err != nil {
				log.Fatalf("%s: %v", spec, err)
			}
		}
		fs := flag.NewFlagSet("", flag.ContinueOnError)
		var o llm.Options
		set := llm.BindFlags(fs, &o, "")
		if err := fs.Parse(nil); err != nil {
			log.Fatal(err)
		}
		set()
		o.Backend = llm.BackendCoreAI
		o.Format = a.DefaultFormatFor(model, o.Backend)
		if !strings.HasPrefix(o.Format, "int") {
			o.Quant = "kmeans-minmax"
		}
		o.Start, o.Layers, o.Seq, o.Paged = n[0], n[1], n[2], n[3]
		o.Cache = *cache
		if o.Paged == 0 {
			o.Cache = 8192
		}
		if len(f) == 5 {
			for _, kv := range strings.Split(f[4], ",") {
				k, v, _ := strings.Cut(kv, "=")
				if k == "last" {
					o.LastRow = v == "1"
					continue
				}
				o.Params[k] = v
			}
		}
		fns = append(fns, llm.AIFunction{Name: name, Model: model, Options: o})
	}
	recs, err := llm.WriteAIFunctions(*out, w, fns)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range recs {
		fmt.Printf("%s: %d inputs, %d state outputs, %.1f MB\n", r.Path, len(r.Inputs), len(r.State), float64(r.WeightBytes)/1e6)
	}
}
