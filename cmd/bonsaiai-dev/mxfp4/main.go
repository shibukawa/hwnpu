// Command bonsaiai-dev mxfp4 writes a checkpoint of a Qwen3.5-MoE model (Qwen3.6-35B-A3B) whose
// experts are MXFP4, which bonsaiai runs on the CPU from the checkpoint mapped into memory
// (qwen35.DeriveMXFP4): the release's experts are bf16, 64 GB of its 72, and as MXFP4 17 GB. The
// source is read a layer at a time, from a Hugging Face repository with Range requests (no copy
// of the release on the disk) or from a directory; a run cut off goes on where it stopped.
//
//	bonsaiai-dev mxfp4 -o DIR hf:Qwen/Qwen3.6-35B-A3B
//	bonsaiai-dev mxfp4 -o DIR ~/models/Qwen3.6-35B-A3B
//
// Then convert from DIR as from any checkpoint (bonsaiai-dev convert -config DIR/config.json
// -weights DIR/model.safetensors.index.json -backend coreai ..., or bonsaiai-chat -model DIR).
package mxfp4

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/llm/qwen35"
)

func Main() {
	out := flag.String("o", "", "the directory to write the checkpoint into")
	rev := flag.String("revision", "", "the repository's revision (hf: sources; default main)")
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), "usage: bonsaiai-dev mxfp4 -o DIR hf:OWNER/NAME | DIR")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *out == "" || flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	from := flag.Arg(0)
	var src qwen35.Source
	if repo, ok := strings.CutPrefix(from, "hf:"); ok {
		s := qwen35.NewHTTPSource(ctx, repo)
		s.Revision = *rev
		src = s
	} else {
		src = qwen35.DirSource(from)
	}
	if err := qwen35.DeriveMXFP4(src, from, *out, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "bonsaiai-dev mxfp4:", err)
		os.Exit(1)
	}
}
