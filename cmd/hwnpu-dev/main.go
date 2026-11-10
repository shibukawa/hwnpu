// hwnpu-dev holds the tools for developing hwnpu: converting and checking models by hand,
// measuring NPUs, packaging xclbins. Each subcommand is the tool it was before (hwnpu-check,
// hwnpu-convert, npuprobe, ...) with its own flags, which follow it as they did:
//
//	hwnpu-dev check -config DIR/config.json -weights DIR/model.safetensors -ref DIR/ref
//	hwnpu-dev convert -config DIR/config.json -weights DIR/model.safetensors -seq 64 -o out.mlpackage
//	hwnpu-dev generate -model DIR -package out.mlpackage "Hello"
//	hwnpu-dev formats -formats int4-g128 -ref int8
//	hwnpu-dev info
//	hwnpu-dev probe bench
//	hwnpu-dev powerbench -decode 8,16
//	hwnpu-dev report -o report.md
//	hwnpu-dev xclbin ...                     # xclbinutil's arguments, as aiecc passes them
//	hwnpu-dev check -h                       # each tool's flags
//
// aiecc runs xclbinutil by AIE_XCLBINUTIL alone, without a subcommand: a link to hwnpu-dev named
// hwnpu-xclbin (or xclbinutil) runs xclbin.
package main

import (
	"flag"
	"os"
	"path/filepath"
	"strings"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/check"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/convert"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/formats"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/generate"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/info"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/powerbench"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/probe"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/report"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/xclbin"
)

// tool is a subcommand: the arguments after its name go to the tool's own flags.
type tool struct {
	Args []string `arg:"" optional:"" passthrough:"all" help:"The tool's arguments (-h lists its flags)."`
}

type cli struct {
	Check      tool `cmd:"" passthrough:"all" help:"Compare a conversion with the Hugging Face reference layer by layer, on the CPU interpreter (was hwnpu-check)."`
	Convert    tool `cmd:"" passthrough:"all" help:"Convert a Hugging Face checkpoint into a Core ML package or OpenVINO IR by hand (was hwnpu-convert)."`
	Generate   tool `cmd:"" passthrough:"all" help:"Write text with a causal language model converted by convert (was hwnpu-generate)."`
	Formats    tool `cmd:"" passthrough:"all" help:"Compare the weight formats of a chat model on the NPU: agreement, logits, perplexity, time (was hwnpu-formats)."`
	Info       tool `cmd:"" passthrough:"all" help:"List the NPUs hwnpu can use, and where a model's operations would run (was hwnpuinfo)."`
	Probe      tool `cmd:"" passthrough:"all" help:"Measure this machine's NPU with generated LLM-shaped models: speed, size limit, KV cache (was npuprobe)."`
	Powerbench tool `cmd:"" passthrough:"all" help:"Measure the power Core ML models draw with macOS powermetrics (was powerbench)."`
	Report     tool `cmd:"" passthrough:"all" help:"Check every NPU with bundled test models and print a report for an issue (was report)."`
	Xclbin     tool `cmd:"" passthrough:"all" help:"Package an mlir-aie design into an xclbin without XRT, in place of xclbinutil (was hwnpu-xclbin)."`
}

var mains = map[string]func(){
	"check": check.Main, "convert": convert.Main, "generate": generate.Main, "formats": formats.Main,
	"info": info.Main, "probe": probe.Main, "powerbench": powerbench.Main, "report": report.Main,
	"xclbin": xclbin.Main,
}

func main() {
	switch strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") {
	case "hwnpu-xclbin", "xclbinutil":
		xclbin.Main()
		return
	}
	probe.ChildArgs = []string{"probe"} // the prober runs this executable as its child
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-dev"),
		kong.Description("Tools for developing hwnpu: converting and checking models by hand, measuring NPUs, packaging xclbins."),
		kong.UsageOnError())
	k.FatalIfErrorf(k.Run())
}

// Run runs the tool the subcommand names with its arguments, as its own main did: it parses
// os.Args with the flag package.
func (t *tool) Run(k *kong.Context) error {
	name, _, _ := strings.Cut(k.Command(), " ")
	os.Args = append([]string{"hwnpu-dev " + name}, t.Args...)
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	mains[name]()
	return nil
}
