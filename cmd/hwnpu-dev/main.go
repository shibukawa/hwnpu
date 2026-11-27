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
//	hwnpu-dev sources                         # the cache's root and the sources
//	hwnpu-dev models fetch MODEL/coreai       # bring a model's conversions here, converting nothing
//	hwnpu-dev models package MODEL/coreai     # make them (any OS); MODEL/coreai-h14g compiles (macOS)
//	hwnpu-dev xclbin ...                     # xclbinutil's arguments, as aiecc passes them
//	hwnpu-dev mxfp4 -o DIR hf:Qwen/Qwen3.6-35B-A3B  # a Qwen3.5-MoE checkpoint with MXFP4 experts
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

	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/check"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/convert"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/daemoncmd"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/formats"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/generate"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/info"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/irdiff"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/modelscmd"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/mxfp4"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/powerbench"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/prebuild"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/probe"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/report"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/sourcescmd"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/xclbin"
)

// tool is a subcommand: the arguments after its name go to the tool's own flags.
type tool struct {
	Args []string `arg:"" optional:"" passthrough:"all" help:"The tool's arguments (-h lists its flags)."`
}

type cli struct {
	Check      tool           `cmd:"" passthrough:"all" help:"Compare a conversion with the Hugging Face reference layer by layer, on the CPU interpreter (was hwnpu-check)."`
	Convert    tool           `cmd:"" passthrough:"all" help:"Convert a Hugging Face checkpoint into a Core ML package or OpenVINO IR by hand (was hwnpu-convert)."`
	Generate   tool           `cmd:"" passthrough:"all" help:"Write text with a causal language model converted by convert (was hwnpu-generate)."`
	Formats    tool           `cmd:"" passthrough:"all" help:"Compare the weight formats of a chat model on the NPU: agreement, logits, perplexity, time (was hwnpu-formats)."`
	Irdiff     tool           `cmd:"" passthrough:"all" help:"Lower Core ML packages to OpenVINO IR and compare the CPU interpreter with OpenVINO's CPU on random inputs (Intel's path without its NPU)."`
	Info       tool           `cmd:"" passthrough:"all" help:"List the NPUs hwnpu can use, and where a model's operations would run (was hwnpuinfo)."`
	Probe      tool           `cmd:"" passthrough:"all" help:"Measure this machine's NPU with generated LLM-shaped models: speed, size limit, KV cache (was npuprobe)."`
	Powerbench tool           `cmd:"" passthrough:"all" help:"Measure the power Core ML models draw with macOS powermetrics (was powerbench)."`
	Prebuild   tool           `cmd:"" passthrough:"all" help:"Compile Core AI programs ahead of time with coreai-build, X.ARCH.aimodelc beside each X.aimodel of an export (macOS, Xcode 27's Metal Toolchain)."`
	Report     tool           `cmd:"" passthrough:"all" help:"Check every NPU with bundled test models and print a report for an issue (was report)."`
	Mxfp4      tool           `cmd:"" name:"mxfp4" passthrough:"all" help:"Write a Qwen3.5-MoE checkpoint (Qwen3.6-35B-A3B) with its experts in MXFP4, read a layer at a time from Hugging Face or a directory, for the experts the CPU runs."`
	Xclbin     tool           `cmd:"" passthrough:"all" help:"Package an mlir-aie design into an xclbin without XRT, in place of xclbinutil (was hwnpu-xclbin)."`
	Models     modelscmd.Cmd  `cmd:"" help:"The models' conversions, MODEL/TARGET (amd-xdna2, coreai, coreai-ARCH), the same commands on every OS: fetch them here, package them (made here unless a source has them, and put in the sources), push what was made by hand, put them on Hugging Face (hf)."`
	Sources    sourcescmd.Cmd `cmd:"" help:"The upkeep of the sources (HWNPU_MODEL_SOURCE): which are in effect, what they hold, putting folders made by hand in them, moving locks to a new commit, checking and removing entries."`
	Daemon     daemoncmd.Cmd  `cmd:"" help:"Build hwnpud from this checkout and serve on a socket of its own, beside the installed one: the commands are its clients, so a worktree's code is tried through it (hwnpu-dev daemon -- go run ./cmd/hwnpu-chat chat -m lfm3 hi)."`
}

var mains = map[string]func(){
	"check": check.Main, "convert": convert.Main, "generate": generate.Main, "formats": formats.Main,
	"info": info.Main, "irdiff": irdiff.Main, "probe": probe.Main, "powerbench": powerbench.Main, "prebuild": prebuild.Main, "report": report.Main,
	"xclbin": xclbin.Main, "mxfp4": mxfp4.Main,
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
		kong.Vars(devopt.Vars),
		kong.UsageOnError())
	err := k.Run()
	shared.Wait(os.Stderr) // what was made here goes in the sources written back to
	k.FatalIfErrorf(err)
	console.WriteProfile(os.Stderr) // HWNPU_PROFILE's downloads and loads
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
