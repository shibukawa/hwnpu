// Command hwnpuinfo lists the NPUs hwnpu can use, and optionally checks where a model's
// operations would run.
//
//	hwnpuinfo                 list devices
//	hwnpuinfo -model PATH     also compile PATH on the first device and print its placement
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/shibukawa/hwnpu"
)

func main() {
	model := flag.String("model", "", "compile this model on the first device and report its placement")
	flag.Parse()
	ctx := context.Background()

	infos, err := hwnpu.Enumerate(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning:", err)
	}
	if len(infos) == 0 {
		fmt.Println("no NPU found")
		os.Exit(1)
	}
	for _, d := range infos {
		fmt.Printf("[%d] %s\n", d.Index, d.Name)
		fmt.Printf("    vendor:   %s\n    backend:  %s\n    caps:     %s\n", d.Vendor, d.Backend, d.Caps)
		if d.Path != "" {
			fmt.Printf("    path:     %s\n", d.Path)
		}
		if d.Driver != "" {
			fmt.Printf("    driver:   %s\n", d.Driver)
		}
		if d.Firmware != "" {
			fmt.Printf("    firmware: %s\n", d.Firmware)
		}
	}
	if *model == "" {
		return
	}

	dev, err := hwnpu.OpenDevice(ctx, infos[0], hwnpu.WithPlacement(hwnpu.PlacementPrefer))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer dev.Close()
	gr, err := dev.Graph()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: *model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer g.Close()
	fmt.Printf("\nmodel %s\n", *model)
	for _, a := range g.Args() {
		fmt.Printf("    %s\n", a)
	}
	rep := g.Placement()
	switch {
	case !rep.Verified:
		fmt.Printf("placement: not verified (%s)\n", rep.Reason)
	case rep.OnNPU():
		fmt.Printf("placement: all %d operations on the NPU (%s)\n", rep.NPUOps, rep.Method)
	default:
		fmt.Printf("placement: %d on the NPU, %d elsewhere (%s)\n", rep.NPUOps, len(rep.Ops), rep.Method)
		for _, op := range rep.Ops {
			fmt.Printf("    %s %s -> %s\n", op.Type, op.Name, op.Device)
		}
	}
	if !rep.OnNPU() {
		os.Exit(2)
	}
}
