// Command hwnpu-dev xclbin packages an mlir-aie (IRON) design into an xclbin without XRT. It takes
// the subset of xclbinutil arguments that aiecc passes, so aiecc can call it in place of
// xclbinutil:
//
//	AIE_XCLBINUTIL=/path/to/hwnpu-xclbin python3 design.py --xclbin-path final.xclbin --insts-path insts.bin
//
// The xclbin holds what hwnpu reads (kernel metadata, compute units and the AIE partition with
// its PDIs). It is not meant for XRT, which also expects MEM_TOPOLOGY and other sections.
package xclbin

import (
	"fmt"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu/internal/xclbin"
)

func Main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-dev xclbin:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	var kernels, partition, output string
	for i := 0; i < len(args); i++ {
		next := func() (string, error) {
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s needs a value", args[i])
			}
			i++
			return args[i], nil
		}
		var err error
		switch args[i] {
		case "--add-kernel":
			kernels, err = next()
		case "--add-replace-section", "--add-section":
			var v string
			if v, err = next(); err == nil {
				kind, path, ok := strings.Cut(v, ":JSON:")
				switch {
				case !ok:
					err = fmt.Errorf("unsupported section %q (only KIND:JSON:FILE)", v)
				case kind == "AIE_PARTITION":
					partition = path
				case kind == "MEM_TOPOLOGY":
					// The NPU does not use it.
				default:
					err = fmt.Errorf("unsupported section %s", kind)
				}
			}
		case "--output", "-o":
			output, err = next()
		case "--force", "-f", "--quiet", "-q":
		default:
			err = fmt.Errorf("unsupported argument %q", args[i])
		}
		if err != nil {
			return err
		}
	}
	if kernels == "" || partition == "" || output == "" {
		return fmt.Errorf("usage: hwnpu-dev xclbin --add-kernel kernels.json --add-replace-section AIE_PARTITION:JSON:partition.json --output final.xclbin")
	}
	kj, err := os.ReadFile(kernels)
	if err != nil {
		return err
	}
	pj, err := os.ReadFile(partition)
	if err != nil {
		return err
	}
	spec, err := xclbin.SpecFromAiecc(kj, pj, os.ReadFile)
	if err != nil {
		return err
	}
	data := xclbin.Build(spec)
	if _, err := xclbin.Parse(data); err != nil {
		return fmt.Errorf("the packaged xclbin does not parse: %w", err)
	}
	return os.WriteFile(output, data, 0o644)
}
