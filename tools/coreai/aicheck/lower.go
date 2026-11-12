package main

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
)

// lowerTo lowers each package and writes OUT/NAME.aimodel.
func lowerTo(out string, pkgs []string) error {
	for _, p := range pkgs {
		prog, err := ailower.LowerPackage(p)
		if err != nil {
			return err
		}
		dst := out + "/" + baseName(p) + ".aimodel"
		if err := prog.WriteAsset(dst); err != nil {
			return err
		}
		fmt.Printf("%s: %d bytes, %d inputs, %d outputs\n", dst, len(prog.Bytecode), len(prog.Inputs), len(prog.Outputs))
	}
	return nil
}

func baseName(p string) string {
	for len(p) > 0 && p[len(p)-1] == '/' {
		p = p[:len(p)-1]
	}
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			p = p[i+1:]
			break
		}
	}
	if n := len(p) - len(".mlpackage"); n > 0 && p[n:] == ".mlpackage" {
		p = p[:n]
	}
	return p
}
