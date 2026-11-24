package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/coreai/ailower"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// opTypes prints the MIL operation types of the packages and how often each occurs.
func opTypes(pkgs []string) error {
	count := map[string]int{}
	for _, p := range pkgs {
		m, err := mlpackage.ReadModel(p)
		if err != nil {
			return err
		}
		fn := m.GetMlProgram().GetFunctions()["main"]
		for _, op := range fn.GetBlockSpecializations()[fn.GetOpset()].GetOperations() {
			count[op.GetType()]++
		}
	}
	var names []string
	for k := range count {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		fmt.Printf("%6d %s\n", count[k], k)
	}
	return nil
}

// listOps prints the first n non-constant operations of a package: type, output, shape.
func listOps(pkg string, n int) error {
	m, err := mlpackage.ReadModel(pkg)
	if err != nil {
		return err
	}
	fn := m.GetMlProgram().GetFunctions()["main"]
	for i, op := range fn.GetBlockSpecializations()[fn.GetOpset()].GetOperations() {
		if i >= n {
			break
		}
		o := op.GetOutputs()[0]
		var dims []uint64
		for _, d := range o.GetType().GetTensorType().GetDimensions() {
			dims = append(dims, d.GetConstant().GetSize())
		}
		var ins []string
		for k, a := range op.GetInputs() {
			for _, b := range a.GetArguments() {
				if b.GetName() != "" {
					ins = append(ins, k+"="+b.GetName())
				}
			}
		}
		sort.Strings(ins)
		fmt.Printf("%4d %-32s %-28s %v %v\n", i, op.GetType(), o.GetName(), dims, ins)
	}
	return nil
}

// lint lowers each package without its weights and prints what the Neural Engine would get
// wrong for some values in it (ailower.LintPackage, docs/design/apple-neural-engine.md), counting each note once a
// package.
func lint(pkgs []string) error {
	for _, p := range pkgs {
		lintOne := ailower.LintPackage
		if strings.HasSuffix(strings.TrimSuffix(p, "/"), ".aimodel") {
			lintOne = lintProgram
		}
		notes, err := lintOne(p)
		if err != nil {
			fmt.Printf("%s: FAIL %v\n", p, err)
			continue
		}
		seen := map[string]int{}
		var order []string
		for _, n := range notes {
			if seen[n] == 0 {
				order = append(order, n)
			}
			seen[n]++
		}
		for _, n := range order {
			fmt.Printf("%s: %s\n", p, n)
		}
	}
	return nil
}
