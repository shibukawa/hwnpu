package main

import (
	"fmt"
	"sort"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
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
