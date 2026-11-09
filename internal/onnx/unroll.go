package onnx

import (
	"fmt"
	"maps"
)

// Unroll replaces each While whose condition depends only on loop variables that start as
// constants (a counter) by copies of its body, one an iteration: the loop variables computed
// from constants alone stay constants (the counter of each copy, so that a copy's slices of
// stacked weights fold to that layer's weights when lowered), the others are the copies'
// values. Values of the copies are prefixed with the node's name and the iteration.
func (g *Graph) Unroll() error {
	var nodes []*Node
	for _, n := range g.Nodes {
		if n.Op != "While" {
			nodes = append(nodes, n)
			continue
		}
		copies, err := g.unroll(n)
		if err != nil {
			return fmt.Errorf("onnx: unrolling %s: %w", n.Name, err)
		}
		nodes = append(nodes, copies...)
	}
	g.Nodes = nodes
	return nil
}

func (g *Graph) unroll(n *Node) ([]*Node, error) {
	if len(n.Graphs) != 2 {
		return nil, fmt.Errorf("a loop without its condition and body")
	}
	cond, body := n.Graphs[0], n.Graphs[1]
	if len(body.Inputs) != len(n.In) || len(body.Outputs) != len(n.In) || len(n.Out) != len(n.In) {
		return nil, fmt.Errorf("loop variables: %d in, %d body inputs, %d body outputs", len(n.In), len(body.Inputs), len(body.Outputs))
	}
	// each loop variable: a constant, or a value of the graph
	consts := make([]*Tensor, len(n.In))
	names := make([]string, len(n.In))
	for i, in := range n.In {
		consts[i], names[i] = g.Init[in], in
	}
	var out []*Node
	for it := 0; ; it++ {
		if it > 4096 {
			return nil, fmt.Errorf("more than 4096 iterations")
		}
		given := map[string]*Tensor{}
		for i, vi := range cond.Inputs {
			if consts[i] != nil {
				given[vi.Name] = consts[i]
			}
		}
		c, err := cond.Eval(given, []string{cond.Outputs[0].Name})
		if err != nil {
			return nil, fmt.Errorf("the condition does not depend on constants alone: %w", err)
		}
		if c[cond.Outputs[0].Name].F[0] == 0 {
			break
		}
		prefix := fmt.Sprintf("%s_it%d/", n.Name, it)
		rename := map[string]string{}
		for i, vi := range body.Inputs {
			if consts[i] != nil {
				k := prefix + vi.Name
				g.Init[k] = consts[i]
				rename[vi.Name] = k
			} else {
				rename[vi.Name] = names[i]
			}
		}
		for k, t := range body.Init {
			if _, input := rename[k]; !input {
				g.Init[prefix+k] = t
				rename[k] = prefix + k
			}
		}
		to := func(v string) string {
			if v == "" {
				return ""
			}
			if r, ok := rename[v]; ok {
				return r
			}
			return prefix + v
		}
		for _, bn := range body.Nodes {
			cp := &Node{Op: bn.Op, Name: prefix + bn.Name, Domain: bn.Domain, Attrs: maps.Clone(bn.Attrs), Graphs: bn.Graphs}
			for _, v := range bn.In {
				cp.In = append(cp.In, to(v))
			}
			for _, v := range bn.Out {
				cp.Out = append(cp.Out, to(v))
			}
			out = append(out, cp)
		}
		// the next loop variables: constants where the body computes them from constants
		given = map[string]*Tensor{}
		for i, vi := range body.Inputs {
			if consts[i] != nil {
				given[vi.Name] = consts[i]
			}
		}
		next := make([]*Tensor, len(n.In))
		for i, vo := range body.Outputs {
			if body.dependsOnly(vo.Name, given) {
				v, err := body.Eval(given, []string{vo.Name})
				if err != nil {
					return nil, err
				}
				next[i] = v[vo.Name]
			}
			names[i] = to(vo.Name)
		}
		consts = next
	}
	for i, o := range n.Out {
		if consts[i] != nil {
			g.Init[o] = consts[i]
		} else {
			out = append(out, &Node{Op: "Identity", Name: n.Name + "_out", In: []string{names[i]}, Out: []string{o}, Attrs: map[string]Attr{}})
		}
	}
	return out, nil
}

// dependsOnly reports whether value v is computed from the given values and constants alone.
func (g *Graph) dependsOnly(v string, given map[string]*Tensor) bool {
	made := map[string]*Node{}
	for _, n := range g.Nodes {
		for _, o := range n.Out {
			made[o] = n
		}
	}
	seen := map[string]bool{}
	var ok func(v string) bool
	ok = func(v string) bool {
		if v == "" || given[v] != nil || g.Init[v] != nil {
			return true
		}
		if r, done := seen[v]; done {
			return r
		}
		seen[v] = false
		n := made[v]
		if n == nil {
			return false
		}
		for _, in := range n.In {
			if !ok(in) {
				return false
			}
		}
		seen[v] = true
		return true
	}
	return ok(v)
}
