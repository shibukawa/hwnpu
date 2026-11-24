//go:build !darwin

package main

import (
	"errors"

	"github.com/shibukawa/bonsaiai/internal/llm"
)

func runANE(llm.Model, llm.Weights, llm.Options, []float32, string, int) ([]float32, error) {
	return nil, errors.New("-ane needs Core AI (macOS 27 on Apple silicon)")
}
