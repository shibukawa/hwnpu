// Package dev reads the developers' switches: experiments, measurements and traces, none of them a
// setting of the user's. They are items of one variable, separated by commas:
//
//	HWNPU_DEV=moe.time,moe.prefetch=0,ailower.trace
//
// An item is a key (on) or KEY=VALUE; a key's group goes before a dot (moe., iron., coreai.).
// Keys lists them all, with what they do; an item of a key it does not know is told once on
// standard error. The variable is read at each call, so that a test's t.Setenv takes.
//
// They change what a process does for the one who set them: hwnpud does not take a client's
// (docs/design/hwnpud.md), and a run with them is a run of the developer's own process
// (HWNPU_DAEMON=off).
package dev

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// Env is the variable.
const Env = "HWNPU_DEV"

// Keys are the keys an item may have, with what each does.
var Keys = map[string]string{
	// conversions
	"rewrite":        "OP:WAY[+OP:WAY]: this Mac's programs write OP so, over model-recipes.yaml's rewrite (sin:series, sin:native), in an export of their name (recipe.RewritesFor)",
	"ailower.cut":    "N: ailower lowers only the first N operations, the last one's value the output",
	"ailower.divide": "lowers x / c as a divide (not times 1 / c), which Core AI's Neural Engine miscomputes (docs/handoff-coreai-vision.md 3)",
	"ailower.notes":  "writes ailower's notes of each operation lowered",
	"ailower.trace":  "traces ailower's lowering",
	"ov.device":      "DEVICE: the OpenVINO device (NPU, CPU, GPU)",
	"ov.precision":   "PRECISION: OpenVINO's inference precision hint",
	"coreai.debug":   "tells why a prebuilt program's record was not written and why a cached program was compiled again",
	"coreai.units":   "cpu: runs Core AI's programs on its CPU, off the Neural Engine (to tell a lowering's errors from the Neural Engine's)",
	"coreml.units":   "cpu or gpu: loads Core ML's models there instead of the Neural Engine (hwnpu-dev powerbench's baselines)",
	"coreai.dump":    "DIR: writes every call's outputs to DIR (a folder a call), to compare two runs",
	"coreml.trace":   "traces Core ML's graph calls",
	// decoding
	"mtp.drafts":        "N: the multi-token prediction's drafts a step",
	"mtp.draftscan":     "0: Core AI's qwen35-mtp scores a part of the vocabulary on the Neural Engine (dv2) in place of its hidden state scored on the CPU over all of it (DraftScan)",
	"store.kvq":         "4, 5 or 6: the prompt store keeps its rows in that many bits a value (not exact; 0, the default, as they are)",
	"iron.drafts":       "0 runs IRON's chat models without drafts",
	"iron.lookup.cost":  "R: what a check of the prompt lookup's drafts costs on IRON, in decode steps (1.8): a check is made when the tokens it is expected to bring are more",
	"iron.lookup.trace": "PATH: writes IRON's choice of checking the prompt lookup's drafts, a line a call (tools/amd/draft_analyze.py)",
	"iron.sync":         "waits for each IRON command",
	"iron.barrier":      "1 or 0: a step barrier in IRON's runs over the manifest's choice (for measuring)",
	"iron.timing":       "times IRON's commands",
	"iron.moe.check":    "checks IRON's experts' results against the CPU's",
	"iron.moe.direct":   "1 or 0: IRON's banks read the experts straight from the file or not (by the memory when not set)",
	"cpu.f16":           "rounds the CPU interpreter's values to fp16, to find a program's overflows on the NPU",
	"klein.decode":      "FLUX.2 klein decodes --noise as final latents (a check of the VAE alone)",
	// mixtures of experts
	"moe.time":     "the experts' times",
	"moe.prefetch": "N of the next layer's guessed experts brought in (0: none)",
	"moe.usage":    "PATH: writes the experts a run used (tools/moeusage)",
	"moe.trace":    "PATH: writes the experts each token ran (tools/moeusage/place.py)",
	"moe.prefault":         "wait: a host's experts wired with every page in before the first run (for measuring)",
	"moe.prefill":          "0 or cpu: keeps a host's multi-row prefill experts on the CPU instead of offloading hot ones to the Neural Engine",
	"moe.prefill-min-rows": "N: minimum rows an expert takes in a prefill batch to run on the Neural Engine (24)",
	"moe.min-ratio":        "R: skips trailing routed experts whose weight is below R times the top expert's",
	"mxfp4.bench":          "LAYERS: the layers TestTokenTime times; cache or wake: runs that benchmark (internal/mxfp4's tests)",
}

var warned sync.Map

// Get is the value of key's item, and whether there is one (the last when several).
func Get(key string) (string, bool) {
	var v string
	found := false
	for _, it := range strings.Split(os.Getenv(Env), ",") {
		it = strings.TrimSpace(it)
		if it == "" {
			continue
		}
		k, val, _ := strings.Cut(it, "=")
		k = strings.TrimSpace(k)
		if _, known := Keys[k]; !known {
			if _, done := warned.LoadOrStore(k, true); !done {
				fmt.Fprintf(os.Stderr, "%s: %q is not a key hwnpu knows (internal/dev.Keys)\n", Env, k)
			}
			continue
		}
		if k == key {
			v, found = strings.TrimSpace(val), true
		}
	}
	return v, found
}

// Value is the value of key's item, "" when there is none.
func Value(key string) string {
	v, _ := Get(key)
	return v
}

// On is whether key's item is there with no value or a value other than 0, off, false and no.
func On(key string) bool {
	v, ok := Get(key)
	return ok && !Off(v)
}

// IsOff is whether key's item is there with the value 0, off, false or no.
func IsOff(key string) bool {
	v, ok := Get(key)
	return ok && v != "" && Off(v)
}

// Off is whether v is 0, off, false or no.
func Off(v string) bool {
	return slices.Contains([]string{"0", "off", "false", "no"}, strings.ToLower(v))
}

// Int is key's value as an integer, def when there is none or it is not one.
func Int(key string, def int) int {
	if n, err := strconv.Atoi(Value(key)); err == nil {
		return n
	}
	return def
}
