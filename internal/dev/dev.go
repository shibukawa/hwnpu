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
	"ailower.divide": "lowers x / c as a divide (not times 1 / c), which Core AI's Neural Engine miscomputes",
	"ailower.notes":  "writes ailower's notes of each operation lowered",
	"ailower.trace":  "traces ailower's lowering",
	"ov.device":      "DEVICE: the OpenVINO device (NPU, CPU, GPU)",
	"ov.precision":   "PRECISION: OpenVINO's inference precision hint",
	"coreai.debug":   "logs Core AI's calls",
	"coreml.trace":   "traces Core ML's graph calls",
	// decoding
	"draft.policy": "0 checks the lookup's drafts at every step (no policy)",
	"draft.ratio":  "R: the steps a draft's check must save to be kept on",
	"draft.trace":  "PATH: writes the drafts' checks",
	"mtp.drafts":   "N: the multi-token prediction's drafts a step",
	"iron.drafts":  "0 runs IRON's chat models without drafts",
	"iron.sync":    "waits for each IRON command",
	"iron.barrier": "1 or 0: a step barrier in IRON's runs over the manifest's choice (for measuring)",
	"iron.timing":  "times IRON's commands",
	"cpu.f16":      "rounds the CPU interpreter's values to fp16, to find a program's overflows on the NPU",
	"klein.decode": "FLUX.2 klein decodes --noise as final latents (a check of the VAE alone)",
	// mixtures of experts
	"moe.time":     "the experts' times",
	"moe.prefetch": "N of the next layer's guessed experts brought in (0: none)",
	"moe.row":      "0 runs a token's experts as the rows' three runs",
	"moe.usage":    "PATH: writes the experts a run used (tools/moeusage)",
	"moe.trace":    "PATH: writes the experts each token ran (tools/moeusage/place.py)",
	"moe.direct":   "1 or 0: IRON's banks read the experts directly or not",
	"moe.prefault": "0: a host's experts not faulted in ahead (they are when they fit in memory); wait: every page in before the first run",
	"moe.check":    "checks the experts' results against the CPU's",
	"mxfp4.bench":  "LAYERS: the layers TestTokenTime times",
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
