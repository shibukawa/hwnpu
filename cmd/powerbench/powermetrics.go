package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Rails reports average power in milliwatts. Combined is what powermetrics calls
// combined_power (CPU + GPU + ANE); when it is missing the three rails are added.
type Rails struct {
	CPU      float64 `json:"cpu_mw"`
	GPU      float64 `json:"gpu_mw"`
	ANE      float64 `json:"ane_mw"`
	Combined float64 `json:"combined_mw"`
}

func (r Rails) sub(o Rails) Rails {
	return Rails{r.CPU - o.CPU, r.GPU - o.GPU, r.ANE - o.ANE, r.Combined - o.Combined}
}

func (r Rails) mean(o Rails) Rails {
	return Rails{(r.CPU + o.CPU) / 2, (r.GPU + o.GPU) / 2, (r.ANE + o.ANE) / 2, (r.Combined + o.Combined) / 2}
}

// sampler runs powermetrics, as root directly or through sudo.
type sampler struct {
	interval time.Duration
	samplers string
	rawDir   string
	useSudo  bool
	command  string // HWNPU_POWERMETRICS: a stand-in for powermetrics, used for testing
}

// authorize asks for the sudo password once, before any measurement, so the prompt does not
// end up inside a measured phase. Later calls use sudo -n with the cached credentials.
func (s *sampler) authorize() error {
	if s.command = os.Getenv("HWNPU_POWERMETRICS"); s.command != "" || os.Geteuid() == 0 {
		return nil
	}
	s.useSudo = true
	fmt.Fprintln(os.Stderr, "powermetrics needs root: sudo asks for your password once.")
	cmd := exec.Command("sudo", "-v")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stderr, os.Stderr
	if err := cmd.Run(); err != nil {
		return err
	}
	// Loading a large model can take longer than sudo's credential timeout (5 minutes by
	// default), so keep the credentials fresh for the whole run.
	go func() {
		for range time.Tick(time.Minute) {
			exec.Command("sudo", "-n", "-v").Run()
		}
	}()
	return nil
}

// sample takes n samples and returns them all; the caller drops the first one.
func (s *sampler) sample(ctx context.Context, n int, rawName string) ([]map[string]float64, error) {
	args := []string{"--samplers", s.samplers, "-f", "plist", "-i", strconv.Itoa(int(s.interval / time.Millisecond)), "-n", strconv.Itoa(n)}
	name := "powermetrics"
	if s.command != "" {
		name = s.command
	} else if s.useSudo {
		args = append([]string{"-n", name}, args...)
		name = "sudo"
	}
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if s.rawDir != "" {
		os.WriteFile(s.rawDir+"/"+rawName+".plist", stdout.Bytes(), 0o644)
	}
	if err != nil {
		return nil, fmt.Errorf("%s %s: %v: %s", name, strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return parseSamples(stdout.Bytes())
}

// average returns the power averaged over the samples, weighted by each sample's duration.
func average(samples []map[string]float64) (Rails, error) {
	if len(samples) == 0 {
		return Rails{}, fmt.Errorf("powermetrics returned no samples")
	}
	var sum Rails
	var total float64
	found := false
	for _, s := range samples {
		w := s["elapsed_ns"]
		if w <= 0 {
			w = 1
		}
		cpu, okC := lookup(s, "cpu_power")
		gpu, okG := lookup(s, "gpu_power")
		ane, okA := lookup(s, "ane_power")
		comb, ok := lookup(s, "combined_power")
		if !ok {
			comb = cpu + gpu + ane
		}
		found = found || okC || okG || okA || ok
		sum.CPU += cpu * w
		sum.GPU += gpu * w
		sum.ANE += ane * w
		sum.Combined += comb * w
		total += w
	}
	if !found {
		keys := make([]string, 0, len(samples[0]))
		for k := range samples[0] {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		return Rails{}, fmt.Errorf("no *_power values in the samples (keys: %s); rerun with -raw to keep the output", strings.Join(keys, ", "))
	}
	return Rails{sum.CPU / total, sum.GPU / total, sum.ANE / total, sum.Combined / total}, nil
}

// lookup prefers processor.<name> and falls back to any key ending in <name>, since the
// layout differs between macOS releases and chips.
func lookup(s map[string]float64, name string) (float64, bool) {
	if v, ok := s["processor."+name]; ok {
		return v, true
	}
	if v, ok := s[name]; ok {
		return v, true
	}
	keys := make([]string, 0, 1)
	for k := range s {
		if strings.HasSuffix(k, "."+name) {
			keys = append(keys, k)
		}
	}
	if len(keys) == 0 {
		return 0, false
	}
	slices.Sort(keys)
	return s[keys[0]], true
}
