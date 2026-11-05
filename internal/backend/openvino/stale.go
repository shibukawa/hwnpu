package openvino

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// dirPrefix starts the name of a device's directory of shared buffers, followed by the
// process id of the hwnpu process that made it: hwnpu-openvino-PID-RANDOM.
const dirPrefix = "hwnpu-openvino-"

// makeDir makes the directory of a device's shared buffers in shmDir, named by this process.
func makeDir() (string, error) {
	return os.MkdirTemp(shmDir(), fmt.Sprintf("%s%d-", dirPrefix, os.Getpid()))
}

// removeStale removes the directories of shared buffers left by processes that have ended
// without closing their device (killed, interrupted). In /dev/shm their files are memory that
// stays taken until removed: 8 GiB built up over a day of tests on a Linux machine.
func removeStale() {
	dir := shmDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		rest, ok := strings.CutPrefix(e.Name(), dirPrefix)
		if !ok || !e.IsDir() {
			continue
		}
		s, _, ok := strings.Cut(rest, "-")
		if !ok {
			continue
		}
		pid, err := strconv.Atoi(s)
		if err != nil || pid == os.Getpid() || processAlive(pid) {
			continue
		}
		os.RemoveAll(filepath.Join(dir, e.Name()))
	}
}
