package mem

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDRMMemory(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string]string{
		"3": "pos:\t0\nflags:\t02\n",                                                               // not a DRM device
		"4": "drm-driver:\tamdxdna_accel_driver\ndrm-client-id:\t7\ndrm-total-memory:\t2048 KiB\n", // the NPU
		"5": "drm-driver:\tamdxdna_accel_driver\ndrm-client-id:\t7\ndrm-total-memory:\t2048 KiB\n", // the same client again
		"6": "drm-client-id:\t9\ndrm-total-memory:\t3 MiB\n",
	} {
		os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644)
	}
	n, ok := drmMemory(dir)
	if !ok || n != 2048<<10+3<<20 {
		t.Fatalf("drmMemory = %d, %v", n, ok)
	}
	if _, ok := drmMemory(t.TempDir()); ok {
		t.Fatal("ok without a DRM device")
	}
}
