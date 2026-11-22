package sourcescmd

import (
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// gpt-oss-20b has two exports for AMD's NPU: the 16 GB machines' (16 experts a layer placed)
// and the 24 GB ones' (all of them); the others one.
func TestAMDForms(t *testing.T) {
	m, ok := chat.Lookup("gpt-oss-20b")
	if !ok {
		t.Fatal("no gpt-oss-20b")
	}
	forms := AMDForms(m)
	if len(forms) != 2 {
		t.Fatalf("%d forms, want 2", len(forms))
	}
	if !strings.Contains(strings.Join(forms[0].AMDExport, " "), "--expert-places") ||
		strings.Contains(strings.Join(forms[1].AMDExport, " "), "--expert-places") {
		t.Errorf("forms %v and %v", forms[0].AMDExport, forms[1].AMDExport)
	}
	if m, ok := chat.Lookup("LFM3-1.7B"); !ok || len(AMDForms(m)) != 1 {
		t.Errorf("LFM3-1.7B: %d forms, want 1", len(AMDForms(m)))
	}
}
