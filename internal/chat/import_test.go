package chat

import "testing"

// TestLFSPointer checks the git object of an LFS pointer against one the hub lists
// (black-forest-labs/FLUX.1-schnell's tokenizer_2/spiece.model: its SHA-256 is masked without a
// token, the pointer's object is not).
func TestLFSPointer(t *testing.T) {
	const sum = "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"
	h := HubFile{LFS: true, GitSHA1: "317a5ccbde45300f5d1d970d4d449af2108b147e", Size: 791656}
	if !h.matches(sum, "", 791656) {
		t.Errorf("pointer %s, want %s", lfsPointerSHA1(sum, 791656), h.GitSHA1)
	}
	if h.matches("0"+sum[1:], "", 791656) {
		t.Error("another file matched")
	}
}
