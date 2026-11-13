package coreai

// Programs of one model converted for different shapes (8 and 128 tokens per call, KV caches of
// different lengths, an embedder's 128 and 1024 tokens) have weights of the same bytes: the same
// weight file (which llm.ShareWeights clones on disk), or the same blobs besides a few that depend
// on the shape (the embedder's). Lowered one by one, each .aimodel and each of Core AI's cache
// entries holds its own copy. A package is lowered instead with the packages beside it that have
// its largest weights into one .aimodel with a function each (ailower.LowerFunctions, design.md
// 16.174), which writes a blob of the same bytes once, so that the asset and Core AI's cache
// entry hold the weights once.
//
// The compile cache keeps, for each package (by its contents, as before), a record of the asset
// and the function it runs as (KEY.aifn), so that a later load hashes only its own package. A
// package without such siblings is lowered alone as before and keeps its entry. An entry of a
// package lowered alone before is replaced by the group's (with Core AI's cache entry of it).

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
)

// groupMin is the least weight file grouped: the programs without weights (attention over a
// chunk) gain nothing.
const groupMin = 1 << 20

// groupBlobs is how many of a package's largest weight blobs a sibling must have too (sameWeights).
const groupBlobs = 3

// recordSuffix names a package's record in the compile cache (record).
const recordSuffix = ".aifn"

// record is what a package of the compile cache runs as: the asset (an entry's name, KEY.aimodel)
// and its function, main for a package lowered alone, by the rule of groupRule.
type record struct {
	Asset    string `json:"asset"`
	Function string `json:"function"`
	Rule     int    `json:"rule"`
}

// groupRule is the version of how siblings are found (siblings): a record of another rule is
// looked at again, so that a package found alone before can join a group.
const groupRule = 2

// The lowering, replaced by tests.
var (
	lowerOne = func(pkg, dst string) error {
		p, err := ailower.LowerPackage(pkg)
		if err != nil {
			return err
		}
		return p.WriteAsset(dst)
	}
	lowerMany = func(pkgs, names []string, dst string) error {
		p, err := ailower.LowerFunctions(pkgs, names)
		if err != nil {
			return err
		}
		return p.WriteAsset(dst)
	}
	// forget deletes Core AI's cache entries of the asset at path (before it is removed).
	forget = func(path string) {
		if load() == nil && caiForget != nil {
			caiForget(cString(path))
		}
	}
)

// singleKey is the cache key of a package by its contents (cache.TreeDigest): its asset when
// lowered alone, and its record.
func singleKey(digest []byte) string {
	return cache.Key([]byte("hwnpu-coreai-v1"), digest, []byte(ailower.Producer))
}

func weightFile(pkg string) string {
	return filepath.Join(pkg, "Data", "com.apple.CoreML", "weights", "weight.bin")
}

// digests remembers the packages hashed in this process (a package is hashed again when a file
// of it changes size or time).
var digests struct {
	sync.Mutex
	m map[string]digestMemo
}

type digestMemo struct {
	stamp string
	sum   []byte
}

func stampOf(pkg string) string {
	var b strings.Builder
	filepath.Walk(pkg, func(p string, fi os.FileInfo, err error) error {
		if err == nil && fi.Mode().IsRegular() {
			fmt.Fprintf(&b, "%s %d %d\n", p, fi.Size(), fi.ModTime().UnixNano())
		}
		return nil
	})
	return b.String()
}

func digestOf(pkg string) ([]byte, error) {
	if r, err := filepath.EvalSymlinks(pkg); err == nil {
		pkg = r
	}
	st := stampOf(pkg)
	digests.Lock()
	if m, ok := digests.m[pkg]; ok && m.stamp == st {
		digests.Unlock()
		return m.sum, nil
	}
	digests.Unlock()
	sum, err := cache.TreeDigest(pkg)
	if err != nil {
		return nil, err
	}
	digests.Lock()
	if digests.m == nil {
		digests.m = map[string]digestMemo{}
	}
	digests.m[pkg] = digestMemo{st, sum}
	digests.Unlock()
	return sum, nil
}

// readRecord returns the record of key whose asset is in the cache, marking both used.
func readRecord(dir, key string) (asset string, r record, ok bool) {
	p, ok := cache.Lookup(dir, key, recordSuffix)
	if !ok {
		return "", r, false
	}
	b, err := os.ReadFile(p)
	if err != nil || json.Unmarshal(b, &r) != nil || r.Rule != groupRule || r.Function == "" || !strings.HasSuffix(r.Asset, ".aimodel") {
		return "", r, false
	}
	asset, ok = cache.Lookup(dir, strings.TrimSuffix(r.Asset, ".aimodel"), ".aimodel")
	return asset, r, ok
}

func writeRecord(dir, key string, r record) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	tmp := cache.Path(dir, key, recordSuffix) + fmt.Sprintf(".%d.tmp", os.Getpid())
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, cache.Path(dir, key, recordSuffix))
}

type member struct {
	path string
	sum  []byte
}

// blob is one weight blob of a weight file (mlpackage.BlobWriter's format): where its data is.
type blob struct{ off, size int64 }

// blobsOf lists the blobs of a weight file from their metadata records, largest first.
func blobsOf(path string) ([]blob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var hdr [64]byte
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		return nil, err
	}
	n := int(binary.LittleEndian.Uint32(hdr[:]))
	var out []blob
	pos := int64(64)
	for i := 0; i < n; i++ {
		var meta [64]byte
		if _, err := f.ReadAt(meta[:], pos); err != nil {
			return nil, err
		}
		if binary.LittleEndian.Uint32(meta[:]) != 0xDEADBEEF {
			return nil, fmt.Errorf("coreai: %s: no blob at %d", path, pos)
		}
		b := blob{off: int64(binary.LittleEndian.Uint64(meta[16:])), size: int64(binary.LittleEndian.Uint64(meta[8:]))}
		if b.off < pos || b.off+b.size > fi.Size() {
			return nil, fmt.Errorf("coreai: %s: a blob out of the file at %d", path, pos)
		}
		out = append(out, b)
		pos = (b.off + b.size + 63) &^ 63
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].size > out[j].size })
	return out, nil
}

func hashBlob(path string, b blob) ([32]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return [32]byte{}, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(f, b.off, b.size)); err != nil {
		return [32]byte{}, err
	}
	var sum [32]byte
	h.Sum(sum[:0])
	return sum, nil
}

// sameWeights reports whether the weight file other has the groupBlobs largest blobs of w (by
// their bytes): two programs of one model. Only blobs of the sizes looked for are read.
func sameWeights(w string, mine []blob, other string) (bool, error) {
	theirs, err := blobsOf(other)
	if err != nil {
		return false, err
	}
	for _, b := range mine[:min(groupBlobs, len(mine))] {
		sum, err := hashBlob(w, b)
		if err != nil {
			return false, err
		}
		found := false
		for _, t := range theirs {
			if t.size != b.size {
				continue
			}
			tsum, err := hashBlob(other, t)
			if err != nil {
				return false, err
			}
			if tsum == sum {
				found = true
				break
			}
		}
		if !found {
			return false, nil
		}
	}
	return true, nil
}

// siblings are pkg and the packages beside it (in its directory) with its largest weights
// (sameWeights) that are not in a group yet, by their digests, without two of the same contents.
func siblings(dir, pkg string, sum []byte) ([]member, error) {
	own := []member{{pkg, sum}}
	w := weightFile(pkg)
	fi, err := os.Stat(w)
	if err != nil || fi.Size() < groupMin {
		return own, nil
	}
	mine, err := blobsOf(w)
	if err != nil || len(mine) == 0 {
		return own, nil // not a weight file this reads: alone, as before
	}
	des, err := os.ReadDir(filepath.Dir(pkg))
	if err != nil {
		return own, nil
	}
	seen := map[string]bool{hex.EncodeToString(sum): true}
	for _, de := range des {
		other := filepath.Join(filepath.Dir(pkg), de.Name())
		if other == pkg || !strings.EqualFold(filepath.Ext(other), ".mlpackage") {
			continue
		}
		if ofi, err := os.Stat(weightFile(other)); err != nil || ofi.Size() < groupMin {
			continue
		}
		if same, err := sameWeights(w, mine, weightFile(other)); err != nil || !same {
			continue
		}
		osum, err := digestOf(other)
		if err != nil {
			return nil, err
		}
		if seen[hex.EncodeToString(osum)] {
			continue
		}
		if _, r, ok := readRecord(dir, singleKey(osum)); ok && r.Function != "main" {
			continue // in a group already
		}
		seen[hex.EncodeToString(osum)] = true
		own = append(own, member{other, osum})
	}
	return own, nil
}

// resolve returns the asset of the compile cache dir that runs the package pkg and its function,
// lowering pkg (with its siblings) when there is none; cached when the asset was there.
func resolve(dir, pkg string) (asset, fn string, cached bool, err error) {
	if abs, err := filepath.Abs(pkg); err == nil {
		pkg = abs
	}
	sum, err := digestOf(pkg)
	if err != nil {
		return "", "", false, fmt.Errorf("coreai: %w", err)
	}
	key := singleKey(sum)
	if a, r, ok := readRecord(dir, key); ok {
		return a, r.Function, true, nil
	}
	unlock, err := cache.Lock(context.Background(), "coreai-group:"+filepath.Dir(pkg), "the Core AI programs of "+filepath.Dir(pkg), nil)
	if err != nil {
		return "", "", false, err
	}
	defer unlock()
	if a, r, ok := readRecord(dir, key); ok {
		return a, r.Function, true, nil
	}
	members, err := siblings(dir, pkg, sum)
	if err != nil {
		return "", "", false, fmt.Errorf("coreai: %w", err)
	}
	if len(members) == 1 {
		a, ok := cache.Lookup(dir, key, ".aimodel")
		if !ok {
			if a, err = cache.Store(dir, key, ".aimodel", func(tmp string) error { return lowerOne(pkg, tmp) }); err != nil {
				return "", "", false, err
			}
		}
		cache.SetSource(dir, key, pkg)
		writeRecord(dir, key, record{Asset: key + ".aimodel", Function: "main", Rule: groupRule})
		return a, "main", ok, nil
	}
	sort.Slice(members, func(i, j int) bool { return hex.EncodeToString(members[i].sum) < hex.EncodeToString(members[j].sum) })
	parts := [][]byte{[]byte("hwnpu-coreai-group-v1"), []byte(ailower.Producer)}
	var paths, names []string
	for _, m := range members {
		parts = append(parts, m.sum)
		paths = append(paths, m.path)
		names = append(names, functionName(m.sum))
	}
	gkey := cache.Key(parts...)
	a, ok := cache.Lookup(dir, gkey, ".aimodel")
	if !ok {
		if a, err = cache.Store(dir, gkey, ".aimodel", func(tmp string) error { return lowerMany(paths, names, tmp) }); err != nil {
			return "", "", false, err
		}
	}
	cache.SetSource(dir, gkey, members[0].path)
	for _, m := range members {
		k := singleKey(m.sum)
		cache.SetSource(dir, k, m.path)
		if err := writeRecord(dir, k, record{Asset: gkey + ".aimodel", Function: functionName(m.sum), Rule: groupRule}); err != nil {
			return "", "", false, err
		}
		// the package's program lowered alone before: the group's holds its weights now
		if old := cache.Path(dir, k, ".aimodel"); exists(old) {
			forget(old)
			os.RemoveAll(old)
		}
	}
	return a, functionName(sum), ok, nil
}

// functionName is the function of a package in a group: by its digest, so that it does not
// depend on the files' names.
func functionName(sum []byte) string { return "p" + hex.EncodeToString(sum)[:16] }

func exists(p string) bool { _, err := os.Stat(p); return err == nil }
