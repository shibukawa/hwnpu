package chat

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// DraftScan performs coarse-to-fine top-1 vocabulary argmax on CPU over the full vocabulary
// (0% OOV), allowing the MTP drafter to omit the large lm_head projection from the NPU and
// output only its final hidden state h.
//
// Embeddings are quantized to per-row symmetric int8 and reordered contiguously into C
// balanced clusters built on mean-centered embeddings (defeating embedding anisotropy).
// Each query scores all C centroids and then scans only the top-B clusters using SIMD
// vecops.DotI8 (ARM64 NEON SDOT / amd64 AVX2).
type DraftScan struct {
	V, D, C        int
	TopClusters    int
	Q8             []int8
	Scales         []float32
	IDs            []int32
	Offsets        []int32
	CentroidsQ8    []int8
	CentroidScales []float32
}

// BuildDraftScan builds a DraftScan index from a V x D embedding matrix supplied by readRow.
func BuildDraftScan(V, D int, readRow func(r int, dst []float32) error) (*DraftScan, error) {
	if V <= 0 || D <= 0 {
		return nil, fmt.Errorf("chat: draftscan: invalid shape %dx%d", V, D)
	}
	C := 512
	if V < 4096 {
		C = max(1, min(V/8, 64))
	}
	nCPU := runtime.GOMAXPROCS(0)

	// 1. Quantize all V rows to symmetric int8 and compute the global mean vector.
	rawQ8 := make([]int8, V*D)
	rawScales := make([]float32, V)
	partMeans := make([][]float64, nCPU)
	var firstErr error
	var errOnce sync.Once
	{
		var wg sync.WaitGroup
		chunk := (V + nCPU - 1) / nCPU
		for w := range nCPU {
			lo, hi := w*chunk, min((w+1)*chunk, V)
			if lo >= hi {
				continue
			}
			partMeans[w] = make([]float64, D)
			wg.Add(1)
			go func() {
				defer wg.Done()
				row := make([]float32, D)
				pm := partMeans[w]
				for r := lo; r < hi; r++ {
					if err := readRow(r, row); err != nil {
						errOnce.Do(func() { firstErr = err })
						return
					}
					for j, v := range row {
						pm[j] += float64(v)
					}
					top, _ := vecops.AbsMax(row)
					s := top / 127.0
					if s == 0 {
						s = 1
					}
					rawScales[r] = s
					vecops.QuantizeSym(rawQ8[r*D:(r+1)*D], row, s, -127, 127)
				}
			}()
		}
		wg.Wait()
	}
	if firstErr != nil {
		return nil, firstErr
	}
	mean := make([]float32, D)
	for w := range nCPU {
		if partMeans[w] == nil {
			continue
		}
		for j := range D {
			mean[j] += float32(partMeans[w][j] / float64(V))
		}
	}

	// Helper: quantize a float32 vector to unit-normalized int8 + scale.
	quantUnit := func(dst []int8, src []float32) float32 {
		var norm float64
		for _, v := range src {
			norm += float64(v) * float64(v)
		}
		if norm == 0 {
			clear(dst)
			return 1
		}
		inv := float32(1.0 / math.Sqrt(norm))
		var amax float32
		for _, v := range src {
			if a := float32(math.Abs(float64(v * inv))); a > amax {
				amax = a
			}
		}
		s := amax / 127.0
		if s == 0 {
			s = 1
		}
		// Scale src by inv/s directly into int8.
		scale := s / inv
		vecops.QuantizeSym(dst, src, scale, -127, 127)
		return s
	}

	// 2. Fit C centroids on up to 32768 evenly-sampled mean-centered rows using DotI8.
	S := min(V, 32768)
	sampleQ8 := make([]int8, S*D)
	sampleSc := make([]float32, S)
	sampleIdx := make([]int, S)
	for i := range S {
		sampleIdx[i] = int(int64(i) * int64(V) / int64(S))
	}
	{
		var wg sync.WaitGroup
		chunk := (S + nCPU - 1) / nCPU
		for w := range nCPU {
			lo, hi := w*chunk, min((w+1)*chunk, S)
			if lo >= hi {
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				tmp := make([]float32, D)
				for i := lo; i < hi; i++ {
					r := sampleIdx[i]
					sc := rawScales[r]
					src := rawQ8[r*D : (r+1)*D]
					for j := range D {
						tmp[j] = float32(src[j])*sc - mean[j]
					}
					sampleSc[i] = quantUnit(sampleQ8[i*D:(i+1)*D], tmp)
				}
			}()
		}
		wg.Wait()
	}

	centQ8 := make([]int8, C*D)
	centSc := make([]float32, C)
	for c := range C {
		si := c * S / C
		copy(centQ8[c*D:(c+1)*D], sampleQ8[si*D:(si+1)*D])
		centSc[c] = sampleSc[si]
	}

	const kmIters = 6
	for range kmIters {
		partSums := make([][]float32, nCPU)
		partCnts := make([][]int, nCPU)
		var wg sync.WaitGroup
		chunk := (S + nCPU - 1) / nCPU
		for w := range nCPU {
			lo, hi := w*chunk, min((w+1)*chunk, S)
			if lo >= hi {
				continue
			}
			partSums[w] = make([]float32, C*D)
			partCnts[w] = make([]int, C)
			wg.Add(1)
			go func() {
				defer wg.Done()
				sums := partSums[w]
				cnts := partCnts[w]
				for i := lo; i < hi; i++ {
					xq := sampleQ8[i*D : (i+1)*D]
					bestC, bestS := 0, float32(-1e30)
					for c := range C {
						d := float32(vecops.DotI8(xq, centQ8[c*D:(c+1)*D])) * centSc[c]
						if d > bestS {
							bestS, bestC = d, c
						}
					}
					cnts[bestC]++
					dst := sums[bestC*D : (bestC+1)*D]
					ssc := sampleSc[i]
					for j := range D {
						dst[j] += float32(xq[j]) * ssc
					}
				}
			}()
		}
		wg.Wait()
		acc := make([]float32, D)
		for c := range C {
			clear(acc)
			cnt := 0
			for w := range nCPU {
				if partSums[w] == nil {
					continue
				}
				cnt += partCnts[w][c]
				src := partSums[w][c*D : (c+1)*D]
				for j := range D {
					acc[j] += src[j]
				}
			}
			if cnt > 0 {
				centSc[c] = quantUnit(centQ8[c*D:(c+1)*D], acc)
			}
		}
	}

	// 3. Assign all V rows (mean-centered) to their nearest centroid in parallel.
	assign := make([]int32, V)
	{
		var wg sync.WaitGroup
		chunk := (V + nCPU - 1) / nCPU
		for w := range nCPU {
			lo, hi := w*chunk, min((w+1)*chunk, V)
			if lo >= hi {
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				tmp := make([]float32, D)
				xq := make([]int8, D)
				for r := lo; r < hi; r++ {
					sc := rawScales[r]
					src := rawQ8[r*D : (r+1)*D]
					for j := range D {
						tmp[j] = float32(src[j])*sc - mean[j]
					}
					quantUnit(xq, tmp)
					bestC, bestS := 0, float32(-1e30)
					for c := range C {
						d := float32(vecops.DotI8(xq, centQ8[c*D:(c+1)*D])) * centSc[c]
						if d > bestS {
							bestS, bestC = d, c
						}
					}
					assign[r] = int32(bestC)
				}
			}()
		}
		wg.Wait()
	}

	// 4. Split any oversized cluster (> maxCap) so no single hub cluster blows up candidate count.
	maxCap := max(2*V/C, 64)
	counts := make([]int, C)
	for _, c := range assign {
		counts[c]++
	}
	buckets := make([][]int32, C)
	for c, n := range counts {
		buckets[c] = make([]int32, 0, n)
	}
	for r, c := range assign {
		buckets[c] = append(buckets[c], int32(r))
	}
	// Recursively bisect any bucket > maxCap by projecting onto the difference of two members.
	var finalBuckets [][]int32
	tmpA, tmpB := make([]float32, D), make([]float32, D)
	var splitBucket func(b []int32)
	splitBucket = func(b []int32) {
		if len(b) == 0 {
			return
		}
		if len(b) <= maxCap {
			finalBuckets = append(finalBuckets, b)
			return
		}
		// Pick two well-separated members (first and middle) to define a split hyperplane.
		r0, r1 := int(b[0]), int(b[len(b)/2])
		s0, s1 := rawScales[r0], rawScales[r1]
		for j := range D {
			tmpA[j] = float32(rawQ8[r0*D+j])*s0 - float32(rawQ8[r1*D+j])*s1
		}
		dirQ8 := make([]int8, D)
		quantUnit(dirQ8, tmpA)
		type proj struct {
			id int32
			v  float32
		}
		ps := make([]proj, len(b))
		for i, id := range b {
			r := int(id)
			ps[i] = proj{id, float32(vecops.DotI8(rawQ8[r*D:(r+1)*D], dirQ8)) * rawScales[r]}
		}
		slices.SortFunc(ps, func(a, b proj) int {
			if a.v < b.v {
				return -1
			}
			if a.v > b.v {
				return 1
			}
			return 0
		})
		mid := len(ps) / 2
		left, right := make([]int32, mid), make([]int32, len(ps)-mid)
		for i := range mid {
			left[i] = ps[i].id
		}
		for i := mid; i < len(ps); i++ {
			right[i-mid] = ps[i].id
		}
		splitBucket(left)
		splitBucket(right)
	}
	for _, b := range buckets {
		splitBucket(b)
	}
	_ = tmpB

	// 5. Pack final balanced clusters contiguously and compute their uncentered unit centroids
	// (so Stage 1 directly approximates h · E[id] without bias).
	C = len(finalBuckets)
	ds := &DraftScan{
		V:              V,
		D:              D,
		C:              C,
		TopClusters:    min(max(C/6, 16), C),
		Q8:             make([]int8, V*D),
		Scales:         make([]float32, V),
		IDs:            make([]int32, V),
		Offsets:        make([]int32, C+1),
		CentroidsQ8:    make([]int8, C*D),
		CentroidScales: make([]float32, C),
	}
	pos := 0
	for c, b := range finalBuckets {
		ds.Offsets[c] = int32(pos)
		clear(tmpA)
		for _, id := range b {
			r := int(id)
			copy(ds.Q8[pos*D:(pos+1)*D], rawQ8[r*D:(r+1)*D])
			ds.Scales[pos] = rawScales[r]
			ds.IDs[pos] = id
			sc := rawScales[r]
			src := rawQ8[r*D : (r+1)*D]
			for j := range D {
				tmpA[j] += float32(src[j])*sc - mean[j]
			}
			pos++
		}
		invN := float32(1.0 / float64(len(b)))
		for j := range D {
			tmpA[j] *= invN
		}
		top, _ := vecops.AbsMax(tmpA)
		cs := top / 127.0
		if cs == 0 {
			cs = 1
		}
		vecops.QuantizeSym(ds.CentroidsQ8[c*D:(c+1)*D], tmpA, cs, -127, 127)
		ds.CentroidScales[c] = cs
	}
	ds.Offsets[C] = int32(pos)
	return ds, nil
}

// BuildDraftScanFromWeights builds a DraftScan index directly from an llm.Weights table.
func BuildDraftScanFromWeights(w llm.Weights, tensorName string, V, D int) (*DraftScan, error) {
	if r, ok := w.(interface {
		Float32Range(name string, start, end int) ([]float32, error)
	}); ok {
		return BuildDraftScan(V, D, func(row int, dst []float32) error {
			v, err := r.Float32Range(tensorName, row*D, (row+1)*D)
			if err != nil {
				return err
			}
			copy(dst, v)
			return nil
		})
	}
	raw, _, err := w.Get(tensorName)
	if err != nil {
		return nil, err
	}
	return BuildDraftScan(V, D, func(row int, dst []float32) error {
		copy(dst, raw[row*D:(row+1)*D])
		return nil
	})
}

// Scan returns the top-1 token ID maximizing h · E[id] across the full vocabulary using
// coarse-to-fine centroid pruning with SIMD DotI8.
func (ds *DraftScan) Scan(h []float32) int32 {
	return ds.ScanTop(h, ds.TopClusters)
}

// ScanTop is Scan with an explicit number of top clusters B to inspect.
func (ds *DraftScan) ScanTop(h []float32, B int) int32 {
	if B <= 0 || B > ds.C {
		B = ds.C
	}
	D := ds.D
	hq := make([]int8, D)
	top, _ := vecops.AbsMax(h[:D])
	hs := top / 127.0
	if hs == 0 {
		return 0
	}
	vecops.QuantizeSym(hq, h[:D], hs, -127, 127)

	// Stage 1: score all C centroids (~10 us with NEON SDOT).
	type cs struct {
		c int
		s float32
	}
	scores := make([]cs, ds.C)
	for c := range ds.C {
		scores[c] = cs{c, float32(vecops.DotI8(hq, ds.CentroidsQ8[c*D:(c+1)*D])) * ds.CentroidScales[c]}
	}
	// Partial selection for top-B clusters (O(C log C) for C ~ 512 is < 5 us).
	slices.SortFunc(scores, func(a, b cs) int {
		if a.s > b.s {
			return -1
		}
		if a.s < b.s {
			return 1
		}
		return 0
	})
	chosen := scores[:B]

	// Stage 2: scan chosen clusters in parallel using SIMD DotI8.
	nCPU := min(runtime.GOMAXPROCS(0), B)
	if nCPU <= 1 || B < 4 {
		bestID, bestV := int32(0), float32(-1e30)
		for _, ch := range chosen {
			lo, hi := int(ds.Offsets[ch.c]), int(ds.Offsets[ch.c+1])
			for p := lo; p < hi; p++ {
				v := float32(vecops.DotI8(hq, ds.Q8[p*D:(p+1)*D])) * ds.Scales[p]
				if v > bestV {
					bestV, bestID = v, ds.IDs[p]
				}
			}
		}
		return bestID
	}
	bestIDs := make([]int32, nCPU)
	bestVals := make([]float32, nCPU)
	var wg sync.WaitGroup
	for w := range nCPU {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bID, bV := int32(0), float32(-1e30)
			for i := w; i < len(chosen); i += nCPU {
				c := chosen[i].c
				lo, hi := int(ds.Offsets[c]), int(ds.Offsets[c+1])
				for p := lo; p < hi; p++ {
					v := float32(vecops.DotI8(hq, ds.Q8[p*D:(p+1)*D])) * ds.Scales[p]
					if v > bV {
						bV, bID = v, ds.IDs[p]
					}
				}
			}
			bestIDs[w], bestVals[w] = bID, bV
		}()
	}
	wg.Wait()
	bID, bV := bestIDs[0], bestVals[0]
	for w := 1; w < nCPU; w++ {
		if bestVals[w] > bV {
			bV, bID = bestVals[w], bestIDs[w]
		}
	}
	return bID
}

const (
	draftScanMagic   = "HWNPUDS1"
	draftScanVersion = uint32(1)
	draftScanFile    = "draftscan.bin"
)

// Save writes ds to path atomically (via a temp file + rename), tagging it with srcSize and
// srcMtime (UnixNano) of the backing weights file so stale caches are automatically rebuilt.
func (ds *DraftScan) Save(path string, srcSize, srcMtime int64) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 256<<10)
	var hdr [40]byte
	copy(hdr[0:8], draftScanMagic)
	binary.LittleEndian.PutUint32(hdr[8:12], draftScanVersion)
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(ds.V))
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(ds.D))
	binary.LittleEndian.PutUint32(hdr[20:24], uint32(ds.C))
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(ds.TopClusters))
	binary.LittleEndian.PutUint64(hdr[28:36], uint64(srcSize))
	binary.LittleEndian.PutUint32(hdr[36:40], uint32(srcMtime)) // low 32 bits of mtime sec/nsec hash
	// Write full 64-bit mtime right after header (8 bytes).
	var mt [8]byte
	binary.LittleEndian.PutUint64(mt[:], uint64(srcMtime))
	write := func(b []byte) error {
		_, err := bw.Write(b)
		return err
	}
	err = write(hdr[:])
	if err == nil {
		err = write(mt[:])
	}
	if err == nil {
		err = write(sliceAsBytes(ds.Offsets))
	}
	if err == nil {
		err = write(sliceAsBytes(ds.CentroidScales))
	}
	if err == nil {
		err = write(sliceAsBytes(ds.CentroidsQ8))
	}
	if err == nil {
		err = write(sliceAsBytes(ds.IDs))
	}
	if err == nil {
		err = write(sliceAsBytes(ds.Scales))
	}
	if err == nil {
		err = write(sliceAsBytes(ds.Q8))
	}
	if err == nil {
		err = bw.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadDraftScan reads a cached DraftScan index from path and validates its dimensions and
// backing checkpoint metadata.
func LoadDraftScan(path string, expectedV, expectedD int, srcSize, srcMtime int64) (*DraftScan, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 256<<10)
	var hdr [48]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, err
	}
	if string(hdr[0:8]) != draftScanMagic || binary.LittleEndian.Uint32(hdr[8:12]) != draftScanVersion {
		return nil, fmt.Errorf("chat: draftscan: bad header in %s", path)
	}
	V := int(binary.LittleEndian.Uint32(hdr[12:16]))
	D := int(binary.LittleEndian.Uint32(hdr[16:20]))
	C := int(binary.LittleEndian.Uint32(hdr[20:24]))
	topC := int(binary.LittleEndian.Uint32(hdr[24:28]))
	fSize := int64(binary.LittleEndian.Uint64(hdr[28:36]))
	fMtime := int64(binary.LittleEndian.Uint64(hdr[40:48]))
	if V != expectedV || D != expectedD || C <= 0 || topC <= 0 || topC > C {
		return nil, fmt.Errorf("chat: draftscan: shape mismatch in %s", path)
	}
	if (srcSize != 0 && fSize != srcSize) || (srcMtime != 0 && fMtime != srcMtime) {
		return nil, fmt.Errorf("chat: draftscan: stale cache %s", path)
	}
	ds := &DraftScan{
		V:              V,
		D:              D,
		C:              C,
		TopClusters:    topC,
		Offsets:        make([]int32, C+1),
		CentroidScales: make([]float32, C),
		CentroidsQ8:    make([]int8, C*D),
		IDs:            make([]int32, V),
		Scales:         make([]float32, V),
		Q8:             make([]int8, V*D),
	}
	for _, buf := range [][]byte{
		sliceAsBytes(ds.Offsets),
		sliceAsBytes(ds.CentroidScales),
		sliceAsBytes(ds.CentroidsQ8),
		sliceAsBytes(ds.IDs),
		sliceAsBytes(ds.Scales),
		sliceAsBytes(ds.Q8),
	} {
		if _, err := io.ReadFull(br, buf); err != nil {
			return nil, err
		}
	}
	return ds, nil
}

// LoadOrBuildDraftScan loads a cached DraftScan index from cachePath if valid, or builds it
// from w (tensorName, V x D) and saves it to cachePath for fast future startups.
func LoadOrBuildDraftScan(cachePath, srcPath string, w llm.Weights, tensorName string, V, D int) (*DraftScan, error) {
	var srcSize, srcMtime int64
	if srcPath != "" {
		if st, err := os.Stat(srcPath); err == nil {
			srcSize, srcMtime = st.Size(), st.ModTime().UnixNano()
		}
	}
	if cachePath != "" {
		if ds, err := LoadDraftScan(cachePath, V, D, srcSize, srcMtime); err == nil {
			return ds, nil
		}
	}
	ds, err := BuildDraftScanFromWeights(w, tensorName, V, D)
	if err != nil {
		return nil, err
	}
	if cachePath != "" {
		_ = ds.Save(cachePath, srcSize, srcMtime) // best-effort cache save
	}
	return ds, nil
}

func sliceAsBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var zero T
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(zero)))
}
