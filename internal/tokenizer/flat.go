package tokenizer

import (
	"unsafe"
)

// The tables a tokenizer looks its strings and pairs up in. They are flat arrays, the same in
// memory and in a binary tokenizer file (tokbin.go): read from a tokenizer.json they are built on
// the heap, from a binary file they are its mapped bytes, looked up in place. A Go map of the
// 262144 tokens and 514653 merges of Gemma's vocabulary was 50 MB of heap and 0.2 s to build at
// every start; the file is 21 MB the system pages in as it is read, and opened in 0.1 ms.

// strTable is strings, each with a value (their index when val is nil), and an open addressing
// hash index of them (none when slots is nil: str only).
type strTable struct {
	off   []uint32 // len n+1: string i is data[off[i]:off[i+1]]
	data  []byte
	val   []int32 // by string, or nil
	slots []int32 // string index + 1 at the slot of its hash (linear probing), 0 for none; a power of two long
}

// newStrTable builds the table of strs with their values (nil: their indexes). The strings must
// be different from one another when index is set.
func newStrTable(strs []string, val []int32, index bool) *strTable {
	t := &strTable{off: make([]uint32, len(strs)+1), val: val}
	n := 0
	for _, s := range strs {
		n += len(s)
	}
	t.data = make([]byte, 0, n)
	for i, s := range strs {
		t.off[i] = uint32(len(t.data))
		t.data = append(t.data, s...)
	}
	t.off[len(strs)] = uint32(len(t.data))
	if index {
		size := 2
		for size < 2*len(strs) {
			size <<= 1
		}
		t.slots = make([]int32, size)
		mask := uint64(size - 1)
		for i, s := range strs {
			h := strHash(s) & mask
			for t.slots[h] != 0 {
				h = (h + 1) & mask
			}
			t.slots[h] = int32(i + 1)
		}
	}
	return t
}

// indexOf is an index of the strings ids of t (different strings), sharing t's strings: find
// gives the id.
func indexOf(t *strTable, ids []int32) *strTable {
	size := 2
	for size < 2*len(ids) {
		size <<= 1
	}
	x := &strTable{off: t.off, data: t.data, slots: make([]int32, size)}
	mask := uint64(size - 1)
	for _, id := range ids {
		h := strHash(t.str(int(id))) & mask
		for x.slots[h] != 0 {
			h = (h + 1) & mask
		}
		x.slots[h] = id + 1
	}
	return x
}

// shares reports whether t's strings are u's (an index of them, indexOf).
func (t *strTable) shares(u *strTable) bool {
	return len(t.off) > 0 && len(u.off) > 0 && &t.off[0] == &u.off[0]
}

// n is the number of strings.
func (t *strTable) n() int { return len(t.off) - 1 }

// str is string i, without a copy: it lives as long as the table (and the file it is mapped from).
func (t *strTable) str(i int) string {
	lo, hi := t.off[i], t.off[i+1]
	if lo == hi {
		return ""
	}
	return unsafe.String(&t.data[lo], int(hi-lo))
}

// find is the value of s.
func (t *strTable) find(s string) (int32, bool) {
	if len(t.slots) == 0 {
		return 0, false
	}
	mask := uint64(len(t.slots) - 1)
	for h := strHash(s) & mask; ; h = (h + 1) & mask {
		k := t.slots[h]
		if k == 0 {
			return 0, false
		}
		i := int(k - 1)
		if int(t.off[i+1]-t.off[i]) == len(s) && t.str(i) == s {
			if t.val == nil {
				return int32(i), true
			}
			return t.val[i], true
		}
	}
}

// has reports whether s is in the table.
func (t *strTable) has(s string) bool {
	_, ok := t.find(s)
	return ok
}

// strHash is FNV-1a: what a file's index was built with, so it must not change (tokbinVersion).
func strHash(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

// pairTable maps pairs of ids to a rank and an id (a merge: the pair's rank and what it makes),
// an open addressing table of four int32s a slot: a, b, rank, id, with a < 0 for an empty slot.
type pairTable struct {
	slots []int32
}

type pairEntry struct{ a, b, rank, id int32 }

// newPairTable builds the table of entries, which must be of different pairs: at most three
// quarters full (16 MB for Gemma's 514653 merges).
func newPairTable(entries []pairEntry) *pairTable {
	size := 2
	for 3*size < 4*len(entries)+4 {
		size <<= 1
	}
	p := &pairTable{slots: make([]int32, 4*size)}
	for i := 0; i < size; i++ {
		p.slots[4*i] = -1
	}
	mask := uint64(size - 1)
	for _, e := range entries {
		h := pairHash(e.a, e.b) & mask
		for p.slots[4*h] >= 0 {
			h = (h + 1) & mask
		}
		copy(p.slots[4*h:], []int32{e.a, e.b, e.rank, e.id})
	}
	return p
}

// find is the rank and id of the pair a, b.
func (p *pairTable) find(a, b int32) (rank, id int32, ok bool) {
	size := uint64(len(p.slots) / 4)
	if size == 0 {
		return 0, 0, false
	}
	mask := size - 1
	for h := pairHash(a, b) & mask; ; h = (h + 1) & mask {
		s := p.slots[4*h : 4*h+4 : 4*h+4]
		if s[0] < 0 {
			return 0, 0, false
		}
		if s[0] == a && s[1] == b {
			return s[2], s[3], true
		}
	}
}

// pairHash mixes the two ids (murmur3's finalizer); like strHash, fixed by the file format.
func pairHash(a, b int32) uint64 {
	x := uint64(uint32(a))<<32 | uint64(uint32(b))
	x ^= x >> 33
	x *= 0xff51afd7ed558ccd
	x ^= x >> 33
	x *= 0xc4ceb9fe1a85ec53
	x ^= x >> 33
	return x
}
