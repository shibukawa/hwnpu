package coreml

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// Thin helpers over purego/objc. Every function that creates autoreleased objects must run
// inside withPool, which also pins the goroutine to its OS thread for the pool's lifetime.

var (
	loadOnce sync.Once
	loadErr  error

	autoreleasePoolPush func() uintptr
	autoreleasePoolPop  func(uintptr)
	mlAllComputeDevices uintptr // optional C function, macOS 14+
	creatorDefinedKey   objc.ID // the NSString constant MLModelCreatorDefinedKey
)

const coreMLPath = "/System/Library/Frameworks/CoreML.framework/CoreML"

func load() error {
	loadOnce.Do(func() {
		if _, err := purego.Dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
			loadErr = fmt.Errorf("coreml: load Foundation: %w", err)
			return
		}
		h, err := purego.Dlopen(coreMLPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("coreml: load CoreML: %w", err)
			return
		}
		libobjc, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("coreml: load libobjc: %w", err)
			return
		}
		purego.RegisterLibFunc(&autoreleasePoolPush, libobjc, "objc_autoreleasePoolPush")
		purego.RegisterLibFunc(&autoreleasePoolPop, libobjc, "objc_autoreleasePoolPop")
		if p, err := purego.Dlsym(h, "MLAllComputeDevices"); err == nil {
			mlAllComputeDevices = p
		}
		if p, err := purego.Dlsym(h, "MLModelCreatorDefinedKey"); err == nil && p != 0 {
			creatorDefinedKey = **(**objc.ID)(unsafe.Pointer(&p))
		}
		if objc.GetClass("MLModel") == 0 {
			loadErr = errors.New("coreml: MLModel class not found")
		}
	})
	return loadErr
}

// withPool runs fn inside an autorelease pool on a locked OS thread.
func withPool(fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := autoreleasePoolPush()
	defer autoreleasePoolPop(pool)
	return fn()
}

// selector cache: sel_registerName takes a global lock.
var (
	selMu    sync.Mutex
	selCache = map[string]objc.SEL{}
)

func sel(name string) objc.SEL {
	selMu.Lock()
	defer selMu.Unlock()
	s, ok := selCache[name]
	if !ok {
		s = objc.RegisterName(name)
		selCache[name] = s
	}
	return s
}

func class(name string) objc.ID { return objc.ID(objc.GetClass(name)) }

func send(id objc.ID, name string, args ...any) objc.ID { return id.Send(sel(name), args...) }

func sendInt(id objc.ID, name string, args ...any) int64 {
	return objc.Send[int64](id, sel(name), args...)
}

func sendBool(id objc.ID, name string, args ...any) bool {
	return objc.Send[bool](id, sel(name), args...)
}

func sendPtr(id objc.ID, name string, args ...any) unsafe.Pointer {
	return objc.Send[unsafe.Pointer](id, sel(name), args...)
}

func retain(id objc.ID) objc.ID {
	if id != 0 {
		send(id, "retain")
	}
	return id
}

func release(id objc.ID) {
	if id != 0 {
		send(id, "release")
	}
}

func isKind(id objc.ID, className string) bool {
	c := objc.GetClass(className)
	if id == 0 || c == 0 {
		return false
	}
	return sendBool(id, "isKindOfClass:", c)
}

// nsString returns an autoreleased NSString.
func nsString(s string) objc.ID {
	return send(class("NSString"), "stringWithUTF8String:", s)
}

// goString copies an NSString into Go memory.
func goString(s objc.ID) string {
	if s == 0 {
		return ""
	}
	p := sendPtr(s, "UTF8String")
	if p == nil {
		return ""
	}
	return cString(p)
}

// cString copies a NUL terminated native string.
func cString(p unsafe.Pointer) string {
	n := 0
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(p), n))
}

func describe(id objc.ID) string {
	if id == 0 {
		return "<nil>"
	}
	return goString(send(id, "description"))
}

// nsError converts an NSError into a Go error.
func nsError(op string, e objc.ID) error {
	if e == 0 {
		return fmt.Errorf("coreml: %s failed", op)
	}
	return fmt.Errorf("coreml: %s: %s (domain=%s code=%d)", op,
		goString(send(e, "localizedDescription")), goString(send(e, "domain")), sendInt(e, "code"))
}

func fileURL(path string) objc.ID {
	return send(class("NSURL"), "fileURLWithPath:", nsString(path))
}

func nsNumberInt(v int64) objc.ID {
	return send(class("NSNumber"), "numberWithLongLong:", v)
}

func arrayItems(a objc.ID) []objc.ID {
	if a == 0 {
		return nil
	}
	n := sendInt(a, "count")
	out := make([]objc.ID, n)
	for i := range out {
		out[i] = send(a, "objectAtIndex:", uint64(i))
	}
	return out
}

func nsArrayOfInts(vs []int) objc.ID {
	arr := send(class("NSMutableArray"), "arrayWithCapacity:", uint64(len(vs)))
	for _, v := range vs {
		send(arr, "addObject:", nsNumberInt(int64(v)))
	}
	return arr
}

func intsOfArray(a objc.ID) []int {
	items := arrayItems(a)
	out := make([]int, len(items))
	for i, it := range items {
		out[i] = int(sendInt(it, "longLongValue"))
	}
	return out
}

// dictItems returns the keys (as Go strings) and values of an NSDictionary.
func dictItems(d objc.ID) (keys []string, values []objc.ID) {
	for _, k := range arrayItems(send(d, "allKeys")) {
		keys = append(keys, goString(k))
		values = append(values, send(d, "objectForKey:", k))
	}
	return keys, values
}
