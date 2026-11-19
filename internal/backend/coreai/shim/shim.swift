// The C ABI over Core AI (macOS 27) that internal/backend/coreai calls through purego. It has no
// types of its own: a program is an allocated Fn, reached through an opaque pointer.
//
// Build (tools/coreai/build-shim.sh): swiftc -emit-library with the Xcode SDK (Core AI's
// subframeworks are not in the Command Line Tools' SDK).
import CoreAI
import Foundation

private func cstr(_ s: String) -> UnsafeMutablePointer<CChar> { strdup(s)! }

private struct Fn {
    var model: AIModel
    var fn: InferenceFunction
    var inputs: [String]
    var outputs: [String]
    /// The input arrays, made when an input is first given a buffer of the caller's and filled
    /// again on each run: Core AI runs arrays it has seen before faster (M1: PLaMo 3 Translate
    /// 0.8B's first part 7.2 ms against 10.0 with new arrays, 10.7 with views of the caller's
    /// buffers). An input given an array of the caller's (cai_input_array) holds that one.
    var feed: [String: NDArray]? = nil
    /// The inputs whose array in feed is the caller's, by the address of its storage.
    var lent: [String: UnsafeMutableRawPointer] = [:]
}

/// The arrays the callers fill in place (cai_input_array), by the address of their storage.
private final class Lent: @unchecked Sendable {
    private let lock = NSLock()
    private var arrays: [UnsafeMutableRawPointer: NDArray] = [:]
    func put(_ p: UnsafeMutableRawPointer, _ a: NDArray) { lock.lock(); arrays[p] = a; lock.unlock() }
    func get(_ p: UnsafeMutableRawPointer) -> NDArray? { lock.lock(); defer { lock.unlock() }; return arrays[p] }
    func drop(_ p: UnsafeMutableRawPointer) { lock.lock(); arrays[p] = nil; lock.unlock() }
}

private let lent = Lent()

/// Blocks on an async body (purego calls on a Go thread).
private func block<T>(_ f: @escaping @Sendable () async throws -> T) -> Result<T, any Error> {
    nonisolated(unsafe) var r: Result<T, any Error>!
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        do { r = .success(try await f()) } catch { r = .failure(error) }
        sem.signal()
    }
    sem.wait()
    return r
}

private func options(_ kind: Int32) -> SpecializationOptions {
    switch kind {
    case 1: return .cpuOnly
    case 2: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    default: return .default
    }
}

/// A flag of cai_open's kind: fail rather than delete the cache entries and load again.
private let noRetry: Int32 = 0x100

@_cdecl("cai_arch")
public func cai_arch() -> UnsafeMutablePointer<CChar> { cstr(AIModel.deviceArchitectureName) }

@_cdecl("cai_free")
public func cai_free(_ p: UnsafeMutablePointer<CChar>?) { free(p) }

/// Loads the .aimodel (or coreai-build's .aimodelc) at path and its function main, or the function
/// NAME of PATH#NAME (a program with several, whose weights they share). Core AI's cache sometimes keeps an entry
/// whose function then fails to load at once (_GenericObjCError 0) and never recompiles: the
/// entries of the model are deleted and it is loaded again (*retried set to 1, *errOut the first
/// attempt's error), unless kind has
/// noRetry (an asset whose bytecode is cut, which loads only from its entry: parsing it aborts).
@_cdecl("cai_open")
public func cai_open(_ path: UnsafePointer<CChar>, _ kind: Int32, _ retried: UnsafeMutablePointer<Int32>,
                     _ errOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutableRawPointer? {
    var file = String(cString: path)
    var name = "main"
    if let i = file.lastIndex(of: "#"), file[..<i].hasSuffix(".aimodel") || file[..<i].hasSuffix(".aimodelc") {
        name = String(file[file.index(after: i)...])
        file = String(file[..<i])
    }
    let url = URL(fileURLWithPath: file)
    let o = options(kind & 0xff)
    retried.pointee = 0
    func once() -> Result<Fn, any Error> {
        let r = block { () async throws -> AIModel in try await AIModel(contentsOf: url, options: o) }
        switch r {
        case .failure(let e): return .failure(e)
        case .success(let m):
            do {
                guard let f = try m.loadFunction(named: name) else {
                    return .failure(NSError(domain: "hwnpu", code: 1, userInfo: [NSLocalizedDescriptionKey: "no function \(name)"]))
                }
                return .success(Fn(model: m, fn: f, inputs: f.descriptor.inputNames, outputs: f.descriptor.outputNames))
            } catch {
                return .failure(error)
            }
        }
    }
    var r = once()
    if case .failure(let e) = r, kind & noRetry == 0 {
        try? AIModelCache.default.deleteEntries(for: url)
        retried.pointee = 1
        r = once()
        // loaded the second time: *errOut is the first attempt's error (the caller frees it)
        if case .success = r { errOut.pointee = cstr(String(describing: e)) }
    }
    switch r {
    case .success(let f):
        let p = UnsafeMutablePointer<Fn>.allocate(capacity: 1)
        p.initialize(to: f)
        return UnsafeMutableRawPointer(p)
    case .failure(let e):
        errOut.pointee = cstr(String(describing: e))
        return nil
    }
}

/// Deletes Core AI's cache entries of the .aimodel at path (before hwnpu removes it: an asset
/// replaced by one that holds several programs' weights once).
@_cdecl("cai_forget")
public func cai_forget(_ path: UnsafePointer<CChar>) {
    try? AIModelCache.default.deleteEntries(for: URL(fileURLWithPath: String(cString: path)))
}

@_cdecl("cai_close")
public func cai_close(_ h: UnsafeMutableRawPointer) {
    let p = h.assumingMemoryBound(to: Fn.self)
    p.deinitialize(count: 1)
    p.deallocate()
}

/// The names of the inputs (which 0) or the outputs (1), joined by newlines.
@_cdecl("cai_names")
public func cai_names(_ h: UnsafeMutableRawPointer, _ which: Int32) -> UnsafeMutablePointer<CChar> {
    let f = h.assumingMemoryBound(to: Fn.self).pointee
    return cstr((which == 0 ? f.inputs : f.outputs).joined(separator: "\n"))
}

private func ints(_ s: Span<Int>) -> [Int] {
    var a = [Int]()
    for i in 0..<s.count { a.append(s[i]) }
    return a
}

/// Whether the arrays of the type hold two values a byte (int4, uint4), dense.
private func isNibbles(_ t: NDArray.ScalarType) -> Bool { t == .int4 || t == .uint4 }

private func scalarBytes(_ t: NDArray.ScalarType) -> Int {
    switch t {
    case .float16, .int16, .uint16, .bfloat16: return 2
    case .float32, .int32, .uint32: return 4
    case .int8, .uint8, .bool: return 1
    default: return 0
    }
}

/// Whether an array of the shape and strides (in elements) lies as a dense row major buffer, and
/// its number of elements.
private func isDense(_ shape: [Int], _ strides: [Int]) -> (Bool, Int) {
    var dense = true
    var want = 1
    for i in stride(from: shape.count - 1, through: 0, by: -1) {
        if shape[i] > 1 && strides[i] != want { dense = false }
        want *= shape[i]
    }
    return (dense, want)
}

/// Copies between the dense buffer d (row major, elem bytes each) and p, the array's storage of
/// the given strides (in elements; the last one 1).
private func copyStrided(_ d: UnsafeMutableRawPointer, _ p: UnsafeMutableRawPointer, _ shape: [Int], _ strides: [Int], _ elem: Int, toArray: Bool) {
    let r = shape.count
    if r == 0 {
        if toArray { memcpy(p, d, elem) } else { memcpy(d, p, elem) }
        return
    }
    let inner = shape[r - 1] * elem
    let (dense, want) = isDense(shape, strides)
    if dense {
        let n = want * elem
        if toArray { memcpy(p, d, n) } else { memcpy(d, p, n) }
        return
    }
    let rows = want / max(shape[r - 1], 1)
    var idx = [Int](repeating: 0, count: r)
    for row in 0..<rows {
        var off = 0
        for i in 0..<(r - 1) { off += idx[i] * strides[i] }
        let src = d + row * inner
        let dst = p + off * elem
        if toArray { memcpy(dst, src, inner) } else { memcpy(src, dst, inner) }
        var i = r - 2
        while i >= 0 {
            idx[i] += 1
            if idx[i] < shape[i] { break }
            idx[i] = 0
            i -= 1
        }
    }
}

/// Makes an array of the input index (in the order of cai_names 0) of the function and returns
/// its storage, a dense row major buffer of zeros the caller fills in place: cai_run takes a pointer to
/// it for an input of that shape, of this function or another, as the array itself, with no
/// copy (a program's caches, 84 MiB a call of PLaMo 3 Translate 2B's, were a tenth of its
/// step); for an input of another shape it is a buffer as any, copied. nil when the array does
/// not lie densely: the caller then gives a buffer of its own.
/// cai_array_free frees it.
@_cdecl("cai_input_array")
public func cai_input_array(_ h: UnsafeMutableRawPointer, _ index: Int32) -> UnsafeMutableRawPointer? {
    let f = h.assumingMemoryBound(to: Fn.self).pointee
    guard index >= 0, Int(index) < f.inputs.count,
          case .ndArray(let d)? = f.fn.descriptor.inputDescriptor(of: f.inputs[Int(index)]) else { return nil }
    var a = NDArray(shape: d.shape, scalarType: d.scalarType)
    let elem = scalarBytes(d.scalarType)
    var made: UnsafeMutableRawPointer? = nil
    a.mutableRawView().withUnsafeMutableBytes { p, shape, strides in
        let (dense, count) = isDense(ints(shape), ints(strides))
        if dense && elem > 0 {
            memset(p, 0, count * elem) // as a buffer of the caller's: zeros
            made = p
        } else if dense && isNibbles(d.scalarType) {
            memset(p, 0, (count + 1) / 2) // two a byte
            made = p
        }
    }
    guard let p = made else { return nil }
    lent.put(p, a)
    // the storage must stay where it is once the array is kept: the caller writes to it
    var kept: UnsafeRawPointer? = nil
    lent.get(p)?.rawView().withUnsafeBytes { q, _, _ in kept = q }
    if kept != UnsafeRawPointer(p) {
        lent.drop(p)
        return nil
    }
    return p
}

/// Frees an array of cai_input_array, by its storage. A function that ran it last keeps it
/// until it runs another.
@_cdecl("cai_array_free")
public func cai_array_free(_ p: UnsafeMutableRawPointer) { lent.drop(p) }

/// Runs the function on the inputs ins (dense row major buffers in the order of cai_names 0, or
/// the storage of arrays of cai_input_array) into outs (in the order of cai_names 1). Returns 0,
/// or 1 with *errOut set. Calls on one handle must not overlap (the caller holds a lock per
/// program).
@_cdecl("cai_run")
public func cai_run(_ h: UnsafeMutableRawPointer, _ ins: UnsafePointer<UnsafeMutableRawPointer?>, _ outs: UnsafePointer<UnsafeMutableRawPointer?>,
                    _ errOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32 {
    let fp = h.assumingMemoryBound(to: Fn.self)
    var feed = fp.pointee.feed ?? [:] // moved out while it is written, so that it is not copied
    fp.pointee.feed = nil
    for (i, name) in fp.pointee.inputs.enumerated() {
        let src = ins[i]!
        if fp.pointee.lent[name] == src { continue } // the caller's array, as at the last run
        if let a = lent.get(src), case .ndArray(let d)? = fp.pointee.fn.descriptor.inputDescriptor(of: name),
           d.shape == a.shape, d.scalarType == a.scalarType {
            // an array of the caller's of the input's shape, filled in place: passed as it is.
            // It is never written through here (a mutable view of an array held twice would
            // copy it). One of another shape is a buffer as any, copied below
            feed[name] = a
            fp.pointee.lent[name] = src
            continue
        }
        if feed[name] == nil || fp.pointee.lent[name] != nil {
            guard case .ndArray(let d)? = fp.pointee.fn.descriptor.inputDescriptor(of: name) else {
                fp.pointee.feed = feed
                errOut.pointee = cstr("input \(name) is not an array"); return 1
            }
            feed[name] = NDArray(shape: d.shape, scalarType: d.scalarType)
            fp.pointee.lent[name] = nil
        }
        let elem = scalarBytes(feed[name]!.scalarType)
        let nibbles = isNibbles(feed[name]!.scalarType)
        feed[name]!.mutableRawView().withUnsafeMutableBytes { p, shape, strides in
            if nibbles { // two a byte, dense
                let (_, count) = isDense(ints(shape), ints(strides))
                memcpy(p, src, (count + 1) / 2)
                return
            }
            copyStrided(src, p, ints(shape), ints(strides), elem, toArray: true)
        }
    }
    fp.pointee.feed = feed
    let f = fp.pointee
    nonisolated(unsafe) let input = feed
    let fn = f.fn
    nonisolated(unsafe) var failure: String?
    nonisolated(unsafe) let outPtrs = outs
    let names = f.outputs
    let r = block { () async throws -> Int in
        var o = try await fn.run(inputs: input)
        for (i, name) in names.enumerated() {
            guard let v = o.remove(name), let a = v.ndArray else {
                failure = "output \(name) missing"; return 1
            }
            let elem = scalarBytes(a.scalarType)
            let dst = outPtrs[i]!
            a.rawView().withUnsafeBytes { p, shape, strides in
                copyStrided(dst, UnsafeMutableRawPointer(mutating: p), ints(shape), ints(strides), elem, toArray: false)
            }
        }
        return 0
    }
    switch r {
    case .success(let c):
        if c != 0 { errOut.pointee = cstr(failure ?? "run failed"); return 1 }
        return 0
    case .failure(let e):
        errOut.pointee = cstr(String(describing: e)); return 1
    }
}
