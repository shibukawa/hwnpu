// The C ABI over Core AI (macOS 27) that internal/backend/coreai calls through purego. It has no
// types of its own: a program is an allocated Fn, reached through an opaque pointer.
//
// Build (tools/coreai/build-shim.sh): swiftc -emit-library with the Xcode SDK (Core AI's
// subframeworks are not in the Command Line Tools' SDK).
import CoreAI
import Foundation

private func cstr(_ s: String) -> UnsafeMutablePointer<CChar> { strdup(s)! }

/// Single-owner box for an NDArray whose dense backing storage pointer `ptr` is exposed to Go.
/// Moving `a` out into a local stack variable during `outputViews` binding guarantees CoW refcount
/// == 1 so Core AI writes directly into `ptr` without reallocation or copying.
private final class ArrayBox: @unchecked Sendable {
    var a: NDArray?
    let ptr: UnsafeMutableRawPointer
    let shape: [Int]
    let scalarType: NDArray.ScalarType
    let byteCount: Int

    init(_ a: NDArray, _ ptr: UnsafeMutableRawPointer, _ byteCount: Int) {
        self.a = a
        self.ptr = ptr
        self.shape = a.shape
        self.scalarType = a.scalarType
        self.byteCount = byteCount
    }

    func take() -> NDArray {
        let arr = a!
        a = nil
        return arr
    }

    func put(_ arr: NDArray) {
        a = arr
    }
}

private struct Fn {
    var model: AIModel
    var fn: InferenceFunction
    var inputs: [String]
    var outputs: [String]
    var inDescs: [NDArrayDescriptor?]
    var outDescs: [NDArrayDescriptor?]
    /// Cached private input arrays for plain host buffers (not allocated via cai_input_array /
    /// cai_output_array): Core AI runs arrays it has seen before faster (M1: PLaMo 3 Translate
    /// 0.8B's first part 7.2 ms against 10.0 with new arrays, 10.7 with views of caller buffers).
    var inCache: [ArrayBox?]
    /// Cached private output arrays for plain host buffers (or aliased buffers), passed via
    /// outputViews so Core AI does not reallocate an output NDArray on every call.
    var outCache: [ArrayBox?]
}

/// The arrays callers access in place (cai_input_array / cai_output_array), keyed by storage address.
private final class Lent: @unchecked Sendable {
    private let lock = NSLock()
    private var arrays: [UnsafeMutableRawPointer: ArrayBox] = [:]
    func put(_ p: UnsafeMutableRawPointer, _ b: ArrayBox) { lock.lock(); arrays[p] = b; lock.unlock() }
    func get(_ p: UnsafeMutableRawPointer) -> ArrayBox? { lock.lock(); defer { lock.unlock() }; return arrays[p] }
    func drop(_ p: UnsafeMutableRawPointer) { lock.lock(); arrays[p] = nil; lock.unlock() }
    func withLock<R>(_ body: ([UnsafeMutableRawPointer: ArrayBox]) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(arrays)
    }
}

private let lent = Lent()

/// Counter incremented if a lent output NDArray's backing pointer ever moves during outputViews
/// execution (indicating a CoW reallocation). Must stay 0 for true zero-copy outputs.
private final class CowCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count: Int64 = 0
    func inc() { lock.lock(); count += 1; lock.unlock() }
    func get() -> Int64 { lock.lock(); defer { lock.unlock() }; return count }
}

private let cowMoves = CowCounter()

@_cdecl("cai_cow_moves")
public func cai_cow_moves() -> Int64 { cowMoves.get() }

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

/// Reports whether Core AI's cache already has a specialized entry for the .aimodel at path
/// under the given specialization kind, without triggering compilation.
@_cdecl("cai_has_entry")
public func cai_has_entry(_ path: UnsafePointer<CChar>, _ kind: Int32) -> Int32 {
    let url = URL(fileURLWithPath: String(cString: path))
    let o = options(kind & 0xff)
    if let _ = try? AIModelCache.default.model(for: url, options: o) {
        return 1
    }
    return 0
}

/// Reports whether path is a valid Core AI model asset (.aimodel / .aimodelc) via
/// AIModelAsset.isValid(at:) without loading weights or triggering compilation.
@_cdecl("cai_asset_valid")
public func cai_asset_valid(_ path: UnsafePointer<CChar>) -> Int32 {
    let url = URL(fileURLWithPath: String(cString: path))
    return AIModelAsset.isValid(at: url) ? 1 : 0
}

/// Returns the asset's function names (newline-separated) using
/// AIModelAsset.summary(includingStatistics: false) without materializing weights.
/// Returns nil and sets *errOut on failure.
@_cdecl("cai_asset_functions")
public func cai_asset_functions(_ path: UnsafePointer<CChar>,
                                _ errOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutablePointer<CChar>? {
    let url = URL(fileURLWithPath: String(cString: path))
    guard AIModelAsset.isValid(at: url) else {
        errOut.pointee = cstr("invalid Core AI model asset")
        return nil
    }
    do {
        let asset = try AIModelAsset(contentsOf: url)
        guard let summary = try asset.summary(includingStatistics: false) else {
            errOut.pointee = cstr("Core AI model asset has no summary")
            return nil
        }
        return cstr(summary.functions.map { $0.name }.joined(separator: "\n"))
    } catch {
        errOut.pointee = cstr(String(describing: error))
        return nil
    }
}

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
                let ins = f.descriptor.inputNames
                let outs = f.descriptor.outputNames
                let inDescs: [NDArrayDescriptor?] = ins.map {
                    if case .ndArray(let d)? = f.descriptor.inputDescriptor(of: $0) { return d }
                    return nil
                }
                let outDescs: [NDArrayDescriptor?] = outs.map {
                    if case .ndArray(let d)? = f.descriptor.outputDescriptor(of: $0) { return d }
                    return nil
                }
                return .success(Fn(
                    model: m, fn: f, inputs: ins, outputs: outs,
                    inDescs: inDescs, outDescs: outDescs,
                    inCache: [ArrayBox?](repeating: nil, count: ins.count),
                    outCache: [ArrayBox?](repeating: nil, count: outs.count)
                ))
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

/// Returns the dense byte count expected by Core AI for input (which 0) or output (1) at index,
/// or -1 if invalid or dynamic-shaped. Used to validate static shapes before cai_run (preventing
/// uncatchable SIGKILL on shape mismatch).
@_cdecl("cai_arg_bytes")
public func cai_arg_bytes(_ h: UnsafeMutableRawPointer, _ which: Int32, _ index: Int32) -> Int64 {
    let f = h.assumingMemoryBound(to: Fn.self).pointee
    let descs = which == 0 ? f.inDescs : f.outDescs
    guard index >= 0, Int(index) < descs.count, let d = descs[Int(index)], !d.hasDynamicShape else {
        return -1
    }
    var count = 1
    for dim in d.shape {
        if dim < 0 { return -1 }
        count *= dim
    }
    let elem = scalarBytes(d.scalarType)
    if elem > 0 { return Int64(count * elem) }
    if isNibbles(d.scalarType) { return Int64((count + 1) / 2) }
    return -1
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

/// Allocates a dense zeroed NDArray matching descriptor d and wraps it in a single-owner ArrayBox.
/// Returns nil if the descriptor is dynamic or Core AI lays out the array non-densely.
private func makeDenseBox(_ d: NDArrayDescriptor) -> ArrayBox? {
    guard !d.hasDynamicShape else { return nil }
    var a = NDArray(shape: d.shape, scalarType: d.scalarType)
    let elem = scalarBytes(d.scalarType)
    var made: UnsafeMutableRawPointer? = nil
    var byteCount = 0
    a.mutableRawView().withUnsafeMutableBytes { p, shape, strides in
        let (dense, count) = isDense(ints(shape), ints(strides))
        if dense && elem > 0 {
            byteCount = count * elem
            memset(p, 0, byteCount)
            made = p
        } else if dense && isNibbles(d.scalarType) {
            byteCount = (count + 1) / 2
            memset(p, 0, byteCount)
            made = p
        }
    }
    guard let p = made else { return nil }
    let box = ArrayBox(a, p, byteCount)
    var kept: UnsafeRawPointer? = nil
    let check = box.take()
    check.rawView().withUnsafeBytes { q, _, _ in kept = q }
    box.put(check)
    if kept != UnsafeRawPointer(p) { return nil }
    return box
}

private func makeLentArray(_ d: NDArrayDescriptor?) -> UnsafeMutableRawPointer? {
    guard let d = d, let box = makeDenseBox(d) else { return nil }
    lent.put(box.ptr, box)
    return box.ptr
}

/// Makes an array of the input index (in the order of cai_names 0) of the function and returns
/// its storage, a dense row major buffer of zeros the caller fills in place: cai_run takes a pointer to
/// it for an input or output of that shape, of this function or another, as the array itself, with
/// no copy. nil when the array does not lie densely: the caller then gives a buffer of its own.
/// cai_array_free frees it.
@_cdecl("cai_input_array")
public func cai_input_array(_ h: UnsafeMutableRawPointer, _ index: Int32) -> UnsafeMutableRawPointer? {
    let f = h.assumingMemoryBound(to: Fn.self).pointee
    guard index >= 0, Int(index) < f.inDescs.count else { return nil }
    return makeLentArray(f.inDescs[Int(index)])
}

/// Makes an array of the output index (in the order of cai_names 1) of the function and returns
/// its storage, a dense row major buffer of zeros that cai_run writes directly via outputViews
/// (zero-copy output, and zero-copy input when bound to a subsequent function's input of the same
/// shape). nil when the array does not lie densely. cai_array_free frees it.
@_cdecl("cai_output_array")
public func cai_output_array(_ h: UnsafeMutableRawPointer, _ index: Int32) -> UnsafeMutableRawPointer? {
    let f = h.assumingMemoryBound(to: Fn.self).pointee
    guard index >= 0, Int(index) < f.outDescs.count else { return nil }
    return makeLentArray(f.outDescs[Int(index)])
}

/// Frees an array of cai_input_array or cai_output_array, by its storage.
@_cdecl("cai_array_free")
public func cai_array_free(_ p: UnsafeMutableRawPointer) { lent.drop(p) }

/// Recursively binds each output ArrayBox's NDArray (moved out onto the stack so refcount == 1
/// and its lifetime spans the nested call) into MutableViews, then executes fn.run.
private func runWithOutputs(
    _ fn: InferenceFunction,
    inputs: [String: NDArray],
    outs: ArraySlice<(String, ArrayBox)>,
    views: consuming InferenceFunction.MutableViews,
    moved: inout Bool
) async throws -> InferenceFunction.Outputs {
    guard let first = outs.first else {
        return try await fn.run(inputs: inputs, outputViews: views)
    }
    var arr = first.1.take()
    defer {
        // Verify zero-copy: if CoW ever moved the backing storage, copy back into ptr so Go
        // never observes stale bytes.
        var cur: UnsafeRawPointer? = nil
        arr.rawView().withUnsafeBytes { q, _, _ in cur = q }
        if let cur = cur, cur != UnsafeRawPointer(first.1.ptr) {
            memcpy(first.1.ptr, cur, first.1.byteCount)
            moved = true
        }
        first.1.put(arr)
    }
    var v = views
    v.insert(&arr, for: first.0)
    return try await runWithOutputs(fn, inputs: inputs, outs: outs.dropFirst(), views: v, moved: &moved)
}

/// Runs the function on the inputs ins (dense row major buffers in the order of cai_names 0, or
/// the storage of arrays of cai_input_array / cai_output_array) into outs (in the order of
/// cai_names 1). Lent arrays matching the output's shape and scalarType are written directly via
/// outputViews (zero allocation, zero copy); plain host output buffers reuse a per-function cached
/// NDArray via outputViews and copy out. Returns 0, or 1 with *errOut set. Calls on one handle
/// must not overlap (the caller holds a lock per program).
@_cdecl("cai_run")
public func cai_run(_ h: UnsafeMutableRawPointer, _ ins: UnsafePointer<UnsafeMutableRawPointer?>, _ outs: UnsafePointer<UnsafeMutableRawPointer?>,
                    _ errOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32 {
    let fp = h.assumingMemoryBound(to: Fn.self)
    let nIn = fp.pointee.inputs.count
    let nOut = fp.pointee.outputs.count

    var inputDict = [String: NDArray](minimumCapacity: nIn)
    var outPairs = [(String, ArrayBox)]()
    outPairs.reserveCapacity(nOut)
    var outCopyFrom = [(Int, ArrayBox)]()   // copy from cached box into outs[i] after run
    var outFallback = [Int]()               // read from returned Outputs (non-dense only)

    // Snapshot lent lookups under a single lock acquisition (instead of locking per tensor).
    var lentIn = [ArrayBox?](repeating: nil, count: nIn)
    var lentOut = [ArrayBox?](repeating: nil, count: nOut)
    var nullErr: String? = nil
    lent.withLock { map in
        for i in 0..<nIn {
            guard let src = ins[i] else {
                nullErr = "input \(fp.pointee.inputs[i]) is null"; return
            }
            lentIn[i] = map[src]
        }
        for i in 0..<nOut {
            guard let dst = outs[i] else {
                nullErr = "output \(fp.pointee.outputs[i]) is null"; return
            }
            lentOut[i] = map[dst]
        }
    }
    if let err = nullErr {
        errOut.pointee = cstr(err); return 1
    }

    for i in 0..<nIn {
        let src = ins[i]!
        guard let d = fp.pointee.inDescs[i] else {
            errOut.pointee = cstr("input \(fp.pointee.inputs[i]) is not an array"); return 1
        }
        let name = fp.pointee.inputs[i]
        if let box = lentIn[i], box.shape == d.shape, box.scalarType == d.scalarType, let arr = box.a {
            inputDict[name] = arr
            continue
        }
        if let box = fp.pointee.inCache[i], let arr = box.a {
            memcpy(box.ptr, src, box.byteCount)
            inputDict[name] = arr
            continue
        }
        if let box = makeDenseBox(d), let arr = box.a {
            memcpy(box.ptr, src, box.byteCount)
            fp.pointee.inCache[i] = box
            inputDict[name] = arr
            continue
        }
        // Non-dense fallback: create a one-off strided NDArray.
        var a = NDArray(shape: d.shape, scalarType: d.scalarType)
        let elem = scalarBytes(d.scalarType)
        let nibbles = isNibbles(d.scalarType)
        a.mutableRawView().withUnsafeMutableBytes { p, shape, strides in
            if nibbles {
                let (_, count) = isDense(ints(shape), ints(strides))
                memcpy(p, src, (count + 1) / 2)
                return
            }
            copyStrided(src, p, ints(shape), ints(strides), elem, toArray: true)
        }
        inputDict[name] = a
    }

    // Prepare outputs: use direct zero-copy outputViews for matching lent arrays (not aliased
    // with any input or earlier output), and cached dense output NDArrays for plain host buffers.
    for i in 0..<nOut {
        let dst = outs[i]!
        let name = fp.pointee.outputs[i]
        guard let d = fp.pointee.outDescs[i] else {
            outFallback.append(i)
            continue
        }
        if let box = lentOut[i], box.shape == d.shape, box.scalarType == d.scalarType {
            var aliased = false
            for j in 0..<nIn {
                if ins[j] == dst { aliased = true; break }
            }
            if !aliased {
                for j in 0..<i {
                    if outs[j] == dst { aliased = true; break }
                }
            }
            if !aliased {
                outPairs.append((name, box))
                continue
            }
        }
        if let box = fp.pointee.outCache[i] {
            outPairs.append((name, box))
            outCopyFrom.append((i, box))
            continue
        }
        if let box = makeDenseBox(d) {
            fp.pointee.outCache[i] = box
            outPairs.append((name, box))
            outCopyFrom.append((i, box))
            continue
        }
        outFallback.append(i)
    }

    let fn = fp.pointee.fn
    let names = fp.pointee.outputs
    nonisolated(unsafe) let outPtrs = outs
    // Move inputDict into a single-owner holder that is cleared BEFORE block() signals the
    // semaphore, guaranteeing every lent input NDArray drops back to refcount == 1 strictly
    // before cai_run returns (so a subsequent cai_run using that same array as an output never
    // races with Task.detached teardown and triggers CoW).
    final class InputHolder: @unchecked Sendable {
        var dict: [String: NDArray]
        init(_ d: [String: NDArray]) { self.dict = d }
    }
    let inputHolder = InputHolder(inputDict)
    inputDict = [:]
    let frozenOutPairs = outPairs
    let frozenCopyFrom = outCopyFrom
    let frozenFallback = outFallback
    nonisolated(unsafe) var failure: String?

    let r = block { () async throws -> Int in
        let localInputs = inputHolder.dict
        inputHolder.dict = [:]
        var moved = false
        var o = try await runWithOutputs(fn, inputs: localInputs, outs: frozenOutPairs[...], views: InferenceFunction.MutableViews(), moved: &moved)
        if moved { cowMoves.inc() }
        for (i, box) in frozenCopyFrom {
            memcpy(outPtrs[i]!, box.ptr, box.byteCount)
        }
        for i in frozenFallback {
            let name = names[i]
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
