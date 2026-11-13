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
    /// The input arrays, made on the first run and filled again on each: Core AI runs arrays it
    /// has seen before faster (M1: PLaMo 3 Translate 0.8B's first part 7.2 ms against 10.0 with
    /// new arrays, 10.7 with views of the caller's buffers).
    var feed: [String: NDArray]? = nil
}

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

@_cdecl("cai_arch")
public func cai_arch() -> UnsafeMutablePointer<CChar> { cstr(AIModel.deviceArchitectureName) }

@_cdecl("cai_free")
public func cai_free(_ p: UnsafeMutablePointer<CChar>?) { free(p) }

/// Loads the .aimodel at path and its function main, or the function NAME of PATH#NAME (a program
/// with several, whose weights they share). Core AI's cache sometimes keeps an entry
/// whose function then fails to load at once (_GenericObjCError 0) and never recompiles: the
/// entries of the model are deleted and it is loaded again (*retried set to 1).
@_cdecl("cai_open")
public func cai_open(_ path: UnsafePointer<CChar>, _ kind: Int32, _ retried: UnsafeMutablePointer<Int32>,
                     _ errOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutableRawPointer? {
    var file = String(cString: path)
    var name = "main"
    if let i = file.lastIndex(of: "#"), file[..<i].hasSuffix(".aimodel") {
        name = String(file[file.index(after: i)...])
        file = String(file[..<i])
    }
    let url = URL(fileURLWithPath: file)
    let o = options(kind)
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
    if case .failure = r {
        try? AIModelCache.default.deleteEntries(for: url)
        retried.pointee = 1
        r = once()
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

private func scalarBytes(_ t: NDArray.ScalarType) -> Int {
    switch t {
    case .float16, .int16, .uint16, .bfloat16: return 2
    case .float32, .int32, .uint32: return 4
    case .int8, .uint8, .bool: return 1
    default: return 0
    }
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
    var dense = true
    var want = 1
    for i in stride(from: r - 1, through: 0, by: -1) {
        if shape[i] > 1 && strides[i] != want { dense = false }
        want *= shape[i]
    }
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

/// Runs the function on the inputs ins (dense row major buffers in the order of cai_names 0)
/// into outs (in the order of cai_names 1). Returns 0, or 1 with *errOut set. Calls on one
/// handle must not overlap (the caller holds a lock per program).
@_cdecl("cai_run")
public func cai_run(_ h: UnsafeMutableRawPointer, _ ins: UnsafePointer<UnsafeMutableRawPointer?>, _ outs: UnsafePointer<UnsafeMutableRawPointer?>,
                    _ errOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32 {
    let fp = h.assumingMemoryBound(to: Fn.self)
    if fp.pointee.feed == nil {
        var feed: [String: NDArray] = [:]
        for name in fp.pointee.inputs {
            guard case .ndArray(let d)? = fp.pointee.fn.descriptor.inputDescriptor(of: name) else {
                errOut.pointee = cstr("input \(name) is not an array"); return 1
            }
            feed[name] = NDArray(shape: d.shape, scalarType: d.scalarType)
        }
        fp.pointee.feed = feed
    }
    var feed = fp.pointee.feed! // moved out while it is written, so that it is not copied
    fp.pointee.feed = nil
    for (i, name) in fp.pointee.inputs.enumerated() {
        let src = ins[i]!
        let elem = scalarBytes(feed[name]!.scalarType)
        feed[name]!.mutableRawView().withUnsafeMutableBytes { p, shape, strides in
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
