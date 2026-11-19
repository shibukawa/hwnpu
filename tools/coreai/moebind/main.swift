// moebind times how a mixture of experts program (aicheck moe's experts_i8, lowered with
// aicheck lower) takes the experts' weights passed at run time in three ways: Core AI arrays
// (NDArray, as hwnpu's shim passes them), views of IOSurfaces and views of Metal buffers,
// each with the same experts at every call and with others chosen at every call.
//
//	swiftc -O -parse-as-library tools/coreai/moebind/main.swift -o /tmp/moebind
//	/tmp/moebind experts_i8.aimodel HIDDEN INTER TOPK EXPERTS [CALLS]
import CoreAI
import Foundation
import IOSurface
import Metal

@main
struct MoEBind {
    static func main() async throws {
        let a = CommandLine.arguments
        guard a.count >= 6 else {
            print("usage: moebind experts_i8.aimodel HIDDEN INTER TOPK EXPERTS [CALLS]")
            exit(2)
        }
        let url = URL(fileURLWithPath: a[1])
        let H = Int(a[2])!, I = Int(a[3])!, K = Int(a[4])!, E = Int(a[5])!
        let calls = a.count > 6 ? Int(a[6])! : 100
        let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
        guard let fn = try model.loadFunction(named: "main") else { fatalError("no main") }
        print("inputs:", fn.descriptor.inputNames.sorted().joined(separator: " "))

        let guShape = [1, H, 2 * I], dnShape = [1, I, H], sguShape = [1, 1, 2 * I], sdnShape = [1, 1, H]
        let guBytes = H * 2 * I, dnBytes = I * H

        func array(_ shape: [Int], _ t: NDArray.ScalarType, fill: (UnsafeMutableRawPointer, Int) -> Void) -> NDArray {
            var x = NDArray(shape: shape, scalarType: t)
            x.mutableRawView().withUnsafeMutableBytes { p, _, _ in
                let n = shape.reduce(1, *) * (t == .int8 ? 1 : 2)
                fill(p, n)
            }
            return x
        }
        func randBytes(_ p: UnsafeMutableRawPointer, _ n: Int) {
            let b = p.assumingMemoryBound(to: Int8.self)
            for i in 0..<n { b[i] = Int8.random(in: -127...127) }
        }
        func halves(_ v: Float) -> (UnsafeMutableRawPointer, Int) -> Void {
            return { p, n in
                let h = p.assumingMemoryBound(to: Float16.self)
                for i in 0..<(n / 2) { h[i] = Float16(v) }
            }
        }
        // the inputs every call shares
        let x = array([1, 1, H], .float16, fill: halves(0.5))
        let r = (0..<K).map { _ in array([1, 1, 1], .float16, fill: halves(1 / Float(K))) }
        let sg = 1 / (73 * Float(H).squareRoot()), sd = 1 / (73 * Float(I).squareRoot())

        func time(_ label: String, _ body: (Int) async throws -> Void) async throws {
            for i in 0..<5 { try await body(i) }
            let c = ContinuousClock()
            let t = c.now
            for i in 0..<calls { try await body(i) }
            let d = c.now - t
            let ms = Double(d.components.attoseconds) / 1e15 / Double(calls) + Double(d.components.seconds) * 1e3 / Double(calls)
            print(String(format: "  %-44@ %8.3f ms", label as NSString, ms))
        }
        var rng = SystemRandomNumberGenerator()
        func pick() -> [Int] { Array((0..<E).shuffled(using: &rng).prefix(K)) }
        // slot k only ever takes the experts k, k + K, ..., N of them: N arrays an input
        func sticky(_ n: Int) -> [Int] { (0..<K).map { k in k + K * Int.random(in: 0..<n, using: &rng) } }
        // K of the first s experts, in any slot: s arrays in all
        func pool(_ s: Int) -> [Int] { Array((0..<s).shuffled(using: &rng).prefix(K)) }
        var ns: [Int] = [], ss: [Int] = []
        var n = 1
        while K * n <= E { ns.append(n); n *= 2 }
        var m = K
        while m <= E { ss.append(m); m *= 2 }
        if ss.last != E { ss.append(E) }
        let shapes: [(String, () -> [Int])] = ns.map { n in ("an input keeps to \(n) expert(s)", { sticky(n) }) }
            + ss.map { s in ("any slot, \(s) experts in all", { pool(s) }) }

        // A. Core AI arrays
        do {
            print("A. NDArray per expert")
            var gu: [NDArray] = [], dn: [NDArray] = [], sgu: [NDArray] = [], sdn: [NDArray] = []
            for _ in 0..<E {
                gu.append(array(guShape, .int8, fill: randBytes))
                dn.append(array(dnShape, .int8, fill: randBytes))
                sgu.append(array(sguShape, .float16, fill: halves(sg)))
                sdn.append(array(sdnShape, .float16, fill: halves(sd)))
            }
            func feed(_ chosen: [Int]) -> [String: NDArray] {
                var d: [String: NDArray] = ["x": x]
                for (k, e) in chosen.enumerated() {
                    d["gu\(k)"] = gu[e]; d["dn\(k)"] = dn[e]; d["sgu\(k)"] = sgu[e]; d["sdn\(k)"] = sdn[e]; d["r\(k)"] = r[k]
                }
                return d
            }
            let same = feed(Array(0..<K))
            try await time("the same experts") { _ in _ = try await fn.run(inputs: same) }
            try await time("experts chosen at every call") { _ in _ = try await fn.run(inputs: feed(pick())) }
            let all = (0..<4).map { _ in feed(pick()) }
            try await time("four sets of experts in turn") { i in _ = try await fn.run(inputs: all[i % 4]) }
            for (label, f) in shapes {
                try await time(label) { _ in _ = try await fn.run(inputs: feed(f())) }
            }
        }

        // B and C: Metal buffers given as AsyncValues to encode on a compute stream
        let dev = MTLCreateSystemDefaultDevice()!
        let stream = ComputeStream()
        let page = 16384
        func rounded(_ n: Int) -> Int { (n + page - 1) / page * page }
        func shared(_ bytes: Int) -> any MTLBuffer {
            let b = dev.makeBuffer(length: rounded(bytes), options: .storageModeShared)!
            randBytes(b.contents(), bytes)
            return b
        }
        var surfaces: [IOSurface] = [] // kept alive under their buffers
        func onSurface(_ bytes: Int) -> any MTLBuffer {
            let n = rounded(bytes)
            let s = IOSurface(properties: [.width: n, .height: 1, .bytesPerElement: 1, .bytesPerRow: n, .allocSize: n, .pixelFormat: 0x4c303038 /* L008 */])!
            surfaces.append(s)
            s.lock(options: [], seed: nil)
            randBytes(s.baseAddress, bytes)
            let b = dev.makeBuffer(bytesNoCopy: s.baseAddress, length: n, options: .storageModeShared, deallocator: nil)!
            s.unlock(options: [], seed: nil)
            return b
        }
        for (label, make) in [("B. MTLBuffer per expert (AsyncValue)", shared), ("C. IOSurface-backed MTLBuffer per expert (AsyncValue)", onSurface)] {
            print(label)
            var gu: [any MTLBuffer] = [], dn: [any MTLBuffer] = [], sgu: [NDArray] = [], sdn: [NDArray] = []
            for _ in 0..<E {
                gu.append(make(guBytes)); dn.append(make(dnBytes))
                sgu.append(array(sguShape, .float16, fill: halves(sg))); sdn.append(array(sdnShape, .float16, fill: halves(sd)))
            }
            let gv = gu.map { InferenceFunction.AsyncValue(unsafeBuffer: $0, scalarType: .int8, shape: guShape) }
            let dv = dn.map { InferenceFunction.AsyncValue(unsafeBuffer: $0, scalarType: .int8, shape: dnShape) }
            let sgv = sgu.map { InferenceFunction.AsyncValue($0) }, sdv = sdn.map { InferenceFunction.AsyncValue($0) }
            let xv = InferenceFunction.AsyncValue(x), rv = r.map { InferenceFunction.AsyncValue($0) }
            func run(_ chosen: [Int]) async throws {
                var d: [String: InferenceFunction.AsyncValue] = ["x": xv]
                for (k, e) in chosen.enumerated() {
                    d["gu\(k)"] = gv[e]; d["dn\(k)"] = dv[e]; d["sgu\(k)"] = sgv[e]; d["sdn\(k)"] = sdv[e]; d["r\(k)"] = rv[k]
                }
                let out = try fn.encode(inputs: d, to: stream)
                _ = try await out["y"]!.ndArray
            }
            let first = Array(0..<K)
            try await time("the same experts") { _ in try await run(first) }
            try await time("experts chosen at every call") { _ in try await run(pick()) }
            for (label, f) in shapes {
                try await time(label) { _ in try await run(f()) }
            }
        }
    }
}
