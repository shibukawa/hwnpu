// caiload: a small Swift command over Core AI for measuring loads apart from hwnpu (design.md
// 16.229): specializes .aimodel or .aimodelc files on the Neural Engine and prints the time of each.
//
//	caiload PATH...            specialize in turn (the first load compiles; the second is the cache)
//	caiload -par PATH...       specialize all at once in this process
//	caiload -check PATH...     is the program in Core AI's cache (AIModelCache.model(for:), no compile)?
//	caiload -run N PATH...     load the function main and run it N times on zero inputs (ms/call)
//	caiload -cycle R PATH...   load all, call each alone R times, then all in turn R rounds: the
//	                           cost of switching between programs (ms a round, ms a program)
//
// Build: swiftc -sdk "$(xcrun --sdk macosx --show-sdk-path)" -target arm64-apple-macos27.0 -O main.swift -o caiload
// (the SDK must have Core AI's subframeworks: Xcode 27's, as tools/coreai/build-shim.sh picks).
// Core AI's cache is kept per executable name, so caiload's entries are its own.
import CoreAI
import Foundation

var args = Array(CommandLine.arguments.dropFirst())
var par = false
if args.first == "-par" { par = true; args.removeFirst() }
// CAILOAD_UNITS=gpu or cpu prefers that unit instead of the Neural Engine (what its entry holds).
let opts = SpecializationOptions(preferredComputeUnitKind: ProcessInfo.processInfo.environment["CAILOAD_UNITS"] == "gpu" ? .gpu : ProcessInfo.processInfo.environment["CAILOAD_UNITS"] == "cpu" ? .cpu : .neuralEngine)
// -check: asks the cache whether each model is specialized already (no compile), and exits.
if args.first == "-check" {
    args.removeFirst()
    for p in args {
        let t = Date()
        let hit = (try? AIModelCache.default.model(for: URL(fileURLWithPath: p), options: opts)) ?? nil
        print("\(URL(fileURLWithPath: p).lastPathComponent) \(hit == nil ? "not cached" : "cached") \(String(format: "%.3f", Date().timeIntervalSince(t)))s")
    }
    exit(0)
}
let fmt: (Double) -> String = { String(format: "%.2f", $0) }

// -run N: loads each model's function main and runs it N times on zero inputs, printing the
// mean time of a call after the first (the device's time of the program).
if args.first == "-run" {
    args.removeFirst()
    let n = Int(args.removeFirst()) ?? 5
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        for p in args {
            do {
                let t0 = Date()
                let m = try await AIModel(contentsOf: URL(fileURLWithPath: p), options: opts)
                guard let f = try m.loadFunction(named: "main") else { print("\(p): no main"); continue }
                var feed: [String: NDArray] = [:]
                for name in f.descriptor.inputNames {
                    guard case .ndArray(let d)? = f.descriptor.inputDescriptor(of: name) else { continue }
                    feed[name] = NDArray(shape: d.shape, scalarType: d.scalarType)
                }
                let tl = Date().timeIntervalSince(t0)
                _ = try await f.run(inputs: feed)
                let t1 = Date()
                for _ in 0..<n { _ = try await f.run(inputs: feed) }
                let per = Date().timeIntervalSince(t1) / Double(n)
                print("\(URL(fileURLWithPath: p).lastPathComponent) load \(fmt(tl))s, \(String(format: "%.1f", per * 1000)) ms/call over \(n) calls")
            } catch { print("\(p) FAILED: \(error)") }
            fflush(stdout)
        }
        sem.signal()
    }
    sem.wait()
    exit(0)
}

// -cycle R PATH...: loads every program, calls each alone R times (ms/call), then all in turn for
// R rounds (ms a round and the sum of the solo times): what switching between programs costs.
if args.first == "-cycle" {
    args.removeFirst()
    let r = Int(args.removeFirst()) ?? 3
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        var fns: [(String, InferenceFunction, [String: NDArray])] = []
        var models: [AIModel] = []
        let t0 = Date()
        for p in args {
            do {
                let m = try await AIModel(contentsOf: URL(fileURLWithPath: p), options: opts)
                guard let f = try m.loadFunction(named: "main") else { print("\(p): no main"); continue }
                var feed: [String: NDArray] = [:]
                for name in f.descriptor.inputNames {
                    guard case .ndArray(let d)? = f.descriptor.inputDescriptor(of: name) else { continue }
                    feed[name] = NDArray(shape: d.shape, scalarType: d.scalarType)
                }
                models.append(m); fns.append((URL(fileURLWithPath: p).lastPathComponent, f, feed))
            } catch { print("\(p) FAILED: \(error)"); fflush(stdout) }
        }
        print("loaded \(fns.count) programs in \(fmt(Date().timeIntervalSince(t0)))s"); fflush(stdout)
        do {
            var solo = 0.0
            for (name, f, feed) in fns {
                _ = try await f.run(inputs: feed)
                let t = Date()
                for _ in 0..<r { _ = try await f.run(inputs: feed) }
                let per = Date().timeIntervalSince(t) / Double(r)
                solo += per
                print("  solo \(name) \(String(format: "%.1f", per * 1000)) ms"); fflush(stdout)
            }
            for (_, f, feed) in fns { _ = try await f.run(inputs: feed) } // one round to settle
            let t = Date()
            for _ in 0..<r { for (_, f, feed) in fns { _ = try await f.run(inputs: feed) } }
            let round = Date().timeIntervalSince(t) / Double(r)
            print("in turn: \(String(format: "%.1f", round * 1000)) ms a round of \(fns.count) programs; solo sum \(String(format: "%.1f", solo * 1000)) ms; switching \(String(format: "%+.1f", (round - solo) * 1000)) ms a round, \(String(format: "%+.1f", (round - solo) * 1000 / Double(fns.count))) ms a program")
        } catch { print("FAILED: \(error)") }
        _ = models.count
        sem.signal()
    }
    sem.wait()
    exit(0)
}
// -run N: loads each model's function main and runs it N times on zero inputs, printing the
// mean time of a call after the first (the device's time of the program).
if args.first == "-run" {
    args.removeFirst()
    let n = Int(args.removeFirst()) ?? 5
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        for p in args {
            do {
                let t0 = Date()
                let m = try await AIModel(contentsOf: URL(fileURLWithPath: p), options: opts)
                guard let f = try m.loadFunction(named: "main") else { print("\(p): no main"); continue }
                var feed: [String: NDArray] = [:]
                for name in f.descriptor.inputNames {
                    guard case .ndArray(let d)? = f.descriptor.inputDescriptor(of: name) else { continue }
                    feed[name] = NDArray(shape: d.shape, scalarType: d.scalarType)
                }
                let tl = Date().timeIntervalSince(t0)
                _ = try await f.run(inputs: feed)
                let t1 = Date()
                for _ in 0..<n { _ = try await f.run(inputs: feed) }
                let per = Date().timeIntervalSince(t1) / Double(n)
                print("\(URL(fileURLWithPath: p).lastPathComponent) load \(fmt(tl))s, \(String(format: "%.1f", per * 1000)) ms/call over \(n) calls")
            } catch { print("\(p) FAILED: \(error)") }
            fflush(stdout)
        }
        sem.signal()
    }
    sem.wait()
    exit(0)
}

func one(_ p: String) async -> String {
    let t = Date()
    do {
        _ = try await AIModel.specialize(contentsOf: URL(fileURLWithPath: p), options: opts)
        return "\(URL(fileURLWithPath: p).lastPathComponent) \(fmt(Date().timeIntervalSince(t)))s"
    } catch {
        return "\(URL(fileURLWithPath: p).lastPathComponent) FAILED \(fmt(Date().timeIntervalSince(t)))s: \(error)"
    }
}

let sem = DispatchSemaphore(value: 0)
let t0 = Date()
Task.detached {
    if par {
        await withTaskGroup(of: String.self) { g in
            for p in args { g.addTask { await one(p) } }
            for await s in g { print(s); fflush(stdout) }
        }
    } else {
        for p in args { print(await one(p)); fflush(stdout) }
    }
    print("total \(fmt(Date().timeIntervalSince(t0)))s (\(par ? "parallel" : "sequential"), pid \(getpid()))")
    sem.signal()
}
sem.wait()
