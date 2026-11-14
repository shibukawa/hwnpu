#!/usr/bin/env python3
"""GPTQ for Qwen3.6-35B-A3B's 4-bit parts, a decoder layer at a time.

q4_gptq.py holds the whole model in float32, which Qwen3.6-35B-A3B (36 B parameters, 140 GB in
float32) does not fit. This runs the calibration texts through the model a layer at a time
instead: a layer's weights are read from the shards (iron_qwen35_check.streamed's way), the
texts' hidden states before it are kept, and the layer's 4-bit weights are chosen by GPTQ
(q4_gptq.gptq's method, here for many experts at once) before the texts go through it,
quantized, to the next layer.

The 4-bit weights are the ones the export (iron_qwen35, --q4) quantizes to q4nx:

- each expert's gate and up as one table (gu_table: column blocks alternating),
  against H of the tokens the router sends it; its down against H of its activations (silu(g) u
  of the quantized gate and up)
- the shared expert's gate, up and down (for the prompt chunk's Linears), and its table (the
  decode step's MoEGateUp reads it after the routed experts'), against every token
- the head, against the final norm's output

The router, the mixers (q8 in the export) and the norms stay as they are. An expert the
calibration sends fewer than --min-tokens tokens keeps the export's rounding.

The evaluation texts run alongside, three times: with the checkpoint's weights, with the
export's rounding, with GPTQ's codes. At the end the head gives each its logits and the script
prints the KL from the checkpoint's (mean over the positions past --eval-skip).

    ~/.cache/hwnpu/dev/ref-venv/bin/python tools/amd/qwen36_gptq.py   # 40 layers about 4.3 hours ~/models/Qwen3.6-35B-A3B OUT
    python tools/amd/iron_export_lm.py qwen35moe CKPT EXPORT --q4 --q4-codes OUT/codes

OUT/codes is iron_q4.load_codes' layout (index.json and KEY.npy, keyed by iron_q4.weight_key).
OUT/result.json holds the measurements. About 60 GB of memory is too little for anything else
big beside it (see docs/amd-npu-models.md, 13).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_q4  # noqa: E402
from q4_gptq import CALIB_DATA, kl_stats  # noqa: E402

CB = iron_q4.CB
G = iron_q4.GROUP

CALIB = [CALIB_DATA / "wikitext2-train.txt", CALIB_DATA / "wiki40b-ja-calib.txt"]
EVAL = [CALIB_DATA / "wikitext2-test.txt", CALIB_DATA / "wiki40b-ja-eval.txt",
        Path(__file__).resolve().parents[2] / "testdata/texts/readme.md"]


def texts(paths, tok, length: int, count: int, skip_chars: int = 0):
    """Up to count sequences of length tokens cut from the files, in turn (Qwen has no BOS)."""
    rr = []
    for p in paths:
        if not Path(p).is_file():
            print(f"{p}: missing, left out")
            continue
        text = Path(p).read_text()[skip_chars: skip_chars + 16 * length * count]
        ids = tok.encode(text, add_special_tokens=False)
        rr.append([ids[s: s + length] for s in range(0, len(ids) - length + 1, length)])
    out = []
    while len(out) < count and any(rr):
        for r in rr:
            if r and len(out) < count:
                out.append(r.pop(0))
    return out


# --- GPTQ of a batch of weights, each against its own H ---------------------------------------


def gptq_q4(W, H, damp: float, block: int = 128):
    """q4_gptq.gptq (q4nx) of W (B, N, K) float32 against H (B, K, K) float64, torch: each
    weight's codes (B, N, K) and its groups' scale and minimum (B, N, K / 32), numpy."""
    import torch

    W = W.clone()
    B, N, K = W.shape
    H = H.clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag == 0
    b, k = dead.nonzero(as_tuple=True)
    H[b, k, k] = 1
    W.masked_fill_(dead[:, None, :], 0)
    eye = torch.eye(K, dtype=H.dtype)
    H += damp * torch.diagonal(H, dim1=1, dim2=2).mean(1)[:, None, None] * eye
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    codes = torch.empty((B, N, K), dtype=torch.uint8)
    scale = np.empty((B, N, K // G), np.float32)
    mn = np.empty((B, N, K // G), np.float32)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, :, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            x = W1[:, :, i]
            if col % G == 0:
                s, m = iron_q4._fit(W1[:, :, i: i + G].reshape(-1, G).numpy())
                s, m = s.reshape(B, N), m.reshape(B, N)
                scale[:, :, col // G], mn[:, :, col // G] = s, m
                st = torch.as_tensor(np.where(s != 0, s, 1))
                mt, s0 = torch.as_tensor(m), torch.as_tensor(s)
            c = torch.clamp(torch.round((x - mt) / st), 0, 15)
            q = mt + s0 * c
            codes[:, :, col] = c.to(torch.uint8)
            err = (x - q) / Hi[:, i, i][:, None]
            W1[:, :, i:] -= err[:, :, None] * Hi[:, i, i:][:, None, :]
            E1[:, :, i] = err
        W[:, :, i2:] -= E1 @ Hinv[:, i1:i2, i2:]
    return codes.numpy(), scale, mn


def rtn(w) -> np.ndarray:
    n, k = w.shape
    return iron_q4.dequantize(iron_q4.quantize(w), n, k)


class Codes:
    """OUT/codes: each weight's q4nx blocks by iron_q4.weight_key, and index.json."""

    def __init__(self, out: Path):
        self.dir = out / "codes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = {}

    def put(self, name: str, w, codes, scale, mn) -> np.ndarray:
        """Save the blocks of w's codes; returns the weight they give (float32)."""
        blocks = iron_q4.pack(codes, scale, mn)
        k = iron_q4.weight_key(w)
        np.save(self.dir / f"{k}.npy", blocks)
        self.index[k] = dict(name=name, fmt="q4", shape=list(np.shape(w)))
        return iron_q4.dequantize(blocks, *np.shape(w))

    def write(self):
        (self.dir / "index.json").write_text(json.dumps(self.index, indent=1))


def gu_table(gate, up) -> np.ndarray:
    """iron_moeprobe.gu_table (that module needs IRON): an expert's gate and up as the (2 NE, K)
    table MoEGateUp reads, their column blocks of 64 out-features alternating, gate first."""
    ne, k = gate.shape
    return np.stack([gate.reshape(ne // CB, CB, k), up.reshape(ne // CB, CB, k)], 1).reshape(2 * ne, k)


def untable(t, ne: int):
    """gu_table's inverse: the table (2 NE, K) back to gate (NE, K) and up (NE, K)."""
    k = t.shape[1]
    b = t.reshape(ne // CB, 2, CB, k)
    return b[:, 0].reshape(ne, k), b[:, 1].reshape(ne, k)


# --- the model, a layer at a time -------------------------------------------------------------


def build(model_dir: Path):
    """The model on the meta device and a reader of its tensors (float32)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    where = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def tensor(name):
        stored = name if name in where else name.replace("model.", "model.language_model.", 1)
        f = where[stored]
        if f not in files:
            files[f] = safe_open(str(model_dir / f), framework="pt")
        return files[f].get_tensor(stored).float()

    def put(module, prefix, load):
        for n, p in list(module.named_parameters(recurse=True)):
            owner, _, leaf = n.rpartition(".")
            m = module.get_submodule(owner) if owner else module
            t = tensor(prefix + n) if load else torch.empty(p.shape, device="meta")
            m._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)

    for name in ("model.embed_tokens", "model.norm", "lm_head"):
        put(model.get_submodule(name), name + ".", True)
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=rotary.config)
    model.eval()
    return model, put


def layer_kwargs(model, B: int, L: int):
    """Each layer's keyword arguments for a batch of B sequences of L tokens: the model's forward
    run once with every layer an identity that records them."""
    import torch

    layers = model.model.layers
    got = [None] * len(layers)

    class Record(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, h, **kw):
            got[self.i] = kw
            return h

    saved = list(layers)
    for i in range(len(layers)):
        layers[i] = Record(i)
    with torch.no_grad():
        model.model(input_ids=torch.zeros((B, L), dtype=torch.long), use_cache=False)
    for i, m in enumerate(saved):
        layers[i] = m
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--calib-seqs", type=int, default=64)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--eval-seqs", type=int, default=6)
    ap.add_argument("--eval-len", type=int, default=512)
    ap.add_argument("--eval-skip", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8, help="sequences a forward call")
    ap.add_argument("--damp", type=float, default=0.1)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--experts-a-batch", type=int, default=32)
    ap.add_argument("--layers", type=int, default=0, help="only the first ones (a trial; no head)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    calib = texts(CALIB, tok, a.calib_len, a.calib_seqs)
    calib = calib[: len(calib) // a.batch * a.batch]  # whole batches (layer_kwargs' shape)
    ev = texts(EVAL, tok, a.eval_len, a.eval_seqs)
    print(f"{len(calib)} calibration sequences of {a.calib_len}, {len(ev)} evaluation of {a.eval_len}", flush=True)
    model, put = build(a.ckpt)
    layers = model.model.layers
    n_layers = a.layers or len(layers)
    kw_c = layer_kwargs(model, a.batch, a.calib_len)
    kw_e = layer_kwargs(model, 1, a.eval_len)
    emb = model.model.embed_tokens
    with torch.no_grad():
        X = torch.cat([emb(torch.tensor(calib[i: i + a.batch])) for i in range(0, len(calib), a.batch)])
        E = [emb(torch.tensor([s])) for s in ev]
    streams = dict(ref=list(E), rtn=list(E), gptq=list(E))
    codes = Codes(a.out)
    cfg = model.config.get_text_config()
    NE, n_exp = cfg.moe_intermediate_size, cfg.num_experts
    log = []

    def run(layer, x, kw):
        out = layer(x, **kw)
        return out[0] if isinstance(out, tuple) else out

    for li in range(n_layers):
        tl = time.time()
        layer = layers[li]
        put(layer, f"model.layers.{li}.", True)
        mlp = layer.mlp
        ex, sh = mlp.experts, mlp.shared_expert
        orig = dict(gu=ex.gate_up_proj.data.clone(), down=ex.down_proj.data.clone(),
                    sg=sh.gate_proj.weight.data.clone(), su=sh.up_proj.weight.data.clone(),
                    sd=sh.down_proj.weight.data.clone())

        # the experts' inputs: the calibration's tokens after the mixer and the norm
        seen = []

        class Stop(Exception):
            pass

        def grab(m, args):
            seen.append(args[0].reshape(-1, args[0].shape[-1]))
            raise Stop

        hook = mlp.register_forward_pre_hook(grab)
        with torch.no_grad():
            for i in range(0, len(X), a.batch):
                try:
                    run(layer, X[i: i + a.batch], kw_c[li])
                except Stop:
                    pass
        hook.remove()
        hn = torch.cat(seen)
        del seen
        with torch.no_grad():
            _, _, top = mlp.gate(hn)
        tc = time.time() - tl

        # routed experts: gate and up as the table, then down against the quantized activations
        gptq_w = dict(gu=orig["gu"].clone(), down=orig["down"].clone())
        rtn_w = dict(gu=orig["gu"].clone(), down=orig["down"].clone())
        tokens = [torch.nonzero((top == e).any(-1)).flatten() for e in range(n_exp)]
        counts = [len(t) for t in tokens]
        done = [e for e in range(n_exp) if counts[e] >= a.min_tokens]
        for e in range(n_exp):
            g, u = orig["gu"][e, :NE].numpy(), orig["gu"][e, NE:].numpy()
            gq, uq = untable(rtn(gu_table(g, u)), NE)
            rtn_w["gu"][e] = torch.as_tensor(np.concatenate([gq, uq]))
            rtn_w["down"][e] = torch.as_tensor(rtn(orig["down"][e].numpy()))
            if e not in done:  # the export rounds it: no codes
                gptq_w["gu"][e], gptq_w["down"][e] = rtn_w["gu"][e], rtn_w["down"][e]
        for b0 in range(0, len(done), a.experts_a_batch):
            batch = done[b0: b0 + a.experts_a_batch]
            tabs = [gu_table(orig["gu"][e, :NE].numpy(), orig["gu"][e, NE:].numpy()) for e in batch]
            H = torch.stack([(hn[tokens[e]].T @ hn[tokens[e]]).double() for e in batch])
            c, s, m = gptq_q4(torch.as_tensor(np.stack(tabs)), H, a.damp)
            acts = []
            for j, e in enumerate(batch):
                q = codes.put(f"layers.{li}.experts.{e}.gate_up", tabs[j], c[j], s[j], m[j])
                gq, uq = untable(q, NE)
                gptq_w["gu"][e] = torch.as_tensor(np.concatenate([gq, uq]))
                x = hn[tokens[e]]
                acts.append(F.silu(x @ torch.as_tensor(gq).T) * (x @ torch.as_tensor(uq).T))
            H = torch.stack([(t.T @ t).double() for t in acts])
            c, s, m = gptq_q4(torch.stack([orig["down"][e] for e in batch]), H, a.damp)
            for j, e in enumerate(batch):
                q = codes.put(f"layers.{li}.experts.{e}.down", orig["down"][e].numpy(), c[j], s[j], m[j])
                gptq_w["down"][e] = torch.as_tensor(q)
        # the shared expert: its Linears' gate and up, its table, its down
        Hs = (hn.T @ hn).double()[None]
        sg, su, sd = orig["sg"].numpy(), orig["su"].numpy(), orig["sd"].numpy()
        c, s, m = gptq_q4(torch.as_tensor(np.stack([sg, su])), torch.cat([Hs, Hs]), a.damp)
        gq = codes.put(f"layers.{li}.shared.gate", sg, c[0], s[0], m[0])
        uq = codes.put(f"layers.{li}.shared.up", su, c[1], s[1], m[1])
        tab = gu_table(sg, su)
        c, s, m = gptq_q4(torch.as_tensor(tab[None]), Hs, a.damp)
        codes.put(f"layers.{li}.shared.table", tab, c[0], s[0], m[0])
        act = F.silu(hn @ torch.as_tensor(gq).T) * (hn @ torch.as_tensor(uq).T)
        c, s, m = gptq_q4(torch.as_tensor(sd[None]), (act.T @ act).double()[None], a.damp)
        dq = codes.put(f"layers.{li}.shared.down", sd, c[0], s[0], m[0])
        gptq_w.update(sg=torch.as_tensor(gq), su=torch.as_tensor(uq), sd=torch.as_tensor(dq))
        rtn_w.update(sg=torch.as_tensor(rtn(sg)), su=torch.as_tensor(rtn(su)), sd=torch.as_tensor(rtn(sd)))
        del hn
        tq = time.time() - tl - tc

        def use(w):
            ex.gate_up_proj.data, ex.down_proj.data = w["gu"], w["down"]
            sh.gate_proj.weight.data, sh.up_proj.weight.data, sh.down_proj.weight.data = w["sg"], w["su"], w["sd"]

        with torch.no_grad():
            for name, w in (("ref", orig), ("rtn", rtn_w), ("gptq", gptq_w)):
                use(w)
                streams[name] = [run(layer, x, kw_e[li]) for x in streams[name]]
            use(gptq_w)
            X = torch.cat([run(layer, X[i: i + a.batch], kw_c[li]) for i in range(0, len(X), a.batch)])
        put(layer, f"model.layers.{li}.", False)
        del orig, rtn_w, gptq_w
        codes.write()
        rec = dict(layer=li, calibrated=len(done), fewest=min(counts), most=max(counts),
                   calib_s=round(tc), gptq_s=round(tq), all_s=round(time.time() - tl))
        log.append(rec)
        print(json.dumps(rec), flush=True)

    with torch.no_grad():
        norm, head = model.model.norm, model.lm_head
        result = dict(layers=log, damp=a.damp, calib_seqs=a.calib_seqs, calib_len=a.calib_len)
        if n_layers == len(layers):
            hf = torch.cat([norm(X[i: i + a.batch]).reshape(-1, X.shape[-1]) for i in range(0, len(X), a.batch)])
            W = head.weight.data.clone()
            Wr = torch.as_tensor(rtn(W.numpy()))
            Hh = (hf.T @ hf).double()[None]
            del hf
            Wg = torch.empty_like(W)
            rows = 8192
            blocks, scales, mins = [], [], []
            for r in range(0, W.shape[0], rows):
                c, s, m = gptq_q4(W[None, r: r + rows], Hh, a.damp)
                blocks.append(c[0]), scales.append(s[0]), mins.append(m[0])
            Wg = torch.as_tensor(codes.put("lm_head", W.numpy(), np.concatenate(blocks),
                                           np.concatenate(scales), np.concatenate(mins)))
            codes.write()
            heads = dict(ref=W, rtn=Wr, gptq=Wg)
        else:
            heads = dict(ref=head.weight.data, rtn=head.weight.data, gptq=head.weight.data)
        refs = [norm(x)[0, a.eval_skip:] @ heads["ref"].T for x in streams["ref"]]
        for name in ("rtn", "gptq"):
            st = [kl_stats(r, norm(x)[0, a.eval_skip:] @ heads[name].T) for r, x in zip(refs, streams[name])]
            result[name] = dict(kl=float(np.mean([s[0] for s in st])), median=float(np.mean([s[1] for s in st])),
                                top=float(np.mean([s[3] for s in st])), per_text=st)
            print(f"{name}: KL {result[name]['kl']:.4f} (median {result[name]['median']:.4f}), "
                  f"top-1 {result[name]['top']:.3f}", flush=True)
    result["seconds"] = round(time.time() - t0)
    (a.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{a.out}: {len(codes.index)} weights' codes, {result['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
