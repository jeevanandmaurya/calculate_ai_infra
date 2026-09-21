/* ============================================================
   calc.js — pure calculation engine (no DOM access).
   All formulas are documented in docs/calculations.md.
   Units: params in billions (B params), memory in GB (1e9 B),
   bandwidth in GB/s, compute in TFLOP/s, times in seconds.
   ============================================================ */
"use strict";

const Calc = (() => {

  /* ------------------------- hardware table ------------------------- */
  /* vram: usable HBM/GDDR in GB. bw: peak HBM bandwidth GB/s.
     bf16/fp8: dense tensor TFLOP/s (no sparsity). price: ~cloud $/h. */
  const GPUS = [
    { id: "h100-sxm", name: "H100 SXM 80GB",     vram: 80,  bw: 3350, bf16: 989,  fp8: 1979, price: 2.5,  tdp: 700 },
    { id: "h200",     name: "H200 141GB",        vram: 141, bw: 4800, bf16: 989,  fp8: 1979, price: 3.0,  tdp: 700 },
    { id: "b200",     name: "B200 180GB",        vram: 180, bw: 8000, bf16: 2250, fp8: 4500, price: 4.5,  tdp: 1000 },
    { id: "b300",     name: "B300 288GB",        vram: 288, bw: 8000, bf16: 2500, fp8: 5000, price: 5.5,  tdp: 1400 },
    { id: "a100-80",  name: "A100 80GB",         vram: 80,  bw: 2039, bf16: 312,  fp8: 624,  price: 1.5,  tdp: 400 },
    { id: "a100-40",  name: "A100 40GB",         vram: 40,  bw: 1555, bf16: 312,  fp8: 624,  price: 1.2,  tdp: 400 },
    { id: "mi300x",   name: "MI300X 192GB",      vram: 192, bw: 5300, bf16: 1300, fp8: 2600, price: 2.5,  tdp: 750 },
    { id: "mi325x",   name: "MI325X 256GB",      vram: 256, bw: 6000, bf16: 1300, fp8: 2600, price: 3.0,  tdp: 1000 },
    { id: "l40s",     name: "L40S 48GB",         vram: 48,  bw: 864,  bf16: 362,  fp8: 733,  price: 1.0,  tdp: 350 },
    { id: "rtx6000a", name: "RTX 6000 Ada 48GB", vram: 48,  bw: 960,  bf16: 91,   fp8: 182,  price: 1.0,  tdp: 300 },
    { id: "rtx-5090", name: "RTX 5090 32GB",     vram: 32,  bw: 1792, bf16: 209,  fp8: 419,  price: 0.5,  tdp: 575 },
    { id: "rtx-4090", name: "RTX 4090 24GB",     vram: 24,  bw: 1008, bf16: 165,  fp8: 330,  price: 0.4,  tdp: 450 },
    { id: "rtx-3090", name: "RTX 3090 24GB",     vram: 24,  bw: 936,  bf16: 71,   fp8: 142,  price: 0.3,  tdp: 350 },
    { id: "l4",       name: "L4 24GB",           vram: 24,  bw: 300,  bf16: 121,  fp8: 242,  price: 0.6,  tdp: 72 },
    { id: "custom",   name: "Custom (editable)", vram: 80,  bw: 3350, bf16: 989,  fp8: 1979, price: 2.0,  tdp: 700 },
  ];

  /* ------------------- verified architecture presets -------------------
     Configs cross-checked against Hugging Face config.json (2026-09-21).
     Meta/Google entries use vendor-published values (repos are gated).
     type: gqa (covers MHA when k=a, MQA when k=1) · mla: DeepSeek-style. */
  const PRESETS = [
    { id: "custom",       name: "Custom",                params: 8,    active: 8,    type: "gqa", L: 36,  h: 4096,  a: 32,  k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 128256, ctx: 40960 },
    { id: "llama31-8b",   name: "Llama 3.1 8B",          params: 8.0,  active: 8.0,  type: "gqa", L: 32,  h: 4096,  a: 32,  k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 128256, ctx: 131072 },
    { id: "llama33-70b",  name: "Llama 3.3 70B",         params: 70.6, active: 70.6, type: "gqa", L: 80,  h: 8192,  a: 64,  k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 128256, ctx: 131072 },
    { id: "llama31-405b", name: "Llama 3.1 405B",        params: 405,  active: 405,  type: "gqa", L: 126, h: 16384, a: 128, k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 128256, ctx: 131072 },
    { id: "qwen3-8b",     name: "Qwen3 8B",              params: 8.2,  active: 8.2,  type: "gqa", L: 36,  h: 4096,  a: 32,  k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 151936, ctx: 40960 },
    { id: "qwen3-30b",    name: "Qwen3 30B-A3B (MoE)",   params: 30.5, active: 3.3,  type: "gqa", L: 48,  h: 2048,  a: 32,  k: 4,  d: 128, mlaRank: 512, mlaRope: 64, experts: 128, ept: 8, vocab: 151936, ctx: 40960 },
    { id: "qwen3-235b",   name: "Qwen3 235B-A22B (MoE)", params: 235,  active: 22,   type: "gqa", L: 94,  h: 4096,  a: 64,  k: 4,  d: 128, mlaRank: 512, mlaRope: 64, experts: 128, ept: 8, vocab: 151936, ctx: 262144 },
    { id: "deepseek-v3",  name: "DeepSeek-V3 (MoE+MLA)", params: 671,  active: 37,   type: "mla", L: 61,  h: 7168,  a: 128, k: 128, d: 128, mlaRank: 512, mlaRope: 64, experts: 256, ept: 8, vocab: 129280, ctx: 163840 },
    { id: "mistral-7b",   name: "Mistral 7B v0.3",       params: 7.3,  active: 7.3,  type: "gqa", L: 32,  h: 4096,  a: 32,  k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 32768,  ctx: 32768 },
    { id: "mixtral-8x7b", name: "Mixtral 8x7B (MoE)",    params: 46.7, active: 12.9, type: "gqa", L: 32,  h: 4096,  a: 32,  k: 8,  d: 128, mlaRank: 512, mlaRope: 64, experts: 8,   ept: 2, vocab: 32000,  ctx: 32768 },
    { id: "gemma3-27b",   name: "Gemma 3 27B",           params: 27.4, active: 27.4, type: "gqa", L: 62,  h: 5376,  a: 32,  k: 16, d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 262144, ctx: 131072 },
    { id: "phi4-14b",     name: "Phi-4 14B",             params: 14.7, active: 14.7, type: "gqa", L: 40,  h: 5120,  a: 40,  k: 10, d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 100352, ctx: 16384 },
    { id: "llama2-7b",    name: "Llama 2 7B (MHA)",      params: 6.7,  active: 6.7,  type: "gqa", L: 32,  h: 4096,  a: 32,  k: 32, d: 128, mlaRank: 512, mlaRope: 64, experts: 0,   ept: 0, vocab: 32000,  ctx: 4096 },
  ];

  /* Optimizer memory, bytes per parameter: {p params, g grads, master fp32
     copy, m 1st moment, v 2nd moment}. */
  const OPTIMIZERS = {
    adamw:     { p: 2, g: 2, master: 4, m: 4, v: 4, label: "AdamW (16 B/param)" },
    adamw8:    { p: 2, g: 2, master: 4, m: 1, v: 1, label: "AdamW 8-bit (10 B/param)" },
    adafactor: { p: 2, g: 2, master: 4, m: 0, v: 0, label: "Adafactor (8 B/param, +ε factored)" },
    sgd:       { p: 2, g: 2, master: 4, m: 4, v: 0, label: "SGD + momentum (12 B/param)" },
  };

  const G = 1e9; /* bytes per GB (decimal, matches vendor VRAM specs) */

  /* ---------------------- marginal (electricity-only) cost ----------------------
     For owned hardware — sunk capex, so NOT comparable to the amortized
     self-host figure or an API price; it is the marginal cost of one more
     token. Returns null when the electricity rate is unset (no safe default). */
  function computeElectricityCost(o) {
    if (!o.elecRate || o.elecRate <= 0) return null;
    const tdpW = (o.gpuTdpWatts || 0) > 0 ? o.gpuTdpWatts : 0;
    if (!tdpW) return null;
    const gpuPowerKw = o.numGpus * tdpW / 1000;
    const hostPowerKw = gpuPowerKw * (o.hostFrac ?? 15) / 100;
    const facilityPowerKw = (gpuPowerKw + hostPowerKw) * (o.pue ?? 1.3);
    const costPerHour = facilityPowerKw * o.elecRate;
    const costPer1M = o.throughput > 0 ? (costPerHour / (o.throughput * 3600)) * 1e6 : null;
    return { gpuPowerKw, hostPowerKw, facilityPowerKw, costPerHour, costPer1M };
  }

  /* ------------------------- KV cache per token -------------------------
     GQA/MHA/MQA: 2 (K+V) × layers × kv_heads × head_dim × bytes.
     MLA: layers × (kv_lora_rank + rope_dim) × bytes — one compressed
     latent per layer replaces all K and V heads (DeepSeek-V3 paper). */
  function kvBytesPerToken(arch, kvBytes) {
    if (arch.type === "mla") {
      return arch.L * (arch.mlaRank + arch.mlaRope) * kvBytes;
    }
    return 2 * arch.L * arch.k * arch.d * kvBytes;
  }

  /* ------------------------------ inference ------------------------------
     Memory bookkeeping (kept additive so every row sums to the total):
       rawWeightsGb = P × b_w                             (checkpoint bytes)
       weightOvhGb  = raw × overheadPct                   (fragmentation, workspace)
       kvAvgGb      = KV_tok × S_avg × batch              (resident KV state)
       fixedGb      = fixedGb × tp                        (per-GPU CUDA/framework runtime)
       totalGb      = rawWeightsGb + weightOvhGb + kvAvgGb + fixedGb
     Fit check compares only what shards across GPUs (raw + overhead + KV)
     against usable VRAM (VRAM × usable% − fixed), because the fixed runtime
     was already subtracted from the usable-per-GPU figure. */
  function inference(o) {
    const arch = o.arch;
    const rawWeightsGb = o.paramsB * o.wBytes;
    const weightOvhGb = rawWeightsGb * o.overheadPct / 100;
    const weightsGb = rawWeightsGb + weightOvhGb;
    const kvTok = kvBytesPerToken(arch, o.kvBytes);
    const kvAvgGb = kvTok * o.avgCtx * o.batch / G;
    const kvMaxGb = kvTok * o.ctx / G;
    const modelGb = weightsGb + kvAvgGb;              /* sharded across TP */
    const usablePerGpu = o.gpu.vram * o.usablePct / 100 - o.fixedGb;
    const tpAuto = Math.max(Math.ceil(weightsGb / usablePerGpu),
                            Math.ceil(modelGb / usablePerGpu), 1);
    const tp = o.tp > 0 ? o.tp : tpAuto;
    const fixedAllGb = o.fixedGb * tp;
    const totalGb = modelGb + fixedAllGb;

    /* memory-bound decode: bytes read per token-step across the TP group */
    const weightTokBytes = o.activeB * G * o.wBytes;
    /* TP all-reduce: 2 all-reduces/layer (attn out, MLP out) × 2× data
       volume of an all-reduce, on h-sized activations, scaled (tp-1)/tp */
    const comm = (tp > 1 && o.linkGbps > 0 && arch.h)
      ? 4 * arch.L * arch.h * o.wBytes * (tp - 1) / tp : 0;
    const bwEff = o.gpu.bw * G * tp * o.bwEff / 100;

    const tps1 = bwEff / (weightTokBytes + kvTok * o.avgCtx + comm);
    const sysTps = bwEff * o.batch /
      (weightTokBytes + o.batch * (kvTok * o.avgCtx + comm));

    /* prefill is compute-bound: 2 × P_active × prompt tokens FLOPs */
    const ttft = 2 * o.activeB * G * o.promptTokens /
      (o.gpu.bf16 * 1e12 * tp * o.mfuPrefill / 100);

    const fits = modelGb <= usablePerGpu * tp + 1e-9;

    return {
      rawWeightsGb, weightOvhGb, weightsGb,
      kvTok, kvAvgGb, kvMaxGb, modelGb, totalGb, usablePerGpu,
      tp, tpAuto, fits, comm, weightTokBytes, tps1, sysTps, ttft,
      overheadGb: rawWeightsGb * o.overheadPct / 100 + o.fixedGb * tp,
    };
  }

  /* ------------------------------ training ------------------------------ */
  function shardDivisor(stage, n) {
    switch (stage) {
      case "zero1": return { opt: n, g: 1, p: 1 };
      case "zero2": return { opt: n, g: n, p: 1 };
      case "zero3": return { opt: n, g: n, p: n };
      default:      return { opt: 1, g: 1, p: 1 };
    }
  }

  /* Korthikanti et al. 2022 (bf16, no sequence parallelism):
     none: (34 + 5·a·s/h) · selective: 34 · full: ~2 bytes/element·layer */
  function actCoefficient(o) {
    const over = parseFloat(o.actMult);
    if (Number.isFinite(over) && over > 0) return over;
    if (o.actMode === "full") return 2;
    if (o.actMode === "selective") return 34;
    return 34 + 5 * o.arch.a * o.seq / o.arch.h;
  }

  function trainingPerGpu(o, n) {
    const s = shardDivisor(o.shard, n);
    const paramsGb = o.paramsB * o.bytes.p / s.p;
    const gradsGb = o.paramsB * o.bytes.g / s.g;
    const optGb = o.paramsB * (o.bytes.master + o.bytes.m + o.bytes.v) / s.opt;
    const actGb = o.arch.L * o.seq * o.mbatch * o.arch.h * actCoefficient(o) / G;
    const logitsGb = o.logits ? o.seq * o.mbatch * o.vocab * 4 * 2 / G : 0;
    return { paramsGb, gradsGb, optGb, actGb, logitsGb,
             total: paramsGb + gradsGb + optGb + actGb + logitsGb };
  }

  function training(o) {
    const usable = o.gpu.vram * (o.usablePct ?? 90) / 100 - (o.fixedGb ?? 2);
    let n = o.gpus > 0 ? o.gpus : 1;
    let mem = trainingPerGpu(o, n);
    while (o.gpus <= 0 && mem.total > usable && n < 1024) {
      n = n < 8 ? n + 1 : n * 2;
      mem = trainingPerGpu(o, n);
    }
    const fits = mem.total <= usable + 1e-9;
    const flops = 6 * o.activeB * G * o.tokensB * G;   /* 6PD, Kaplan et al. 2020 */
    const timeS = flops / (n * o.gpu.bf16 * 1e12 * o.mfu / 100);
    const tps = o.tokensB * G / timeS;
    const accum = Math.max(1, Math.round(o.gbatch / (n * o.mbatch)));
    const ckptGb = o.paramsB * (o.bytes.p + o.bytes.master + o.bytes.m + o.bytes.v);
    const cost = n * (timeS / 3600) * o.price;
    return { n, mem, usable, fits, flops, timeS, tps, accum, ckptGb, cost, actC: actCoefficient(o) };
  }

  return { GPUS, PRESETS, OPTIMIZERS, G, kvBytesPerToken, inference, training, trainingPerGpu, computeElectricityCost };
})();

