/* Smoke tests for calc.js and openrouter.js — run with: node tests/smoke.mjs
   No framework: plain assertions, exits non-zero on failure. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const load = (rel, name) => {
  const src = readFileSync(join(root, rel), "utf8");
  (0, eval)(src + `\nglobalThis.${name} = ${name};`);
  return globalThis[name];
};

const Calc = load("assets/js/calc.js", "Calc");
const ORAPI = load("assets/js/openrouter.js", "ORAPI");

let failed = 0;
function eq(actual, expected, label, tol = 1e-6) {
  const ok = typeof expected === "number"
    ? Math.abs(actual - expected) <= Math.abs(expected) * tol + 1e-9
    : Object.is(actual, expected);
  console.log((ok ? "  ok  " : "FAIL  ") + label + `  (${actual} vs ${expected})`);
  if (!ok) failed++;
}
function truthy(cond, label) {
  console.log((cond ? "  ok  " : "FAIL  ") + label);
  if (!cond) failed++;
}

console.log("— KV cache —");
const gqa = { type: "gqa", L: 80, h: 8192, k: 8, d: 128 };
eq(Calc.kvBytesPerToken(gqa, 2), 327680, "Llama-3.3-70B GQA KV/token (bf16)");
const mla = { type: "mla", L: 61, h: 7168, mlaRank: 512, mlaRope: 64 };
eq(Calc.kvBytesPerToken(mla, 2), 70272, "DeepSeek-V3 MLA KV/token (bf16)");
const mha = { type: "gqa", L: 32, h: 4096, k: 32, d: 128 };
eq(Calc.kvBytesPerToken(mha, 2), 524288, "Llama-2-7B MHA KV/token (bf16)");

console.log("— inference —");
const h100 = Calc.GPUS.find((g) => g.id === "h100-sxm");
const llama70 = { paramsB: 70.6, activeB: 70.6, wBytes: 2, kvBytes: 2, arch: gqa,
  ctx: 131072, avgCtx: 8192, batch: 32, promptTokens: 4096, gpu: h100, tp: 0,
  bwEff: 75, mfuPrefill: 50, overheadPct: 5, fixedGb: 2, usablePct: 90, linkGbps: 900 };
let r = Calc.inference(llama70);
eq(r.rawWeightsGb, 141.2, "70B bf16 raw weights = P × bytes (no overhead inside)");
eq(r.weightOvhGb, 7.06, "70B bf16 weight overhead = 5% of raw");
eq(r.totalGb, r.rawWeightsGb + r.weightOvhGb + r.kvAvgGb + 2 * r.tp, "display rows sum exactly to total");
eq(r.tp, 4, "auto TP: weights (148 GB) + KV (86 GB) across 70 GB usable → 4 GPUs");
eq(r.kvTok, 327680, "KV/token");
const expTps1 = (3350e9 * 4 * 0.75) / (70.6e9 * 2 + 327680 * 8192 + 4 * 80 * 8192 * 2 * 3 / 4);
eq(r.tps1, expTps1, "single-stream decode roofline (bandwidth-shared across TP)");
truthy(r.fits, "70B bf16 fits on 4×H100");
truthy(r.tps1 > 30 && r.tps1 < 120, "decode speed in a plausible 30–120 tok/s band (got " + r.tps1.toFixed(1) + ")");
truthy(r.ttft > 0.05 && r.ttft < 5, "TTFT for 4k prompt in 0.05–5 s band (got " + r.ttft.toFixed(2) + " s)");

const qwen30 = { ...llama70, paramsB: 30.5, activeB: 3.3, wBytes: 1, kvBytes: 2, arch: { type: "gqa", L: 48, k: 4, d: 128 } };
r = Calc.inference(qwen30);
eq(r.rawWeightsGb, 30.5, "Qwen3-30B-A3B fp8 raw weights, overhead shown separately");
eq(r.tp, 1, "fits on a single H100");
truthy(r.tps1 > 80, "MoE fp8 decode fast on one GPU (got " + r.tps1.toFixed(0) + " tok/s)");

const ds = { ...llama70, paramsB: 671, activeB: 37, arch: mla };
r = Calc.inference(ds);
eq(r.tp, Math.ceil(671 * 2 * 1.05 / 70), "DeepSeek-V3 bf16 needs ~21 H100s (weights-bound)");
eq(r.tps1, h100.bw * 1e9 * r.tp * 0.75 / (37e9 * 2 + 70272 * 8192 + 4 * 61 * 7168 * 2 * (r.tp - 1) / r.tp), "MLA decode roofline");

console.log("— training —");
const t8 = { paramsB: 8, activeB: 8, tokensB: 15000, bytes: Calc.OPTIMIZERS.adamw,
  arch: { type: "gqa", L: 32, h: 4096, a: 32, k: 8, d: 128 },
  shard: "none", gpus: 1, gpu: h100, mfu: 40, seq: 4096, gbatch: 512, mbatch: 1,
  actMode: "selective", actMult: 0, logits: true, vocab: 128256, price: 2,
  usablePct: 90, fixedGb: 2 };
r = Calc.training(t8);
eq(r.mem.paramsGb + r.mem.gradsGb + r.mem.optGb, 8 * 16, "AdamW static = 16 B/param");
truthy(!r.fits, "8B AdamW replica (128 GB) does not fit one H100");

const tZero3 = { ...t8, shard: "zero3", gpus: 0 };
r = Calc.training(tZero3);
truthy(r.fits && r.n <= 8, "auto-fit finds ≤8 GPUs for 8B with ZeRO-3 (found " + r.n + ")");
eq(r.flops, 6 * 8e9 * 15000e9, "6PD FLOPs for 8B × 15T tokens");
const days405 = (6 * 405e9 * 15600e9) / (16384 * h100.bf16 * 1e12 * 0.45) / 86400;
truthy(days405 > 50 && days405 < 75, "405B/15.6T on 16k H100 ≈ 50–75 days (got " + days405.toFixed(0) + ")");

console.log("— OpenRouter naming parser —");
const cases = [
  [{ id: "qwen/qwen3-235b-a22b", name: "Qwen: Qwen3 235B A22B" }, 235, 22],
  [{ id: "meta-llama/llama-3.3-70b-instruct", name: "Meta: Llama 3.3 70B Instruct" }, 70, 70],
  [{ id: "qwen/qwen3.8-2.4t-a95b", name: "Qwen: Qwen3.8 2.4T A95B" }, 2400, 95],
  [{ id: "tencent/hunyuan-a13b-instruct", name: "Tencent: Hunyuan A13B" }, null, 13],
  [{ id: "mistralai/mixtral-8x7b-instruct", name: "Mixtral 8x7B" }, 56, null],
  [{ id: "x-ai/grok-4.7", name: "SpaceXAI: Grok 4.7" }, null, null],
  [{ id: "qwen/qwen3-coder", name: "Qwen: Qwen3 Coder 480B A35B" }, 480, 35],
];
for (const [m, total, active] of cases) {
  const s = ORAPI.parseSize(m);
  eq(s.total, total, `total of ${m.id}`);
  eq(s.active, active, `active of ${m.id}`);
}

console.log(failed === 0 ? "\nALL TESTS PASSED" : `\n${failed} TEST(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
