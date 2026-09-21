/* Live integration tests for hf-arch.js — network required, no API key.
   Run: node tests/hf-arch.mjs */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = readFileSync(join(root, "assets/js/hf-arch.js"), "utf8");
(0, eval)(src + "\nglobalThis.HFArch = HFArch;");
const HFArch = globalThis.HFArch;

let failed = 0;
function check(cond, label, extra = "") {
  console.log((cond ? "  ok  " : "FAIL  ") + label + (extra ? `  (${extra})` : ""));
  if (!cond) failed++;
}

console.log("— config.json path (public repo) —");
{
  const r = await HFArch.autoFill("Qwen/Qwen3-30B-A3B");
  const a = r.arch;
  check(a.source === "config.json", "source is config.json");
  check(a.L === 48 && a.h === 2048, "Qwen3-30B-A3B layers/hidden", `L=${a.L} h=${a.h}`);
  check(a.k === 4, "GQA kv heads = 4", `k=${a.k}`);
  check(a.experts === 128 && a.ept === 8, "MoE 128 experts, 8 used");
  check(a.vocab === 151936 && a.ctx === 40960, "vocab + context", `${a.vocab}/${a.ctx}`);
  check(r.paramsB && Math.abs(r.paramsB - 30.5) < 1, "exact params from safetensors", r.paramsB && r.paramsB.toFixed(2) + "B");
}

console.log("— GGUF header path (repo without config.json) —");
{
  const r = await HFArch.autoFill("Qwen/Qwen3-8B-GGUF");
  const a = r.arch;
  check(a.source === "gguf-header", "source is gguf-header");
  check(a.L === 36 && a.h === 4096, "Qwen3-8B layers/hidden", `L=${a.L} h=${a.h}`);
  check(a.k === 8 && a.d === 128, "kv heads / head dim", `k=${a.k} d=${a.d}`);
  check(a.ctx === 40960, "context from GGUF", String(a.ctx));
  check(r.paramsB === 8, "params from general.size_label", r.paramsB + "B");
}

console.log("— gated repo → public GGUF mirror —");
{
  const r = await HFArch.autoFill("meta-llama/Llama-3.3-70B-Instruct");
  const a = r.arch;
  check(r.notes.some((n) => /gated|mirror/i.test(n)), "gated path noted", r.notes.join(" | "));
  check(a.L === 80 && a.h === 8192, "Llama-3.3-70B layers/hidden from mirror", `L=${a.L} h=${a.h}`);
  check(a.k === 8 && a.d === 128, "kv heads / head dim", `k=${a.k} d=${a.d}`);
  check(r.paramsB && Math.abs(r.paramsB - 70.6) < 3, "params from size_label", r.paramsB && r.paramsB + "B");
  check(a.ctx >= 131072, "context ≥ 128k", String(a.ctx));
}

console.log("— MLA model (DeepSeek) —");
{
  const r = await HFArch.autoFill("deepseek-ai/DeepSeek-V3.1");
  const a = r.arch;
  check(a.type === "mla" && a.mlaRank === 512 && a.mlaRope === 64, "MLA rank/rope", `r=${a.mlaRank} d_r=${a.mlaRope}`);
  check(a.L === 61 && a.h === 7168, "layers/hidden", `L=${a.L} h=${a.h}`);
  check(a.experts === 256 && a.ept === 8, "n_routed_experts mapping", `E=${a.experts} ept=${a.ept}`);
  check(r.paramsB && r.paramsB > 660 && r.paramsB < 690, "params ≈ 671B", r.paramsB && r.paramsB.toFixed(1) + "B");
}

console.log("— MoE under text_config + n_routed_experts (V4 arch) —");
{
  const r = await HFArch.autoFill("deepseek-ai/DeepSeek-V4.1-Flash");
  const a = r.arch;
  check(a.experts === 384 && a.ept === 6, "n_routed_experts=384, ept=6", `E=${a.experts} ept=${a.ept}`);
  check(a.L === 40 && a.h === 5120 && a.k === 1, "layers/hidden/kv", `L=${a.L} h=${a.h} k=${a.k}`);
  check(a.moe === true, "MoE flag set");
  check(a.activeEstimateB && a.activeEstimateB > 10 && a.activeEstimateB < 80,
    "active-params estimate in a sane 10–80B band", a.activeEstimateB && a.activeEstimateB.toFixed(1) + "B");
  check(r.paramsB > 700, "total params from safetensors", r.paramsB && r.paramsB.toFixed(1) + "B");
}

console.log(failed === 0 ? "\nALL HF-ARCH TESTS PASSED" : `\n${failed} TEST(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
