/* Render smoke test — runs app.js against a minimal DOM built from index.html
   defaults, then asserts the results panel gets real values.
   Catches TDZ / wrong-branch / missing-id bugs in render paths.
   Run: node tests/render.mjs */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "index.html"), "utf8");

/* ---- build a stub DOM whose values come from the real markup defaults ---- */
const registry = new Map();
const listeners = new Map();

function mkEl(id) {
  let _value = "";
  const el = {
    id, textContent: "", innerHTML: "", hidden: false, checked: false,
    className: "", dataset: {}, children: [], style: {}, title: "",
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild(c) { this.children.push(c); },
    setAttribute() {},
    addEventListener(type, fn) {
      if (!listeners.has(id + ":" + type)) listeners.set(id + ":" + type, []);
      listeners.get(id + ":" + type).push(fn);
    },
    querySelectorAll() { return []; },
  };
  /* browsers coerce input.value assignments to strings — mirror that */
  Object.defineProperty(el, "value", {
    get: () => _value,
    set: (v) => { _value = v === null || v === undefined ? "" : String(v); },
    enumerable: true, configurable: true,
  });
  registry.set(id, el);
  return el;
}

/* inputs / selects: pull id + value (or first option) from the markup */
for (const m of html.matchAll(/<(input|select)\b[^>]*>/g)) {
  const tag = m[0];
  const id = /id="([^"]+)"/.exec(tag)?.[1];
  if (!id) continue;
  const el = mkEl(id);
  if (m[1] === "input") {
    el.value = /value="([^"]*)"/.exec(tag)?.[1] ?? "";
    el.checked = /\bchecked\b/.test(tag);
  } else {
    /* select: emulate the browser — honour `selected` (either attribute order),
       otherwise take the first option */
    const open = html.indexOf(tag);
    const close = html.indexOf("</select>", open);
    const body = html.slice(open, close);
    let chosen = null, first = null;
    for (const om of body.matchAll(/<option\b[^>]*>/g)) {
      const optTag = om[0];
      const v = /value="([^"]*)"/.exec(optTag)?.[1] ?? "";
      if (first === null) first = v;
      if (/\bselected\b/.test(optTag)) { chosen = v; break; }
    }
    el.value = chosen ?? first ?? "";
  }
}
/* every id that JS writes to must exist even if it is not an input */
for (const m of html.matchAll(/id="([a-z0-9-]+)"/gi)) if (!registry.has(m[1])) mkEl(m[1]);

const tabs = ["inference", "training", "method"].map((t) => {
  const el = mkEl("tab-" + t);
  el.dataset.tab = t;
  return el;
});
const panels = ["inference", "training", "method"].map((p) => {
  const el = mkEl("panel-" + p);
  el.dataset.panel = p;
  return el;
});

globalThis.document = {
  documentElement: { dataset: {} },
  getElementById: (id) => registry.get(id) || null,
  createElement: () => mkEl("_created" + Math.random()),
  querySelectorAll: (sel) => {
    if (sel === ".tab") return tabs;
    if (sel.includes("input")) return [...registry.values()];
    if (sel === ".tab-panel") return panels;
    if (sel.startsWith(".tab-panel")) return byPrefix(sel);
    return byPrefix(sel);
  },
  addEventListener() {},
};
function byPrefix(sel) {
  const m = /\[id\^="([^"]+)"\]/.exec(sel);
  if (!m) return [];
  return [...registry.values()].filter((el) => el.id.startsWith(m[1]));
}
globalThis.localStorage = { getItem: () => null, setItem() {} };
Object.defineProperty(globalThis, "navigator", {
  value: { clipboard: { writeText: async (t) => { globalThis.__copied = t; } } },
  configurable: true,
});
globalThis.matchMedia = () => ({ matches: false });

/* stub network: one fake repo fills successfully, anything with "no-such" fails */
const jsonRes = (obj) => Promise.resolve({
  ok: true, status: 200, json: () => Promise.resolve(obj),
});
globalThis.fetch = (url) => {
  const u = String(url);
  if (u.includes("no-such")) return Promise.reject(new Error("offline in test"));
  if (u.includes("/api/models/")) {
    return jsonRes({ id: "stub/Model-70B", gated: false, safetensors: { total: 70.6e9, parameters: { BF16: 70.6e9 } },
                     siblings: [{ rfilename: "config.json" }] });
  }
  if (u.includes("config.json")) {
    return jsonRes({ num_hidden_layers: 80, hidden_size: 8192, num_attention_heads: 64, num_key_value_heads: 8,
                     head_dim: 128, vocab_size: 128256, max_position_embeddings: 131072 });
  }
  return Promise.reject(new Error("offline in test"));
};

/* ------------------------------- run app.js ------------------------------- */
const appSrc = readFileSync(join(root, "assets/js/app.js"), "utf8");
const load = (f, name) => {
  const src = readFileSync(join(root, "assets/js", f), "utf8");
  (0, eval)(src + `\nglobalThis.${name} = ${name};`);
};
load("calc.js", "Calc");
load("openrouter.js", "ORAPI");
load("hf-arch.js", "HFArch");

let failed = 0;
function check(cond, label, extra = "") {
  console.log((cond ? "  ok  " : "FAIL  ") + label + (extra ? `  (${extra})` : ""));
  if (!cond) failed++;
}
const text = (id) => registry.get(id)?.textContent ?? "";

if (process.argv.includes("--dump")) {
  for (const k of ["t-actmult", "t-seq", "t-mbatch", "t-gbatch", "t-gpus", "t-shard", "t-mfu", "t-tokens",
                   "a-layers", "a-hidden", "a-heads", "a-kvheads", "a-experts", "a-ept",
                   "m-params", "m-active", "m-vocab", "i-fixedgb", "i-price", "i-gpu"]) {
    console.log(`  ${k.padEnd(12)} = ${JSON.stringify(registry.get(k)?.value)}`);
  }
}

let threw = null;
try { (0, eval)(appSrc); } catch (err) { threw = err; }
check(!threw, "app.js initialises without throwing", threw ? threw.message : "");

const notes = (id) => (registry.get(id)?.children ?? []).map((c) => c.textContent).join(" | ");
const invoke = async (key, arg) => {
  const fns = listeners.get(key);
  if (!Array.isArray(fns)) return null;
  let out;
  for (const fn of fns) out = await fn(arg);
  return out;
};

console.log("— empty start (nothing preloaded) —");
check(text("r-i-total") === "—", "results are blank, not stale", text("r-i-total"));
check(/no model loaded/.test(text("r-i-model")), "title says no model loaded", text("r-i-model"));
check(/No architecture loaded/.test(notes("r-i-notes")), "explicit unavailable note", notes("r-i-notes"));
check(registry.get("m-params").value === "", "params start empty", JSON.stringify(registry.get("m-params").value));
check(registry.get("a-layers").value === "", "architecture starts empty");
check(/Catalog not loaded/.test(registry.get("or-status").innerHTML), "catalog is not auto-fetched");

console.log("— preset selection computes —");
registry.get("m-preset").value = "llama33-70b";
await invoke("m-preset:change");
check(/GB|TB/.test(text("r-i-total")), "total memory rendered", text("r-i-total"));
check(/\d/.test(text("r-i-gpus")), "min GPUs rendered", text("r-i-gpus"));
check(/tok\/s/.test(text("r-i-tps")), "decode throughput rendered", text("r-i-tps"));
check(/\$[\d.]+\s*\/\s*1M/.test(text("r-i-cost-out")), "self-host output cost rendered", text("r-i-cost-out"));
check(/\$[\d.]+\s*\/\s*1M/.test(text("r-i-cost-in")), "self-host input cost rendered", text("r-i-cost-in"));
check(/KB|MB|GB/.test(text("r-i-kvtok")), "KV/token rendered", text("r-i-kvtok"));
check(/kW/.test(text("r-i-marginal")), "marginal shown by default ($0.12/kWh)", text("r-i-marginal"));
check(registry.get("i-price").value === "2.5", "inference GPU price synced to H100", registry.get("i-price").value);
registry.get("i-elec").value = "";
await invoke("i-elec:input");
check(/set electricity rate/.test(text("r-i-marginal")), "marginal row hidden with hint when rate cleared", text("r-i-marginal"));
registry.get("i-elec").value = "0.12";
await invoke("i-elec:input");

console.log("— failed auto-fill resets to not available —");
registry.get("a-hfrepo").value = "no-such-repo/x";
await invoke("a-autofill:click");
check(registry.get("a-layers").value === "", "stale layers cleared", JSON.stringify(registry.get("a-layers").value));
check(registry.get("a-kvheads").value === "", "stale KV heads cleared");
check(text("r-i-total") === "—", "results blanked after failure", text("r-i-total"));
check(/Auto-fill failed/.test(registry.get("a-hfstatus").textContent), "failure is reported, not silent", registry.get("a-hfstatus").textContent.slice(0, 90));

console.log("— successful auto-fill populates + computes —");
registry.get("a-hfrepo").value = "stub/Model-70B";
await invoke("a-autofill:click");
check(registry.get("a-layers").value === "80", "layers filled", registry.get("a-layers").value);
check(registry.get("a-kvheads").value === "8", "KV heads filled");
check(registry.get("m-params").value === "70.6", "exact params filled", registry.get("m-params").value);
check(/GB|TB/.test(text("r-i-total")), "results computed from filled arch", text("r-i-total"));
check(/exact params 70\.60B/.test(registry.get("a-hfstatus").innerHTML), "status cites source", registry.get("a-hfstatus").innerHTML.slice(0, 80));

console.log("— gpu price follows selection + marginal electricity cost —");
registry.get("i-gpu").value = "b200";
await invoke("i-gpu:change");
check(registry.get("i-price").value === "4.5", "switching GPU updates $/h price", registry.get("i-price").value);
registry.get("i-pue").value = "1.3";
await invoke("i-elec:input");
const marg = text("r-i-marginal");
check(/kW/.test(marg) && /\$[\d.]+\s*\/\s*1M/.test(marg), "marginal row shows $/1M + facility kW", marg);

console.log("— copy report (while model loaded) —");
globalThis.__copied = null;
await invoke("tab-inference:click");
await invoke("copy-inference:click");
const cp = globalThis.__copied || "";
check(/AI Infra Calculator — Inference report/.test(cp), "inference report header");
check(/Source: /.test(cp), "inference report has provenance");
check(/MODEL[\s\S]*RESULTS[\s\S]*Total memory:/.test(cp), "inference report has model + results", cp.slice(0, 120));
globalThis.__copied = null;
await invoke("tab-training:click");
await invoke("copy-training:click");
const cpt = globalThis.__copied || "";
check(/AI Infra Calculator — Training report/.test(cpt), "training report header");
check(/MODEL[\s\S]*RESULTS[\s\S]*Memory\/GPU:/.test(cpt), "training report has model + results");
globalThis.__copied = null;
await invoke("tab-inference:click");

console.log("— custom preset clears —");
registry.get("m-preset").value = "custom";
await invoke("m-preset:change");
check(registry.get("m-params").value === "", "custom clears params");
check(text("r-i-total") === "—", "results blank for custom");

console.log("— tab switch → training render —");
registry.get("m-preset").value = "llama33-70b";
await invoke("m-preset:change");
const clickTraining = (listeners.get("tab-training:click") || []).find((f) => typeof f === "function");
check(typeof clickTraining === "function", "training tab handler registered");
if (clickTraining) {
  try { clickTraining(); } catch (err) { threw = err; }
  check(!threw, "training render does not throw", threw ? threw.message : "");
  check(/GB|TB/.test(text("r-t-mem")), "memory/GPU rendered", text("r-t-mem"));
  check(/\d/.test(text("r-t-gpus")), "GPU count rendered", text("r-t-gpus"));
  check(/FLOP/.test(text("r-t-flops")), "compute rendered", text("r-t-flops"));
  check(/\$/.test(text("r-t-cost")), "training cost rendered", text("r-t-cost"));
  check(/\d/.test(text("r-t-accum")), "gradient accumulation rendered", text("r-t-accum"));
}

console.log(failed === 0 ? "\nRENDER TEST PASSED" : `\n${failed} RENDER CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
