/* ============================================================
   app.js — UI wiring: state collection, rendering, presets,
   tabs, theme, and the OpenRouter picker.
   ============================================================ */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const num = (id, fallback = 0) => {
    const el = $(id);
    const v = el ? parseFloat(el.value) : NaN;
    return Number.isFinite(v) ? v : fallback;
  };

  /* ------------------------- formatting helpers ------------------------- */
  const fmtGB = (gb) => {
    if (!Number.isFinite(gb)) return "—";
    if (gb >= 1024) return (gb / 1024).toFixed(2) + " TB";
    if (gb >= 100) return gb.toFixed(0) + " GB";
    return gb.toFixed(1) + " GB";
  };
  const fmtBytes = (b) => {
    if (!Number.isFinite(b)) return "—";
    if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
    return b.toFixed(0) + " B";
  };
  const fmtTps = (x) => {
    if (!Number.isFinite(x)) return "—";
    if (x >= 1e6) return (x / 1e6).toFixed(2) + "M tok/s";
    if (x >= 1e3) return (x / 1e3).toFixed(1) + "k tok/s";
    return x.toFixed(1) + " tok/s";
  };
  const fmtFlops = (f) => (Number.isFinite(f) ? f.toExponential(2) + " FLOP" : "—");
  const fmtTime = (s) => {
    if (!Number.isFinite(s) || s < 0) return "—";
    if (s < 0.1) return (s * 1000).toFixed(1) + " ms";
    if (s < 90) return s.toFixed(1) + " s";
    if (s < 5400) return (s / 60).toFixed(1) + " min";
    if (s < 259200) return (s / 3600).toFixed(1) + " h";
    if (s < 15778800) return (s / 86400).toFixed(1) + " days";
    return (s / 31557600).toFixed(2) + " yr";
  };
  const fmtUSD = (c) => {
    if (!Number.isFinite(c)) return "—";
    if (c >= 1e6) return "$" + (c / 1e6).toFixed(2) + "M";
    if (c >= 1e3) return "$" + (c / 1e3).toFixed(1) + "k";
    return "$" + c.toFixed(0);
  };

  /* ------------------------------ populates ------------------------------ */
  function fillSelect(el, items, getLabel, getValue) {
    el.innerHTML = "";
    for (const it of items) {
      const opt = document.createElement("option");
      opt.value = getValue(it);
      opt.textContent = getLabel(it);
      el.appendChild(opt);
    }
  }

  fillSelect($("m-preset"), Calc.PRESETS, (p) => p.name, (p) => p.id);
  for (const id of ["i-gpu", "t-gpu"]) {
    fillSelect($(id), Calc.GPUS, (g) => `${g.name} · ${g.vram} GB · ${g.bw} GB/s`, (g) => g.id);
  }
  $("i-gpu").value = "h100-sxm";
  $("t-gpu").value = "h100-sxm";

  function applyPreset(id) {
    if (id === "custom") {
      /* Custom means empty — never invent values for the user. */
      $("m-name").value = "";
      $("m-params").value = "";
      $("m-active").value = "";
      clearArchitecture("Custom model — enter parameters and architecture manually, or auto-fill from a Hugging Face repo.");
      recalc();
      return;
    }
    const p = Calc.PRESETS.find((x) => x.id === id);
    if (!p) return;
    lastSource = "preset: " + p.name;
    $("a-hfstatus").innerHTML =
      `Preset “${escapeHtml(p.name)}” — values from the vendor-published config (verified against Hugging Face).`;
    $("m-name").value = p.name;
    $("m-params").value = p.params;
    $("m-active").value = p.active;
    $("m-vocab").value = p.vocab;
    $("a-type").value = p.type;
    $("a-layers").value = p.L;
    $("a-hidden").value = p.h;
    $("a-heads").value = p.a;
    $("a-kvheads").value = p.k;
    $("a-headdim").value = p.d;
    $("a-mla-rank").value = p.mlaRank;
    $("a-mla-rope").value = p.mlaRope;
    $("a-experts").value = p.experts;
    $("a-ept").value = p.ept;
    $("i-ctx").value = p.ctx;
    $("i-avgctx").value = Math.min(8192, p.ctx);
    syncArchVisibility();
  }

  function syncArchVisibility() {
    const mla = $("a-type").value === "mla";
    for (const el of document.querySelectorAll(".field-gqa")) el.hidden = mla;
    for (const el of document.querySelectorAll(".field-mla")) el.hidden = !mla;
  }

  /* ------------------------------ gpu / custom ------------------------------ */
  function gpuFor(prefix) {
    const id = $(prefix + "-gpu").value;
    const base = Calc.GPUS.find((g) => g.id === id);
    const priceId = prefix === "i" ? "i-price" : "t-price";
    if (id !== "custom") return { ...base, price: num(priceId, base.price) };
    return {
      ...base,
      vram: num(prefix + "-cvram", 80),
      bw: num(prefix + "-cbw", 3350),
      bf16: num(prefix + "-ctf", 989),
      fp8: num(prefix + "-ctf", 989),
      tdp: num(prefix + "-ctdp", 700),
      price: num(priceId, 2),
    };
  }
  function syncCustomHw(prefix) {
    $(prefix + "-custom-hw").hidden = $(prefix + "-gpu").value !== "custom";
  }
  /* Keep the $/h price input in sync with the selected GPU so the
     self-host $/1M estimate never uses a stale price from the previous
     selection. A manual edit afterwards still wins until the next change. */
  function syncGpuPrice(prefix) {
    const id = $(prefix + "-gpu").value;
    if (id === "custom") return;
    const base = Calc.GPUS.find((g) => g.id === id);
    if (!base) return;
    $(prefix === "i" ? "i-price" : "t-price").value = base.price;
  }

  /* ------------------------------ tabs & theme ------------------------------ */
  let activeTab = "inference";
  function setTab(name) {
    activeTab = name;
    for (const btn of document.querySelectorAll(".tab")) {
      const on = btn.dataset.tab === name;
      btn.classList.toggle("is-active", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    }
    for (const el of document.querySelectorAll(".tab-panel")) {
      el.hidden = el.dataset.panel !== name;
    }
    recalc();
  }

  const theme = localStorage.getItem("aiic-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("aiic-theme", next);
  });

  /* ------------------------------ collect state ------------------------------ */
  function archState() {
    return {
      type: $("a-type").value,
      L: num("a-layers", 0), h: num("a-hidden", 0), a: num("a-heads", 0),
      k: num("a-kvheads", 0), d: num("a-headdim", 0),
      mlaRank: num("a-mla-rank", 0), mlaRope: num("a-mla-rope", 0),
      experts: num("a-experts", 0), ept: num("a-ept", 0),
    };
  }

  /* An architecture is only usable when every field the math needs is present. */
  function archReady(a) {
    if (!a.L || !a.h || !a.a || !a.k || !a.d) return false;
    if (a.type === "mla" && !(a.mlaRank + a.mlaRope)) return false;
    return true;
  }

  /* Result fields per panel — explicit so blanking never depends on DOM traversal. */
  const RESULT_IDS = {
    inference: ["r-i-total", "r-i-gpus", "r-i-tps", "r-i-ttft", "r-i-tpn", "r-i-weights",
                "r-i-ovhpct", "r-i-weightovh", "r-i-kv", "r-i-kvmax", "r-i-fixedn", "r-i-ovh",
                "r-i-kvtok", "r-i-totalt", "r-i-sys",
                "r-i-cost-out", "r-i-marginal", "r-i-cost-in", "r-i-api", "r-i-comm", "r-i-wtok"],
    training: ["r-t-mem", "r-t-gpus", "r-t-flops", "r-t-time", "r-t-params", "r-t-grads",
               "r-t-opt", "r-t-act", "r-t-logits", "r-t-total", "r-t-tps", "r-t-accum",
               "r-t-ckpt", "r-t-cost"],
  };

  /* Wipe a results panel so stale numbers can never masquerade as fresh ones. */
  function unavailable(panel, msg) {
    const prefix = panel === "inference" ? "r-i" : "r-t";
    if (panel === "inference") lastInference = null;
    else lastTraining = null;
    for (const id of RESULT_IDS[panel]) setText(id, "—");
    const bar = $(`${prefix}-bar`);
    if (bar) bar.innerHTML = "";
    setText(`${prefix}-model`, "· no model loaded");
    renderNotes($(`${prefix}-notes`), [{ kind: "warn", text: msg }]);
  }

  /* Reset every model-derived field to "not available". */
  function clearArchitecture(msg) {
    $("a-type").value = "gqa";
    syncArchVisibility();
    for (const id of ["a-layers", "a-hidden", "a-heads", "a-kvheads", "a-headdim",
                      "a-mla-rank", "a-mla-rope", "a-experts", "a-ept", "m-vocab"]) {
      $(id).value = "";
    }
    if (msg) $("a-hfstatus").textContent = msg;
  }

  function collectInference() {
    return {
      paramsB: num("m-params"), activeB: num("m-active") || num("m-params"),
      arch: archState(), vocab: num("m-vocab", 0),
      wBytes: num("i-wdtype", 2), kvBytes: num("i-kvdtype", 2),
      ctx: num("i-ctx", 1), avgCtx: Math.min(num("i-avgctx", 1), num("i-ctx", 1)),
      batch: num("i-conc", 1), promptTokens: num("i-prompt", 1),
      gpu: gpuFor("i"), tp: num("i-tp", 0),
      bwEff: num("i-bweff", 75), mfuPrefill: num("i-mfu-prefill", 50),
      overheadPct: num("i-ovhpct", 5), fixedGb: num("i-fixedgb", 2),
      usablePct: num("i-usable", 90), linkGbps: num("i-link", 0),
      elecRate: num("i-elec", 0), pue: num("i-pue", 1.3), hostFrac: num("i-hostfrac", 15),
    };
  }

  function collectTraining() {
    return {
      paramsB: num("m-params"), activeB: num("m-active") || num("m-params"),
      arch: archState(), vocab: num("m-vocab", 0),
      tokensB: num("t-tokens", 0),
      bytes: { p: num("t-bparam", 2), g: num("t-bgrad", 2), master: num("t-bmaster", 4), m: num("t-bm", 4), v: num("t-bv", 4) },
      shard: $("t-shard").value, gpus: num("t-gpus", 0),
      gpu: gpuFor("t"), mfu: num("t-mfu", 40),
      seq: num("t-seq", 1), gbatch: num("t-gbatch", 1), mbatch: num("t-mbatch", 1),
      actMode: $("t-act").value, actMult: num("t-actmult", 0),
      logits: $("t-logits").checked,
      price: num("t-price", 2), usablePct: 90, fixedGb: num("i-fixedgb", 2),
    };
  }


  /* ------------------------------ rendering ------------------------------ */
  function setText(id, v) { $(id).textContent = v; }

  function renderBar(el, segs) {
    const total = segs.reduce((s, x) => s + x.v, 0);
    el.innerHTML = "";
    if (total <= 0) return;
    for (const s of segs) {
      const sp = document.createElement("span");
      sp.className = s.c;
      sp.style.width = Math.max(0, (s.v / total) * 100) + "%";
      sp.title = s.label + ": " + fmtGB(s.v);
      el.appendChild(sp);
    }
  }

  function renderNotes(el, notes) {
    el.innerHTML = "";
    for (const n of notes) {
      const li = document.createElement("li");
      li.className = n.kind;
      li.textContent = n.text;
      el.appendChild(li);
    }
  }

  function renderInference() {
    const o = collectInference();
    if (!archReady(o.arch)) {
      unavailable("inference", "No architecture loaded — pick an architecture preset, auto-fill from a Hugging Face repo, or enter layers / heads / hidden size. Nothing is computed until the model is defined.");
      return;
    }
    const r = Calc.inference(o);
    const name = $("m-name").value || "model";
    const notes = [];

    setText("r-i-model", "· " + name + " on " + o.gpu.name);
    setText("r-i-total", fmtGB(r.totalGb));
    setText("r-i-gpus", String(r.tp));
    setText("r-i-tps", fmtTps(r.tps1));
    setText("r-i-ttft", fmtTime(r.ttft));
    setText("r-i-tpn", String(r.tp));
    setText("r-i-weights", fmtGB(r.rawWeightsGb));
    setText("r-i-ovhpct", String(o.overheadPct));
    setText("r-i-weightovh", fmtGB(r.weightOvhGb));
    setText("r-i-kv", fmtGB(r.kvAvgGb) + " · " + o.batch + " × " + o.avgCtx.toLocaleString("en-US") + " tok");
    setText("r-i-kvmax", fmtGB(r.kvMaxGb) + " · " + o.ctx.toLocaleString("en-US") + " tok");
    setText("r-i-fixedn", String(r.tp));
    setText("r-i-ovh", fmtGB(r.overheadGb));
    setText("r-i-kvtok", fmtBytes(r.kvTok));
    setText("r-i-totalt", fmtGB(r.totalGb));
    setText("r-i-sys", fmtTps(r.sysTps));
    setText("r-i-comm", r.comm > 0 ? fmtBytes(r.comm) + " (" + fmtBytes(r.comm * o.batch) + "/step)" : "—");
    setText("r-i-wtok", fmtBytes(r.weightTokBytes));

    /* self-host cost: GPU-hour rate ÷ achieved throughput, per 1M tokens */
    const gph = r.tp * o.gpu.price;
    const costOut = r.sysTps > 0 ? (gph / (r.sysTps * 3600)) * 1e6 : NaN;
    const prefillTps = r.ttft > 0 ? o.promptTokens / r.ttft : 0;
    const costIn = prefillTps > 0 ? (gph / (prefillTps * 3600)) * 1e6 : NaN;
    setText("r-i-cost-out", "$" + costOut.toFixed(2) + " / 1M tok");
    const elec = Calc.computeElectricityCost({
      numGpus: r.tp, gpuTdpWatts: o.gpu.tdp || 0,
      pue: o.pue, elecRate: o.elecRate, hostFrac: o.hostFrac,
      throughput: r.sysTps,
    });
    if (elec && Number.isFinite(elec.costPer1M)) {
      setText("r-i-marginal", `$${elec.costPer1M.toFixed(4)} / 1M tok · ${elec.facilityPowerKw.toFixed(1)} kW`);
    } else {
      setText("r-i-marginal", o.elecRate > 0 ? "— (GPU TDP unknown)" : "— (set electricity rate)");
    }
    setText("r-i-cost-in", "$" + costIn.toFixed(2) + " / 1M tok");
    const api = lastPicked;
    if (api && api.priceIn !== null && Number.isFinite(api.priceIn)) {
      setText("r-i-api", `$${api.priceIn} / $${api.priceOut} per 1M`);
      if (Number.isFinite(costOut) && api.priceOut > 0) {
        const ratio = api.priceOut / costOut;
        notes.push({
          kind: ratio > 1 ? "ok" : "warn",
          text: ratio > 1
            ? `API output costs ${ratio.toFixed(1)}× the self-host estimate ($${api.priceOut} vs $${costOut.toFixed(2)} per 1M at this batch/precision) — serving your own GPUs is cheaper only above this utilization.`
            : `API ($${api.priceOut}/1M out) is cheaper than self-hosting at ${o.batch} concurrent streams — self-host breaks even at ~${Math.ceil(o.batch * ratio)} streams.`,
        });
      }
    } else {
      setText("r-i-api", "—");
    }

    renderBar($("r-i-bar"), [
      { label: "Weights (checkpoint)", v: r.rawWeightsGb, c: "seg-1" },
      { label: "Weights overhead", v: r.weightOvhGb, c: "seg-3" },
      { label: "KV cache (avg)", v: r.kvAvgGb, c: "seg-2" },
      { label: "Runtime", v: o.fixedGb * r.tp, c: "seg-4" },
    ]);

    if (!r.fits) {
      notes.push({ kind: "danger", text: `Does not fit: ${fmtGB(r.modelGb)} of model memory vs ${fmtGB(r.usablePerGpu * r.tp)} usable across ${r.tp} GPU(s). Auto-fit suggests ${r.tpAuto}.` });
    } else if (r.tpAuto > 1 && o.tp === 0) {
      notes.push({ kind: "ok", text: `Auto tensor parallelism: ${r.tpAuto} GPU(s) — minimum to fit weights + KV.` });
    }
    if (r.kvMaxGb > r.kvAvgGb * 4) {
      notes.push({ kind: "warn", text: `KV at full context (${fmtGB(r.kvMaxGb)}/seq) is ${(r.kvMaxGb / Math.max(r.kvAvgGb, 1e-9)).toFixed(0)}× the average-load estimate — capacity drops sharply at long prompts.` });
    }
    const vramShare = (r.modelGb / (o.gpu.vram * r.tp)) * 100;
    if (r.fits && vramShare < 45) {
      notes.push({ kind: "ok", text: `Only ${vramShare.toFixed(0)}% of VRAM used — a smaller or cheaper GPU may serve this workload.` });
    }
    if (r.tp > 1 && o.linkGbps > 0 && o.linkGbps < 100) {
      notes.push({ kind: "warn", text: `Interconnect is only ${o.linkGbps} GB/s — TP all-reduce (${fmtBytes(r.comm)}/token) may dominate decode latency.` });
    }
    if (o.arch.ept > 0 && o.arch.experts === 0) {
      notes.push({ kind: "warn", text: `Experts/token is set (${o.arch.ept}) but expert count is 0 — architecture is inconsistent. Re-run Auto-fill or enter the expert count.` });
    }
    if ((o.arch.experts > 0 || o.arch.ept > 0) && o.activeB >= o.paramsB) {
      notes.push({ kind: "danger", text: `MoE configured but active params = total (${o.paramsB}B) — decode speed is wildly overstated as dense. Set the real active slice (top-k experts × expert size + attention).` });
    }
    if (o.arch.experts > 0) {
      notes.push({ kind: "ok", text: `MoE: decode reads only ${o.activeB}B of ${o.paramsB}B params/token, but VRAM must hold all ${o.paramsB}B.` });
    }
    renderNotes($("r-i-notes"), notes);
    lastInference = { o, r, notes, costOut, costIn, elec, source: lastSource };
  }


  function renderTraining() {
    const o = collectTraining();
    if (!archReady(o.arch)) {
      unavailable("training", "No architecture loaded — pick an architecture preset, auto-fill from a Hugging Face repo, or enter layers / heads / hidden size. Nothing is computed until the model is defined.");
      return;
    }
    const r = Calc.training(o);
    const name = $("m-name").value || "model";

    setText("r-t-model", "· " + name + " on " + o.gpu.name);
    setText("r-t-mem", fmtGB(r.mem.total));
    setText("r-t-gpus", (!r.fits && o.gpus === 0) ? "> 1024" : String(r.n));
    setText("r-t-flops", fmtFlops(r.flops));
    setText("r-t-time", fmtTime(r.timeS));
    setText("r-t-params", fmtGB(r.mem.paramsGb));
    setText("r-t-grads", fmtGB(r.mem.gradsGb));
    setText("r-t-opt", fmtGB(r.mem.optGb));
    setText("r-t-act", fmtGB(r.mem.actGb) + " · c=" + (Number.isFinite(r.actC) ? r.actC.toFixed(1) : "—"));
    setText("r-t-logits", o.logits ? fmtGB(r.mem.logitsGb) : "off");
    setText("r-t-total", fmtGB(r.mem.total));
    setText("r-t-tps", fmtTps(r.tps));
    setText("r-t-accum", String(r.accum));
    setText("r-t-ckpt", fmtGB(r.ckptGb));
    setText("r-t-cost", fmtUSD(r.cost));

    renderBar($("r-t-bar"), [
      { label: "Params", v: r.mem.paramsGb, c: "seg-1" },
      { label: "Gradients", v: r.mem.gradsGb, c: "seg-2" },
      { label: "Optimizer", v: r.mem.optGb, c: "seg-3" },
      { label: "Activations", v: r.mem.actGb, c: "seg-4" },
      { label: "Logits", v: r.mem.logitsGb, c: "seg-5" },
    ]);

    const notes = [];
    if (!r.fits) {
      notes.push({ kind: "danger", text: `Does not fit: ${fmtGB(r.mem.total)}/GPU needed vs ${fmtGB(r.usable)} usable. Add GPUs, enable activation checkpointing, or shard more (ZeRO-3).` });
    }
    if (r.accum > 32) {
      notes.push({ kind: "warn", text: `Gradient accumulation is ${r.accum} steps — effective step time gets long; consider more GPUs or a smaller global batch.` });
    }
    if (o.tokensB > 0 && o.tokensB < 20 * o.paramsB) {
      notes.push({ kind: "warn", text: `Tokens (${o.tokensB}B) < 20 × params (${(20 * o.paramsB).toFixed(0)}B) — below the Chinchilla-optimal ratio.` });
    }
    if (o.shard === "zero3") {
      notes.push({ kind: "ok", text: "ZeRO-3/FSDP adds ~3 collective ops over the full parameter set per step (2 all-gathers + 1 reduce-scatter); interconnect time is not included in the estimate — MFU absorbs it." });
    }
    if (o.logits && r.mem.logitsGb > r.mem.actGb) {
      notes.push({ kind: "warn", text: "Logits + loss activations exceed layer activations — large vocab × long seq. Consider logit sharding (fused CE) in production." });
    }
    if (r.mem.actGb > r.mem.total * 0.5 && o.actMode !== "full") {
      notes.push({ kind: "warn", text: `Activation memory dominates (${fmtGB(r.mem.actGb)} of ${fmtGB(r.mem.total)} per GPU) — switch checkpointing to “Full” (or shorten the sequence) before adding GPUs; sharding does not reduce activations.` });
    }
    if (o.arch.experts > 0) {
      notes.push({ kind: "ok", text: `MoE training uses active params (${o.activeB}B) for the 6PD compute estimate; expert-parallel routing overhead is not modeled.` });
    }
    renderNotes($("r-t-notes"), notes);
    lastTraining = { o, r, notes, source: lastSource };
  }

  function recalc() {
    try {
      if (activeTab === "training") renderTraining();
      else if (activeTab === "inference") renderInference();
      renderGraph();
    } catch (err) {
      console.error(err);
    }
  }


  /* ------------------------- copy report -------------------------
     Structured, paste-ready report built from the computed inputs and
     results (not from DOM text), so it always matches what is shown. */
  const GRAPH_X = {
    params: { label: "Params", apply: (o, v) => { o.paramsB = v; o.activeB = v; } },
    tp: { label: "GPUs", apply: (o, v) => { o.tp = Math.max(1, Math.round(v)); } },
    batch: { label: "Batch", apply: (o, v) => { o.batch = Math.max(1, Math.round(v)); } },
    avgCtx: { label: "AvgCtx", apply: (o, v) => { o.avgCtx = Math.max(1, Math.min(Math.round(v), o.ctx)); } },
    prompt: { label: "Prompt", apply: (o, v) => { o.promptTokens = Math.max(1, Math.round(v)); } },
  };
  const GRAPH_Y = {
    costOut: { label: "out", get: (o, r) => { const g = r.tp * o.gpu.price; return r.sysTps > 0 ? (g / (r.sysTps * 3600)) * 1e6 : NaN; } },
    costIn: { label: "in", get: (o, r) => { const g = r.tp * o.gpu.price; const p = r.ttft > 0 ? o.promptTokens / r.ttft : 0; return p > 0 ? (g / (p * 3600)) * 1e6 : NaN; } },
    marginal: { label: "marg", get: (o, r) => { const e = Calc.computeElectricityCost({ numGpus: r.tp, gpuTdpWatts: o.gpu.tdp || 0, pue: o.pue, elecRate: o.elecRate, hostFrac: o.hostFrac, throughput: r.sysTps }); return (e && e.costPer1M) || NaN; } },
    sysTps: { label: "sys", get: (o, r) => r.sysTps },
    tps1: { label: "tps1", get: (o, r) => r.tps1 },
    totalGb: { label: "GB", get: (o, r) => r.totalGb },
    ttft: { label: "ttft", get: (o, r) => r.ttft },
  };
  function fmtVal(v) {
    if (!Number.isFinite(v)) return "--";
    if (Math.abs(v) >= 1000) return Math.round(v) + "";
    return (+v.toFixed(3)) + "";
  }
  function renderGraph() {
    const svg = $("g-chart"), cap = $("g-cap");
    if (!svg) return;
    svg.innerHTML = "";
    const base = collectInference();
    if (!archReady(base.arch)) { if (cap) cap.textContent = "Load a model to sweep."; return; }
    const xKey = ($("g-x").value in GRAPH_X) ? $("g-x").value : "params";
    const yKey = ($("g-y").value in GRAPH_Y) ? $("g-y").value : "costOut";
    let lo = parseFloat($("g-min").value), hi = parseFloat($("g-max").value);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
    if (hi < lo) { const t = lo; lo = hi; hi = t; }
    let n = Math.max(3, Math.min(40, Math.round(num("g-steps", 12))));
    const log = $("g-log").checked && lo > 0 && hi > 0;
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? 0 : i / (n - 1);
      const x = log ? lo * Math.pow(hi / lo, t) : lo + (hi - lo) * t;
      const o = { paramsB: base.paramsB, activeB: base.activeB, arch: base.arch, vocab: base.vocab, wBytes: base.wBytes, kvBytes: base.kvBytes, ctx: base.ctx, avgCtx: base.avgCtx, batch: base.batch, promptTokens: base.promptTokens, gpu: base.gpu, tp: base.tp, bwEff: base.bwEff, mfuPrefill: base.mfuPrefill, overheadPct: base.overheadPct, fixedGb: base.fixedGb, usablePct: base.usablePct, linkGbps: base.linkGbps, elecRate: base.elecRate, pue: base.pue, hostFrac: base.hostFrac };
      GRAPH_X[xKey].apply(o, x);
      let y = NaN;
      try { y = GRAPH_Y[yKey].get(o, Calc.inference(o)); } catch (e) { y = NaN; }
      pts.push({ x: x, y: Number.isFinite(y) ? y : null });
    }
    const ys = pts.map((p) => p.y).filter((v) => v !== null);
    if (!ys.length) { if (cap) cap.textContent = "No finite values in range."; return; }
    const W = 400, H = 160, P = 28;
    let yLo = Math.min.apply(null, ys), yHi = Math.max.apply(null, ys);
    if (yHi <= yLo) { yHi = yLo + 1; }
    const xLo = pts[0].x, xHi = pts[pts.length - 1].x;
    const X = (x) => P + ((x - xLo) / Math.max(1e-9, xHi - xLo)) * (W - 2 * P);
    const Y = (y) => H - P - ((y - yLo) / Math.max(1e-9, yHi - yLo)) * (H - 2 * P);
    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    let s = "";
    for (let i = 0; i <= 3; i++) {
      const yy = (10 + ((H - 30) * i) / 3).toFixed(1);
      s += '<line x1="' + P + '" x2="' + (W - 8) + '" y1="' + yy + '" y2="' + yy + '" stroke="currentColor" opacity="0.15"/>';
    }
    let d = "";
    pts.forEach((p) => { if (p.y === null) return; d += (d ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p.y).toFixed(1); });
    s += '<path d="' + d + '" fill="none" stroke="currentColor" stroke-width="2"/>';
    pts.forEach((p) => {
      if (p.y === null) return;
      s += '<circle cx="' + X(p.x).toFixed(1) + '" cy="' + Y(p.y).toFixed(1) + '" r="2.5" fill="currentColor"><title>' + esc(fmtVal(p.x) + " -> " + fmtVal(p.y)) + "</title></circle>";
    });
    s += '<text x="' + P + '" y="' + (H - 8) + '" font-size="9" fill="currentColor">' + esc(fmtVal(xLo)) + "</text>";
    s += '<text x="' + (W - 8) + '" y="' + (H - 8) + '" font-size="9" text-anchor="end" fill="currentColor">' + esc(fmtVal(xHi)) + "</text>";
    s += '<text x="' + P + '" y="12" font-size="9" fill="currentColor">' + esc(GRAPH_Y[yKey].label + ": " + fmtVal(yLo) + " - " + fmtVal(yHi)) + "</text>";
    svg.innerHTML = s;
    if (cap) cap.textContent = GRAPH_Y[yKey].label + " vs " + GRAPH_X[xKey].label + ": " + fmtVal(yLo) + " - " + fmtVal(yHi) + " over " + n + " steps. Hover dots for values.";
  }
  function archLine(a) {
    if (a.type === "mla") return `MLA L=${a.L} h=${a.h} a=${a.a} rank=${a.mlaRank} rope=${a.mlaRope}`;
    const kind = a.k === a.a ? "MHA" : a.k === 1 ? "MQA" : "GQA";
    return `${kind} L=${a.L} h=${a.h} a=${a.a} k=${a.k} d=${a.d}`;
  }
  function moeLine(a) {
    return a.experts > 0 ? `MoE E=${a.experts} top-k=${a.ept} · ` : "dense · ";
  }

  function buildInferenceReport() {
    if (!lastInference) {
      return "AI Infra Calculator — Inference report\n\nNo results: no model is loaded. Load a preset, auto-fill from a Hugging Face repo, or enter an architecture.";
    }
    const { o, r, notes, costOut, costIn, elec, source } = lastInference;
    const L = [];
    L.push("AI Infra Calculator — Inference report");
    L.push(`Generated: ${new Date().toISOString()} · Source: ${source || "manual entry"}`);
    L.push("");
    L.push("MODEL");
    L.push(`  Name: ${$("m-name").value || "—"} · Hardware: ${o.gpu.name}${r.tp > 1 ? ` × ${r.tp} (TP)` : ""}`);
    L.push(`  Total params: ${o.paramsB}B · Active/token: ${o.activeB}B · Vocab: ${o.vocab || "—"}`);
    L.push(`  Architecture: ${moeLine(o.arch)}${archLine(o.arch)}`);
    L.push(`  Precision: weights ${o.wBytes} B/param · KV ${o.kvBytes} B/elem`);
    L.push(`  Runtime: max ctx ${o.ctx} · avg ${o.avgCtx} tok/seq × ${o.batch} streams · prompt ${o.promptTokens}`);
    L.push("");
    L.push("RESULTS");
    L.push(`  Total memory: ${fmtGB(r.totalGb)} = weights ${fmtGB(r.rawWeightsGb)} + weight overhead (${o.overheadPct}%) ${fmtGB(r.weightOvhGb)} + KV (avg) ${fmtGB(r.kvAvgGb)} + runtime ${r.tp}×${fmtGB(o.fixedGb)}`);
    L.push(`  KV/token: ${fmtBytes(r.kvTok)} · KV (1 seq, max ctx): ${fmtGB(r.kvMaxGb)}`);
    L.push(`  Usable/GPU: ${fmtGB(r.usablePerGpu)} · Fits: ${r.fits ? "yes" : "no"}`);
    L.push(`  Decode: ${fmtTps(r.tps1)} (1 stream) · System: ${fmtTps(r.sysTps)}`);
    L.push(`  TTFT @ ${o.promptTokens} prompt: ${fmtTime(r.ttft)} · Read/token: ${fmtBytes(r.weightTokBytes)} weights + ${fmtBytes(r.comm)} TP comm`);
    if (lastPicked && lastPicked.priceIn !== null && Number.isFinite(lastPicked.priceIn)) {
      L.push(`  OpenRouter API $/1M: in $${lastPicked.priceIn} · out $${lastPicked.priceOut}`);
    }
    if (Number.isFinite(costOut)) L.push(`  Self-host $/1M: out $${costOut.toFixed(2)} · in $${costIn.toFixed(2)}`);
    if (elec && Number.isFinite(elec.costPer1M)) {
      L.push(`  Marginal (electricity only, owned hw): $${elec.costPer1M.toFixed(4)}/1M · ${elec.facilityPowerKw.toFixed(1)} kW at wall (TDP ${o.gpu.tdp}W × ${r.tp}, host +${o.hostFrac}%, PUE ${o.pue}, $${o.elecRate}/kWh)`);
    } else {
      L.push(`  Marginal (electricity only, owned hw): not computed — ${o.elecRate > 0 ? "GPU TDP unknown" : "set electricity rate $/kWh"}`);
    }
    if (notes.length) {
      L.push("");
      L.push("NOTES");
      for (const n of notes) L.push(`  - [${String(n.kind).toUpperCase()}] ${n.text}`);
    }
    return L.join("\n");
  }

  function buildTrainingReport() {
    if (!lastTraining) {
      return "AI Infra Calculator — Training report\n\nNo results: no model is loaded. Load a preset, auto-fill from a Hugging Face repo, or enter an architecture.";
    }
    const { o, r, notes, source } = lastTraining;
    const L = [];
    L.push("AI Infra Calculator — Training report");
    L.push(`Generated: ${new Date().toISOString()} · Source: ${source || "manual entry"}`);
    L.push("");
    L.push("MODEL");
    L.push(`  Name: ${$("m-name").value || "—"} · Hardware: ${o.gpu.name} × ${r.n}`);
    L.push(`  Total params: ${o.paramsB}B · Active/token: ${o.activeB}B · Vocab: ${o.vocab || "—"}`);
    L.push(`  Architecture: ${moeLine(o.arch)}${archLine(o.arch)}`);
    L.push(`  Run: ${o.tokensB}B tokens · seq ${o.seq} · global batch ${o.gbatch} (micro ${o.mbatch}/GPU, accum ${r.accum})`);
    L.push(`  Optimizer: ${o.bytes.p}/${o.bytes.g}/${o.bytes.master}/${o.bytes.m}/${o.bytes.v} B (param/grad/master/m/v) · sharding ${o.shard} · checkpointing ${o.actMode} (c=${r.actC})`);
    L.push(`  MFU: ${o.mfu}% · GPU price: $${o.price}/h`);
    L.push("");
    L.push("RESULTS");
    L.push(`  Memory/GPU: ${fmtGB(r.mem.total)} · Params: ${fmtGB(r.mem.paramsGb)} · Gradients: ${fmtGB(r.mem.gradsGb)} · Optimizer: ${fmtGB(r.mem.optGb)} · Activations: ${fmtGB(r.mem.actGb)} · Logits: ${fmtGB(r.mem.logitsGb)}`);
    L.push(`  Fits: ${r.fits ? "yes" : "no"} · Checkpoint: ${fmtGB(r.ckptGb)}`);
    L.push(`  Compute: ${fmtFlops(r.flops)} · Time: ${fmtTime(r.timeS)} · Throughput: ${fmtTps(r.tps)}`);
    L.push(`  Cost: ${fmtUSD(r.cost)}`);
    if (notes.length) {
      L.push("");
      L.push("NOTES");
      for (const n of notes) L.push(`  - [${String(n.kind).toUpperCase()}] ${n.text}`);
    }
    return L.join("\n");
  }

  async function copyText(btn, text) {
    const done = (label) => {
      const prev = btn.textContent;
      btn.textContent = label;
      setTimeout(() => { btn.textContent = prev; }, 1500);
    };
    try {
      if (globalThis.navigator && globalThis.navigator.clipboard && globalThis.navigator.clipboard.writeText) {
        await globalThis.navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        if (typeof ta.select === "function") ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      done("Copied");
    } catch {
      done("Copy failed");
    }
  }

  $("copy-inference").addEventListener("click", () => copyText($("copy-inference"), buildInferenceReport()));
  $("copy-training").addEventListener("click", () => copyText($("copy-training"), buildTrainingReport()));

  /* ------------------------- openrouter picker ------------------------- */
  let orModels = [];
  let orIndex = -1;
  let orMatches = [];
  let lastPicked = null;
  let lastSource = "manual entry";   /* provenance line for copy reports */
  let lastInference = null;          /* { o, r, notes, costOut, costIn } */
  let lastTraining = null;           /* { o, r, notes } */

  function orStatus(msg) { $("or-status").innerHTML = msg; }

  async function loadCatalog() {
    const btn = $("or-fetch");
    btn.classList.add("is-loading");
    btn.textContent = "Loading…";
    try {
      orModels = await ORAPI.fetchModels();
      const sized = orModels.filter((m) => m.size.total !== null).length;
      orStatus(`${orModels.length.toLocaleString("en-US")} models · ${sized} with a parseable size · updated ${new Date().toLocaleTimeString()}`);
      $("footer-stats").textContent = `openrouter: ${orModels.length} models live`;
    } catch (err) {
      orStatus("Fetch failed — " + err.message + ". Enter parameters manually.");
    }
    btn.classList.remove("is-loading");
    btn.textContent = "Reload catalog";
  }

  function filterModels(q) {
    q = q.trim().toLowerCase();
    if (!q) return [];
    const starts = [], contains = [];
    for (const m of orModels) {
      if (m.alias) continue;                       /* hide ~...-latest aliases */
      const hay = (m.id + " " + m.name).toLowerCase();
      if (hay.includes(q)) {
        (m.id.toLowerCase().includes(q) ? starts : contains).push(m);
        if (starts.length + contains.length >= 80) break;
      }
    }
    return starts.concat(contains).slice(0, 40);
  }

  function sizeLabel(m) {
    if (m.size.total !== null) {
      const t = m.size.total >= 1000 ? (m.size.total / 1000).toFixed(1) + "T" : m.size.total + "B";
      return m.size.active !== null && m.size.active !== m.size.total
        ? t + "-A" + m.size.active + "B" : t;
    }
    return "?";
  }

  function renderMenu() {
    const menu = $("or-menu");
    menu.innerHTML = "";
    if (!orMatches.length) { menu.hidden = true; return; }
    orMatches.forEach((m, i) => {
      const div = document.createElement("div");
      div.className = "or-item" + (i === orIndex ? " is-sel" : "");
      div.setAttribute("role", "option");
      const meta = [sizeLabel(m), m.ctx ? (m.ctx / 1024) + "k ctx" : null, m.hf ? "open" : "closed"].filter(Boolean).join(" · ");
      div.innerHTML = `<span class="mono">${escapeHtml(m.name.replace(/^[^:]+:\s*/, ""))}</span><span class="or-id mono">${escapeHtml(m.id)}</span><span class="or-meta mono">${escapeHtml(meta)}</span>`;
      div.addEventListener("mousedown", (e) => { e.preventDefault(); pickModel(m); });
      menu.appendChild(div);
    });
    menu.hidden = false;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function pickModel(m) {
    $("or-menu").hidden = true;
    $("or-search").value = "";
    lastPicked = m;
    lastSource = "openrouter: " + m.id;
    $("a-hfrepo").value = m.hf || "";
    $("m-name").value = m.name.replace(/^[^:]+:\s*/, "");
    /* never inherit the previously selected model's architecture */
    clearArchitecture();

    if (m.size.total !== null) $("m-params").value = m.size.total;
    else $("m-params").value = "";
    if (m.size.active !== null) $("m-active").value = m.size.active;
    else $("m-active").value = m.size.total !== null ? m.size.total : "";
    if (m.ctx) { $("i-ctx").value = m.ctx; $("i-avgctx").value = Math.min(8192, m.ctx); }

    const parts = [];
    parts.push(`<b>${escapeHtml(m.name)}</b>`);
    parts.push(`<span class="mono">${escapeHtml(m.id)}</span>`);
    if (m.size.total !== null) {
      const src = m.size.from === "name" ? "name" : "id";
      parts.push(`<span>Size <b class="mono">${sizeLabel(m)}</b> (parsed from ${src}, ~±3–15%)</span>`);
    } else {
      parts.push(`<span class="warn-text">No size token in naming — enter parameters manually (OpenRouter does not publish parameter counts).</span>`);
    }
    if (m.size.confidence === "active-only") {
      parts.push(`<span class="warn-text">Naming encodes active params only (a${m.size.active}B); set total params manually.</span>`);
    }
    if (m.ctx) parts.push(`<span>Context <b class="mono">${m.ctx.toLocaleString("en-US")}</b></span>`);
    if (m.priceIn !== null) parts.push(`<span>Price <b class="mono">$${m.priceIn} / $${m.priceOut}</b> per 1M tok</span>`);
    if (m.aa !== null) parts.push(`<span>AA index <b class="mono">${m.aa}</b></span>`);
    if (m.hf) parts.push(`<a class="mono" href="https://huggingface.co/${escapeHtml(m.hf)}" target="_blank" rel="noopener">${escapeHtml(m.hf)}</a>`);
    parts.push(`<span class="warn-text">${m.hf
      ? "Fetching the architecture from its Hugging Face repo… (status under the Auto-fill button)"
      : "Closed model — no Hugging Face repo, so architecture and exact parameters are unavailable. Enter them manually or reuse a preset."}</span>`);
    const box = $("or-picked");
    box.innerHTML = parts.join("");
    box.hidden = false;
    recalc();
    if (m.hf) runAutoFill(m.hf);
    else $("a-hfstatus").textContent = "No Hugging Face repo for this model — architecture stays “not available” until you enter it or pick a preset.";
  }

  $("or-search").addEventListener("input", () => {
    orMatches = filterModels($("or-search").value);
    orIndex = orMatches.length ? 0 : -1;
    renderMenu();
  });
  $("or-search").addEventListener("keydown", (e) => {
    const menu = $("or-menu");
    if (menu.hidden) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      orIndex = (orIndex + (e.key === "ArrowDown" ? 1 : -1) + orMatches.length) % orMatches.length;
      renderMenu();
    } else if (e.key === "Enter" && orIndex >= 0) {
      e.preventDefault();
      pickModel(orMatches[orIndex]);
    } else if (e.key === "Escape") {
      menu.hidden = true;
    }
  });
  $("or-search").addEventListener("blur", () => setTimeout(() => { $("or-menu").hidden = true; }, 150));
  $("or-fetch").addEventListener("click", loadCatalog);

  /* ---------------------- HF architecture auto-fill ---------------------- */
  async function runAutoFill(repo) {
    const btn = $("a-autofill");
    const status = $("a-hfstatus");
    if (!repo) {
      clearArchitecture("Enter a Hugging Face repo id (org/model), or pick an OpenRouter model that has an HF link.");
      recalc();
      return;
    }
    btn.classList.add("is-loading");
    btn.textContent = "Fetching…";
    try {
      const r = await HFArch.autoFill(repo);
      const a = r.arch;
      const isMoe = (a.experts || 0) > 0 || (a.ept || 0) > 0 || !!a.moe;

      $("m-preset").value = "custom";
      $("a-type").value = a.type;
      syncArchVisibility();
      $("a-layers").value = a.L || "";
      $("a-hidden").value = a.h || "";
      $("a-heads").value = a.a || "";
      $("a-kvheads").value = a.k || "";
      $("a-headdim").value = a.d || "";
      $("a-mla-rank").value = a.mlaRank || "";
      $("a-mla-rope").value = a.mlaRope || "";
      $("a-experts").value = isMoe ? (a.experts || 0) : 0;
      $("a-ept").value = isMoe ? (a.ept || 0) : 0;
      if (a.vocab) $("m-vocab").value = a.vocab;
      if (a.ctx) {
        $("i-ctx").value = a.ctx;
        $("i-avgctx").value = Math.min(8192, a.ctx);
      }
      if (r.paramsB) {
        $("m-params").value = +r.paramsB.toFixed(1);
        if (!isMoe) $("m-active").value = +r.paramsB.toFixed(1);
      }
      if (isMoe && a.activeEstimateB) {
        $("m-active").value = +a.activeEstimateB.toFixed(1);
      } else if (isMoe) {
        $("m-active").value = "";   /* never claim dense-active for a MoE */
      }
      const src = a.source === "gguf-header" ? "GGUF header (64 KB range request)" : "config.json";
      let statusMsg =
        `Filled from <a href="https://huggingface.co/${escapeHtml(r.repo)}" target="_blank" rel="noopener">${escapeHtml(r.repo)}</a>` +
        ` · ${src}${r.paramsB ? " · exact params " + r.paramsB.toFixed(2) + "B" : ""}` +
        (r.notes.length ? " · " + escapeHtml(r.notes.join(" ")) : "");
      if (isMoe && a.activeEstimateB) {
        statusMsg += ` · <span class="warn-text">active params ≈ ${a.activeEstimateB.toFixed(1)}B estimated from config (top-${a.ept} of ${a.experts} experts + shared + attention) — override with the vendor figure</span>`;
      } else if (isMoe) {
        statusMsg += ` · <span class="warn-text">MoE detected — enter “Active params / token” manually (not published in config)</span>`;
      }
      status.innerHTML = statusMsg;
      lastSource = "huggingface: " + r.repo + " (" +
        (a.source === "gguf-header" ? "GGUF header" : "config.json") + ")";
      recalc();
    } catch (err) {
      /* Never keep the previous model's architecture after a failed lookup. */
      clearArchitecture(`Auto-fill failed: ${err.message} — architecture reset to “not available”. Enter values manually or pick a preset.`);
      lastSource = "manual entry";
      recalc();
    }
    btn.classList.remove("is-loading");
    btn.textContent = "Auto-fill";
  }

  $("a-autofill").addEventListener("click", () =>
    runAutoFill($("a-hfrepo").value.trim() || (lastPicked && lastPicked.hf) || ""));



  /* ------------------------------ event wiring ------------------------------ */
  document.querySelectorAll(".tab").forEach((btn) =>
    btn.addEventListener("click", () => setTab(btn.dataset.tab)));

  document.querySelectorAll("input, select").forEach((el) => {
    if (el.id === "or-search" || el.id === "m-preset" || el.id === "t-optimizer" || el.id === "t-act") return;
    el.addEventListener("input", recalc);
    el.addEventListener("change", recalc);
  });

  $("m-preset").addEventListener("change", () => { applyPreset($("m-preset").value); recalc(); });
  $("a-type").addEventListener("change", () => { lastSource = "manual entry"; syncArchVisibility(); });

  /* keep head dim = h / a in sync unless user overrides afterwards */
  for (const id of ["a-hidden", "a-heads"]) {
    $(id).addEventListener("input", () => {
      const h = num("a-hidden"), a = num("a-heads");
      if (h > 0 && a > 0) $("a-headdim").value = Math.round(h / a);
    });
  }

  for (const p of ["i", "t"]) {
    $(p + "-gpu").addEventListener("change", () => { syncGpuPrice(p); syncCustomHw(p); recalc(); });
  }

  /* graph X presets: sensible sweep ranges per variable */
  $("g-x").addEventListener("change", () => {
    const presets = { params: [1, 70], tp: [1, 8], batch: [1, 128], avgCtx: [1024, 32768], prompt: [512, 16384] };
    const r = presets[$("g-x").value] || [1, 70];
    $("g-min").value = r[0]; $("g-max").value = r[1];
    recalc();
  });

  /* optimizer select fills the editable byte breakdown */
  $("t-optimizer").addEventListener("change", () => {
    const o = Calc.OPTIMIZERS[$("t-optimizer").value];
    $("t-bparam").value = o.p; $("t-bgrad").value = o.g;
    $("t-bmaster").value = o.master; $("t-bm").value = o.m; $("t-bv").value = o.v;
    recalc();
  });

  /* checkpointing mode seeds the editable activation coefficient */
  $("t-act").addEventListener("change", () => {
    const mode = $("t-act").value;
    if (mode === "selective") $("t-actmult").value = 34;
    else if (mode === "full") $("t-actmult").value = 2;
    else $("t-actmult").value = (34 + 5 * num("a-heads") * num("t-seq") / Math.max(1, num("a-hidden"))).toFixed(1);
    recalc();
  });

  /* ------------------------- live GPU market prices -------------------------
     Optional Vast.ai refresh (keyless). Overwrites the static $/h defaults
     with median rentable $/GPU-h per model; static table stays the fallback. */
  async function refreshGpuPrices() {
    const btn = $("gpu-refresh");
    const status = $("gpu-price-status");
    if (typeof GPUPrices === "undefined") {
      if (status) status.textContent = "price module not loaded";
      return;
    }
    btn.classList.add("is-loading");
    try {
      const med = await GPUPrices.fetchMedians();
      let updated = 0;
      for (const p of ["i", "t"]) {
        const id = $(p + "-gpu").value;
        if (med[id]) {
          $(p === "i" ? "i-price" : "t-price").value = +med[id].median.toFixed(2);
          updated++;
        }
      }
      const keys = Object.keys(med).length;
      if (status) status.textContent = `market medians for ${keys} models · updated ${new Date().toLocaleTimeString()}`;
      $("footer-stats").textContent = `vast.ai: medians for ${keys} GPU models live`;
      recalc();
      if (!updated && status) status.textContent += " · current selection not in market set";
    } catch (err) {
      if (status) status.textContent = "market refresh failed — static prices kept (" + err.message + ")";
    }
    btn.classList.remove("is-loading");
  }

  $("gpu-refresh").addEventListener("click", refreshGpuPrices);

  /* ------------------------------ init ------------------------------ */
  /* Nothing is preloaded: no default model, no catalog fetch. Every number in
     the results panel stays "—" until a model is defined by the user. */
  syncCustomHw("i");
  syncCustomHw("t");
  syncGpuPrice("i");
  syncGpuPrice("t");
  syncArchVisibility();
  orStatus("Catalog not loaded — press “Load catalog” to search live OpenRouter models.");
  $("footer-stats").textContent = "no catalog loaded";
  recalc();
})();

