/* ============================================================
   gpu-prices.js — optional live GPU rental prices from the Vast.ai
   public marketplace (no key). Static Calc.GPUS prices stay the
   default; this only overwrites the $/h inputs on explicit refresh.
   Endpoint: https://console.vast.ai/api/v0/bundles/ (keyless for
   read-only search today, verified 2026-09-21 — may require auth
   later; failures fall back to the static table with a message).
   Shadeform is NOT used: its OpenAPI spec requires X-API-KEY on
   /instances/types, unsuitable for a keyless static app.
   ============================================================ */
"use strict";

const GPUPrices = (() => {
  const URL = "https://console.vast.ai/api/v0/bundles/";

  /* Vast gpu_name → Calc GPUS id. Unlisted models are ignored. */
  const NAME_TO_ID = [
    [/^H100.*SXM/i, "h100-sxm"],
    [/^H100.*NVL/i, "h100-sxm"],
    [/^H100/i, "h100-sxm"],
    [/^H200/i, "h200"],
    [/^B200/i, "b200"],
    [/^B300/i, "b300"],
    [/^A100.*80|80.*A100/i, "a100-80"],
    [/^A100/i, "a100-40"],
    [/^MI300X/i, "mi300x"],
    [/^MI325X/i, "mi325x"],
    [/^L40S/i, "l40s"],
    [/^RTX\s*6000.*Ada|6000.*Ada/i, "rtx6000a"],
    [/^RTX\s*5090/i, "rtx-5090"],
    [/^RTX\s*4090/i, "rtx-4090"],
    [/^RTX\s*3090/i, "rtx-3090"],
    [/^\s*L4\s/i, "l4"],
    [/^L4$/i, "l4"],
  ];

  function mapId(gpuName) {
    if (!gpuName) return null;
    for (const [re, id] of NAME_TO_ID) if (re.test(gpuName)) return id;
    return null;
  }

  function median(xs) {
    if (!xs.length) return null;
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /* Per-GPU $/h = dph_total / num_gpus. Keep rentable + sane rows only. */
  function aggregate(offers) {
    const buckets = new Map();
    for (const o of offers || []) {
      if (!o || o.rentable === false || o.rented) continue;
      const id = mapId(o.gpu_name);
      const n = o.num_gpus || 1;
      const perGpu = typeof o.dph_total === "number" ? o.dph_total / n : null;
      if (!id || !Number.isFinite(perGpu) || perGpu <= 0 || perGpu > 50) continue;
      if (!buckets.has(id)) buckets.set(id, []);
      buckets.get(id).push(perGpu);
    }
    const out = {};
    for (const [id, xs] of buckets) {
      if (xs.length < 3) continue; /* too thin to trust */
      out[id] = { median: median(xs), n: xs.length };
    }
    return out;
  }

  async function fetchMedians() {
    const res = await fetch(URL, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error("Vast.ai HTTP " + res.status);
    const json = await res.json();
    const offers = json.offers || json.data || json.results || [];
    const medians = aggregate(offers);
    if (!Object.keys(medians).length) throw new Error("no rentable offers aggregated");
    return medians;
  }

  return { URL, mapId, aggregate, fetchMedians };
})();
