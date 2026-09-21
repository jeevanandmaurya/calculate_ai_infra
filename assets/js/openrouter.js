/* ============================================================
   openrouter.js — fetch the live OpenRouter catalog and parse
   vendor naming conventions into size estimates.
   Endpoint: https://openrouter.ai/api/v1/models (CORS: *, no key).
   Parsing rules validated against Hugging Face safetensors totals
   (median error ~3% for models >= 24B; see docs/calculations.md).
   ============================================================ */
"use strict";

const ORAPI = (() => {
  const URL = "https://openrouter.ai/api/v1/models";

  const RE_MOE_AB  = /(\d+(?:\.\d+)?)\s?[bB][\s-]?[aA](\d+(?:\.\d+)?)\s?[bB](?![a-z0-9])/i; /* 235B-A22B */
  const RE_MOE_NM  = /(\d+)\s?[xX]\s?(\d+(?:\.\d+)?)\s?[bB](?![a-z0-9])/;                 /* 8x7B */
  const RE_ACTIVE  = /(?:^|[\s-])a(\d+(?:\.\d+)?)[bB](?![a-z0-9])/i;                      /* a13B = active only */
  const RE_TERA    = /(\d+(?:\.\d+)?)\s?[tT](?![a-z0-9])/;                                /* 2.4T */
  const RE_DENSE   = /(?<![aA])(\d+(?:\.\d+)?)\s?[bB](?![a-z0-9])/;                       /* 70B (not -a13B) */

  /* Parse one string pair (name first, id as fallback) into a size record.
     Priority per text: MoE total+active > T-scale (+optional active) >
     NxM MoE > active-only > dense B-scale. The dense pattern ignores
     tokens preceded by "a" so `-a13b` (active-only) is never read as a total.
     Returns { total, active, moe, from, confidence } — params in B. */
  function parseSize(m) {
    for (const [text, from] of [[m.name || "", "name"], [m.id || "", "id"]]) {
      if (!text) continue;
      let hit = RE_MOE_AB.exec(text);
      if (hit) return { total: +hit[1], active: +hit[2], moe: true, from, confidence: "high" };
      const tera = RE_TERA.exec(text);
      if (tera) {
        const act = RE_ACTIVE.exec(text);
        return { total: +tera[1] * 1000, active: act ? +act[1] : null,
                 moe: !!act, from, confidence: "high" };
      }
      hit = RE_MOE_NM.exec(text);
      if (hit) return { total: +hit[1] * +hit[2], active: null, moe: true, from, confidence: "medium" };
      hit = RE_ACTIVE.exec(text);
      if (hit) return { total: null, active: +hit[1], moe: true, from, confidence: "active-only" };
      hit = RE_DENSE.exec(text);
      if (hit) return { total: +hit[1], active: +hit[1], moe: false, from, confidence: "high" };
    }
    return { total: null, active: null, moe: false, from: null, confidence: "none" };
  }

  /* Normalize one API entry: resolve ~aliases, mark :free/:batch variants. */
  function normalize(m) {
    const alias = (m.id || "").startsWith("~");
    const variant = /:(free|batch|extended|nitro|floor|online)$/i.exec(m.id || "");
    return {
      id: m.id,
      alias, variant: variant ? variant[1].toLowerCase() : null,
      aliasTarget: m.alias_target && m.alias_target.slug ? m.alias_target.slug : null,
      name: m.name || m.id,
      ctx: m.context_length || (m.top_provider && m.top_provider.context_length) || null,
      maxOut: m.top_provider && m.top_provider.max_completion_tokens,
      modality: m.architecture && m.architecture.modality,
      priceIn: m.pricing ? +m.pricing.prompt * 1e6 : null,       /* $ / 1M tokens */
      priceOut: m.pricing ? +m.pricing.completion * 1e6 : null,
      hf: m.hugging_face_id || null,
      created: m.created || null,
      cutoff: m.knowledge_cutoff || null,
      aa: m.benchmarks && m.benchmarks.artificial_analysis
        ? m.benchmarks.artificial_analysis.intelligence_index : null,
      size: parseSize(m),
    };
  }

  async function fetchModels() {
    const res = await fetch(URL, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error("OpenRouter HTTP " + res.status);
    const json = await res.json();
    return (json.data || []).map(normalize);
  }

  return { URL, fetchModels, parseSize };
})();
