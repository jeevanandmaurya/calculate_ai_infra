/* ============================================================
   hf-arch.js — keyless architecture auto-fill from Hugging Face.

   Chain (all public, no API key, CORS-safe):
     1. /api/models/{repo}?expand[]=safetensors&siblings&gated  → params, access
     2. {repo}/resolve/main/config.json                          → full arch
     3. fallback: first *.gguf sibling → 64 KB range request → GGUF header
     4. gated/missing repo → search public "-GGUF" mirrors by name

   Returns a normalizable arch record for the calculator.
   ============================================================ */
"use strict";

const HFArch = (() => {
  const HF = "https://huggingface.co";

  async function getJson(url) {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) { const err = new Error("HTTP " + res.status); err.status = res.status; throw err; }
    return res.json();
  }

  async function fetchInfo(repo) {
    return getJson(`${HF}/api/models/${repo}?expand[]=safetensors&expand[]=siblings&expand[]=gated&expand[]=cardData`);
  }

  /* ------------------------- config.json path -------------------------
     Vision/multimodal models nest the LLM under text_config. DeepSeek
     names its expert count n_routed_experts (not num_experts). */
  async function fromConfigJson(repo) {
    const raw = await getJson(`${HF}/${repo}/resolve/main/config.json`);
    const c = raw.text_config || raw;
    const heads = c.num_attention_heads || 0;
    const experts = c.num_experts || c.n_routed_experts || 0;
    const ept = c.num_experts_per_tok || 0;
    const arch = {
      source: "config.json",
      type: c.kv_lora_rank ? "mla" : "gqa",
      L: c.num_hidden_layers, h: c.hidden_size, a: heads,
      k: c.num_key_value_heads || heads || 1,
      d: c.head_dim || (heads ? Math.round(c.hidden_size / heads) : 128),
      mlaRank: c.kv_lora_rank || 512, mlaRope: c.qk_rope_head_dim || 64,
      experts, ept,
      moe: !!(experts || ept),
      vocab: c.vocab_size || 0,
      ctx: c.max_position_embeddings || 0,
    };
    arch.activeEstimateB = arch.moe ? estimateActiveParamsB(c, arch) : null;
    return arch;
  }

  /* Rough active-params estimate for MoE from config alone:
     attention (Q/O: h·q each, K/V: h·k·d each) + (top-k + shared) experts
     (3·h·moe_intermediate each: gate+up+down) + router + embeddings.
     Accuracy ±30% — always overridable, meant to avoid dense=total mistakes. */
  function estimateActiveParamsB(c, arch) {
    const h = c.hidden_size, L = c.num_hidden_layers;
    if (!h || !L) return null;
    const a = c.num_attention_heads || 0;
    const d = c.head_dim || (a ? Math.round(h / a) : 0);
    const q = a * d, k = c.num_key_value_heads || a || 1;
    const attn = 2 * h * q + 2 * h * k * d;
    const inter = c.moe_intermediate_size || c.intermediate_size || 0;
    const expertP = inter ? 3 * h * inter : 0;
    const shared = c.n_shared_experts || 0;
    const ept = c.num_experts_per_tok || 0;
    const experts = c.num_experts || c.n_routed_experts || 0;
    const perLayer = attn + (ept + shared) * expertP + (experts ? h * experts : 0)
      + (arch && arch.type === "gqa" && inter && !experts ? 3 * h * inter : 0);
    const embed = c.tie_word_embeddings ? h * (c.vocab_size || 0) : 2 * h * (c.vocab_size || 0);
    return (perLayer * L + embed) / 1e9;
  }

  /* ------------------------- GGUF header path -------------------------
     64 KB range request decodes the GGUF key/value header — no download. */
  const GGUF_FIXED = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

  async function ggufHeader(repo, file) {
    const url = `${HF}/${repo}/resolve/main/${encodeURIComponent(file)}`;
    const res = await fetch(url, { headers: { Range: "bytes=0-65535" } });
    if (!res.ok && res.status !== 206) {
      const err = new Error("HTTP " + res.status); err.status = res.status; throw err;
    }
    const buf = new DataView(await res.arrayBuffer());
    const dec = new TextDecoder();
    if (dec.decode(new Uint8Array(buf.buffer, 0, 4)) !== "GGUF") {
      throw new Error("not a GGUF file");
    }
    let off = 24; /* magic(4) + version(4) + tensor_count(8) + kv_count(8) */
    const kvCount = Number(buf.getBigUint64(16, true));

    function readStr() {
      const len = Number(buf.getBigUint64(off, true)); off += 8;
      const s = dec.decode(new Uint8Array(buf.buffer, off, len)); off += len;
      return s;
    }
    function readVal(type) {
      if (type === 8) return readStr();
      if (type === 9) {
        const elem = buf.getUint32(off, true); off += 4;
        const count = Number(buf.getBigUint64(off, true)); off += 8;
        if (elem === 8) {                 /* array of strings: walk len-prefixed */
          const items = [];
          for (let i = 0; i < count; i++) {
            if (off + 8 > buf.byteLength) throw new Error("unskippable string array");
            const len = Number(buf.getBigUint64(off, true)); off += 8;
            if (i < 4 && off + len <= buf.byteLength) {
              items.push(dec.decode(new Uint8Array(buf.buffer, off, len)));
            }
            off += len;
          }
          return items;
        }
        if (GGUF_FIXED[elem]) {           /* numeric: skip arithmetically */
          const skip = count - Math.min(count, 4);
          const items = [];
          for (let i = 0; i < Math.min(count, 4); i++) items.push(readVal(elem));
          off += skip * GGUF_FIXED[elem];
          return items;
        }
        throw new Error("nested array in GGUF header");
      }
      if (type === 7) { const v = buf.getUint8(off) !== 0; off += 1; return v; }
      let v = 0;
      if (type === 0) v = buf.getUint8(off);
      else if (type === 1) v = buf.getInt8(off);
      else if (type === 2) v = buf.getUint16(off, true);
      else if (type === 3) v = buf.getInt16(off, true);
      else if (type === 4) v = buf.getUint32(off, true);
      else if (type === 5) v = buf.getInt32(off, true);
      else if (type === 6) v = buf.getFloat32(off, true);
      else if (type === 10) v = buf.getFloat64(off, true);
      else if (type === 11) v = Number(buf.getBigInt64(off, true));
      else if (type === 12) v = Number(buf.getBigUint64(off, true));
      off += GGUF_FIXED[type];
      return v;
    }

    const meta = {};
    for (let i = 0; i < kvCount; i++) {
      /* tokenizer.* string arrays are huge and unskippable within a 64 KB
         window — architecture keys always precede them, so stop cleanly. */
      try {
        const key = readStr();
        const type = buf.getUint32(off, true); off += 4;
        meta[key] = readVal(type);
      } catch {
        break;
      }
      if (off > buf.byteLength - 24) break;   /* header truncated — good enough */
    }
    return meta;
  }

  /* Map {prefix}.{key} GGUF keys (llama.*, qwen3moe.*, deepseek2.*, …) to fields. */
  function archFromGguf(meta) {
    const pick = (suffix) => {
      for (const [k, v] of Object.entries(meta)) {
        if (k === suffix || k.endsWith("." + suffix)) return v;
      }
      return undefined;
    };
    const heads = pick("attention.head_count") || 0;
    const kvRank = pick("attention.kv_lora_rank");
    return {
      source: "gguf-header",
      type: kvRank ? "mla" : "gqa",
      L: pick("block_count"), h: pick("embedding_length"),
      a: heads,
      k: pick("attention.head_count_kv") || heads || 1,
      d: pick("attention.key_length") || pick("attention.value_length") ||
         (heads && pick("embedding_length") ? Math.round(pick("embedding_length") / heads) : 128),
      mlaRank: kvRank || 512, mlaRope: pick("attention.rope_dimension_count") || 64,
      experts: pick("expert_count") || 0, ept: pick("expert_used_count") || 0,
      vocab: pick("vocab_size") || 0,
      ctx: pick("context_length") || 0,
      sizeLabel: pick("general.size_label") || null,
    };
  }

  function sizeLabelToParams(label) {
    const m = /([\d.]+)\s*([BMT])(?![a-z])/i.exec(label || "");
    if (!m) return null;
    const f = { B: 1, M: 0.001, T: 1000 }[m[2].toUpperCase()];
    return parseFloat(m[1]) * f;
  }

  /* Gated / missing repo → most-downloaded public GGUF mirror. */
  async function findGgufMirror(modelName) {
    const q = modelName.replace(/\s+/g, " ").trim();
    const res = await getJson(`${HF}/api/models?search=${encodeURIComponent(q + " GGUF")}&limit=20&expand[]=gated&expand[]=downloads&expand[]=siblings`);
    const candidates = (res || [])
      .filter((m) => !m.gated && (m.id || "").toLowerCase().endsWith("gguf"))
      .sort((a, b) => (b.downloads || 0) - (a.downloads || 0));
    for (const c of candidates) {
      const ggufs = (c.siblings || []).map((s) => s.rfilename)
        .filter((f) => f.toLowerCase().endsWith(".gguf"));
      if (ggufs.length) return { repo: c.id, file: pickGgufFile(ggufs) };
    }
    return null;
  }

  function pickGgufFile(files) {
    return files.find((f) => /Q4_K_M\.gguf$/i.test(f))
        || files.find((f) => /Q4_K_S\.gguf$/i.test(f))
        || files.find((f) => /Q4_0\.gguf$/i.test(f))
        || files.find((f) => /Q8_0\.gguf$/i.test(f))
        || files[0];
  }


  /* ------------------------------ main entry ------------------------------
     autoFill("Qwen/Qwen3-30B-A3B") or a bare model name ("Llama 3.3 70B"). */
  async function autoFill(repoOrName) {
    const notes = [];
    let repo = repoOrName.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/$/, "");
    const looksLikeRepo = /^[^/\s]+\/[^/\s]+$/.test(repo);

    let info = null;
    if (looksLikeRepo) {
      try { info = await fetchInfo(repo); }
      catch (err) {
        if (err.status === 401 || err.status === 403) notes.push("Repo is gated — using a public GGUF mirror.");
        else if (err.status === 404) notes.push("Repo not found — trying a GGUF mirror by name.");
        info = null;
      }
    } else {
      notes.push("Not a repo id — searching GGUF mirrors by name.");
      repo = null;
    }

    /* resolve exact parameter count from the original repo when possible */
    let paramsB = null;
    if (info && info.safetensors && info.safetensors.total) {
      paramsB = info.safetensors.total / 1e9;
    }

    let arch = null;
    if (info) {
      const siblings = (info.siblings || []).map((s) => s.rfilename);
      try { arch = await fromConfigJson(repo); }
      catch { notes.push("No readable config.json — trying GGUF header."); }
      if (!arch) {
        const ggufs = siblings.filter((f) => f.toLowerCase().endsWith(".gguf"));
        if (ggufs.length) {
          const meta = await ggufHeader(repo, pickGgufFile(ggufs));
          arch = archFromGguf(meta);
          if (!paramsB && arch.sizeLabel) paramsB = sizeLabelToParams(arch.sizeLabel);
        }
      }
    }

    /* gated / not found / bare name → GGUF mirror */
    if (!arch) {
      const nameForSearch = looksLikeRepo ? repo.split("/")[1].replace(/[-_]+/g, " ") : repoOrName;
      const mirror = await findGgufMirror(nameForSearch);
      if (!mirror) throw new Error("No public config.json or GGUF mirror found for “" + repoOrName + "”.");
      notes.push(`Using mirror: ${mirror.repo}`);
      const meta = await ggufHeader(mirror.repo, mirror.file);
      arch = archFromGguf(meta);
      if (!paramsB && arch.sizeLabel) paramsB = sizeLabelToParams(arch.sizeLabel);
      repo = mirror.repo;
    }

    return { repo, arch, paramsB, notes, gated: !!(info && info.gated) };
  }

  return { autoFill, fromConfigJson, ggufHeader, archFromGguf, findGgufMirror, sizeLabelToParams, estimateActiveParamsB };
})();

