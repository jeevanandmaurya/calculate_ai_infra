# How the numbers are calculated

This document is the complete reference for every formula used by the app.
Unless noted otherwise, all quantities are decimal (1 GB = 10⁹ bytes, matching
how GPU vendors spec VRAM), parameter counts `P` are in raw parameter count
(`B` = billions), and times are in seconds.

Notation: `L` layers · `h` hidden size · `a` attention heads · `k` KV heads ·
`d` head dim (`h/a` unless the model overrides it) · `r` MLA KV-LoRA rank ·
`d_r` MLA RoPE dim · `s` sequence length · `b` batch · `V` vocab size ·
`P_act` active parameters per token (dense: `P_act = P`) · `D` training tokens.

---

## 1. Inference

### 1.1 Weight memory — shown raw, overhead as its own additive row

```
W = P × b_w            (checkpoint bytes — the Weights row)
W_overhead = W × overhead%      (fragmentation, alignment, CUDA graphs — its own row)
Total = W + W_overhead + KV_tot + fixed × TP     (rows sum exactly)
```

MoE caveat: VRAM must hold **all** `P` parameters, but each decode step only
**reads** `P_act` — this distinction drives the throughput math below.

### 1.2 KV cache

Standard attention (MHA, GQA, MQA — one formula, via the number of KV heads `k`):

```
KV_tok = 2 · L · k · d · b_kv       bytes per token, per sequence
```

- MHA: `k = a` (Llama 2 7B) · GQA: `1 < k < a` (Llama 3 8B: k = 8) · MQA: `k = 1`

MLA (DeepSeek-V2/V3) caches one compressed latent vector per layer instead of
full K/V per head:

```
KV_tok = L · (r + d_r) · b_kv       r = 512, d_r = 64 for DeepSeek-V3
```

Sanity checks (bf16 KV): Llama-3.3-70B → 80·8·128·2·2 = 327,680 B/token ≈ 0.33 MB
(42 GB at 128k context); DeepSeek-V3 → 61·576·2 = 70,272 B/token ≈ 0.07 MB.

```
KV_tot = KV_tok · S_avg · concurrency
```

The app reports KV both at the average load and at `1 sequence × max context`,
because long prompts are what actually exhaust memory in production.

### 1.3 Overhead & fit

```
usable_per_gpu = VRAM × usable% − fixed_overhead          (defaults: 90%, 2 GB)
W'             = W × (1 + weight_overhead%)                (default 5%)
TP_min         = ceil((W' + KV_tot) / usable_per_gpu)      (weights alone also must fit)
```

The 2 GB fixed overhead covers the CUDA context, framework runtime, allocator
fragmentation and KV block tables (vLLM-style PagedAttention).

### 1.4 Decode throughput — memory-bound roofline

A decode step is bound by HBM bandwidth: it must read the activated weights
plus the KV state of every resident sequence before it can produce one token.

```
bytes/step = P_act·b_w  +  batch · KV_tok · S_avg  +  comm
tps_stream = BW·TP·η / (P_act·b_w + KV_tok·S_avg + comm)        per sequence
tps_system = batch · BW·TP·η / (P_act·b_w + batch·KV_tok·S_avg + batch·comm)
```

`η` = achieved bandwidth fraction (default 75%; real serving stacks land at
60–85%). Batching amortizes the weight read across sequences — that is why
system throughput grows with concurrency until the KV term dominates.

TP communication (all-reduce of hidden activations, 2 collectives per layer,
all-reduce moves ~2× the data volume):

```
comm = 4 · L · h · b_w · (TP−1)/TP        bytes per token-step
```

Compared against the interconnect speed (NVLink ≈ 900 GB/s, PCIe ≈ 64 GB/s).

### 1.5 Prefill / TTFT — compute-bound

```
FLOPs_prefill = 2 · P_act · S_prompt
TTFT ≈ FLOPs_prefill / (TFLOP/s · TP · MFU_prefill)
```

The factor 2 is one multiply-accumulate per parameter per token (Kaplan et
al. 2020). MFU_prefill default 50%.

---

## 2. Training

### 2.1 Static memory — the classic "16 bytes per parameter"

Mixed-precision AdamW keeps: bf16 params (2 B) + bf16 grads (2 B) + fp32
master weights (4 B) + fp32 1st moment m (4 B) + fp32 2nd moment v (4 B):

```
M_model = P · 16 B          (AdamW)
```

Presets (all byte fields editable in "Memory model tuning"):

| Optimizer          | params | grads | master | m | v | total B/param |
|--------------------|--------|-------|--------|---|---|---------------|
| AdamW              | 2      | 2     | 4      | 4 | 4 | 16            |
| AdamW 8-bit states | 2      | 2     | 4      | 1 | 1 | 10            |
| Adafactor          | 2      | 2     | 4      | 0 | 0 | 8 (+ε factored state) |
| SGD + momentum     | 2      | 2     | 4      | 4 | 0 | 12            |

### 2.2 ZeRO / FSDP sharding (per GPU, N GPUs)

| Stage          | params | grads | optimizer |
|----------------|--------|-------|-----------|
| none / DDP     | P·b_p  | P·b_g | P·(b_m+b_v+b_master) |
| ZeRO-1         | P·b_p  | P·b_g | ÷ N |
| ZeRO-2         | P·b_p  | ÷ N   | ÷ N |
| ZeRO-3 / FSDP  | ÷ N    | ÷ N   | ÷ N |

ZeRO-3 pays for this with ~2 all-gathers + 1 reduce-scatter over the full
parameter set per optimizer step; the app folds that cost into the MFU you
choose rather than modeling interconnect directly.

### 2.3 Activation memory (per GPU, per micro-batch)

From Korthikanti et al. 2022 (*Reducing Activation Recomputation in Large
Transformer Models*), bf16, without sequence parallelism:

```
none      c = 34 + 5·a·s/h
selective c = 34                (recompute the quadratic attention term)
full      c ≈ 2                 (store only layer inputs; +30–40% recompute time)

act = L · s · b · h · c         bytes
```

The activation coefficient `c` is directly editable — override it for sequence
parallelism (Megatron-SP roughly divides several terms by TP) or MoE variants.

### 2.4 Logits + cross-entropy

```
logits = s · b · V · 4 B · 2    (fp32 logits + their gradient)
```

With V = 128k–256k this term can exceed all layer activations at small
micro-batch × long sequence; the app flags when it dominates.

### 2.5 Compute, time, cost

```
FLOPs = 6 · P_act · D                    (forward 2PD + backward 4PD)
t     = FLOPs / (N · TFLOP/s · MFU)
tok/s = D / t
cost  = N · t_hours · $/GPU-hour
checkpoint = P · (b_p + b_master + b_m + b_v)
```

Gradient accumulation: `accum = global_batch / (N · micro_batch)`.

Sanity check (Llama-3-405B, 15.6T tokens, 16,384 H100, MFU ≈ 45%):
6·405e9·15.6e12 = 3.8e25 FLOP → ≈ 62 days — matching Meta's reported
~54-day run within MFU uncertainty.

---

## 3. OpenRouter size parsing

OpenRouter's `/api/v1/models` publishes **no parameter field** (the union of
all model keys contains no size), but vendor naming carries it for ~22% of
the live catalog. Rules, ordered by priority:

| Pattern            | Example                | Meaning                    |
|--------------------|------------------------|----------------------------|
| `(\d+)B-A(\d+)B`   | `qwen3-235b-a22b`      | MoE total 235B, active 22B |
| `(\d+)x(\d+)B`     | `mixtral-8x7b`         | MoE total ≈ 8×7 = 56B      |
| `a(\d+)B` alone    | `hunyuan-a13b`         | **active** only (13B); total unknown |
| `(\d+(?:\.\d+)?)T` | `qwen3.8-2.4t-a95b`    | 2.4T = 2400B               |
| `(\d+(?:\.\d+)?)B` | `llama-3.3-70b`        | dense total 70B            |

The `name` field is parsed before `id` (e.g. `qwen/qwen3-coder` hides
"480B A35B" in the name only). `:free`/`:batch` suffixes are pricing variants
of one model; `~…-latest` ids are aliases pointing at a canonical slug.

Validation (2026-09-21, 19 models with both a size token and a
`hugging_face_id`): **median absolute error ≈ 2.7% for models ≥ 24B**; for
sub-10B dense models the name undershoots by 5–25% because vendors quote core
params while safetensors totals include embeddings and vision towers.

What OpenRouter **cannot** tell the calculator: layers, heads, KV heads, MLA
ranks, expert counts, license, exact dtype of the downloadable checkpoint.
That is why the app keeps the architecture section user-editable with verified
presets.

## Cost model (inference)

```
self-host $/1M output tokens = (TP × GPU $/h) ÷ (system tok/s × 3600) × 1e6
self-host $/1M input tokens  = (TP × GPU $/h) ÷ (prompt ÷ TTFT × 3600) × 1e6

marginal $/1M (electricity only, owned hw):
  gpu_kW      = TP × TDP_W / 1000
  facility_kW = (gpu_kW × (1 + host%)) × PUE
  $/h         = facility_kW × elec $/kWh
  $/1M        = $/h ÷ (system tok/s × 3600) × 1e6
break-even concurrency = current batch × (API $/1M out ÷ self-host $/1M out)
```

Display order in the results panel (and copy report): OpenRouter API price →
self-host out/in → marginal. The electricity rate defaults to $0.12/kWh; empty
hides the marginal row.

GPU $/h comes from the static hardware table and follows the GPU selection
(manual edits win until the next change). The optional “refresh from Vast.ai
market” button replaces the $/h inputs with median rentable $/GPU-h per model
(`dph_total ÷ num_gpus`, ≥3 listings, >$50 outliers dropped) from the keyless
`console.vast.ai/api/v0/bundles/` search; on failure static prices are kept.
Costs are pure compute rental — they exclude networking, storage, idle
capacity and ops.

## Pricing graph

The bottom graph sweeps one inference input over [min, max] (linear or log-X,
3–40 steps) through the same §1 engine and plots one metric: self-host $/1M
out/in, marginal $/1M, system/single tok/s, total GB, or TTFT. X presets:
params 1–70 B, TP 1–8, batch 1–128, avg ctx 1024–32768, prompt 512–16384.
Non-finite sweep points are skipped; the caption reports the plotted Y range.

## Assumptions & limitations

> Experimental estimates — verify before spending. Found a bug? Please [file an
> issue](https://github.com/jeevanandmaurya/calculate_ai_infra/issues/new) with the model, your inputs, the copied report, and what
> you expected vs what the app showed.

- Throughput numbers are roofline estimates; real deployments land at
  60–85% of them depending on kernels, batch scheduler and speculative decoding.
- Attention FLOPs (`O(s²·a·d)`) are excluded from prefill estimates — at very
  long prompts this underestimates TTFT.
- Activation formulas assume the standard pre-norm transformer block; MoE
  MLP activations scale with experts-per-token (approximated, not exact).
- Speculative decoding, prefix caching and disaggregated prefill change the
  decode/prefill split and are not modeled.
- Training time excludes communication, data loading, checkpointing stalls and
  restarts — all folded into your MFU choice.

## 5. References

1. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 — 6PD rule, 2 FLOPs/param/token.
2. Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, 2022 — activation constants 34 and 5·a·s/h.
3. Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2020 — sharding stages.
4. DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 — MLA KV-cache structure.
5. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention (vLLM)*, 2023 — KV management & overhead.
6. Kipply, *Transformer Inference Arithmetic*, 2023 — memory-bound decode roofline.
7. Hugging Face `config.json` files for the preset architectures (verified 2026-09-21).
8. OpenRouter `/api/v1/models` — live catalog, naming-convention analysis above.

