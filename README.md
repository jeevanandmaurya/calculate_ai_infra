# AI Infra Calculator

Web app that computes the infrastructure required to **serve** or
**train** a transformer model of a given size and architecture — VRAM,
GPU count, decode throughput, TTFT, training time and cost — with a live
model picker fed by the public OpenRouter catalog.

No build step, no dependencies, no API keys. Everything runs client-side.

## Run it

```bash
# any static server works; from the repo root:
python -m http.server 8000
# → http://localhost:8000
```

Opening `index.html` directly from disk also works (the OpenRouter endpoint
sends `Access-Control-Allow-Origin: *`).

## Features

- **Inference tab** — weight memory, KV cache (GQA/MHA/MQA and DeepSeek-style
  MLA), minimum GPUs to fit, memory-bound decode tok/s (single stream and
  batched), compute-bound TTFT, TP all-reduce overhead, and **cost** in display
  order: OpenRouter API price → self-host $/1M output tokens (decode) and $/1M
  input tokens (prefill) at your GPU price → marginal electricity-only $/1M
  (owned hardware). The API-vs-self-host break-even hint is unchanged.
- **Electricity default** — `i-elec` defaults to `$0.12/kWh` (US average), so the
  marginal row shows immediately; clearing the field hides it again.
- **GPU pricing** — static `Calc.GPUS` $/h table stays the default and the price
  input now follows the GPU selection (manual edits win until the next change).
  An opt-in “refresh from Vast.ai market” button (`assets/js/gpu-prices.js`)
  overwrites the $/h inputs with median rentable $/GPU-h from the keyless
  Vast.ai bundle search; failures keep static prices with a message. Shadeform
  is intentionally not used (its API requires `X-API-KEY`).
- **Training tab** — per-GPU memory with DDP / ZeRO-1 / ZeRO-2 / ZeRO-3
  sharding, optimizer presets (AdamW, AdamW-8bit, Adafactor, SGD) with
  editable byte breakdowns, activation memory with selectable checkpointing,
  logits/loss term, 6PD compute, wall-clock time and cost.
- **OpenRouter fetch** — live `openrouter.ai/api/v1/models` catalog (no key);
  size parsed from vendor naming (`70B`, `235B-A22B`, `8x7B`, `a13B`, `2.4T`),
  context window, price and Hugging Face link filled in automatically. Picking a
  model clears the previous architecture first and auto-fills from HF when possible.
- **Keyless architecture auto-fill** — one button reads the model's real
  structure from Hugging Face with no API key: `config.json` first (layers,
  heads, KV heads, experts, MLA rank, context, vocab, exact parameter count
  from safetensors); falls back to the GGUF header via a 64 KB HTTP range
  request for GGUF-only repos; gated repos (meta-llama, gemma) resolve
  automatically through their most-downloaded public GGUF mirror.
- **Verified presets** — Llama 3.x, Qwen3 (dense + MoE), DeepSeek-V3 (MLA),
  Mistral/Mixtral, Gemma 3, Phi-4, Llama 2 (MHA). Every field stays editable.
- **Copy report** — a “Copy report” button on each results card copies a
  structured text report (model config, all results, cost, tagged notes),
  including a provenance line (preset / Hugging Face repo / OpenRouter model /
  manual) and a generation timestamp. When nothing is loaded it copies an
  explicit “No results: no model is loaded” message instead of blanks.
- **Pricing graph** — customizable sweep at the bottom of the page: X = params,
  GPUs/TP, batch, avg ctx, prompt tokens; Y = self-host $/1M out/in, marginal
  $/1M, system/single tok/s, total GB, TTFT. Min/max/steps + log-X, ranges
  preset per X, hover dots for exact values, recomputed from current inputs.
- Responsive single-column layout on phones, two-column with sticky results
  on desktop; light/dark themes.
- **Hardware table** — H100/H200/B200/B300/A100/MI300X/MI325X/L40S/RTX…
  plus a fully custom GPU entry.

## Layout

```
index.html               app shell
assets/css/style.css     design system (light + dark)
assets/js/calc.js        pure calculation engine (no DOM)
assets/js/gpu-prices.js  optional Vast.ai market medians (no key)
assets/js/openrouter.js  catalog fetch + naming-parser
assets/js/app.js         UI wiring / rendering
docs/calculations.md     every formula, assumptions, references
tests/smoke.mjs          engine + parser assertions (34 tests)
tests/check-dom.mjs      JS↔HTML id cross-check
```

## Tests

```bash
node tests/smoke.mjs      # calculation engine + OpenRouter parser
node tests/check-dom.mjs  # every DOM id used by JS exists in the HTML
node tests/gguf-meta.mjs Qwen/Qwen3-8B-GGUF   # read full architecture
                          # from any GGUF repo via a 64 KB range request
node tests/hf-arch.mjs    # live auto-fill chain: config.json → GGUF
                          # header → gated-repo GGUF mirror (network, no key)
node tests/render.mjs     # runs app.js against a stub DOM built from the real
                          # markup and asserts the results panel renders
```

The smoke suite validates hand-computed values (Llama-70B KV = 327,680 B/tok,
DeepSeek-V3 MLA = 70,272 B/tok, 405B/15.6T ≈ 60 days on 16k H100, auto-TP and
ZeRO auto-fit) and the OpenRouter naming parser against known ids
(`235B-A22B`, `2.4T-A95B`, `hunyuan-a13b` active-only trap, `8x7B`, name-only
`480B A35B`).

## No silent stale data

The app starts **empty** — no model, no catalog fetch. Every result cell shows
`—` until a model is actually defined, and the notes panel explains why.

Model-derived fields (parameters, vocab, layers, heads, KV heads, head dim,
MLA ranks, experts) are wiped to “not available” whenever they can no longer be
trusted:


| Event                                                  | What happens                                                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Picking an OpenRouter model                            | Architecture is cleared first, then auto-filled from its HF repo if it has one                    |
| Auto-fill fails (gated, closed model, network, bad id) | All architecture fields cleared, failure reason shown in the status line and in the results notes |
| Model has no`hugging_face_id` (closed model)           | Architecture stays unavailable; you are told to enter values or pick a preset                     |
| Selecting the “Custom” preset                        | Clears model and architecture fields instead of inventing defaults                                |
| Architecture incomplete while typing                   | Results show`—` plus the reason, never numbers from a half-filled form                           |

## Methodology

All formulas and their sources (Kaplan 2020 6PD rule, Korthikanti 2022
activation constants, ZeRO sharding, DeepSeek-V3 MLA, vLLM overhead model)
are documented in [docs/calculations.md](docs/calculations.md) and summarized
in the in-app **Methodology** tab.

## Experimental warning + bug reports

All numbers are roofline **estimates** — verify against vendor docs and a real
run before spending money. Throughput assumes ideal kernels/scheduling, GPU
$/h moves with the market, and the Vast.ai medians are peer listings, not
quotes.

Found a bug? Please [file an issue](https://github.com/jeevanandmaurya/calculate_ai_infra/issues/new) with:

1. The model (preset / HF repo / OpenRouter id) and a screenshot or the copied
   report,
2. Your inputs (GPU, precision, batch, context, TP),
3. What you expected and what the app showed instead.
