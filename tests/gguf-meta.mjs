/* GGUF metadata reader — fetches only the first ~64 KB of a .gguf file
   (HTTP Range request) and decodes the GGUF key/value header to recover the
   full architecture: layers, heads, KV heads, experts, context length.
   Run: node tests/gguf-meta.mjs <repo> [file]
   e.g. node tests/gguf-meta.mjs Qwen/Qwen3-8B-GGUF */
import { readFileSync } from "node:fs";

const repo = process.argv[2] || "Qwen/Qwen3-8B-GGUF";

/* --- list *.gguf files via the public Hub API (no key needed) --- */
const api = await fetch(`https://huggingface.co/api/models/${repo}`);
if (!api.ok) { console.error("HF API error", api.status); process.exit(1); }
const info = await api.json();
const ggufs = (info.siblings || []).map((s) => s.rfilename)
  .filter((f) => f.toLowerCase().endsWith(".gguf"));
if (!ggufs.length) { console.error("no .gguf files in", repo); process.exit(1); }
const file = process.argv[3] || ggufs.find((f) => /Q4_K_M|Q4_0|Q8_0/i.test(f)) || ggufs[0];
console.log(`repo: ${repo}\nfile: ${file} (${ggufs.length} gguf files available)\n`);

/* --- range-download just the header --- */
const url = `https://huggingface.co/${repo}/resolve/main/${encodeURIComponent(file)}`;
const res = await fetch(url, { headers: { Range: "bytes=0-65535" } });
if (!res.ok && res.status !== 206) { console.error("download error", res.status); process.exit(1); }
const buf = Buffer.from(await res.arrayBuffer());

/* --- GGUF v2/v3 header parser --- */
let off = 0;
const magic = buf.toString("ascii", 0, 4);
if (magic !== "GGUF") { console.error("not a GGUF file"); process.exit(1); }
off = 4;
const version = buf.readUInt32LE(off); off += 4;
off += 16;                                    /* tensor_count(u64) + kv_count(u64) */

const T = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
function readStr() {
  const len = Number(buf.readBigUInt64LE(off)); off += 8;
  const s = buf.toString("utf8", off, off + len); off += len;
  return s;
}
function readVal(type) {
  if (type === 8) return readStr();
  if (type === 9) {
    const elemType = buf.readUInt32LE(off); off += 4;
    const count = Number(buf.readBigUInt64LE(off)); off += 8;
    const arr = [];
    for (let i = 0; i < Math.min(count, 8); i++) arr.push(readVal(elemType));
    return count > 8 ? [...arr, `…(+${count - 8})`] : arr;
  }
  if (type === 7) { const v = buf.readUInt8(off) !== 0; off += 1; return v; }
  const size = T[type];
  let v;
  if (type === 0) v = buf.readUInt8(off);
  else if (type === 1) v = buf.readInt8(off);
  else if (type === 2) v = buf.readUInt16LE(off);
  else if (type === 3) v = buf.readInt16LE(off);
  else if (type === 4) v = buf.readUInt32LE(off);
  else if (type === 5) v = buf.readInt32LE(off);
  else if (type === 6) v = buf.readFloatLE(off);
  else if (type === 10) v = buf.readDoubleLE(off);
  else if (type === 11) v = Number(buf.readBigInt64LE(off));
  else if (type === 12) v = Number(buf.readBigUInt64LE(off));
  off += size;
  return v;
}

const meta = {};
const kvCount = Number(buf.readBigUInt64LE(16));   /* magic(4)+ver(4)+tensors(8) */
try {
  for (let i = 0; i < kvCount; i++) {
    const key = readStr();
    const type = buf.readUInt32LE(off); off += 4;
    meta[key] = readVal(type);
  }
} catch { /* header longer than 64 KB — the keys we need come first */ }

console.log(`GGUF v${version} · ${kvCount} metadata keys\n`);
const interesting = Object.entries(meta)
  .filter(([k]) => /general\.|block_count|head_count|key_length|value_length|embedding_length|expert|context_length|rope\.|attention\.|feed_forward/i.test(k))
  .filter(([k]) => !/^tokenizer/i.test(k));
for (const [k, v] of interesting) console.log(`${k.padEnd(45)} ${JSON.stringify(v)}`);
