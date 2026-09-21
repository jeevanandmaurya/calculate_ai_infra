/* DOM cross-check: every $("id") referenced in JS must exist in index.html.
   Run: node tests/check-dom.mjs */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "index.html"), "utf8");
const js = ["app.js", "calc.js", "openrouter.js"]
  .map((f) => readFileSync(join(root, "assets/js", f), "utf8"))
  .join("\n");

const used = new Set();
for (const m of js.matchAll(/\$\("([a-z0-9-]+)"\)/gi)) used.add(m[1]);
for (const m of js.matchAll(/getElementById\("([a-z0-9-]+)"\)/gi)) used.add(m[1]);
/* result ids referenced as plain strings (e.g. RESULT_IDS lists) */
for (const m of js.matchAll(/"([rt]-[it]-[a-z0-9-]+)"/g)) used.add(m[1]);

const missing = [...used].filter((id) => !html.includes(`id="${id}"`));
console.log(`ids referenced: ${used.size} · missing in HTML: ${missing.length ? missing.join(", ") : "none"}`);

const panels = new Set([...html.matchAll(/data-panel="([a-z]+)"/g)].map((m) => m[1]));
const tabs = new Set([...html.matchAll(/data-tab="([a-z]+)"/g)].map((m) => m[1]));
const jsPanels = new Set([...js.matchAll(/dataset\.panel !== "([a-z]+)"|dataset\.tab === "([a-z]+)"/g)]
  .flatMap((m) => [m[1], m[2]]).filter(Boolean));
for (const p of jsPanels) {
  const ok = panels.has(p) || tabs.has(p);
  console.log(`${ok ? "  ok  " : "FAIL  "}panel/tab "${p}" exists in HTML`);
}
process.exit(missing.length ? 1 : 0);
