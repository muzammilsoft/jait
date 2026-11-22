/**
 * Phase 1 gate: load TinyBili (local copy if present, else Hugging Face)
 * in the compiled JAIT engine and compare against the Python reference.
 * Pass = same encoder ids, same top-5 ids, max prob diff < 1e-3,
 * and identical greedy decoding.
 */
import fs from 'node:fs';
import { loadModelFromNpz, loadModelFromUrl, generateGreedy, SEP_CHAR } from '../dist/index.js';

const REF = JSON.parse(fs.readFileSync(new URL('./reference.json', import.meta.url), 'utf8'));
const HF_URL = 'https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz';
const LOCAL = process.env.JAIT_TEST_NPZ || '/tmp/jait-check/tinybili.npz';

let loaded;
if (fs.existsSync(LOCAL)) {
  console.log('loading TinyBili from local file:', LOCAL);
  loaded = loadModelFromNpz(new Uint8Array(fs.readFileSync(LOCAL)));
} else {
  console.log('loading TinyBili from Hugging Face:', HF_URL);
  loaded = await loadModelFromUrl(HF_URL);
}
const { model, tokenizer } = loaded;
console.log(`model: V=${model.vocabSize} D=${model.dModel} Dff=${model.dFf} block=${model.blockSize} pad=${model.padId}`);
console.log(`tokenizer: sep=${tokenizer.sepId} end=${tokenizer.endId} pad=${tokenizer.padId} merges=${tokenizer.merges.size}`);

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}

for (const c of REF.cases) {
  const ids = tokenizer.encode(c.prompt + SEP_CHAR);
  check(
    `encode ${JSON.stringify(c.prompt)}`,
    JSON.stringify(ids) === JSON.stringify(c.ids),
    `got [${ids}] want [${c.ids}]`,
  );
  const probs = model.forwardSingle(ids.slice(-model.blockSize));
  const p = Float64Array.from(probs);
  p[model.padId] = 0;
  let sum = 0;
  for (const v of p) sum += v;
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  const top5 = Array.from(p.keys())
    .sort((a, b) => p[b] - p[a])
    .slice(0, 5)
    .map((i) => [i, p[i]]);
  const idsMatch = JSON.stringify(top5.map(([i]) => i)) === JSON.stringify(c.top5.map(([i]) => i));
  let maxDiff = 0;
  for (let k = 0; k < 5; k++) maxDiff = Math.max(maxDiff, Math.abs(top5[k][1] - c.top5[k][1]));
  check(
    `top5 ${JSON.stringify(c.prompt)}`,
    idsMatch && maxDiff < REF.gate.maxProbDiffBelow,
    `ids ${idsMatch ? 'match' : 'MISMATCH got [' + top5.map(([i]) => i) + ']'} maxDiff=${maxDiff.toExponential(2)}`,
  );
}

for (const g of REF.greedy) {
  const full = generateGreedy(model, tokenizer, g.prompt, g.maxNew);
  check(
    `greedy ${JSON.stringify(g.prompt)}`,
    full === g.full,
    `got ${JSON.stringify(full)}`,
  );
}

// A non-TAIT file must be rejected with a clear error.
try {
  loadModelFromNpz(new Uint8Array([80, 75, 3, 4]));
  check('rejects non-TAIT file', false, 'no error thrown');
} catch (e) {
  check('rejects non-TAIT file', /Not a valid|Empty/.test(e.message), e.message);
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
