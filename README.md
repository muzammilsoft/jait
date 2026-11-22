# JAIT — run TAIT models in the browser

JAIT is the TypeScript engine that runs TAIT-trained `.npz` models directly
in the browser. No server, no account, no data leaves the device: the model
is downloaded straight from Hugging Face (or opened from the user's own
file) and executed locally.

Arabic: [README.ar.md](README.ar.md) · 简体中文: [README.zh-CN.md](README.zh-CN.md)

One `.npz`, three runtimes:

- **TAIT** trains (Python / NumPy, on the phone) —
  [muzammilsoft/tait](https://github.com/muzammilsoft/tait)
- **KAIT** runs on the phone (Kotlin) —
  [muzammilsoft/KAIT-Studio](https://github.com/muzammilsoft/KAIT-Studio)
- **JAIT** runs in the browser (TypeScript) — this repo

## Performance

`forward()` allocates nothing: every intermediate tensor lives in
per-model scratch storage that is reused across calls, so a reply
produces no GC churn (before the fix, one forward pass allocated 12
`Float64Array`s — about 0.5 MB of garbage per short reply; now it is
0 for the forward pass). The returned probabilities are a view into
that scratch storage: consume or copy them before the next `forward()`
call on the same model.

## What is verified

Phase 1 gate — `npm test` loads TinyBili
(`muzammilsoft/tinybili`, `tinyGPT0.1.npz`, 1,072,206 bytes) and compares
against Python TAIT 2.3.0:

- encoder ids match exactly for Arabic and English prompts
- top-5 next-token ids match, max probability difference **2.22e-16**
  (gate: below 1e-3)
- greedy decoding is identical to Python
- a non-TAIT file is rejected with a clear error

The test loads from Hugging Face when no local copy is present
(`JAIT_TEST_NPZ=/path/to/model.npz` overrides the source).
Reference values live in `test/reference.json`, generated from the real
TAIT source in `workspace/tait-dev`.

## The model format (what JAIT implements)

Ported line-by-line from `tait/core/model.py` and `tait/core/tokenizer.py`:

- **NPZ**: a ZIP of `.npy` arrays (STORED or DEFLATE). Weights:
  `wte, wpe, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Whead, bhead`, plus the BPE
  data `merge_a, merge_b, merge_id` and scalars `sep_id, end_id,
  tok_pad_id, block_size, pad_id`. `wte`/`wpe` are float32, the rest
  float64; merges are int32.
- **Tokenizer**: byte-level BPE. Vocab is 256 single bytes + merges + 3
  special tokens (SEP 1000, END 1001, PAD 1002 in TinyBili). Encoding
  merges the pair with the lowest merge id first. The NPZ is
  self-contained: no `vocab.json` is ever needed.
- **Model**: a single transformer block with **no LayerNorm** (TAIT has
  none). Causal self-attention with `-1e9` masking, ReLU MLP, residual
  connections. Embedding addition is rounded with `Math.fround` because
  NumPy adds the float32 embeddings in float32.
- **Generation**: same order as Python — pad zeroed, temperature power,
  top-k, top-p, then one draw. TinyBili is a tiny experimental model
  (6,666 examples, 50 epochs), so its answers are often garbled; JAIT
  shows it honestly as an experimental model.

## Honest difference from Python

Python samples with NumPy's PCG64 RNG; JavaScript has no equivalent.
JAIT uses mulberry32, so the same seed gives a *different but repeatable*
sample in JS than in Python. Deterministic comparison must use top-5
probabilities or greedy decoding, never a sampled seed.

## Usage

Install (JAIT is not on npm yet — install from GitHub; the build runs
automatically on install):

```sh
npm install github:muzammilsoft/jait
```

Or clone and build it yourself:

```sh
git clone https://github.com/muzammilsoft/jait
cd jait
npm install
npm run build   # -> dist/
npm test        # build + Phase 1 gate
```

Then:

```ts
import { loadModelFromUrl, generateReply } from 'jait';

const { model, tokenizer } = await loadModelFromUrl(
  'https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz',
);
const reply = generateReply(model, tokenizer, 'مرحبا', {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  maxNewTokens: 64,
});
```

Or load from the user's own file:

```ts
import { loadModelFromNpz } from 'jait';
const bytes = new Uint8Array(await file.arrayBuffer());
const { model, tokenizer } = loadModelFromNpz(bytes);
```

Runtime dependency: only `fflate` (MIT) for ZIP/DEFLATE. TypeScript is a
dev-only tool; the output is plain JavaScript.

JAIT is an independent open-source project by Khartoum Ghoul. It is not
affiliated with Bilibili, Hugging Face, or any model host.
