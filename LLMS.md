# LLMS.md — JAIT, for LLM agents

> This file is written for an LLM agent (or a developer handing this repo
> to one). It is the complete, factual overview of the JAIT library: what
> it is, how to install it, the full API with exact signatures, the model
> format, the numeric guarantees, and the traps. Read this file first;
> you do not need any other document to use JAIT correctly. If this file
> and the code disagree, the code is right — fix this file.

## 1. What JAIT is

JAIT is a TypeScript engine that runs TAIT-trained `.npz` models directly
in the browser (or in Node). No server, no account, no API key, no data
leaves the device: the model file is downloaded straight from Hugging
Face (or opened from the user's own file) and executed locally.

One `.npz`, three runtimes:

- **TAIT** trains the model (Python / NumPy, on a phone) —
  <https://github.com/muzammilsoft/tait>
- **KAIT** runs the model on the phone (Kotlin) —
  <https://github.com/muzammilsoft/KAIT-Studio>
- **JAIT** runs the model in the browser (TypeScript) — this repo.

JAIT is a line-by-line port of TAIT's `tait/core/model.py` (MiniGPT
forward) and `tait/core/tokenizer.py` (byte-level BPE). It is not a
general ML framework: it runs exactly one architecture (a single
transformer block, no LayerNorm — TAIT has none) on exactly one file
format (TAIT `.npz`, see §5).

Live demo of JAIT in a real web app (same engine, same model):
<https://www.bilibili.com/toy/tait-toy/index.html> (TAIT-Toy).

## 2. Install

JAIT is **not on npm**. Install from GitHub (the build runs automatically
on install via the `prepare` script):

```sh
npm install github:muzammilsoft/jait
```

Or clone and build it yourself:

```sh
git clone https://github.com/muzammilsoft/jait
cd jait
npm install
npm run build   # tsc -> dist/
npm test        # build + the verification gate (see §8)
```

Runtime dependency: only `fflate` (MIT) for ZIP/DEFLATE. TypeScript is a
dev-only tool; the output is plain JavaScript (ESM). Package entry:
`dist/index.js` (`main`), types in `dist/index.d.ts`.

## 3. Quick start

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
console.log(reply);
```

Load from the user's own file instead (the file never leaves the device):

```ts
import { loadModelFromNpz } from 'jait';
const bytes = new Uint8Array(await file.arrayBuffer());
const { model, tokenizer } = loadModelFromNpz(bytes);
```

In a web app, bundle JAIT with a bundler (Vite, webpack, …) and run
generation in a Web Worker so the UI never blocks — that is exactly how
the TAIT-Toy demo does it. A plain `<script type="module">` import of
`dist/index.js` also works if the file is served over HTTP(S).

## 4. Full API reference

Everything is exported from `dist/index.js` (source in `src/`).
Types are exported alongside the values.

### Loading

```ts
loadModelFromNpz(bytes: Uint8Array): LoadedModel
loadModelFromUrl(url: string, fetchFn?: typeof fetch): Promise<LoadedModel>
```

- `LoadedModel = { model: MiniGPT; tokenizer: BPETokenizer }`.
- Both throw a clear `Error` when the file is not a TAIT model
  (missing BPE merge data or missing weights). A non-TAIT file is
  rejected, never silently misread.
- `loadModelFromUrl` throws when the download fails
  (`Could not download the model (HTTP <status>)`).

### Model

```ts
class MiniGPT {
  readonly vocabSize: number;
  readonly dModel: number;
  readonly dFf: number;
  readonly blockSize: number;
  readonly padId: number;

  forward(ids: number[], validMask?: boolean[]): Float64Array;
  forwardSingle(idsCond: number[]): Float64Array;
}
```

- `forward(ids)` runs the full forward pass and returns probabilities
  as a flat `[T * V]` array (row `t` = distribution after token `t`).
  `validMask[i] === false` marks a padding position (generation always
  passes all-true; the mask exists so training-style masked calls stay
  exact).
- **`forward()` returns a view into per-model scratch storage (see §7).
  Consume it or copy it before the next `forward()` call on the same
  model — the next call overwrites it.**
- `forwardSingle(idsCond)` returns the probabilities at the last
  position only, as a fresh copy (safe to hold). This is the call
  generation uses.
- `ModelMeta = { vocabSize, dModel, dFf, blockSize, padId }` (all numbers).

### Generation

```ts
generate(model: MiniGPT, tokenizer: BPETokenizer, prompt: string, options?: GenerateOptions): string;
generateReply(model: MiniGPT, tokenizer: BPETokenizer, prompt: string, options?: GenerateOptions): string;
generateGreedy(model: MiniGPT, tokenizer: BPETokenizer, prompt: string, maxNewTokens?: number): string;
extractReply(full: string): string;
sampleNext(probs: Float64Array, padId: number, opts: { temperature: number; topK: number; topP: number }, rng: () => number): number;

interface GenerateOptions {
  maxNewTokens?: number;   // default 64
  temperature?: number;    // default 0.8
  topK?: number;           // default 40
  topP?: number;           // default 0.95
  seed?: number;           // 0 or undefined = unseeded (Math.random); any other value seeds mulberry32
  onToken?: (id: number, replySoFar: string) => void;
  signal?: AbortSignal;    // cancel generation mid-stream
}
const DEFAULT_GENERATE_OPTIONS: { maxNewTokens: 64; temperature: 0.8; topK: 40; topP: 0.95 };
```

- `generate()` returns the full decoded text: prompt + SEP + reply + END.
  `generateReply()` returns only the reply (same as TAIT's
  `generate_reply`: text after SEP, cut at END, trimmed).
- `generateGreedy()` decodes by argmax each step (deterministic;
  `maxNewTokens` defaults to 20). Use it for verification, not for
  user-facing text.
- Sampling order, copied exactly from Python: (1) last-position probs,
  pad set to 0, renormalise; (2) temperature power
  `p = probs ** (1 / max(T, 1e-6))`, renormalise; (3) top-k, renormalise;
  (4) top-p, renormalise; (5) draw one token. Stops on the END token
  or after `maxNewTokens`.
- **RNG difference (honest):** Python samples with NumPy's PCG64;
  JavaScript has no equivalent. JAIT uses mulberry32, so the same seed
  gives a *different but repeatable* sample in JS than in Python.
  Never compare sampled outputs by seed across the two. Compare with
  top-5 probabilities or greedy decoding instead.

### Tokenizer

```ts
class BPETokenizer {
  merges: Map<string, number>;   // "a,b" -> new id
  vocab: Uint8Array[];           // vocab[id] = raw bytes of that token
  sepId: number; endId: number; padId: number;
  get vocabSizeTotal(): number;  // 256 + merges.size + 3
  addMerges(a: Int32Array, b: Int32Array, ids: Int32Array): void;
  encode(text: string): number[];
  decode(ids: number[]): string;
}
const SEP_CHAR = '\u0001';
const END_CHAR = '\u0002';
```

- Byte-level BPE. Vocab starts as 256 single-byte entries; each merge
  creates one new id. Encoding repeatedly merges the pair with the
  LOWEST merge id until no mergeable pair remains.
- `encode()` splits the text on SEP (`\u0001`) and END (`\u0002`) first.
  `decode()` flushes accumulated bytes on SEP/END (emitting the control
  character itself), skips PAD, and decodes the rest as UTF-8.
- `encode()` / `decode()` throw `Tokenizer is not loaded` when
  `sepId === -1` (i.e. before a model was loaded).

### NPZ / NPY reader

```ts
parseNpz(bytes: Uint8Array): Map<string, NpyArray>;
parseNpy(bytes: Uint8Array): NpyArray;
type NpyArray =
  | { kind: 'f64'; shape: number[]; data: Float64Array }
  | { kind: 'f32'; shape: number[]; data: Float32Array }
  | { kind: 'i32'; shape: number[]; data: Int32Array }
  | { kind: 'i64'; shape: number[]; data: BigInt64Array };
scalarInt(a: NpyArray | undefined, name: string): number;
asFloat64(a: NpyArray | undefined, name: string): Float64Array;
asInt32(a: NpyArray | undefined, name: string): Int32Array;
```

- An `.npz` is a ZIP of `.npy` files (STORED or DEFLATE, both handled
  via fflate). Only the dtypes TAIT writes are supported: `<f8`, `<f4`,
  `<i4`, `<i8`. Fortran-order arrays are rejected.
- `parseNpz` throws `Not a valid .npz (ZIP) file: …` on bad input and
  `Empty .npz: no .npy entries found` when there are no `.npy` entries.

### Model metadata (`tait-meta.json`)

```ts
parseTaitMeta(value: unknown): TaitMeta | null;
isTaitMeta(meta: TaitMeta | null): meta is TaitMeta;
interface TaitMeta {
  taitVersion?: string; data?: string; examples?: number; epochs?: number;
  finalLoss?: number; valLoss?: number; dModel?: number; dFf?: number;
  vocabSize?: number; blockSize?: number; seed?: number; createdAt?: string;
}
```

- `tait-meta.json` is the small metadata file TAIT writes next to a
  published model (`tait publish`). JSON keys use snake_case
  (`tait_version`, `final_loss`, `val_loss`, `d_model`, `d_ff`,
  `vocab_size`, `block_size`, `created_at`).
- `isTaitMeta` is true only when `vocabSize`, `dModel` and `blockSize`
  are all numbers — that is the check a model gallery should use
  before offering "Run".

### Low-level math

```ts
matmul(a: Float64Array, aRows: number, aCols: number, b: Float64Array, bRows: number, bCols: number): Float64Array;
matmulInto(out: Float64Array, a: Float64Array, aRows: number, aCols: number, b: Float64Array, bRows: number, bCols: number): void;
```

- `matmul` allocates and returns the product. `matmulInto` writes the
  product into a pre-allocated `out` (zeroed first); same arithmetic
  and summation order, so results are identical. `out` must hold at
  least `aRows * bCols` elements and must not overlap the inputs.

## 5. The model format

A TAIT `.npz` is self-contained: no `vocab.json` is ever needed. It
contains:

| Arrays | Dtype | Meaning |
|---|---|---|
| `wte`, `wpe` | float32 | token + position embeddings |
| `Wq`, `Wk`, `Wv`, `Wo` | float64 | attention weights |
| `W1`, `b1`, `W2`, `b2` | float64 | MLP weights/biases |
| `Whead`, `bhead` | float64 | output head |
| `merge_a`, `merge_b`, `merge_id` | int32 | BPE merges |
| `sep_id`, `end_id`, `tok_pad_id`, `block_size`, `pad_id` | int scalars | special ids + limits |

Architecture: a single transformer block with **no LayerNorm** (TAIT
has none). `h0 = wte[x] + wpe[:T]`; causal self-attention with `-1e9`
masking (`scores = Q @ K^T / sqrt(D)`, masked where `j > i` or the key
is padding); `h1 = h0 + attn`; ReLU MLP `ff1 = relu(h1 @ W1 + b1)`,
`ff2 = ff1 @ W2 + b2`; `h2 = h1 + ff2 + b2`; `logits = h2 @ Whead +
bhead`; row-wise softmax with a `+1e-9` denominator (same as TAIT).

Reference model (the one the tests use): TinyBili,
<https://huggingface.co/muzammilsoft/tinybili> (`tinyGPT0.1.npz`,
1,072,206 bytes): vocab 1003, `dModel` 64, `dFf` 128, `blockSize` 64,
SEP 1000, END 1001, PAD 1002, 744 merges, trained on 6,666 examples
for 50 epochs. TinyBili is a tiny experimental model — its answers
are often garbled. Present it honestly as an experimental model,
never as a strong one.

## 6. Numeric guarantees

- Embedding addition is rounded with `Math.fround`, because NumPy
  adds the float32 embeddings in float32. Everything after that is
  float64 in both.
- Summation order inside a matmul differs from NumPy/BLAS, so results
  agree to ~1e-12, not bit-for-bit.
- Verification gate (see §8): same top-5 next-token ids, max
  probability difference below 1e-3, identical greedy decoding.
  Measured on TinyBili: max difference **2.22e-16**.

## 7. Performance and the buffer-reuse contract

`forward()` allocates **zero** `Float64Array`s: all 12 intermediate
tensors live in per-model scratch storage (sized to `blockSize`, grown
only if a longer sequence is passed) and are overwritten on every
call. A short reply used to churn ~0.56 MB of garbage (78 allocations);
now it churns ~47 KB (6 small copies from `forwardSingle`). This is
what keeps generation smooth in the browser without GC jank.

Consequences an agent must respect:

- `forward()` returns a **view** into scratch. Consume it or copy it
  before the next `forward()` call on the same model. Holding it
  across calls reads corrupted data.
- `forwardSingle()` returns a fresh copy — safe to hold.
- `sampleNext()` copies the probabilities internally before mutating
  them, so it is safe to pass it a `forward()` view.
- Each `MiniGPT` instance owns its own scratch; two models never share
  buffers. `forward()` is synchronous and single-threaded, so there is
  no re-entrancy hazard.

## 8. Verification

```sh
npm test   # build + Phase 1 gate
```

`test/verify.mjs` loads TinyBili (local file when `JAIT_TEST_NPZ` is
set, else from Hugging Face) and checks, against `test/reference.json`
(generated from the real TAIT source):

1. encoder ids match exactly (Arabic and English prompts);
2. top-5 next-token ids match, max probability difference < 1e-3
   (measured 2.22e-16);
3. greedy decoding is identical to Python;
4. a non-TAIT file is rejected with a clear error.

CI (`.github/workflows/ci.yml`) runs `npm ci` + `npm test` on every
push and pull request (Node 20).

## 9. Recipes

**Run a prompt and print the reply** — see §3.

**Load a local `.npz` file and check it is a TAIT model:**

```ts
import { loadModelFromNpz } from 'jait';
const { model, tokenizer } = loadModelFromNpz(bytes);
console.log(`V=${model.vocabSize} D=${model.dModel} block=${model.blockSize}`);
```

**Stream tokens to a UI as they are generated:**

```ts
generateReply(model, tokenizer, prompt, {
  maxNewTokens: 64,
  onToken: (id, replySoFar) => { /* update the UI with replySoFar */ },
});
```

**Cancel generation (e.g. a Stop button):**

```ts
const ctrl = new AbortController();
generateReply(model, tokenizer, prompt, { signal: ctrl.signal });
ctrl.abort();
```

**Verify a new TAIT model file behaves like Python:** load it with
`loadModelFromNpz`, run `generateGreedy` on a fixed prompt, and compare
the output with Python's greedy output for the same prompt and model.
Do not compare sampled outputs by seed (§4, RNG difference).

**Embed in a web app:** bundle with Vite (or any bundler), import
from `'jait'`, run generation inside a Web Worker, post the model
bytes or URL to the worker. The TAIT-Toy demo is the reference for
this pattern.

## 10. Constraints — do not do these

- Do not compare sampled outputs to Python by seed (different RNGs).
- Do not hold a `forward()` result across another `forward()` call.
- Do not present TinyBili (or any tiny TAIT demo model) as a strong
  or general model. It is experimental; say so.
- Do not add a server, an API key, an account, or any tracking. The
  whole point is that everything runs locally.
- Do not change the sampling order or the `+1e-9` softmax denominator
  — they are copied from Python and the verification gate depends on them.
- Do not claim JAIT is affiliated with Bilibili, Hugging Face, or any
  model host. It is an independent open-source project by Khartoum Ghoul.

## 11. Links

- JAIT (this repo): <https://github.com/muzammilsoft/jait>
- TAIT (trainer): <https://github.com/muzammilsoft/tait>
- KAIT Studio (Android runner): <https://github.com/muzammilsoft/KAIT-Studio>
- TinyBili (reference model): <https://huggingface.co/muzammilsoft/tinybili>
- TAIT-Toy (live web demo): <https://www.bilibili.com/toy/tait-toy/index.html>
- License: MIT (see `LICENSE`).
