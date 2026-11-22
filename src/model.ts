/**
 * MiniGPT forward pass — a line-by-line port of TAIT's
 * tait/core/model.py (MiniGPT.forward).
 *
 * Architecture (single transformer block, NO LayerNorm — TAIT has none):
 *   h0  = wte[x] + wpe[:T]
 *   Q,K,V = h0 @ Wq/Wk/Wv
 *   scores = Q @ K^T / sqrt(D), masked with -1e9 where causal (j > i)
 *          or the key position is padding
 *   A = softmax(scores); attn = A @ V @ Wo; h1 = h0 + attn
 *   ff1 = relu(h1 @ W1 + b1); ff2 = ff1 @ W2 + b2; h2 = h1 + ff2
 *   logits = h2 @ Whead + bhead; probs = softmax(logits)
 *
 * Numeric notes:
 * - JavaScript numbers are float64, like the float64 weights. The two
 *   embedding tables are float32 in the file; NumPy adds them in float32,
 *   so the sum is rounded with Math.fround to match Python bit-for-bit
 *   as closely as possible. Everything after that is float64 in Python.
 * - Summation order inside a matmul differs from NumPy/BLAS, so results
 *   agree to ~1e-12, not bit-for-bit. The verification gate is: same
 *   top-5 token ids and max probability difference below 1e-3.
 * - Buffer reuse: forward() writes every intermediate into per-model
 *   scratch storage instead of allocating (12 Float64Arrays per call
 *   before; zero now). The returned probabilities are a view into that
 *   scratch — consume or copy them before the next forward() call.
 */

import { parseNpz, asFloat64, asInt32, scalarInt } from './npz.js';
import { BPETokenizer } from './tokenizer.js';

/**
 * Matmul that writes into a pre-allocated `out` (zeroed first).
 * Same arithmetic and summation order as `matmul`, so results are
 * identical; `out` must hold at least aRows * bCols elements.
 * `out` and the inputs must not overlap.
 */
export function matmulInto(
  out: Float64Array,
  a: Float64Array,
  aRows: number,
  aCols: number,
  b: Float64Array,
  bRows: number,
  bCols: number,
): void {
  if (aCols !== bRows) {
    throw new Error(`matmul shape mismatch (${aRows}x${aCols}) @ (${bRows}x${bCols})`);
  }
  out.fill(0);
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = a[i * aCols + k];
      if (aik === 0) continue;
      for (let j = 0; j < bCols; j++) {
        out[i * bCols + j] += aik * b[k * bCols + j];
      }
    }
  }
}

export function matmul(
  a: Float64Array,
  aRows: number,
  aCols: number,
  b: Float64Array,
  bRows: number,
  bCols: number,
): Float64Array {
  if (aCols !== bRows) {
    throw new Error(`matmul shape mismatch (${aRows}x${aCols}) @ (${bRows}x${bCols})`);
  }
  const out = new Float64Array(aRows * bCols);
  matmulInto(out, a, aRows, aCols, b, bRows, bCols);
  return out;
}

/** In-place row-wise softmax with the same +1e-9 denominator as TAIT. */
export function softmaxRows(x: Float64Array, rows: number, cols: number): void {
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) {
      const v = x[r * cols + c];
      if (v > max) max = v;
    }
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(x[r * cols + c] - max);
      x[r * cols + c] = e;
      sum += e;
    }
    const denom = sum + 1e-9;
    for (let c = 0; c < cols; c++) x[r * cols + c] /= denom;
  }
}

export interface ModelMeta {
  vocabSize: number;
  dModel: number;
  dFf: number;
  blockSize: number;
  padId: number;
}

/**
 * Reusable per-forward scratch storage (see MiniGPT.forward).
 * One set of buffers per model, sized for the longest sequence seen
 * so far; a forward pass overwrites them instead of allocating.
 */
interface ForwardScratch {
  h0: Float64Array;
  Q: Float64Array;
  K: Float64Array;
  Vv: Float64Array;
  scores: Float64Array;
  attnOut: Float64Array;
  attnProj: Float64Array;
  h1: Float64Array;
  pre1: Float64Array;
  ff2: Float64Array;
  h2: Float64Array;
  logits: Float64Array;
  /** Sequence length this storage was sized for. */
  capacity: number;
}

export class MiniGPT {
  readonly vocabSize: number;
  readonly dModel: number;
  readonly dFf: number;
  readonly blockSize: number;
  readonly padId: number;

  private wte: Float64Array;
  private wpe: Float64Array;
  private Wq: Float64Array;
  private Wk: Float64Array;
  private Wv: Float64Array;
  private Wo: Float64Array;
  private W1: Float64Array;
  private b1: Float64Array;
  private W2: Float64Array;
  private b2: Float64Array;
  private Whead: Float64Array;
  private bhead: Float64Array;

  /** Per-forward scratch buffers, allocated once and reused (see forward). */
  private scratch: ForwardScratch | null = null;

  /** Return scratch sized for a sequence of length T, growing it if needed. */
  private ensureScratch(T: number): ForwardScratch {
    const cur = this.scratch;
    if (cur && cur.capacity >= T) return cur;
    // Size for the model's block size, not just the current T: generation
    // grows T by one token per step, and sizing to T would reallocate
    // all 12 buffers on every step. blockSize is the real maximum
    // (generate() slices the context to it), so this allocates once.
    const size = Math.max(T, this.blockSize);
    const D = this.dModel;
    const F = this.dFf;
    const V = this.vocabSize;
    const next: ForwardScratch = {
      h0: new Float64Array(size * D),
      Q: new Float64Array(size * D),
      K: new Float64Array(size * D),
      Vv: new Float64Array(size * D),
      scores: new Float64Array(size * size),
      attnOut: new Float64Array(size * D),
      attnProj: new Float64Array(size * D),
      h1: new Float64Array(size * D),
      pre1: new Float64Array(size * F),
      ff2: new Float64Array(size * D),
      h2: new Float64Array(size * D),
      logits: new Float64Array(size * V),
      capacity: size,
    };
    this.scratch = next;
    return next;
  }

  constructor(meta: ModelMeta, weights: Record<string, Float64Array>) {
    this.vocabSize = meta.vocabSize;
    this.dModel = meta.dModel;
    this.dFf = meta.dFf;
    this.blockSize = meta.blockSize;
    this.padId = meta.padId;
    this.wte = weights['wte']!;
    this.wpe = weights['wpe']!;
    this.Wq = weights['Wq']!;
    this.Wk = weights['Wk']!;
    this.Wv = weights['Wv']!;
    this.Wo = weights['Wo']!;
    this.W1 = weights['W1']!;
    this.b1 = weights['b1']!;
    this.W2 = weights['W2']!;
    this.b2 = weights['b2']!;
    this.Whead = weights['Whead']!;
    this.bhead = weights['bhead']!;
  }

  /**
   * Full forward pass. Returns probabilities as a flat [T * V] array.
   * validMask[i] is false for padding positions (generation always passes
   * all-true; it exists so training-style masked calls stay exact).
   *
   * Buffer reuse: every intermediate tensor lives in per-model scratch
   * storage and is overwritten on each call, so a forward pass allocates
   * nothing and generation produces no GC churn. The returned array is
   * a view into that scratch storage: consume it or copy it before the
   * next forward() call on this model.
   */
  forward(ids: number[], validMask?: boolean[]): Float64Array {
    const T = ids.length;
    const D = this.dModel;
    const V = this.vocabSize;
    const valid: boolean[] | undefined = validMask;
    if (valid && valid.length !== T) throw new Error('validMask length must equal ids length');
    const s = this.ensureScratch(T);

    // h0 = wte[x] + wpe[:T] (float32 addition in NumPy -> Math.fround)
    const h0 = s.h0;
    for (let t = 0; t < T; t++) {
      const tokRow = ids[t] * D;
      for (let d = 0; d < D; d++) {
        h0[t * D + d] = Math.fround(this.wte[tokRow + d] + this.wpe[t * D + d]);
      }
    }

    matmulInto(s.Q, h0, T, D, this.Wq, D, D);
    matmulInto(s.K, h0, T, D, this.Wk, D, D);
    matmulInto(s.Vv, h0, T, D, this.Wv, D, D);

    // scores = Q @ K^T / sqrt(D), with causal + key-pad masking.
    const scores = s.scores;
    const scale = 1 / Math.sqrt(D);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        if (j > i || (valid && !valid[j])) {
          scores[i * T + j] = -1e9;
          continue;
        }
        let ssum = 0;
        for (let d = 0; d < D; d++) ssum += s.Q[i * D + d] * s.K[j * D + d];
        scores[i * T + j] = ssum * scale;
      }
    }
    softmaxRows(scores, T, T);

    matmulInto(s.attnOut, scores, T, T, s.Vv, T, D);
    matmulInto(s.attnProj, s.attnOut, T, D, this.Wo, D, D);
    const h1 = s.h1;
    for (let i = 0; i < T * D; i++) h1[i] = h0[i] + s.attnProj[i];

    matmulInto(s.pre1, h1, T, D, this.W1, D, this.dFf);
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < this.dFf; f++) {
        const idx = t * this.dFf + f;
        s.pre1[idx] = Math.max(0, s.pre1[idx] + this.b1[f]);
      }
    }
    matmulInto(s.ff2, s.pre1, T, this.dFf, this.W2, this.dFf, D);
    const h2 = s.h2;
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        const idx = t * D + d;
        h2[idx] = h1[idx] + s.ff2[idx] + this.b2[d];
      }
    }

    matmulInto(s.logits, h2, T, D, this.Whead, D, V);
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < V; v++) s.logits[t * V + v] += this.bhead[v];
    }
    softmaxRows(s.logits, T, V);
    return s.logits.subarray(0, T * V);
  }

  /** Probabilities at the last position for one sequence (generation). */
  forwardSingle(idsCond: number[]): Float64Array {
    const probs = this.forward(idsCond);
    const T = idsCond.length;
    return probs.slice((T - 1) * this.vocabSize, T * this.vocabSize);
  }
}

export interface LoadedModel {
  model: MiniGPT;
  tokenizer: BPETokenizer;
}

const PARAM_NAMES = [
  'wte',
  'wpe',
  'Wq',
  'Wk',
  'Wv',
  'Wo',
  'W1',
  'b1',
  'W2',
  'b2',
  'Whead',
  'bhead',
] as const;

export function loadModelFromNpz(bytes: Uint8Array): LoadedModel {
  const arrays = parseNpz(bytes);
  if (!arrays.has('merge_a') || !arrays.has('merge_b') || !arrays.has('merge_id')) {
    throw new Error('Not a TAIT model file (missing BPE merge data)');
  }
  for (const p of PARAM_NAMES) {
    if (!arrays.has(p)) throw new Error(`Not a TAIT model file (missing weights: ${p})`);
  }
  const tok = new BPETokenizer();
  tok.addMerges(
    asInt32(arrays.get('merge_a'), 'merge_a'),
    asInt32(arrays.get('merge_b'), 'merge_b'),
    asInt32(arrays.get('merge_id'), 'merge_id'),
  );
  tok.sepId = scalarInt(arrays.get('sep_id'), 'sep_id');
  tok.endId = scalarInt(arrays.get('end_id'), 'end_id');
  tok.padId = scalarInt(arrays.get('tok_pad_id'), 'tok_pad_id');

  const wte = arrays.get('wte')!;
  if (wte.shape.length !== 2) throw new Error('wte must be 2-D');
  const vocabSize = wte.shape[0];
  const dModel = wte.shape[1];
  const W1 = arrays.get('W1')!;
  if (W1.shape.length !== 2) throw new Error('W1 must be 2-D');
  const dFf = W1.shape[1];
  const blockSize = scalarInt(arrays.get('block_size'), 'block_size');
  const padId = scalarInt(arrays.get('pad_id'), 'pad_id');

  const weights: Record<string, Float64Array> = {};
  for (const p of PARAM_NAMES) weights[p] = asFloat64(arrays.get(p), p);
  const model = new MiniGPT({ vocabSize, dModel, dFf, blockSize, padId }, weights);
  return { model, tokenizer: tok };
}

export async function loadModelFromUrl(
  url: string,
  fetchFn: typeof fetch = fetch,
): Promise<LoadedModel> {
  const res = await fetchFn(url);
  if (!res.ok) {
    throw new Error(`Could not download the model (HTTP ${res.status})`);
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  return loadModelFromNpz(buf);
}
