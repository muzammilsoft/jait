/**
 * Text generation — a port of TAIT's MiniGPT.generate
 * (tait/core/model.py) plus generate_reply (tait/core/generation.py).
 *
 * Sampling order copied exactly from Python:
 *   1. probs[last], pad set to 0, renormalise
 *   2. temperature: p = probs ** (1 / max(T, 1e-6)), renormalise
 *   3. top-k: keep the k largest p, zero the rest, renormalise
 *   4. top-p: sort descending, cumulative sum, keep up to the first
 *      entry whose cumulative sum reaches top_p (inclusive), renormalise
 *   5. draw one token from p
 * Stops on the END token or after maxNewTokens.
 *
 * One honest difference: Python draws with numpy's PCG64 RNG, which has
 * no equivalent in JavaScript. JAIT uses mulberry32, a small seeded RNG,
 * so the same seed gives a DIFFERENT (but still seeded and repeatable)
 * sample in JS than in Python. Deterministic comparison between the two
 * must use top-5 probabilities or greedy decoding, never a sampled seed.
 */

import { SEP_CHAR, END_CHAR, BPETokenizer } from './tokenizer.js';
import { MiniGPT } from './model.js';

export interface GenerateOptions {
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  /** 0 or undefined = unseeded (Math.random). Any other value seeds mulberry32. */
  seed?: number;
  /** Called with each new token id and the full decoded reply so far. */
  onToken?: (id: number, replySoFar: string) => void;
  /** AbortSignal to cancel generation mid-stream. */
  signal?: AbortSignal;
}

export const DEFAULT_GENERATE_OPTIONS: Required<
  Pick<GenerateOptions, 'maxNewTokens' | 'temperature' | 'topK' | 'topP'>
> = {
  maxNewTokens: 64,
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function applyTopK(p: Float64Array, topK: number): void {
  if (!topK || topK >= p.length) return;
  const order = Array.from(p.keys()).sort((a, b) => {
    const d = p[b] - p[a];
    return d !== 0 ? d : a - b;
  });
  const keep = new Set(order.slice(0, topK));
  for (let i = 0; i < p.length; i++) if (!keep.has(i)) p[i] = 0;
}

function applyTopP(p: Float64Array, topP: number): void {
  if (!topP || topP >= 1) return;
  const order = Array.from(p.keys()).sort((a, b) => {
    const d = p[b] - p[a];
    return d !== 0 ? d : a - b;
  });
  let cum = 0;
  let cutoff = order.length;
  for (let i = 0; i < order.length; i++) {
    cum += p[order[i]];
    if (cum >= topP) {
      cutoff = i + 1;
      break;
    }
  }
  const keep = new Set(order.slice(0, cutoff));
  for (let i = 0; i < p.length; i++) if (!keep.has(i)) p[i] = 0;
}

function renormalise(p: Float64Array): void {
  let sum = 0;
  for (let i = 0; i < p.length; i++) sum += p[i];
  if (sum <= 0 || !Number.isFinite(sum)) {
    // Should not happen (softmax always has a positive max), but never
    // divide by zero: fall back to a one-hot on the argmax.
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    p.fill(0);
    p[best] = 1;
    return;
  }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
}

function sampleFrom(p: Float64Array, rng: () => number): number {
  const r = rng();
  let cum = 0;
  for (let i = 0; i < p.length; i++) {
    cum += p[i];
    if (r < cum) return i;
  }
  return p.length - 1;
}

/** Sample one next-token distribution exactly as Python does. */
export function sampleNext(
  probs: Float64Array,
  padId: number,
  opts: { temperature: number; topK: number; topP: number },
  rng: () => number,
): number {
  const p = Float64Array.from(probs);
  p[padId] = 0;
  renormalise(p);
  const temp = Math.max(opts.temperature, 1e-6);
  for (let i = 0; i < p.length; i++) p[i] = Math.pow(p[i], 1 / temp);
  renormalise(p);
  applyTopK(p, opts.topK);
  renormalise(p);
  applyTopP(p, opts.topP);
  renormalise(p);
  return sampleFrom(p, rng);
}

/** Full generation: returns the decoded text including prompt + SEP + reply + END. */
export function generate(
  model: MiniGPT,
  tokenizer: BPETokenizer,
  prompt: string,
  options: GenerateOptions = {},
): string {
  const maxNew = options.maxNewTokens ?? DEFAULT_GENERATE_OPTIONS.maxNewTokens;
  const temperature = options.temperature ?? DEFAULT_GENERATE_OPTIONS.temperature;
  const topK = options.topK ?? DEFAULT_GENERATE_OPTIONS.topK;
  const topP = options.topP ?? DEFAULT_GENERATE_OPTIONS.topP;
  const rng =
    options.seed && options.seed !== 0 ? mulberry32(options.seed) : Math.random;

  const ids = tokenizer.encode(prompt + SEP_CHAR);
  for (let step = 0; step < maxNew; step++) {
    if (options.signal?.aborted) break;
    const cond = ids.slice(-model.blockSize);
    const probs = model.forwardSingle(cond);
    const nextId = sampleNext(probs, model.padId, { temperature, topK, topP }, rng);
    ids.push(nextId);
    if (options.onToken) {
      options.onToken(nextId, extractReply(tokenizer.decode(ids)));
    }
    if (nextId === tokenizer.endId) break;
  }
  return tokenizer.decode(ids);
}

/** Deterministic greedy decoding (argmax each step) — used for verification. */
export function generateGreedy(
  model: MiniGPT,
  tokenizer: BPETokenizer,
  prompt: string,
  maxNewTokens = 20,
): string {
  const ids = tokenizer.encode(prompt + SEP_CHAR);
  for (let step = 0; step < maxNewTokens; step++) {
    const cond = ids.slice(-model.blockSize);
    const probs = model.forwardSingle(cond);
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
      if (i !== model.padId && probs[i] > probs[best]) best = i;
    }
    ids.push(best);
    if (best === tokenizer.endId) break;
  }
  return tokenizer.decode(ids);
}

/** Same as TAIT's generate_reply: text after SEP, cut at END, trimmed. */
export function extractReply(full: string): string {
  let out = full;
  if (out.includes(SEP_CHAR)) out = out.split(SEP_CHAR, 2)[1];
  if (out.includes(END_CHAR)) out = out.split(END_CHAR, 2)[0];
  return out.trim();
}

export function generateReply(
  model: MiniGPT,
  tokenizer: BPETokenizer,
  prompt: string,
  options: GenerateOptions = {},
): string {
  return extractReply(generate(model, tokenizer, prompt, options));
}
