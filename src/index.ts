/**
 * JAIT — run TAIT-trained .npz models in the browser.
 *
 * One .npz, three runtimes: TAIT trains (Python/NumPy, phone),
 * KAIT runs on the phone (Kotlin), JAIT runs in the browser (TypeScript).
 * No server, no account, no data leaves the device: the model is
 * downloaded straight from Hugging Face (or opened from the user's own
 * file) and executed locally.
 */
export { parseNpz, parseNpy } from './npz.js';
export { BPETokenizer, SEP_CHAR, END_CHAR } from './tokenizer.js';
export {
  MiniGPT,
  matmul,
  matmulInto,
  loadModelFromNpz,
  loadModelFromUrl,
} from './model.js';
export type { LoadedModel, ModelMeta } from './model.js';
export {
  generate,
  generateGreedy,
  generateReply,
  extractReply,
  sampleNext,
  DEFAULT_GENERATE_OPTIONS,
} from './generate.js';
export type { GenerateOptions } from './generate.js';
export { parseTaitMeta, isTaitMeta } from './meta.js';
export type { TaitMeta } from './meta.js';
