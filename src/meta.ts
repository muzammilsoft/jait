/**
 * tait-meta.json — the small metadata file TAIT writes next to a published
 * model (see tait publish). TAIT-Toy reads it to fill model cards and to
 * validate that a repo is really a TAIT model before offering "Run".
 */

export interface TaitMeta {
  taitVersion?: string;
  data?: string;
  examples?: number;
  epochs?: number;
  finalLoss?: number;
  valLoss?: number;
  dModel?: number;
  dFf?: number;
  vocabSize?: number;
  blockSize?: number;
  seed?: number;
  createdAt?: string;
}

export function parseTaitMeta(value: unknown): TaitMeta | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  const num = (k: string): number | undefined =>
    typeof v[k] === 'number' && Number.isFinite(v[k]) ? (v[k] as number) : undefined;
  const str = (k: string): string | undefined =>
    typeof v[k] === 'string' ? (v[k] as string) : undefined;
  return {
    taitVersion: str('tait_version'),
    data: str('data'),
    examples: num('examples'),
    epochs: num('epochs'),
    finalLoss: num('final_loss'),
    valLoss: num('val_loss'),
    dModel: num('d_model'),
    dFf: num('d_ff'),
    vocabSize: num('vocab_size'),
    blockSize: num('block_size'),
    seed: num('seed'),
    createdAt: str('created_at'),
  };
}

/** A repo counts as a TAIT model when its meta has the fields JAIT needs. */
export function isTaitMeta(meta: TaitMeta | null): meta is TaitMeta {
  return (
    meta !== null &&
    typeof meta.vocabSize === 'number' &&
    typeof meta.dModel === 'number' &&
    typeof meta.blockSize === 'number'
  );
}
