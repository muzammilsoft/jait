/**
 * Byte-level BPE tokenizer — a line-by-line port of TAIT's
 * tait/core/tokenizer.py (BPETokenizer).
 *
 * Rules copied exactly:
 * - vocab starts as 256 single-byte entries; each merge creates one new
 *   id whose bytes are the concatenation of its two parts.
 * - encode() splits the text on the SEP (\x01) and END (\x02) characters
 *   first, then BPE-encodes each chunk: repeatedly merge the pair with
 *   the LOWEST merge id until no mergeable pair remains.
 * - decode() flushes accumulated bytes on SEP/END (emitting the control
 *   character itself), skips PAD, and decodes the rest as UTF-8.
 */

export const SEP_CHAR = '\u0001';
export const END_CHAR = '\u0002';

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8');

export class BPETokenizer {
  /** "a,b" -> new id, where a and b are token ids. */
  merges = new Map<string, number>();
  /** vocab[id] = raw bytes of that token. */
  vocab: Uint8Array[] = [];
  sepId = -1;
  endId = -1;
  padId = -1;

  constructor() {
    for (let i = 0; i < 256; i++) this.vocab[i] = new Uint8Array([i]);
  }

  /** Add merges in merge-id order, rebuilding the vocab as we go. */
  addMerges(a: Int32Array, b: Int32Array, ids: Int32Array): void {
    const order = Array.from(ids.keys()).sort((x, y) => ids[x] - ids[y]);
    for (const i of order) {
      const ai = a[i];
      const bi = b[i];
      const nid = ids[i];
      if (!this.vocab[ai] || !this.vocab[bi]) {
        throw new Error(`BPE merge refers to unknown token id (${ai}, ${bi})`);
      }
      this.merges.set(`${ai},${bi}`, nid);
      const merged = new Uint8Array(
        this.vocab[ai].length + this.vocab[bi].length,
      );
      merged.set(this.vocab[ai], 0);
      merged.set(this.vocab[bi], this.vocab[ai].length);
      this.vocab[nid] = merged;
    }
  }

  get vocabSizeTotal(): number {
    return 256 + this.merges.size + 3;
  }

  private static mergeOnce(ids: number[], pair: [number, number], newId: number): number[] {
    const out: number[] = [];
    let i = 0;
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === pair[0] && ids[i + 1] === pair[1]) {
        out.push(newId);
        i += 2;
      } else {
        out.push(ids[i]);
        i += 1;
      }
    }
    return out;
  }

  private encodeChunk(bytes: number[]): number[] {
    let ids = [...bytes];
    while (ids.length >= 2) {
      const seen = new Set<string>();
      let bestPair: string | null = null;
      let bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const key = `${ids[i]},${ids[i + 1]}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const rank = this.merges.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = key;
        }
      }
      if (bestPair === null) break;
      const [a, b] = bestPair.split(',').map((s) => parseInt(s, 10));
      ids = BPETokenizer.mergeOnce(ids, [a, b], bestRank);
    }
    return ids;
  }

  encode(text: string): number[] {
    if (this.sepId === -1) throw new Error('Tokenizer is not loaded');
    const result: number[] = [];
    let buf = '';
    const flush = () => {
      if (buf.length > 0) {
        result.push(...this.encodeChunk([...textEncoder.encode(buf)]));
        buf = '';
      }
    };
    for (const ch of text) {
      if (ch === SEP_CHAR || ch === END_CHAR) {
        flush();
        result.push(ch === SEP_CHAR ? this.sepId : this.endId);
      } else {
        buf += ch;
      }
    }
    flush();
    return result;
  }

  decode(ids: number[]): string {
    if (this.sepId === -1) throw new Error('Tokenizer is not loaded');
    let out = new Uint8Array(0);
    const parts: string[] = [];
    const flush = () => {
      parts.push(textDecoder.decode(out));
      out = new Uint8Array(0);
    };
    for (const id of ids) {
      if (id === this.sepId) {
        flush();
        parts.push(SEP_CHAR);
      } else if (id === this.endId) {
        flush();
        parts.push(END_CHAR);
      } else if (id === this.padId) {
        continue;
      } else {
        const bytes = this.vocab[id];
        if (!bytes) continue;
        const merged = new Uint8Array(out.length + bytes.length);
        merged.set(out, 0);
        merged.set(bytes, out.length);
        out = merged;
      }
    }
    flush();
    return parts.join('');
  }
}
