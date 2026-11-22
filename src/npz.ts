/**
 * Minimal .npy / .npz reader for TAIT models.
 *
 * An .npz file is a ZIP archive of .npy files. TAIT writes it with
 * np.savez, so entries are usually STORED (no compression), but
 * savez_compressed files use DEFLATE — both are handled here via fflate.
 *
 * Only the dtypes TAIT actually writes are supported: <f8, <f4, <i4, <i8.
 * Fortran-order arrays are rejected (numpy never writes them for TAIT
 * checkpoints, which are always C-contiguous).
 */

import { unzipSync } from 'fflate';

export type NpyArray =
  | { kind: 'f64'; shape: number[]; data: Float64Array }
  | { kind: 'f32'; shape: number[]; data: Float32Array }
  | { kind: 'i32'; shape: number[]; data: Int32Array }
  | { kind: 'i64'; shape: number[]; data: BigInt64Array };

function countOf(shape: number[]): number {
  return shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1);
}

export function parseNpy(bytes: Uint8Array): NpyArray {
  if (
    bytes.length < 10 ||
    bytes[0] !== 0x93 ||
    bytes[1] !== 0x4e ||
    bytes[2] !== 0x55 ||
    bytes[3] !== 0x4d ||
    bytes[4] !== 0x50 ||
    bytes[5] !== 0x59
  ) {
    throw new Error('Not an .npy file (bad magic)');
  }
  const major = bytes[6];
  let headerLen: number;
  let dataStart: number;
  if (major === 1) {
    headerLen =
      bytes[8] | (bytes[9] << 8);
    dataStart = 10 + headerLen;
  } else if (major === 2 || major === 3) {
    headerLen =
      bytes[8] |
      (bytes[9] << 8) |
      (bytes[10] << 16) |
      (bytes[11] << 24);
    dataStart = 12 + headerLen;
  } else {
    throw new Error(`Unsupported .npy version ${major}`);
  }
  const header = new TextDecoder('latin1').decode(
    bytes.subarray(major === 1 ? 10 : 12, dataStart),
  );
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeRaw = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? '';
  if (!descr) throw new Error('Cannot parse .npy header (descr)');
  if (fortran === 'True') {
    throw new Error('Fortran-order .npy arrays are not supported');
  }
  const shape: number[] = shapeRaw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));
  const count = countOf(shape);
  const view = new DataView(bytes.buffer, bytes.byteOffset + dataStart);
  switch (descr) {
    case '<f8': {
      const data = new Float64Array(count);
      for (let i = 0; i < count; i++) data[i] = view.getFloat64(i * 8, true);
      return { kind: 'f64', shape, data };
    }
    case '<f4': {
      const data = new Float32Array(count);
      for (let i = 0; i < count; i++) data[i] = view.getFloat32(i * 4, true);
      return { kind: 'f32', shape, data };
    }
    case '<i4': {
      const data = new Int32Array(count);
      for (let i = 0; i < count; i++) data[i] = view.getInt32(i * 4, true);
      return { kind: 'i32', shape, data };
    }
    case '<i8': {
      const data = new BigInt64Array(count);
      for (let i = 0; i < count; i++) data[i] = view.getBigInt64(i * 8, true);
      return { kind: 'i64', shape, data };
    }
    default:
      throw new Error(`Unsupported .npy dtype ${descr}`);
  }
}

/** Parse a whole .npz buffer into a map from array name to parsed array. */
export function parseNpz(bytes: Uint8Array): Map<string, NpyArray> {
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(bytes);
  } catch (e) {
    throw new Error(
      `Not a valid .npz (ZIP) file: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  const out = new Map<string, NpyArray>();
  for (const [name, data] of Object.entries(files)) {
    if (!name.endsWith('.npy')) continue;
    out.set(name.slice(0, -4), parseNpy(data));
  }
  if (out.size === 0) throw new Error('Empty .npz: no .npy entries found');
  return out;
}

/** Read a 0-d integer array as a JS number (TAIT scalars: sep_id, ...). */
export function scalarInt(a: NpyArray | undefined, name: string): number {
  if (!a) throw new Error(`Missing required array: ${name}`);
  if (a.kind === 'i64') return Number(a.data[0]);
  if (a.kind === 'i32') return a.data[0];
  throw new Error(`Array ${name} is not an integer scalar`);
}

/** View any float array as Float64Array (float32 values convert exactly). */
export function asFloat64(a: NpyArray | undefined, name: string): Float64Array {
  if (!a) throw new Error(`Missing required array: ${name}`);
  if (a.kind === 'f64') return a.data;
  if (a.kind === 'f32') return Float64Array.from(a.data);
  throw new Error(`Array ${name} is not a float array`);
}

/** View any int array as Int32Array. */
export function asInt32(a: NpyArray | undefined, name: string): Int32Array {
  if (!a) throw new Error(`Missing required array: ${name}`);
  if (a.kind === 'i32') return a.data;
  if (a.kind === 'i64') return Int32Array.from(a.data, (v) => Number(v));
  throw new Error(`Array ${name} is not an integer array`);
}
