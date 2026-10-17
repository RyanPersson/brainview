// Fetching + decoding of the binaries written by build/build.py.

const TYPES = {
  uint8: Uint8Array, uint16: Uint16Array, int16: Int16Array,
  uint32: Uint32Array, int32: Int32Array, float32: Float32Array,
};

const cache = new Map();

// Data files are stored gzipped (*.bin.gz) and inflated here, so any static host works
// (GitHub Pages included).  A host that already decoded them (Content-Encoding: gzip)
// hands back exactly the expected byte count, which is then used as-is.
async function fetchBytes(url, expectedBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  if (buf.byteLength === expectedBytes) return buf;
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

export function fetchArray(desc) {
  if (!cache.has(desc.url)) {
    const T = TYPES[desc.dtype];
    cache.set(desc.url, fetchBytes(desc.url, desc.length * T.BYTES_PER_ELEMENT).then((buf) => new T(buf)));
  }
  return cache.get(desc.url);
}

// uint16-quantized Nx3 -> Float32Array (x = min + q * scale)
export async function fetchPositions(desc) {
  const q = await fetchArray(desc);
  const out = new Float32Array(q.length);
  const [m0, m1, m2] = desc.min, [s0, s1, s2] = desc.scale;
  for (let i = 0; i < q.length; i += 3) {
    out[i] = m0 + q[i] * s0;
    out[i + 1] = m1 + q[i + 1] * s1;
    out[i + 2] = m2 + q[i + 2] * s2;
  }
  return out;
}

export async function fetchManifest() {
  const r = await fetch('data/manifest.json');
  return r.json();
}

// Area-weighted vertex normals for an indexed triangle mesh.
export function vertexNormals(pos, index, out = new Float32Array(pos.length)) {
  out.fill(0);
  for (let f = 0; f < index.length; f += 3) {
    const a = index[f] * 3, b = index[f + 1] * 3, c = index[f + 2] * 3;
    const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
    const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    out[a] += nx; out[a + 1] += ny; out[a + 2] += nz;
    out[b] += nx; out[b + 1] += ny; out[b + 2] += nz;
    out[c] += nx; out[c + 1] += ny; out[c + 2] += nz;
  }
  for (let i = 0; i < out.length; i += 3) {
    const l = Math.hypot(out[i], out[i + 1], out[i + 2]) || 1;
    out[i] /= l; out[i + 1] /= l; out[i + 2] /= l;
  }
  return out;
}

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1, 7), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
