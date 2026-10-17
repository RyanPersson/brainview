import * as THREE from 'three';
import { fetchArray, fetchPositions, vertexNormals, hexToRgb, srgbToLinear } from './data.js';
import { makeLUT } from './colormaps.js';
import { makeSolidMaterial, makeXRayMaterial, setOpacity } from './materials.js';

const LIN = new Float32Array(1025);
for (let i = 0; i <= 1024; i++) LIN[i] = srgbToLinear(i / 1024);
const lin = (v) => LIN[Math.max(0, Math.min(1024, (v * 1024) | 0))];

const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// The fsaverage cortical surface (both hemispheres) with shape morphing and per-vertex overlays.
export class Cortex {
  constructor(man, parent) {
    this.man = man;
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    this.shapes = man.surfaces.map((s) => s.id);
    this.hemis = {};
    this.shape = 0;
    this.overlay = 'atlas:dk';
    this.foldShading = true;
    this.outlines = false;
    this.style = 'solid';
    this.opacity = 1;
    this.hemiOn = { L: true, R: true };
    this.highlight = null; // { hemi, label }
    this.atlasCache = new Map();
    this.scalarCache = new Map();
    this.boundaryCache = new Map();
    this.solidMat = makeSolidMaterial({ vertexColors: true, side: THREE.DoubleSide });
    this.xrayMat = makeXRayMaterial({ vertexColors: true, color: '#ffffff', opacity: 0.55, power: 1.6 });
    this.bvhDirty = true;
    this.loading = null;
  }

  // Loaded on first show; settings changed before then are applied once the meshes exist.
  load() {
    if (!this.loading) this.loading = this._load();
    return this.loading;
  }

  async _load() {
    const faces = await fetchArray(this.man.hemis.L.faces);
    const sulc = this.man.scalars.find((s) => s.id === this.man.shading);
    await Promise.all(['L', 'R'].map(async (h) => {
      const hd = this.man.hemis[h];
      const P = await Promise.all(this.shapes.map((s) => fetchPositions(hd[s])));
      const N = P.map((p) => vertexNormals(p, faces));
      const n = hd.vertices;
      const geo = new THREE.BufferGeometry();
      // own copy per hemisphere: building a bounds tree reorders the index buffer in place
      const index = h === 'L' ? faces : faces.slice();
      geo.setIndex(new THREE.BufferAttribute(index, 1));
      const pos = new THREE.BufferAttribute(new Float32Array(P[0]), 3).setUsage(THREE.DynamicDrawUsage);
      const nor = new THREE.BufferAttribute(new Float32Array(N[0]), 3).setUsage(THREE.DynamicDrawUsage);
      const col = new THREE.BufferAttribute(new Float32Array(n * 3).fill(0.6), 3);
      geo.setAttribute('position', pos);
      geo.setAttribute('normal', nor);
      geo.setAttribute('color', col);
      const mesh = new THREE.Mesh(geo, this.solidMat);
      mesh.frustumCulled = false;
      mesh.userData = { kind: 'cortex', hemi: h };
      mesh.visible = this.hemiOn[h];
      this.group.add(mesh);
      this.hemis[h] = { mesh, geo, P, N, n, faces: index, sulc: await fetchArray(sulc.files[h]) };
    }));
    if (this.shape !== 0) this.setShape(this.shape);
    this.setStyle(this.style);
    await this.setOverlay(this.overlay);
  }

  get meshes() { return Object.values(this.hemis).map((h) => h.mesh); }

  setVisible(v) {
    this.group.visible = v;
    if (v) this.load();
  }

  setHemiVisible(h, v) {
    this.hemiOn[h] = v;
    if (this.hemis[h]) this.hemis[h].mesh.visible = v;
  }

  // t in [0, shapes-1]: fractional values blend neighbouring surfaces.
  setShape(t) {
    this.shape = t;
    const last = this.shapes.length - 1;
    const i = Math.min(Math.floor(t), last - 1);
    const f = t - i;
    for (const H of Object.values(this.hemis)) {
      const pos = H.geo.attributes.position.array, nor = H.geo.attributes.normal.array;
      const A = H.P[i], B = H.P[i + 1], NA = H.N[i], NB = H.N[i + 1];
      for (let k = 0; k < pos.length; k += 3) {
        pos[k] = A[k] + (B[k] - A[k]) * f;
        pos[k + 1] = A[k + 1] + (B[k + 1] - A[k + 1]) * f;
        pos[k + 2] = A[k + 2] + (B[k + 2] - A[k + 2]) * f;
        const x = NA[k] + (NB[k] - NA[k]) * f, y = NA[k + 1] + (NB[k + 1] - NA[k + 1]) * f,
          z = NA[k + 2] + (NB[k + 2] - NA[k + 2]) * f;
        const l = Math.hypot(x, y, z) || 1;
        nor[k] = x / l; nor[k + 1] = y / l; nor[k + 2] = z / l;
      }
      H.geo.attributes.position.needsUpdate = true;
      H.geo.attributes.normal.needsUpdate = true;
    }
    this.bvhDirty = true;
  }

  setStyle(style) {
    this.style = style;
    for (const H of Object.values(this.hemis)) H.mesh.material = style === 'xray' ? this.xrayMat : this.solidMat;
    this.setOpacity(this.opacity);
  }

  setOpacity(o) {
    this.opacity = o;
    setOpacity(this.solidMat, o);
    this.xrayMat.uniforms.uOpacity.value = 0.9 * o;
    // let nested structures draw before a see-through cortex
    for (const H of Object.values(this.hemis)) H.mesh.renderOrder = (o < 1 || this.style === 'xray') ? 10 : 0;
  }

  async getAtlas(id) {
    if (!this.atlasCache.has(id)) {
      const a = this.man.atlases.find((x) => x.id === id);
      const p = Promise.all(['L', 'R'].map(async (h) => {
        const e = a.hemis[h];
        const labels = await fetchArray(e.labels);
        const pal = new Float32Array(e.colors.length * 3);
        e.colors.forEach((c, i) => pal.set(hexToRgb(c), i * 3));
        return [h, { labels, pal, names: e.names }];
      })).then((r) => ({ meta: a, ...Object.fromEntries(r) }));
      this.atlasCache.set(id, p);
    }
    return this.atlasCache.get(id);
  }

  async getScalar(id) {
    if (!this.scalarCache.has(id)) {
      const s = this.man.scalars.find((x) => x.id === id);
      const p = Promise.all(['L', 'R'].map(async (h) => [h, await fetchArray(s.files[h])]))
        .then((r) => ({ meta: s, lut: makeLUT(s.cmap), ...Object.fromEntries(r) }));
      this.scalarCache.set(id, p);
    }
    return this.scalarCache.get(id);
  }

  boundaries(atlasId, h, labels) {
    const key = atlasId + h;
    if (!this.boundaryCache.has(key)) {
      const b = new Uint8Array(labels.length);
      const f = this.hemis[h].faces;
      for (let i = 0; i < f.length; i += 3) {
        const a = f[i], c = f[i + 1], d = f[i + 2];
        const la = labels[a], lc = labels[c], ld = labels[d];
        if (la !== lc || la !== ld) { b[a] = 1; b[c] = 1; b[d] = 1; }
      }
      this.boundaryCache.set(key, b);
    }
    return this.boundaryCache.get(key);
  }

  async setOverlay(ov) {
    this.overlay = ov;
    const [kind, id] = ov.split(':');
    this.current = kind === 'atlas' ? await this.getAtlas(id) : kind === 'scalar' ? await this.getScalar(id) : null;
    if (this.overlay !== ov) return; // superseded while loading
    this.highlight = null;
    this.recolor();
  }

  setFoldShading(v) { this.foldShading = v; this.recolor(); }

  setOutlines(v) { this.outlines = v; this.recolor(); }

  setHighlight(hl) {
    const same = (a, b) => (a && b ? a.hemi === b.hemi && a.label === b.label : a === b);
    if (same(hl, this.highlight)) return;
    const prev = this.highlight;
    this.highlight = hl;
    const hs = new Set([prev?.hemi, hl?.hemi].filter(Boolean));
    for (const h of hs) this.recolorHemi(h);
  }

  recolor() { for (const h of Object.keys(this.hemis)) this.recolorHemi(h); }

  recolorHemi(h) {
    const H = this.hemis[h];
    const col = H.geo.attributes.color.array;
    const [kind, id] = this.overlay.split(':');
    const cur = this.current;
    const atlas = kind === 'atlas' ? cur?.[h] : null;
    const scal = kind === 'scalar' ? cur?.[h] : null;
    const lut = kind === 'scalar' ? cur?.lut : null;
    const bound = atlas && this.outlines ? this.boundaries(id, h, atlas.labels) : null;
    const hl = this.highlight && this.highlight.hemi === h && atlas ? this.highlight.label : -1;
    const sulc = H.sulc;
    const fold = this.foldShading;
    for (let i = 0; i < H.n; i++) {
      let r, g, b;
      if (atlas) {
        const L = atlas.labels[i] * 3;
        r = atlas.pal[L]; g = atlas.pal[L + 1]; b = atlas.pal[L + 2];
      } else if (scal) {
        const L = scal[i] * 3;
        r = lut[L]; g = lut[L + 1]; b = lut[L + 2];
      } else {
        r = g = b = 0.8;
      }
      if (fold) {
        const s = (sulc[i] / 255) * 2 - 1; // + = sulcal fundus, - = gyral crown
        const k = 1 - 0.38 * smoothstep(-0.3, 0.55, s);
        r *= k; g *= k; b *= k;
      }
      if (bound && bound[i]) { r *= 0.25; g *= 0.25; b *= 0.25; }
      if (hl >= 0 && atlas.labels[i] === hl) {
        r = r * 0.45 + 0.55; g = g * 0.45 + 0.55; b = b * 0.45 + 0.55;
      }
      col[i * 3] = lin(r); col[i * 3 + 1] = lin(g); col[i * 3 + 2] = lin(b);
    }
    H.geo.attributes.color.needsUpdate = true;
  }

  // Bounds trees are built lazily and refitted after a shape change.
  prepareRaycast() {
    for (const H of Object.values(this.hemis)) {
      if (!H.geo.boundsTree) H.geo.computeBoundsTree();
      else if (this.bvhDirty) H.geo.boundsTree.refit();
    }
    this.bvhDirty = false;
  }

  // Describe a raycast hit on one of the cortex meshes.
  describe(hit) {
    const h = hit.object.userData.hemi;
    const H = this.hemis[h];
    const pos = H.geo.attributes.position.array;
    const { a, b, c } = hit.face;
    const p = hit.object.worldToLocal(hit.point.clone());
    let best = a, bd = Infinity;
    for (const v of [a, b, c]) {
      const d = (pos[v * 3] - p.x) ** 2 + (pos[v * 3 + 1] - p.y) ** 2 + (pos[v * 3 + 2] - p.z) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    const [kind] = this.overlay.split(':');
    const hemiName = h === 'L' ? 'Left' : 'Right';
    // anatomical (pial) position of the vertex, whatever the current shape
    const P0 = H.P[0];
    const mni = [P0[best * 3], P0[best * 3 + 1], P0[best * 3 + 2]];
    const info = { title: `${hemiName} cortex`, lines: [], mni, vertex: best };
    if (kind === 'atlas' && this.current) {
      const lab = this.current[h].labels[best];
      info.title = this.current[h].names[lab];
      info.lines.push(`${hemiName} hemisphere · ${this.current.meta.label}`);
      info.highlight = { hemi: h, label: lab };
      info.swatch = this.current.meta.hemis[h].colors[lab];
    } else if (kind === 'scalar' && this.current) {
      const m = this.current.meta;
      const v = m.min + (this.current[h][best] / 255) * (m.max - m.min);
      info.title = `${m.label}: ${v.toFixed(2)}${m.id === 'thickness' ? ' mm' : ''}`;
      info.lines.push(`${hemiName} hemisphere cortex`);
    }
    info.lines.push(`Surface vertex ${best} (fsaverage)`);
    return info;
  }
}
