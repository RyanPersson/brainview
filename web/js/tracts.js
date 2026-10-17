import * as THREE from 'three';
import { fetchArray, hexToRgb } from './data.js';
import { clipPlanes } from './materials.js';

// Streamlines drawn as camera-facing ribbons shaded like tubes (two vertices per point; the
// vertex shader offsets them sideways in view space, the fragment shader fakes a cylinder normal).
const vertexShader = /* glsl */`
  #include <common>
  #include <clipping_planes_pars_vertex>
  attribute vec3 dir;
  uniform float uWidth;
  uniform int uMode;
  uniform vec3 uColor;
  varying vec3 vColor;
  varying float vSide;
  varying vec3 vSideDir;
  varying vec3 vView;
  varying vec3 vTan;
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vec3 t = normalize(mat3(modelViewMatrix) * dir);
    vec3 v = normalize(-mvPosition.xyz);
    vec3 s = cross(t, v);
    float sl = length(s);
    s = sl > 1e-4 ? s / sl : vec3(1.0, 0.0, 0.0);
    float side = (gl_VertexID % 2 == 0) ? -1.0 : 1.0;
    mvPosition.xyz += s * side * uWidth * 0.5;
    vSide = side;
    vSideDir = s;
    vView = v;
    vTan = t;
    vColor = uMode == 0 ? abs(dir) : uColor;
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
  }`;

const fragmentShader = /* glsl */`
  #include <common>
  #include <clipping_planes_pars_fragment>
  uniform int uMode;
  uniform float uOpacity;
  uniform float uHighlight;
  varying vec3 vColor;
  varying float vSide;
  varying vec3 vSideDir;
  varying vec3 vView;
  varying vec3 vTan;
  void main() {
    #include <clipping_planes_fragment>
    float u = clamp(vSide, -1.0, 1.0);
    vec3 n = normalize(vSideDir * u + vView * sqrt(max(0.0, 1.0 - u * u)));
    vec3 L = normalize(vec3(0.35, 0.6, 1.0));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, normalize(L + vView)), 0.0), 36.0);
    vec3 c = vColor * (0.28 + 0.8 * diff) + 0.22 * spec;
    if (uMode == 2) {
      // gold leaf: a dark metal whose highlight runs along each fibre (hair-style anisotropic sheen), with a
      // faint hue drift by viewing angle like the etched gold of a reflective microetching
      vec3 T = normalize(vTan);
      float th = dot(T, normalize(L + vView));
      float sheen = pow(max(0.0, 1.0 - th * th), 30.0);
      float tl = dot(T, L);
      float lit = sqrt(max(0.0, 1.0 - tl * tl)) * (0.3 + 0.7 * diff);
      vec3 film = 0.5 + 0.5 * cos(6.2832 * (0.5 * dot(T, vView) + vec3(0.0, 0.33, 0.67)));
      vec3 gold = vec3(0.72, 0.5, 0.22);
      c = gold * (0.035 + 0.2 * lit) + mix(vec3(1.0, 0.78, 0.42), film, 0.18) * sheen * (0.12 + 0.4 * diff);
    }
    c = mix(c, vec3(1.0), uHighlight * 0.45);
    gl_FragColor = vec4(c, uOpacity);
  }`;

export class Tracts {
  constructor(desc, parent) {
    this.desc = desc;
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    this.width = 0.5;
    this.density = 0.6;
    this.mode = 0; // 0 = direction, 1 = bundle colour, 2 = gold
    this.opacity = 1;
    this.catOn = Object.fromEntries(desc.categories.map((c) => [c, true]));
    this.items = desc.bundles.map((b) => ({ ...b, on: true, mesh: null }));
    this.base = new THREE.ShaderMaterial({
      uniforms: {
        uWidth: { value: this.width }, uMode: { value: 0 }, uColor: { value: new THREE.Vector3(1, 1, 1) },
        uOpacity: { value: 1 }, uHighlight: { value: 0 },
      },
      vertexShader, fragmentShader, clipping: true, side: THREE.DoubleSide,
    });
    this.loading = null;
  }

  load() {
    if (!this.loading) this.loading = this._load();
    return this.loading;
  }

  async _load() {
    const [q, counts] = await Promise.all([fetchArray(this.desc.positions), fetchArray(this.desc.counts)]);
    const sc = this.desc.posScale;
    const start = new Uint32Array(counts.length + 1);
    for (let i = 0; i < counts.length; i++) start[i + 1] = start[i] + counts[i];
    for (const b of this.items) {
      const s0 = b.first, s1 = b.first + b.count;
      const p0 = start[s0], np = start[s1] - p0;
      const pos = new Float32Array(np * 6), dir = new Float32Array(np * 6);
      const idx = new Uint32Array((np - b.count) * 6);
      b.idxEnd = new Uint32Array(b.count);
      b.ptEnd = new Uint32Array(b.count);
      let ii = 0;
      for (let s = s0; s < s1; s++) {
        const a = start[s] - p0, n = counts[s];
        for (let k = 0; k < n; k++) {
          const j = (a + k) * 3 + p0 * 3, o = (a + k) * 6;
          const x = q[j] * sc, y = q[j + 1] * sc, z = q[j + 2] * sc;
          pos[o] = pos[o + 3] = x; pos[o + 1] = pos[o + 4] = y; pos[o + 2] = pos[o + 5] = z;
          // tangent from neighbours
          const jp = (a + Math.max(0, k - 1)) * 3 + p0 * 3, jn = (a + Math.min(n - 1, k + 1)) * 3 + p0 * 3;
          let dx = q[jn] - q[jp], dy = q[jn + 1] - q[jp + 1], dz = q[jn + 2] - q[jp + 2];
          const l = Math.hypot(dx, dy, dz) || 1;
          dx /= l; dy /= l; dz /= l;
          dir[o] = dir[o + 3] = dx; dir[o + 1] = dir[o + 4] = dy; dir[o + 2] = dir[o + 5] = dz;
          if (k < n - 1) {
            const v = (a + k) * 2;
            idx[ii++] = v; idx[ii++] = v + 1; idx[ii++] = v + 2;
            idx[ii++] = v + 1; idx[ii++] = v + 3; idx[ii++] = v + 2;
          }
        }
        b.idxEnd[s - s0] = ii;
        b.ptEnd[s - s0] = a + n;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('dir', new THREE.BufferAttribute(dir, 3));
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
      geo.computeBoundingSphere();
      const m = this.base.clone();
      m.clippingPlanes = clipPlanes;
      m.uniforms.uColor.value.fromArray(hexToRgb(b.color));
      b.mesh = new THREE.Mesh(geo, m);
      b.mesh.userData = { kind: 'tract', item: b };
      this.group.add(b.mesh);
      // points for hover picking (every point of the drawn streamlines)
      b.points = pos;
    }
    this.apply();
  }

  setVisible(v) {
    this.group.visible = v;
    if (v) this.load();
  }

  apply() {
    for (const b of this.items) {
      if (!b.mesh) continue;
      b.mesh.visible = b.on && this.catOn[b.category];
      const k = Math.max(1, Math.round(b.count * this.density));
      b.mesh.geometry.setDrawRange(0, b.idxEnd[k - 1]);
      const u = b.mesh.material.uniforms;
      u.uWidth.value = this.width;
      u.uMode.value = this.mode;
      u.uOpacity.value = this.opacity;
      u.uHighlight.value = b === this.hovered ? 1 : 0;
      const tr = this.opacity < 0.999;
      if (b.mesh.material.transparent !== tr) {
        b.mesh.material.transparent = tr;
        b.mesh.material.depthWrite = !tr;
        b.mesh.material.needsUpdate = true;
      }
    }
  }

  setHover(b) {
    if (this.hovered === b) return;
    if (this.hovered?.mesh) this.hovered.mesh.material.uniforms.uHighlight.value = 0;
    this.hovered = b;
    if (b?.mesh) b.mesh.material.uniforms.uHighlight.value = 1;
  }

  // Nearest drawn streamline point to a ray given in brain (MNI) coordinates.
  // tolAt(t) = pick radius in mm at distance t along the ray; keep(x,y,z) rejects clipped points.
  pick(ray, tolAt, keep) {
    if (!this.group.visible || !this.loading) return null;
    const o = ray.origin, d = ray.direction;
    let best = null;
    for (const b of this.items) {
      if (!b.mesh || !b.mesh.visible) continue;
      const k = Math.max(1, Math.round(b.count * this.density));
      const lim = b.ptEnd[k - 1], P = b.points;
      for (let i = 0; i < lim; i++) {
        const x = P[i * 6] - o.x, y = P[i * 6 + 1] - o.y, z = P[i * 6 + 2] - o.z;
        const t = x * d.x + y * d.y + z * d.z;
        if (t <= 0 || (best && t >= best.t)) continue;
        const px = x - d.x * t, py = y - d.y * t, pz = z - d.z * t;
        const tol = tolAt(t) + this.width * 0.5;
        if (px * px + py * py + pz * pz < tol * tol && keep(P[i * 6], P[i * 6 + 1], P[i * 6 + 2])) {
          best = { t, item: b, point: [P[i * 6], P[i * 6 + 1], P[i * 6 + 2]] };
        }
      }
    }
    return best;
  }

  describe(b) {
    const side = b.hemi === 'L' ? 'Left ' : b.hemi === 'R' ? 'Right ' : '';
    const name = side && /^[A-Z][a-z]/.test(b.name) ? b.name[0].toLowerCase() + b.name.slice(1) : b.name;
    return {
      title: side + name,
      lines: [`${b.category} tract · ${b.abbr}`,
        `${Math.max(1, Math.round(b.count * this.density))} of ${b.total.toLocaleString()} atlas streamlines shown`],
      swatch: b.color,
    };
  }
}
