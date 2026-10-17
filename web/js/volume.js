import * as THREE from 'three';
import { fetchArray, hexToRgb } from './data.js';

// ---------------------------------------------------------------------------- textures

export class VolumeStore {
  constructor(desc) {
    this.desc = desc;
    this.dims = desc.dims;
    this.origin = desc.origin;
    this.cache = new Map();
  }

  scalar(id) {
    const key = 's:' + id;
    if (!this.cache.has(key)) {
      const meta = this.desc.scalars.find((v) => v.id === id);
      this.cache.set(key, fetchArray(meta.data).then((data) => ({ meta, data, tex: this.tex3d(data, THREE.LinearFilter) })));
    }
    return this.cache.get(key);
  }

  labels(id) {
    const key = 'l:' + id;
    if (!this.cache.has(key)) {
      const meta = this.desc.labels.find((v) => v.id === id);
      this.cache.set(key, fetchArray(meta.data).then((data) => {
        const lut = new Uint8Array(256 * 4);
        meta.colors.forEach((c, i) => {
          if (i > 255) return;
          const [r, g, b] = hexToRgb(c);
          lut.set([r * 255, g * 255, b * 255, i === 0 ? 0 : 255], i * 4);
        });
        const lutTex = new THREE.DataTexture(lut, 256, 1, THREE.RGBAFormat);
        lutTex.needsUpdate = true;
        return { meta, data, tex: this.tex3d(data, THREE.NearestFilter), lut: lutTex };
      }));
    }
    return this.cache.get(key);
  }

  tex3d(data, filter) {
    const [w, h, d] = this.dims;
    const t = new THREE.Data3DTexture(data, w, h, d);
    t.format = THREE.RedFormat;
    t.type = THREE.UnsignedByteType;
    t.minFilter = t.magFilter = filter;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    t.unpackAlignment = 1;
    t.needsUpdate = true;
    return t;
  }

  // voxel value at an MNI position (nearest), or null outside the grid
  sample(data, p) {
    const [w, h, d] = this.dims;
    const i = Math.round(p[0] - this.origin[0]), j = Math.round(p[1] - this.origin[1]), k = Math.round(p[2] - this.origin[2]);
    if (i < 0 || j < 0 || k < 0 || i >= w || j >= h || k >= d) return null;
    return data[i + w * (j + h * k)];
  }

  get bounds() {
    const lo = this.origin.map((o) => o - 0.5);
    const hi = this.origin.map((o, i) => o + this.dims[i] - 0.5);
    return { lo, hi };
  }
}

const dummy3D = (() => {
  const t = new THREE.Data3DTexture(new Uint8Array(1), 1, 1, 1);
  t.format = THREE.RedFormat;
  t.needsUpdate = true;
  return t;
})();
const dummyLut = new THREE.DataTexture(new Uint8Array(4), 1, 1, THREE.RGBAFormat);
dummyLut.needsUpdate = true;

// ---------------------------------------------------------------------------- orthogonal slices

const sliceVS = /* glsl */`
  uniform vec3 uOffset;
  varying vec3 vP;
  void main() {
    vP = position + uOffset;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const sliceFS = /* glsl */`
  precision highp sampler3D;
  uniform sampler3D uVol;
  uniform sampler3D uLab;
  uniform sampler2D uLut;
  uniform bool uHasVol;
  uniform bool uHasLab;
  uniform vec3 uOrigin;
  uniform vec3 uDims;
  uniform float uLo;
  uniform float uHi;
  uniform float uLabAlpha;
  uniform float uOpacity;
  varying vec3 vP;
  void main() {
    vec3 tc = (vP - uOrigin + 0.5) / uDims;
    if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) discard;
    float v = uHasVol ? texture(uVol, tc).r : 0.0;
    float g = clamp((v - uLo) / max(uHi - uLo, 1e-3), 0.0, 1.0);
    vec3 c = vec3(g);
    float idx = 0.0;
    if (uHasLab) {
      idx = floor(texture(uLab, tc).r * 255.0 + 0.5);
      if (idx > 0.5) {
        vec3 lc = texture(uLut, vec2((idx + 0.5) / 256.0, 0.5)).rgb;
        c = uHasVol ? mix(c, lc, uLabAlpha) : lc * (0.35 + 0.65 * uLabAlpha);
      }
    }
    if (v < 0.012 && idx < 0.5) discard;
    gl_FragColor = vec4(c, uOpacity);
  }`;

export class Slices {
  constructor(store, parent) {
    this.store = store;
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    const { lo, hi } = store.bounds;
    const c = lo.map((l, i) => (l + hi[i]) / 2);
    const size = hi.map((h, i) => h - lo[i]);
    this.uniforms = {
      uVol: { value: dummy3D }, uLab: { value: dummy3D }, uLut: { value: dummyLut },
      uHasVol: { value: false }, uHasLab: { value: false },
      uOrigin: { value: new THREE.Vector3(...store.origin) }, uDims: { value: new THREE.Vector3(...store.dims) },
      uLo: { value: 0 }, uHi: { value: 1 }, uLabAlpha: { value: 0.6 }, uOpacity: { value: 1 },
    };
    this.planes = {};
    const mk = (axis, geo) => {
      const mat = new THREE.ShaderMaterial({
        uniforms: { ...this.uniforms, uOffset: { value: new THREE.Vector3() } },
        vertexShader: sliceVS, fragmentShader: sliceFS, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData = { kind: 'slice', axis };
      this.group.add(mesh);
      this.planes[axis] = mesh;
    };
    // x = sagittal, y = coronal, z = axial; each plane geometry is built at 0 along its axis
    mk('x', new THREE.PlaneGeometry(size[2], size[1]).rotateY(Math.PI / 2).translate(0, c[1], c[2]));
    mk('y', new THREE.PlaneGeometry(size[0], size[2]).rotateX(Math.PI / 2).translate(c[0], 0, c[2]));
    mk('z', new THREE.PlaneGeometry(size[0], size[1]).translate(c[0], c[1], 0));
    this.pos = { x: 0, y: -18, z: 8 };
    for (const a of 'xyz') this.setPosition(a, this.pos[a]);
  }

  setVisible(v) { this.group.visible = v; }

  setPlaneVisible(axis, v) { this.planes[axis].visible = v; }

  setPosition(axis, mm) {
    this.pos[axis] = mm;
    const m = this.planes[axis];
    m.position.set(0, 0, 0);
    m.position[axis] = mm;
    m.material.uniforms.uOffset.value.copy(m.position);
  }

  async setImage(id) {
    this.imageId = id;
    if (!id) { this.uniforms.uHasVol.value = false; this.image = null; return; }
    const v = await this.store.scalar(id);
    if (this.imageId !== id) return;
    this.image = v;
    this.uniforms.uVol.value = v.tex;
    this.uniforms.uHasVol.value = true;
  }

  async setLabels(id) {
    this.labelId = id;
    if (!id) { this.uniforms.uHasLab.value = false; this.labels = null; return; }
    const l = await this.store.labels(id);
    if (this.labelId !== id) return;
    this.labels = l;
    this.uniforms.uLab.value = l.tex;
    this.uniforms.uLut.value = l.lut;
    this.uniforms.uHasLab.value = true;
  }

  setWindow(lo, hi) { this.uniforms.uLo.value = lo; this.uniforms.uHi.value = hi; }

  setLabelAlpha(a) { this.uniforms.uLabAlpha.value = a; }

  pickables() { return this.group.visible ? Object.values(this.planes).filter((m) => m.visible) : []; }

  describe(p) {
    const lines = [];
    let title = 'Slice';
    if (this.labels) {
      const k = this.store.sample(this.labels.data, p);
      if (k) title = this.labels.meta.names[k] || title;
      lines.push(this.labels.meta.label);
    }
    if (this.image) {
      const v = this.store.sample(this.image.data, p);
      if (v !== null) {
        const m = this.image.meta;
        const val = m.min + (v / 255) * (m.max - m.min);
        lines.push(`${m.label}: ${m.kind === 'prob' ? val.toFixed(2) : Math.round(val)}`);
        if (title === 'Slice') title = m.label;
      }
    }
    return { title, lines };
  }
}

// ---------------------------------------------------------------------------- ray-marched volume

const rayVS = /* glsl */`
  varying vec3 vP;
  void main() {
    vP = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const rayFS = /* glsl */`
  precision highp float;
  precision highp sampler3D;
  uniform sampler3D uVol;
  uniform sampler3D uLab;
  uniform sampler2D uLut;
  uniform bool uUseLab;
  uniform vec3 uOrigin;
  uniform vec3 uDims;
  uniform vec3 uCam;
  uniform mat4 uMVP;
  uniform int uMode;          // 0 iso-surface, 1 translucent, 2 maximum intensity
  uniform float uThreshold;
  uniform float uDensity;
  uniform float uStep;
  uniform vec3 uIsoColor;
  uniform bool uClipOn;
  uniform vec4 uClip;         // keep points with dot(n, p) + w >= 0
  varying vec3 vP;

  float sampleV(vec3 p) { return texture(uVol, (p - uOrigin + 0.5) / uDims).r; }
  vec3 labColor(vec3 p, vec3 fallback) {
    if (!uUseLab) return fallback;
    float idx = floor(texture(uLab, (p - uOrigin + 0.5) / uDims).r * 255.0 + 0.5);
    return idx > 0.5 ? texture(uLut, vec2((idx + 0.5) / 256.0, 0.5)).rgb : fallback;
  }
  float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float depthOf(vec3 p) {
    vec4 c = uMVP * vec4(p, 1.0);
    return clamp(c.z / c.w * 0.5 + 0.5, 0.0, 1.0);
  }

  void main() {
    vec3 ro = uCam;
    vec3 rd = normalize(vP - uCam);
    vec3 bmin = uOrigin - 0.5, bmax = uOrigin + uDims - 0.5;
    vec3 inv = 1.0 / rd;
    vec3 t0 = (bmin - ro) * inv, t1 = (bmax - ro) * inv;
    vec3 tlo = min(t0, t1), thi = max(t0, t1);
    float tn = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
    float tf = min(min(thi.x, thi.y), thi.z);
    bool clipEntry = false;
    if (uClipOn) {
      float dn = dot(uClip.xyz, rd);
      float d0 = dot(uClip.xyz, ro) + uClip.w;
      if (abs(dn) < 1e-6) { if (d0 < 0.0) discard; }
      else {
        float tc = -d0 / dn;
        if (dn > 0.0) { if (tc > tn) { tn = tc; clipEntry = true; } }
        else tf = min(tf, tc);
      }
    }
    if (tn >= tf) discard;
    float t = tn + uStep * hash(gl_FragCoord.xy);
    float firstT = t;

    if (uMode == 0) {
      for (int i = 0; i < 1400; i++) {
        if (t > tf) break;
        vec3 p = ro + rd * t;
        if (sampleV(p) >= uThreshold) {
          // refine the crossing by bisection
          float a = max(t - uStep, tn), b = t;
          for (int k = 0; k < 5; k++) {
            float m = 0.5 * (a + b);
            if (sampleV(ro + rd * m) >= uThreshold) b = m; else a = m;
          }
          bool onCut = clipEntry && t <= firstT + 1e-3;
          p = ro + rd * (onCut ? tn : b);
          vec3 n;
          if (onCut) n = uClip.xyz;
          else {
            // two-scale central differences: the wide stencil hides the trilinear voxel facets
            vec3 g = vec3(0.0);
            for (int k = 1; k <= 2; k++) {
              vec3 e = vec3(float(k) * 0.9, 0.0, 0.0);
              g += vec3(sampleV(p + e.xyy) - sampleV(p - e.xyy),
                        sampleV(p + e.yxy) - sampleV(p - e.yxy),
                        sampleV(p + e.yyx) - sampleV(p - e.yyx));
            }
            n = -normalize(g + 1e-6);
          }
          if (dot(n, rd) > 0.0) n = -n;
          vec3 L = normalize(-rd + vec3(0.0, 0.0, 0.25));
          float diff = max(dot(n, L), 0.0);
          float spec = pow(max(dot(n, normalize(L - rd)), 0.0), 30.0);
          // look up the label just inside the surface, not on the tissue/background boundary
          vec3 base = labColor(onCut ? p + rd * 0.5 : p + rd * 1.5, uIsoColor);
          if (onCut) base *= 0.92;
          gl_FragColor = vec4(base * (0.25 + 0.8 * diff) + 0.12 * spec, 1.0);
          gl_FragDepth = depthOf(p);
          return;
        }
        t += uStep;
      }
      discard;
    }

    if (uMode == 1) {
      vec4 acc = vec4(0.0);
      for (int i = 0; i < 1400; i++) {
        if (t > tf || acc.a > 0.98) break;
        vec3 p = ro + rd * t;
        float v = sampleV(p);
        float a = clamp((v - uThreshold) / max(1.0 - uThreshold, 1e-3), 0.0, 1.0);
        if (a > 0.0) {
          float alpha = 1.0 - exp(-a * a * uDensity * uStep);
          vec3 c = labColor(p, vec3(0.25 + 0.85 * a));
          acc.rgb += (1.0 - acc.a) * alpha * c;
          acc.a += (1.0 - acc.a) * alpha;
        }
        t += uStep;
      }
      if (acc.a < 0.004) discard;
      gl_FragColor = vec4(acc.rgb / acc.a, acc.a);
      gl_FragDepth = gl_FragCoord.z;
      return;
    }

    float m = 0.0;
    vec3 mp = ro + rd * t;
    for (int i = 0; i < 1400; i++) {
      if (t > tf) break;
      vec3 p = ro + rd * t;
      float v = sampleV(p);
      if (v > m) { m = v; mp = p; }
      t += uStep;
    }
    float g = clamp((m - uThreshold) / max(1.0 - uThreshold, 1e-3), 0.0, 1.0);
    if (g <= 0.0) discard;
    gl_FragColor = vec4(labColor(mp, vec3(1.0)) * g, g);
    gl_FragDepth = gl_FragCoord.z;
  }`;

export class VolumeRender {
  constructor(store, parent) {
    this.store = store;
    const { lo, hi } = store.bounds;
    const size = hi.map((h, i) => h - lo[i]);
    const geo = new THREE.BoxGeometry(...size).translate(...lo.map((l, i) => l + size[i] / 2));
    this.uniforms = {
      uVol: { value: dummy3D }, uLab: { value: dummy3D }, uLut: { value: dummyLut }, uUseLab: { value: false },
      uOrigin: { value: new THREE.Vector3(...store.origin) }, uDims: { value: new THREE.Vector3(...store.dims) },
      uCam: { value: new THREE.Vector3() }, uMVP: { value: new THREE.Matrix4() },
      uMode: { value: 0 }, uThreshold: { value: 0.3 }, uDensity: { value: 0.25 }, uStep: { value: 0.6 },
      uIsoColor: { value: new THREE.Vector3(0.86, 0.80, 0.74) },
      uClipOn: { value: false }, uClip: { value: new THREE.Vector4(1, 0, 0, 0) },
    };
    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: rayVS, fragmentShader: rayFS,
      side: THREE.BackSide, transparent: true, depthWrite: true,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.visible = false;
    this.mesh.renderOrder = 20;
    this.mesh.userData = { kind: 'volume' };
    const camLocal = new THREE.Vector3();
    this.mesh.onBeforeRender = (renderer, scene, camera) => {
      camLocal.copy(camera.position);
      this.mesh.worldToLocal(camLocal);
      this.uniforms.uCam.value.copy(camLocal);
      this.uniforms.uMVP.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(this.mesh.matrixWorld);
    };
    parent.add(this.mesh);
  }

  setVisible(v) { this.mesh.visible = v; }

  async setImage(id) {
    this.imageId = id;
    const v = await this.store.scalar(id);
    if (this.imageId !== id) return;
    this.image = v;
    this.uniforms.uVol.value = v.tex;
  }

  async setLabels(id) {
    this.labelId = id;
    if (!id) { this.uniforms.uUseLab.value = false; return; }
    const l = await this.store.labels(id);
    if (this.labelId !== id) return;
    this.uniforms.uLab.value = l.tex;
    this.uniforms.uLut.value = l.lut;
    this.uniforms.uUseLab.value = true;
  }

  setMode(mode) {
    this.uniforms.uMode.value = mode;
    // only the iso-surface is opaque and writes a real depth
    this.material.depthWrite = mode === 0;
    this.material.transparent = mode !== 0;
    this.material.needsUpdate = true;
  }

  setClip(plane) {
    this.uniforms.uClipOn.value = !!plane;
    if (plane) this.uniforms.uClip.value.set(plane.normal.x, plane.normal.y, plane.normal.z, plane.constant);
  }
}
