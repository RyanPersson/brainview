import * as THREE from 'three';
import { fetchArray } from './data.js';
import { clipPlanes } from './materials.js';

// Signals travelling the white-matter tracts. A small cascade model runs on the CPU: a node (a cortical area or
// a deep nucleus) fires, sends pulses down the streamlines that leave it in their allowed direction, and the
// pulses' arrivals fire the nodes at the far ends. Each streamline's pulses go into a float texture, and a glow
// pass over the tract ribbons draws every pulse as a comet: a sharp bright head with a fading tail behind it,
// so the direction of travel reads at a glance. Model time is in milliseconds.
//
// The measured mode replays the F-TRACT atlas instead of the model: an area is stimulated, pulses run out along
// the fibres to each area that responded, taking the measured axonal delay, and each area lights up at its
// measured response latency.

const ROW = 1024; // streamlines per texture row; 3 texels each: pulse slots 0-1, pulse slots 2-3, end flashes
const SLOTS = 4;
// A pulse slot is (start time, code); code packs the direction (0: from end A, 1: from end B, 2: out both ways
// from the mid point), strength (0-99) and speed (units of 0.02 mm/ms) as one integer, exact in a float32.
const VSTEP = 0.02;
const pack = (dir, amp, vel) => dir + 3 * Math.round(Math.min(1, amp) * 99)
  + 300 * Math.max(1, Math.min(999, Math.round(vel / VSTEP)));
// conduction velocity, mm per ms: 3.9 m/s, the median F-TRACT measured between cortical areas (Lemaréchal 2022)
export const VELOCITY = 3.9;
const TAIL = 14; // comet tail, mm
const FLASH_T = 6; // how long an end glows after its node fires, ms
const FLASH_LEN = 7; // how far the glow reaches in from the end, mm
const IMPACT_T = 2.5; // glow where a pulse lands, ms

const vertexShader = /* glsl */`
  #include <common>
  #include <clipping_planes_pars_vertex>
  attribute vec3 dir;
  attribute vec4 flow; // arc length (mm), streamline length (mm), streamline index, where mid-out pulses start (mm)
  uniform sampler2D uState;
  uniform float uTime;
  uniform float uWidth;
  varying float vSide;
  varying float vArc;
  flat varying vec4 vP0;
  flat varying vec4 vP1;
  flat varying vec4 vF;
  flat varying vec2 vLen;
  float ends(float t0, float code, float len) {
    float vel = max(floor(floor(code + 0.5) / 300.0) * ${VSTEP}, 0.001);
    return t0 + (len + ${TAIL.toFixed(1)}) / vel + ${(4 * IMPACT_T).toFixed(1)};
  }
  void main() {
    int sid = int(flow.z + 0.5);
    ivec2 c = ivec2((sid % ${ROW}) * 3, sid / ${ROW});
    vP0 = texelFetch(uState, c, 0);
    vP1 = texelFetch(uState, c + ivec2(1, 0), 0);
    vF = texelFetch(uState, c + ivec2(2, 0), 0);
    // nothing lit on this streamline: put the vertex outside the clip volume so it rasterises nothing
    float L = flow.y;
    float last = max(max(ends(vP0.x, vP0.y, L), ends(vP0.z, vP0.w, L)),
                     max(ends(vP1.x, vP1.y, L), ends(vP1.z, vP1.w, L)));
    float flashed = max(vF.x, vF.z);
    if (uTime > last && uTime - flashed > ${(6 * FLASH_T).toFixed(1)}) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vec3 t = normalize(mat3(modelViewMatrix) * dir);
    vec3 v = normalize(-mvPosition.xyz);
    vec3 s = cross(t, v);
    float sl = length(s);
    s = sl > 1e-4 ? s / sl : vec3(1.0, 0.0, 0.0);
    float side = (gl_VertexID % 2 == 0) ? -1.0 : 1.0;
    // a wide ribbon for the halo, nudged toward the eye so its own fibre doesn't hide it
    mvPosition.xyz += s * side * uWidth * 0.5 + v * uWidth * 0.3;
    vSide = side;
    vArc = flow.x;
    vLen = flow.yw;
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
  }`;

const fragmentShader = /* glsl */`
  #include <common>
  #include <clipping_planes_pars_fragment>
  uniform float uTime;
  uniform float uGain;
  uniform vec3 uColor;
  varying float vSide;
  varying float vArc;
  flat varying vec4 vP0;
  flat varying vec4 vP1;
  flat varying vec4 vF;
  flat varying vec2 vLen;

  // One pulse (see pack() for the code)
  float comet(float t0, float code) {
    float c = floor(code + 0.5);
    float d = mod(c, 3.0), amp = mod(floor(c / 3.0), 100.0) / 99.0, vel = floor(c / 300.0) * ${VSTEP};
    float h = (uTime - t0) * vel; // how far the head has travelled, mm
    if (h <= 0.0) return 0.0;
    float len = vLen.x, mid = vLen.y;
    float x = d < 0.5 ? vArc : d < 1.5 ? len - vArc : abs(vArc - mid);
    float reach = d < 1.5 ? len : (vArc < mid ? mid : len - mid);
    float b = h - x; // distance behind the head
    float I = b >= 0.0 ? pow(max(0.0, 1.0 - b / ${TAIL.toFixed(1)}), 2.2) : exp(b * 3.0);
    float over = h - reach; // past the end: the landing spot flares, then fades
    if (over > 0.0) I += 1.2 * exp(-over / (vel * ${IMPACT_T.toFixed(1)})) * exp(-(reach - x) / 2.0);
    return I * amp;
  }

  float flash(float t, float amp, float dEnd) {
    float age = uTime - t;
    return age < 0.0 ? 0.0 : amp * exp(-age / ${FLASH_T.toFixed(1)}) * exp(-dEnd / ${FLASH_LEN.toFixed(1)});
  }

  void main() {
    #include <clipping_planes_fragment>
    float I = comet(vP0.x, vP0.y) + comet(vP0.z, vP0.w) + comet(vP1.x, vP1.y) + comet(vP1.z, vP1.w);
    float F = flash(vF.x, vF.y, vArc) + flash(vF.z, vF.w, vLen.x - vArc);
    float u2 = vSide * vSide;
    float core = exp(-u2 * 16.0), halo = exp(-u2 * 3.0);
    float e = I + 0.6 * F;
    vec3 c = uColor * e * (0.3 * halo + 0.8 * core) + vec3(1.0, 0.96, 0.88) * min(I, 1.4) * core * 0.9;
    c *= uGain;
    if (max(c.r, max(c.g, c.b)) < 0.004) discard;
    gl_FragColor = vec4(c, 1.0);
  }`;

// Min-heap of events by time.
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  peek() { return this.a[0]; }
  push(e) {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].t <= e.t) break;
      a[i] = a[p];
      i = p;
    }
    a[i] = e;
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        const mt = () => (m === i ? last.t : a[m].t);
        if (l < a.length && a[l].t < mt()) m = l;
        if (r < a.length && a[r].t < mt()) m = r;
        if (m === i) break;
        a[i] = a[m];
        i = m;
      }
      a[i] = last;
    }
    return top;
  }
  clear() { this.a.length = 0; }
}

// Scenarios: what starts the activity. seeds = [node id, delay ms]; every = repeat period (ms of model time).
// gain = strength kept per hop (cortex = the same through association and commissural bundles, so a scenario can
// follow its main pathway further than the spread across the cortex), hops = how far a burst can spread,
// fanout scales the pulses per bundle, only = bundle categories the activity may use.
const CORTICAL = new Set(['Association', 'Commissural']);
export const SCENARIOS = [
  {
    id: 'rest', label: 'Resting activity', spont: 0.4, hops: 2, gain: 0.65, cortex: 0.6,
    text: 'Areas fire now and then on their own; each burst spreads a short way along its pathways and fades.',
  },
  {
    id: 'see', label: 'Seeing a flash', seeds: [['eye.L', 0], ['eye.R', 0]], every: 150,
    hops: 7, gain: 0.9, cortex: 0.66,
    text: 'Light reaches both eyes. Signals run back along the optic nerves to the thalamus (LGN), up the optic '
      + 'radiations to the visual cortex at the back of the head, then forward along the long association bundles.',
  },
  {
    id: 'hear', label: 'Hearing a sound', seeds: [['ear.L', 0], ['ear.R', 0]], every: 150,
    hops: 9, gain: 0.92, cortex: 0.66,
    text: 'Inner ear, then the cochlear nuclei, the lateral lemniscus, the inferior colliculus and the thalamus '
      + '(MGN), and up the acoustic radiation to the auditory cortex, then on toward the language areas.',
  },
  {
    id: 'touch', label: 'Touch on the right hand', seeds: [['dcn.L', 0], ['cord_sense.L', 2]], every: 150, hops: 6,
    gain: 0.9, cortex: 0.62,
    text: 'Touch signals cross over low in the brainstem and climb the left medial lemniscus to the thalamus, then '
      + 'up to the body-sense strip (postcentral gyrus) of the left hemisphere.',
  },
  {
    id: 'move', label: 'Moving the right hand', every: 180, hops: 7, gain: 0.92, cortex: 0.42,
    seeds: [['L.superiorfrontal', 0], ['L.caudalmiddlefrontal', 5], ['L.precentral', 12]],
    text: 'Planning areas fire, then the left motor strip sends commands down the corticospinal tract, with copies '
      + 'to the pons and on to the cerebellum, whose output returns through the thalamus to the motor cortex.',
  },
  {
    id: 'sleep', label: 'Deep-sleep slow wave', every: 200, hops: 9, gain: 0.9, cortex: 0.9, fanout: 1.4,
    seeds: [['L.superiorfrontal', 0], ['R.superiorfrontal', 0], ['L.rostralmiddlefrontal', 3],
      ['R.rostralmiddlefrontal', 3], ['L.medialorbitofrontal', 4], ['R.medialorbitofrontal', 4]],
    only: ['Association', 'Commissural'],
    text: 'In deep sleep, slow waves usually begin in the frontal lobe and sweep back across the cortex '
      + '(Massimini et al. 2004). Only cortex-to-cortex bundles carry it here, and the gaps between waves are '
      + 'shortened.',
  },
  {
    id: 'ftract', label: 'Stimulation responses (F-TRACT)', measured: true,
    text: 'Measured, not modelled: in epilepsy surgery planning, electrodes inside the brain give one area a single '
      + 'small pulse and record which areas respond, and how soon (F-TRACT, 780 patients). The stimulated area '
      + 'flashes; pulses run along the fibres to each responding area in its estimated axonal delay, and the area '
      + 'lights when its response began. Brighter = responded more often. Areas with no fibre drawn between them '
      + 'just light up.',
  },
];

// Brightness for a response probability. Nine in ten measured pairs respond more often than not (median 0.9), so
// the scale starts at 0.5 to keep the differences visible.
const shade = (p) => Math.max(0.08, Math.min(1, (p - 0.5) / 0.5)) ** 1.5;

export class SignalFlow {
  constructor(desc, tracts, parent, measured = null) {
    this.desc = desc;
    this.measured = measured; // the F-TRACT atlas (man.ftract), when this build has it
    this.stimulus = 'L'; // measured mode: 'L' / 'R' = each area of that side in turn, or a parcel index
    this.minRecordings = 20; // measured mode: leave out pairs tested fewer times (small samples run to extremes)
    this.onStimulus = null; // called with each stimulation's summary
    this.tracts = tracts;
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    this.playing = true;
    this.rate = 10; // model ms per second on screen: 100x slow motion
    this.glowWidth = 2.4;
    this.gain = 1;
    this.time = 0;
    this.nodeIndex = new Map(desc.nodes.map((n, i) => [n.id, i]));
    // Two passes: pulses behind other fibres still show, dimmed (no depth test), and the ones in view get the
    // rest of their brightness from a depth-tested pass on top.
    this.uniforms = {
      uState: { value: null }, uTime: { value: 0 }, uWidth: { value: this.glowWidth },
      uColor: { value: new THREE.Color(1.0, 0.62, 0.2) },
    };
    this.materials = [[false, 0.4], [true, 0.6]].map(([depthTest, share]) => {
      const m = new THREE.ShaderMaterial({
        uniforms: { ...this.uniforms, uGain: { value: share } },
        vertexShader, fragmentShader, clipping: true, side: THREE.DoubleSide,
        transparent: true, depthTest, depthWrite: false, blending: THREE.AdditiveBlending,
      });
      m.userData.share = share;
      m.clippingPlanes = clipPlanes;
      return m;
    });
    this.setScenario('rest');
    this.loading = null;
  }

  load() {
    if (!this.loading) this.loading = this._load();
    return this.loading;
  }

  async _load() {
    const [links, edges] = await Promise.all([fetchArray(this.desc.links),
      this.measured ? fetchArray(this.measured.edges) : null, this.tracts.load()]);
    const nS = links.length / 4, nN = this.desc.nodes.length;
    this.n = nS;
    this.len = new Float32Array(nS);
    this.mid = new Float32Array(nS);
    this.bundleOf = new Uint16Array(nS);
    this.A = new Uint16Array(nS);
    this.B = new Uint16Array(nS);

    // per-vertex flow attribute, sharing the tract geometry's buffers
    this.meshes = [];
    this.tracts.items.forEach((b, bi) => {
      const P = b.points, np = b.ptEnd[b.count - 1];
      const fl = new Float32Array(np * 8);
      let a = 0;
      for (let k = 0; k < b.count; k++) {
        const s = b.first + k, e = b.ptEnd[k];
        const L = links, l4 = s * 4;
        this.A[s] = L[l4];
        this.B[s] = L[l4 + 1];
        this.bundleOf[s] = bi;
        let arc = 0;
        for (let i = a; i < e; i++) {
          if (i > a) {
            arc += Math.hypot(P[i * 6] - P[(i - 1) * 6], P[i * 6 + 1] - P[(i - 1) * 6 + 1],
              P[i * 6 + 2] - P[(i - 1) * 6 + 2]);
          }
          fl[i * 8] = fl[i * 8 + 4] = arc;
          fl[i * 8 + 2] = fl[i * 8 + 6] = s;
        }
        this.len[s] = arc;
        this.mid[s] = ((L[l4 + 3] >> 8) / 255) * arc;
        for (let i = a; i < e; i++) {
          fl[i * 8 + 1] = fl[i * 8 + 5] = arc;
          fl[i * 8 + 3] = fl[i * 8 + 7] = this.mid[s];
        }
        a = e;
      }
      const src = b.mesh.geometry;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', src.getAttribute('position'));
      geo.setAttribute('dir', src.getAttribute('dir'));
      geo.setAttribute('flow', new THREE.BufferAttribute(fl, 4));
      geo.setIndex(src.getIndex());
      geo.boundingSphere = src.boundingSphere;
      this.materials.forEach((mat, i) => {
        const mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = 3 + i;
        mesh.userData = { item: b };
        this.group.add(mesh);
        this.meshes.push(mesh);
      });
    });

    // who sends what: per node, its outgoing streamlines grouped by bundle (encoded s * 4 + direction code)
    const out = Array.from({ length: nN }, () => new Map());
    const ends = Array.from({ length: nN }, () => []);
    const add = (n, s, code) => {
      const m = out[n], bi = this.bundleOf[s];
      if (!m.has(bi)) m.set(bi, []);
      m.get(bi).push(s * 4 + code);
    };
    for (let s = 0; s < nS; s++) {
      const mode = links[s * 4 + 3] & 15;
      if (mode & 1) add(this.A[s], s, 0);
      if (mode & 2) add(this.B[s], s, 1);
      if (mode & 4) add(links[s * 4 + 2], s, 2);
      ends[this.A[s]].push(s * 2);
      ends[this.B[s]].push(s * 2 + 1);
    }
    this.out = out.map((m) => [...m].map(([bi, list]) => ({
      cat: this.tracts.items[bi].category, list: Int32Array.from(list),
    })));
    this.ends = ends.map((l) => Int32Array.from(l));
    this.relays = Array.from({ length: nN }, () => []);
    for (const [f, t, delay] of this.desc.relays) this.relays[f].push([t, delay]);
    this.isCortex = this.desc.nodes.map((n) => n.kind === 'cortex');
    if (edges) this.loadMeasured(edges);

    const rows = Math.ceil(nS / ROW);
    this.state = new Float32Array(ROW * 3 * rows * 4);
    this.tex = new THREE.DataTexture(this.state, ROW * 3, rows, THREE.RGBAFormat, THREE.FloatType);
    this.tex.minFilter = this.tex.magFilter = THREE.NearestFilter;
    this.uniforms.uState.value = this.tex;
    this.slot = new Uint8Array(nS);
    this.nodeFire = new Float32Array(nN);
    this.nodeAmp = new Float32Array(nN);
    this.reset();
  }

  setVisible(v) {
    this.group.visible = v;
    if (v) this.load();
  }

  // F-TRACT responses per stimulated parcel, earliest first, and the streamlines joining each pair of nodes.
  loadMeasured(edges) {
    const m = this.measured, nc = m.columns.length, nP = m.parcels.length, nN = this.desc.nodes.length;
    const col = Object.fromEntries(m.columns.map((c, i) => [c, i]));
    this.responses = Array.from({ length: nP }, () => []);
    for (let i = 0; i < edges.length; i += nc) {
      const e = {};
      for (const c in col) e[c] = edges[i + col[c]];
      if (m.parcels[e.stim].nodes.length) this.responses[e.stim].push(e);
    }
    for (const r of this.responses) r.sort((a, b) => a.onset - b.onset);
    this.between = new Map();
    for (let s = 0; s < this.n; s++) {
      const a = this.A[s], b = this.B[s];
      if (a === b) continue;
      const key = Math.min(a, b) * nN + Math.max(a, b);
      if (!this.between.has(key)) this.between.set(key, []);
      this.between.get(key).push(s);
    }
  }

  // Parcels that can be stimulated: mapped onto the model, with responses.
  stimulable() {
    if (!this.responses) return [];
    return this.measured.parcels.map((p, i) => ({ ...p, index: i })).filter((p) => this.responses[p.index].length);
  }

  setStimulus(v) {
    this.stimulus = v;
    if (this.state) this.reset();
  }

  setScenario(id) {
    this.scenario = SCENARIOS.find((s) => s.id === id);
    if (this.state) this.reset();
  }

  // Clear everything and start the current scenario from the beginning.
  reset() {
    this.time = 0;
    this.events = new Heap();
    this.state.fill(-1e6);
    for (let i = 1; i < this.state.length; i += 2) this.state[i] = 0;
    const nN = this.desc.nodes.length;
    this.refractory = new Float32Array(nN).fill(-1e6);
    this.potential = new Float32Array(nN);
    this.lastInput = new Float32Array(nN);
    this.nodeFire.fill(-1e6);
    this.nodeAmp.fill(0);
    this.nextStim = 0;
    this.spontDebt = 0;
    this.turn = 0;
    this.tex.needsUpdate = true;
  }

  // Advance the model by dt seconds of screen time.
  step(dt) {
    if (!this.state) return;
    const sc = this.scenario;
    const t1 = this.time + Math.min(dt, 0.1) * this.rate;
    if (sc.measured) {
      while (this.responses && this.nextStim <= t1) this.nextStim += this.stimulate(this.nextStim);
    }
    if (sc.seeds) {
      while (this.nextStim <= t1) {
        for (const [id, d] of sc.seeds) {
          const n = this.nodeIndex.get(id);
          if (n !== undefined) this.events.push({ t: this.nextStim + d, fire: true, n, amp: 1, hop: 0 });
        }
        this.nextStim += sc.every;
      }
    }
    if (sc.spont) {
      this.spontDebt += (t1 - this.time) * sc.spont;
      while (this.spontDebt >= 1) {
        this.spontDebt -= 1;
        const n = (Math.random() * this.desc.nodes.length) | 0;
        const t = this.time + Math.random() * (t1 - this.time);
        this.events.push({ t, fire: true, n, amp: 0.55 + 0.45 * Math.random(), hop: 0 });
      }
    }
    while (this.events.size && this.events.peek().t <= t1) {
      const e = this.events.pop();
      if (e.light) this.light(e.t, e.n, e.amp);
      else if (e.fire) this.fire(e);
      else this.arrive(e);
    }
    this.time = t1;
    this.tex.needsUpdate = true;
  }

  // Measured mode: stimulate one parcel at time t0 and lay out its responses. Returns how long until the next.
  stimulate(t0) {
    const parcels = this.measured.parcels, nN = this.desc.nodes.length;
    let p = this.stimulus;
    if (typeof p === 'string') {
      const side = this.stimulable().filter((q) => q.label.startsWith(p === 'L' ? 'Left' : 'Right'));
      p = side[this.turn++ % side.length].index;
    }
    const from = parcels[p].nodes, list = this.responses[p].filter((r) => r.recordings >= this.minRecordings);
    for (const n of from) this.events.push({ t: t0, light: true, n, amp: 1 });
    for (const r of list) {
      const amp = shade(r.probability);
      for (const m of parcels[r.rec].nodes) {
        this.events.push({ t: t0 + r.onset, light: true, n: m, amp });
        if (!(r.delay > 0)) continue;
        for (const n of from) {
          const ss = this.between.get(Math.min(n, m) * nN + Math.max(n, m));
          if (!ss) continue;
          const k = Math.min(ss.length, Math.ceil(1.6 * Math.sqrt(ss.length)));
          for (let i = 0; i < k; i++) {
            const s = ss[(Math.random() * ss.length) | 0];
            this.draw(s, this.A[s] === n ? 0 : 1, t0 + Math.random() * 0.5, amp, Math.min(19.9, this.len[s] / r.delay));
          }
        }
      }
    }
    this.onStimulus?.({ parcel: parcels[p], responses: list.map((r) => ({ ...r, parcel: parcels[r.rec] })) });
    const last = list.length ? list[list.length - 1].onset : 0;
    return Math.max(90, Math.min(220, last + 40));
  }

  // Light up the fibre ends at node n.
  light(t, n, amp) {
    const S = this.state;
    this.nodeFire[n] = t;
    this.nodeAmp[n] = amp;
    for (const e of this.ends[n]) {
      const o = (e >> 1) * 12 + 8 + (e & 1) * 2;
      S[o] = t;
      S[o + 1] = amp;
    }
  }

  fire({ t, n, amp, hop }) {
    const sc = this.scenario;
    this.light(t, n, amp);
    if (hop >= sc.hops) return;
    const fan = sc.fanout ?? 1;
    for (const g of this.out[n]) {
      if (sc.only && !sc.only.includes(g.cat)) continue;
      const list = g.list, k = Math.min(list.length, Math.ceil(1.6 * fan * Math.sqrt(list.length)));
      const gain = CORTICAL.has(g.cat) ? sc.cortex ?? sc.gain : sc.gain;
      for (let i = 0; i < k; i++) {
        const v = list[(Math.random() * list.length) | 0];
        this.pulse(v >> 2, v & 3, t + Math.random() * 2.5, amp, hop, gain);
      }
    }
    if (!sc.only) for (const [m, d] of this.relays[n]) this.events.push({ t: t + d, n: m, amp, hop, gain: sc.gain });
  }

  // Draw a pulse on streamline s (no arrival events).
  draw(s, code, t0, amp, vel) {
    const S = this.state, k = this.slot[s];
    this.slot[s] = (k + 1) % SLOTS;
    const o = s * 12 + (k >> 1) * 4 + (k & 1) * 2;
    S[o] = t0;
    S[o + 1] = pack(code, amp, vel);
  }

  pulse(s, code, t0, amp, hop, gain) {
    this.draw(s, code, t0, amp, VELOCITY);
    const L = this.len[s];
    if (code === 0) this.events.push({ t: t0 + L / VELOCITY, n: this.B[s], amp, hop, gain });
    else if (code === 1) this.events.push({ t: t0 + L / VELOCITY, n: this.A[s], amp, hop, gain });
    else {
      const m = this.mid[s];
      this.events.push({ t: t0 + m / VELOCITY, n: this.A[s], amp, hop, gain });
      this.events.push({ t: t0 + (L - m) / VELOCITY, n: this.B[s], amp, hop, gain });
    }
  }

  // A pulse lands: the node sums its input (leaking over ~15 ms) and fires once it crosses threshold,
  // unless it fired within the last 60 ms.
  arrive({ t, n, amp, hop, gain }) {
    const sc = this.scenario;
    if (sc.only && !this.isCortex[n]) return;
    this.potential[n] = this.potential[n] * Math.exp(-(t - this.lastInput[n]) / 15) + amp;
    this.lastInput[n] = t;
    const next = amp * gain;
    if (t < this.refractory[n] || this.potential[n] < 0.5 || next < 0.12) return;
    this.potential[n] = 0;
    this.refractory[n] = t + 60;
    this.events.push({ t: t + 1.5, fire: true, n, amp: next, hop: hop + 1 });
  }

  // Keep the glow meshes in step with the tract layer's bundle filters and density.
  sync() {
    const tr = this.tracts;
    for (const m of this.meshes) {
      const b = m.userData.item;
      m.visible = b.on && tr.catOn[b.category];
      const k = Math.max(1, Math.round(b.count * tr.density));
      m.geometry.setDrawRange(0, b.idxEnd[k - 1]);
    }
    this.uniforms.uTime.value = this.time;
    this.uniforms.uWidth.value = this.glowWidth;
    for (const m of this.materials) m.uniforms.uGain.value = this.gain * m.userData.share;
  }
}
