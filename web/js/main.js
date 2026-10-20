import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import { fetchManifest } from './data.js';
import { clipPlanes } from './materials.js';
import { cssGradient } from './colormaps.js';
import { Cortex } from './cortex.js';
import { MeshGroup } from './meshes.js';
import { Tracts } from './tracts.js';
import { SignalFlow, SCENARIOS } from './flow.js';
import { TmsTargets } from './tms.js';
import { VolumeStore, Slices, VolumeRender } from './volume.js';
import * as ui from './ui.js';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

// ---------------------------------------------------------------------------- scene

const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#0c0f14');
const camera = new THREE.PerspectiveCamera(30, 1, 2, 4000);
scene.add(camera);
scene.add(new THREE.HemisphereLight('#dfe8ff', '#2a2622', 1.1));
const key = new THREE.DirectionalLight('#ffffff', 2.1);
key.position.set(0.35, 0.6, 1);
camera.add(key);
camera.add(key.target);
key.target.position.set(0, 0, -1);

// All data is in MNI152 RAS millimetres; rotate so superior (+z) is screen-up (+y).
const brain = new THREE.Group();
brain.rotation.x = -Math.PI / 2;
scene.add(brain);
brain.updateMatrixWorld(true);
const CENTER = new THREE.Vector3(0, -18, 12); // MNI
const toWorld = (v) => v.clone().applyMatrix4(brain.matrixWorld);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.rotateSpeed = 0.8;
controls.minDistance = 30;
controls.maxDistance = 1600;
controls.autoRotateSpeed = 1.2;

let needsRender = true;
const requestRender = () => { needsRender = true; };
controls.addEventListener('change', requestRender);

// Keep the brain centred in the part of the window the panel doesn't cover.
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  const panelW = document.body.classList.contains('panel-hidden') ? 0 : document.getElementById('panel').offsetWidth + 10;
  const shift = w > 700 ? panelW / 2 : 0;
  camera.setViewOffset(w, h, -shift, 0, w, h);
  camera.updateProjectionMatrix();
  requestRender();
}
new ResizeObserver(resize).observe(canvas);

const VIEWS = {
  Left: [-1, 0, 0], Right: [1, 0, 0], Front: [0, 1, 0], Back: [0, -1, 0],
  Top: [0, -0.001, 1], Bottom: [0, 0.001, -1], Oblique: [-0.78, 0.42, 0.42],
};
function setView(name, dist) {
  const d = new THREE.Vector3(...VIEWS[name]).normalize();
  const target = toWorld(CENTER);
  const r = dist ?? camera.position.distanceTo(controls.target);
  const dirWorld = d.applyQuaternion(brain.quaternion);
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dirWorld, r);
  camera.lookAt(target);
  controls.update();
  requestRender();
}

// ---------------------------------------------------------------------------- status

const statusEl = document.getElementById('status');
const busy = new Map();
function setBusy(key, text) {
  if (text) busy.set(key, text); else busy.delete(key);
  statusEl.textContent = [...busy.values()].join(' · ');
  statusEl.hidden = busy.size === 0;
}
async function withBusy(key, text, p) {
  setBusy(key, text);
  try { return await p; } finally { setBusy(key, null); requestRender(); }
}

// ---------------------------------------------------------------------------- app

const man = await fetchManifest();
const cortex = new Cortex(man.cortex, brain);
const groups = Object.fromEntries(man.meshGroups.map((g) => [g.id, g]));
const aseg = new MeshGroup(groups.aseg, brain);
const massp = new MeshGroup(groups.massp, brain);
const thalamic = new MeshGroup(groups.thalamic, brain);
const context = new MeshGroup(groups.context, brain, { style: 'xray' });
const tracts = new Tracts(man.tracts, brain);
const flow = new SignalFlow(man.flow, tracts, brain, man.ftract);
const store = new VolumeStore(man.volumes);
const slices = new Slices(store, brain);
const volume = new VolumeRender(store, brain);
const tms = new TmsTargets(man.tms, brain);
const structureGroups = [aseg, massp, thalamic, context];

// Opening view: the white-matter tracts on their own, coloured by fibre direction. Every other layer, signal flow
// included, loads when first switched on, so the first view stays light enough for most devices.
setView('Left', 430);
tracts.setVisible(true);
await withBusy('tracts', 'Loading tracts…', tracts.load());
document.getElementById('loading').remove();

// ---------------------------------------------------------------------------- clipping

const clip = { axis: 'off', pos: { x: -4, y: -18, z: 10 }, flip: false };
const localPlane = new THREE.Plane();
function applyClip() {
  clipPlanes.length = 0;
  if (clip.axis === 'off') {
    volume.setClip(null);
  } else {
    const n = new THREE.Vector3(clip.axis === 'x' ? 1 : 0, clip.axis === 'y' ? 1 : 0, clip.axis === 'z' ? 1 : 0);
    const s = clip.flip ? 1 : -1; // default keeps the low side (left / back / bottom)
    localPlane.set(n.multiplyScalar(s), -s * clip.pos[clip.axis]);
    clipPlanes.push(localPlane.clone().applyMatrix4(brain.matrixWorld));
    volume.setClip(localPlane);
  }
  requestRender();
}
const keepPoint = (p) => clip.axis === 'off' || localPlane.distanceToPoint(p) >= -1e-3;
applyClip();

// ---------------------------------------------------------------------------- panel

const panel = document.getElementById('panel-body');
const { bounds } = store;
const mm = (v) => `${v > 0 ? '+' : ''}${Math.round(v)} mm`;

// Cortex
{
  const s = ui.section(panel, 'Cortex', {
    checked: false,
    onToggle: (v) => {
      cortex.setVisible(v);
      if (v) {
        withBusy('cortex', 'Loading cortex…', cortex.load()).then(() =>
          // build the bounds trees while idle so the first hover doesn't stall
          (window.requestIdleCallback || setTimeout)(() => cortex.prepareRaycast()));
      }
      requestRender();
    },
  });
  const shapeNames = man.cortex.surfaces.map((x) => x.label);
  const sl = ui.slider(s.body, 'Surface', {
    min: 0, max: shapeNames.length - 1, step: 0.01, value: 0,
    format: (v) => {
      const i = Math.round(v);
      return Math.abs(v - i) < 0.04 ? shapeNames[i] : `${shapeNames[Math.floor(v)]} → ${shapeNames[Math.ceil(v)]}`;
    },
    onInput: (v) => { cortex.setShape(v); requestRender(); },
  });
  sl.row.classList.add('wide');
  const ticks = ui.inline(s.body, 'ticks');
  shapeNames.forEach((n, i) => {
    const b = ui.el('button', '', n);
    b.type = 'button';
    b.addEventListener('click', () => { sl.set(i); cortex.setShape(i); requestRender(); });
    ticks.append(b);
  });

  const opts = [{ value: 'none', label: 'Plain', group: 'Shading' }];
  for (const sc of man.cortex.scalars) opts.push({ value: `scalar:${sc.id}`, label: sc.label, group: 'Morphology' });
  const kinds = {
    dk: 'Anatomical', destrieux: 'Anatomical', bodymap: 'Body map', hcpmmp: 'Multimodal', brainnetome: 'Connectivity',
    yeo7: 'Functional networks', yeo17: 'Functional networks', schaefer100: 'Functional networks',
    schaefer400: 'Functional networks', schaefer1000: 'Functional networks',
    lausanne1: 'Multi-scale (Lausanne)', lausanne3: 'Multi-scale (Lausanne)', lausanne5: 'Multi-scale (Lausanne)',
  };
  const atlases = [...man.cortex.atlases].sort((a, b) =>
    Object.values(kinds).indexOf(kinds[a.id]) - Object.values(kinds).indexOf(kinds[b.id]));
  for (const a of atlases) opts.push({ value: `atlas:${a.id}`, label: a.label, group: kinds[a.id] });
  const desc = ui.note(s.body, '', 'desc');
  const legend = ui.inline(s.body, 'legend');
  const keys = ui.inline(s.body, 'keys');
  let labelsBox = null;
  const describe = (v) => {
    const [k, id] = v.split(':');
    const m = k === 'atlas' ? man.cortex.atlases.find((a) => a.id === id)
      : k === 'scalar' ? man.cortex.scalars.find((a) => a.id === id) : null;
    desc.textContent = m ? m.description : 'Uniform grey; fold shading shows sulci (dark) and gyri (light).';
    legend.hidden = k !== 'scalar';
    if (k === 'scalar') {
      const f = (x) => (Math.abs(x) < 10 ? x.toFixed(2) : x.toFixed(0));
      legend.innerHTML = `<span>${f(m.min)}</span><i style="background:${cssGradient(m.cmap)}"></i><span>${f(m.max)}</span>`;
    }
    keys.hidden = !m?.legend;
    keys.replaceChildren(...(m?.legend || []).map((l) => {
      const sw = ui.el('i');
      sw.style.background = l.color;
      const key = ui.el('span');
      key.append(sw, l.name);
      return key;
    }));
    labelsBox?.setEnabled(Boolean(m?.hemis?.L?.anchors));
  };
  ui.select(s.body, 'Colour by', opts, cortex.overlay, (v) => {
    describe(v);
    withBusy('overlay', 'Loading overlay…', cortex.setOverlay(v));
  }).row.after(desc, legend, keys);
  ui.checkbox(s.body, 'Fold shading', true, (v) => { cortex.setFoldShading(v); requestRender(); });
  ui.checkbox(s.body, 'Region outlines', false, (v) => { cortex.setOutlines(v); requestRender(); });
  labelsBox = ui.checkbox(s.body, 'Labels', true, (v) => { cortex.showLabels = v; requestRender(); });
  labelsBox.row.title = 'Names on the surface; the body map has them';
  describe(cortex.overlay);
  const hem = ui.inline(s.body);
  ui.checkbox(hem, 'Left hemisphere', true, (v) => { cortex.setHemiVisible('L', v); requestRender(); });
  ui.checkbox(hem, 'Right hemisphere', true, (v) => { cortex.setHemiVisible('R', v); requestRender(); });
  ui.select(s.body, 'Style', [{ value: 'solid', label: 'Solid' }, { value: 'xray', label: 'Glass (x-ray)' }], 'solid',
    (v) => { cortex.setStyle(v); requestRender(); });
  ui.slider(s.body, 'Opacity', {
    min: 0.05, max: 1, step: 0.01, value: 1, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { cortex.setOpacity(v); requestRender(); },
  });
}

// Structure groups (aseg / nuclei / thalamus / context)
function structureSection(g, title, { hint, open = false, defaultOn = false } = {}) {
  const s = ui.section(panel, title, {
    checked: defaultOn, open, hint,
    onToggle: (v) => { g.setVisible(v); if (v) withBusy(g.desc.id, `Loading ${title.toLowerCase()}…`, g.load()); requestRender(); },
  });
  ui.select(s.body, 'Style', [{ value: 'solid', label: 'Solid' }, { value: 'xray', label: 'Glass (x-ray)' }], g.style,
    (v) => { g.setStyle(v); requestRender(); });
  ui.slider(s.body, 'Opacity', {
    min: 0.05, max: 1, step: 0.01, value: g.opacity, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { g.setOpacity(v); requestRender(); },
  });
  // rows merge left/right copies of a structure; the hemisphere boxes filter by side
  const sides = { L: true, R: true };
  const byName = new Map();
  for (const it of g.items) {
    const k = it.name;
    if (!byName.has(k)) byName.set(k, { name: k, category: it.category, color: it.color, items: [], on: it.on });
    byName.get(k).items.push(it);
  }
  const rows = [...byName.values()];
  const apply = () => {
    for (const r of rows) for (const it of r.items) g.setItemVisible(it, r.on && (it.hemi === '' || sides[it.hemi]));
    requestRender();
  };
  if (g.items.some((it) => it.hemi)) {
    const hem = ui.inline(s.body);
    ui.checkbox(hem, 'Left', true, (v) => { sides.L = v; apply(); });
    ui.checkbox(hem, 'Right', true, (v) => { sides.R = v; apply(); });
  }
  const cats = [...new Set(rows.map((r) => r.category))];
  ui.groupedList(s.body, `Structures (${rows.length})`, cats.map((c) => [cats.length > 1 ? c : '',
    rows.filter((r) => r.category === c).map((r) => ({
      label: r.name, swatch: r.color, checked: r.on, onChange: (v) => { r.on = v; apply(); },
    }))]));
  g.load = ((orig) => () => orig.call(g).then(apply))(g.load);
  return s;
}
structureSection(aseg, 'Subcortical structures', {
  hint: 'Surfaces extracted from the FreeSurfer-style segmentation of the MNI152 2009c template.',
});
structureSection(massp, 'Subcortical nuclei (MASSP)', {
  hint: '63 small nuclei, hippocampal subfields and fibre landmarks from the MASSP atlas.',
});
structureSection(thalamic, 'Thalamic nuclei', { hint: 'Seven nuclear groups per thalamus (MIAL atlas).' });

// Tracts
{
  const s = ui.section(panel, 'White-matter tracts', {
    checked: true, open: true, hint: man.tracts.description,
    onToggle: (v) => { tracts.setVisible(v); if (v) withBusy('tracts', 'Loading tracts…', tracts.load()); requestRender(); },
  });
  const key = ui.note(s.body, 'Red: left–right · green: front–back · blue: up–down', 'desc');
  key.hidden = tracts.mode !== 0 && tracts.mode !== 3;
  ui.select(s.body, 'Colour by', [
    { value: '0', label: 'Fibre direction' }, { value: '3', label: 'Fibre direction (dark)' },
    { value: '1', label: 'Bundle' }, { value: '4', label: 'Bundle (dark)' }, { value: '2', label: 'Gold' }],
  String(tracts.mode),
  (v) => { tracts.mode = +v; key.hidden = v !== '0' && v !== '3'; tracts.apply(); requestRender(); }).row.after(key);
  ui.slider(s.body, 'Width', {
    min: 0.1, max: 2, step: 0.05, value: tracts.width, format: (v) => `${v.toFixed(2)} mm`,
    onInput: (v) => { tracts.width = v; tracts.apply(); requestRender(); },
  });
  ui.slider(s.body, 'Density', {
    min: 0.02, max: 1, step: 0.01, value: tracts.density, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { tracts.density = v; tracts.apply(); requestRender(); },
  });
  ui.slider(s.body, 'Opacity', {
    min: 0.05, max: 1, step: 0.01, value: 1, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { tracts.opacity = v; tracts.apply(); requestRender(); },
  });
  const sides = { L: true, R: true };
  const byName = new Map();
  for (const b of tracts.items) {
    if (!byName.has(b.name)) byName.set(b.name, { name: b.name, category: b.category, color: b.color, items: [], on: true });
    byName.get(b.name).items.push(b);
  }
  const rows = [...byName.values()];
  const apply = () => {
    for (const r of rows) for (const b of r.items) b.on = r.on && (b.hemi === '' || sides[b.hemi]);
    tracts.apply();
    requestRender();
  };
  const hem = ui.inline(s.body);
  ui.checkbox(hem, 'Left', true, (v) => { sides.L = v; apply(); });
  ui.checkbox(hem, 'Right', true, (v) => { sides.R = v; apply(); });
  const cats = ui.inline(s.body, 'cats');
  for (const c of man.tracts.categories) {
    ui.checkbox(cats, c, true, (v) => { tracts.catOn[c] = v; tracts.apply(); requestRender(); });
  }
  ui.groupedList(s.body, `Bundles (${rows.length})`, man.tracts.categories.map((c) => [c,
    rows.filter((r) => r.category === c).map((r) => ({
      label: r.name, swatch: r.color, checked: true, onChange: (v) => { r.on = v; apply(); },
    }))]));
}

// Signal flow
{
  const s = ui.section(panel, 'Signal flow', {
    checked: false,
    hint: man.ftract ? 'Signals travelling the white-matter tracts: illustrative models built from textbook '
      + 'anatomy, or measured responses from intracranial recordings (F-TRACT).' : man.flow.description,
    onToggle: (v) => { flow.setVisible(v); if (v) withBusy('flow', 'Loading tracts…', flow.load()); requestRender(); },
  });
  const desc = ui.note(s.body, '', 'desc');
  const scenarios = SCENARIOS.filter((x) => !x.measured || man.ftract).map((x) => ({
    value: x.id, label: x.label, group: man.ftract ? (x.measured ? 'Measured' : 'Modelled') : undefined,
  }));
  const scenario = ui.select(s.body, 'Scenario', scenarios, flow.scenario.id, (v) => {
    flow.setScenario(v);
    describe();
    requestRender();
  });
  scenario.row.after(desc);
  // measured mode: which area is stimulated, and what responded
  let stimulus = null, fewest = null, readout = null;
  if (man.ftract) {
    const areas = flow.stimulable();
    stimulus = ui.select(s.body, 'Stimulated area', [
      { value: 'L', label: 'Each left area in turn' }, { value: 'R', label: 'Each right area in turn' },
      ...areas.map((p) => ({ value: String(p.index), label: p.label, group: p.label.split(' ')[0] })),
    ], flow.stimulus, (v) => { flow.setStimulus(v === 'L' || v === 'R' ? v : Number(v)); requestRender(); });
    fewest = ui.slider(s.body, 'Fewest recordings', {
      min: 5, max: 200, step: 5, value: flow.minRecordings, format: (v) => `${v}`,
      onInput: (v) => { flow.minRecordings = v; },
    });
    readout = ui.note(s.body, '', 'desc');
    const pct = (x) => `${Math.round(x * 100)}%`;
    flow.onStimulus = ({ parcel, responses }) => {
      const first = responses.slice(0, 4).map((r) => `${r.parcel.label} after ${Math.round(r.onset)} ms `
        + `(${pct(r.probability)} of ${Math.round(r.recordings)} tests)`);
      const med = responses.length ? responses[responses.length >> 1].onset : 0;
      readout.textContent = responses.length
        ? `Stimulating ${parcel.label}: ${responses.length} areas responded, half of them within `
          + `${Math.round(med)} ms. First: ${first.join('; ')}.`
        : `Stimulating ${parcel.label}: no response measured ${flow.minRecordings} or more times.`;
    };
  }
  const describe = () => {
    desc.textContent = flow.scenario.text;
    stimulus?.setEnabled(!!flow.scenario.measured);
    fewest?.setEnabled(!!flow.scenario.measured);
    if (readout) readout.hidden = !flow.scenario.measured;
  };
  describe();
  const btns = ui.buttons(s.body, [
    {
      label: 'Pause',
      onClick: () => { flow.playing = !flow.playing; btns.firstChild.textContent = flow.playing ? 'Pause' : 'Play'; },
    },
    { label: 'Restart', onClick: () => { flow.reset(); requestRender(); } },
  ]);
  // Log scale, faster to the right: the slider holds log10 of model ms per second on screen.
  const slowdown = (u) => Number((1000 / 10 ** u).toPrecision(2));
  ui.slider(s.body, 'Speed', {
    min: 0.3, max: 3, step: 0.01, value: Math.log10(flow.rate),
    format: (u) => (slowdown(u) === 1 ? 'Real time' : `${slowdown(u)}× slower`),
    onInput: (u) => { flow.rate = 1000 / slowdown(u); },
  });
  ui.slider(s.body, 'Brightness', {
    min: 0.2, max: 2.5, step: 0.05, value: flow.gain, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { flow.gain = v; requestRender(); },
  });
  ui.slider(s.body, 'Glow width', {
    min: 0.5, max: 6, step: 0.1, value: flow.glowWidth, format: (v) => `${v.toFixed(1)} mm`,
    onInput: (v) => { flow.glowWidth = v; requestRender(); },
  });
  ui.note(s.body, 'Pulses run on the bundles and density chosen under White-matter tracts. Each pulse leads with '
    + 'its bright head. Hover a tract to see which way it carries signals. Pulses are easiest to see with the '
    + 'brain cut down the middle (Cut-away: Sagittal) and a dark colouring under White-matter tracts. At real time a signal crosses the brain in a frame or two, '
    + 'so repeats are spaced at least a second apart.', 'desc');
}

// Volume slices
const imageOpts = man.volumes.scalars.map((v) => ({ value: v.id, label: v.label }));
const labelOpts = [{ value: '', label: 'None' }, ...man.volumes.labels.map((v) => ({ value: v.id, label: v.label }))];
{
  let loaded = false;
  const s = ui.section(panel, 'MRI slices', {
    checked: false, hint: 'Orthogonal planes through the MNI152 2009c template volumes (1 mm).',
    onToggle: (v) => {
      slices.setVisible(v);
      if (v && !loaded) {
        loaded = true;
        withBusy('slices', 'Loading volume…', Promise.all([slices.setImage('t1brain'), slices.setLabels('')]));
      }
      requestRender();
    },
  });
  ui.select(s.body, 'Image', [{ value: '', label: 'None' }, ...imageOpts], 't1brain',
    (v) => withBusy('slices', 'Loading volume…', slices.setImage(v)));
  ui.select(s.body, 'Labels', labelOpts, '', (v) => withBusy('slices', 'Loading labels…', slices.setLabels(v)));
  ui.slider(s.body, 'Label opacity', {
    min: 0, max: 1, step: 0.01, value: 0.6, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { slices.setLabelAlpha(v); requestRender(); },
  });
  const win = { lo: 0, hi: 1 };
  ui.slider(s.body, 'Black level', {
    min: 0, max: 1, step: 0.01, value: 0, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { win.lo = v; slices.setWindow(win.lo, win.hi); requestRender(); },
  });
  ui.slider(s.body, 'White level', {
    min: 0, max: 1, step: 0.01, value: 1, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { win.hi = v; slices.setWindow(win.lo, win.hi); requestRender(); },
  });
  for (const [axis, name] of [['z', 'Axial'], ['y', 'Coronal'], ['x', 'Sagittal']]) {
    const i = 'xyz'.indexOf(axis);
    const line = ui.inline(s.body, 'slice-row');
    ui.checkbox(line, name, true, (v) => { slices.setPlaneVisible(axis, v); requestRender(); });
    ui.slider(line, '', {
      min: Math.ceil(bounds.lo[i]), max: Math.floor(bounds.hi[i]), step: 1, value: slices.pos[axis], format: mm,
      onInput: (v) => { slices.setPosition(axis, v); requestRender(); },
    });
  }
}

// Volume rendering
{
  let loaded = false;
  const s = ui.section(panel, 'Volume rendering', {
    checked: false, hint: 'Ray-marched rendering of a template volume, directly from the voxels.',
    onToggle: (v) => {
      volume.setVisible(v);
      if (v && !loaded) { loaded = true; withBusy('volume', 'Loading volume…', volume.setImage('t1brain')); }
      requestRender();
    },
  });
  ui.select(s.body, 'Image', imageOpts, 't1brain', (v) => withBusy('volume', 'Loading volume…', volume.setImage(v)));
  let density;
  ui.select(s.body, 'Mode', [
    { value: '0', label: 'Iso-surface' }, { value: '1', label: 'Translucent' }, { value: '2', label: 'Maximum intensity' }],
  '0', (v) => { volume.setMode(+v); density.setEnabled(v === '1'); requestRender(); });
  ui.slider(s.body, 'Threshold', {
    min: 0.01, max: 0.99, step: 0.01, value: 0.3, format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { volume.uniforms.uThreshold.value = v; requestRender(); },
  });
  density = ui.slider(s.body, 'Density', {
    min: 0.01, max: 1.5, step: 0.01, value: 0.25, format: (v) => v.toFixed(2),
    onInput: (v) => { volume.uniforms.uDensity.value = v; requestRender(); },
  });
  density.setEnabled(false);
  ui.select(s.body, 'Colour by', labelOpts.map((o) => (o.value ? o : { value: '', label: 'Intensity' })), '',
    (v) => withBusy('volume', 'Loading labels…', volume.setLabels(v)));
  ui.select(s.body, 'Quality', [
    { value: '1.2', label: 'Fast' }, { value: '0.6', label: 'Normal' }, { value: '0.3', label: 'Fine' }], '0.6',
  (v) => { volume.uniforms.uStep.value = +v; requestRender(); });
}

structureSection(context, 'Context surfaces', { hint: 'Outer hulls of the template brain and head.' });

// TMS targets
{
  const s = ui.section(panel, 'TMS targets', {
    checked: false, hint: man.tms.description,
    onToggle: (v) => { tms.setVisible(v); requestRender(); },
  });
  ui.note(s.body, 'Dot: the brain target · ring: where the coil sits on the scalp. Hover one for details.', 'desc');
  ui.checkbox(s.body, 'Labels', true, (v) => { tms.showLabels = v; requestRender(); });
  ui.groupedList(s.body, `Targets (${tms.items.length})`, man.tms.groups.map((g) => [g,
    tms.items.filter((t) => t.group === g).map((t) => ({
      label: t.name, swatch: t.color, checked: true, onChange: (v) => { tms.setItemVisible(t, v); requestRender(); },
    }))]));
}

// Clipping
{
  const s = ui.section(panel, 'Cut-away', { hint: 'Clips every 3D layer except the MRI slices.' });
  let pos;
  const setRange = () => {
    if (clip.axis === 'off') { pos.setEnabled(false); return; }
    const i = 'xyz'.indexOf(clip.axis);
    pos.input.min = Math.ceil(bounds.lo[i]);
    pos.input.max = Math.floor(bounds.hi[i]);
    pos.set(clip.pos[clip.axis]);
    pos.setEnabled(true);
  };
  ui.select(s.body, 'Plane', [
    { value: 'off', label: 'Off' }, { value: 'x', label: 'Sagittal (left | right)' },
    { value: 'y', label: 'Coronal (back | front)' }, { value: 'z', label: 'Axial (bottom | top)' }], clip.axis,
  (v) => { clip.axis = v; setRange(); applyClip(); });
  pos = ui.slider(s.body, 'Position', {
    min: -100, max: 100, step: 1, value: 0, format: mm,
    onInput: (v) => { if (clip.axis !== 'off') { clip.pos[clip.axis] = v; applyClip(); } },
  });
  ui.checkbox(s.body, 'Keep the other side', clip.flip, (v) => { clip.flip = v; applyClip(); });
  setRange();
}

// View
{
  const s = ui.section(panel, 'View', { open: true });
  ui.buttons(s.body, Object.keys(VIEWS).map((v) => ({ label: v, onClick: () => setView(v) })));
  ui.checkbox(s.body, 'Auto-rotate', false, (v) => { controls.autoRotate = v; requestRender(); });
}

document.getElementById('panel-toggle').addEventListener('click', () => {
  document.body.classList.toggle('panel-hidden');
  resize();
});
const about = document.getElementById('about');
document.getElementById('about-open').addEventListener('click', () => about.showModal());
about.addEventListener('click', (e) => { if (e.target === about) about.close(); });

// ---------------------------------------------------------------------------- hover picking

const tip = document.getElementById('tip');
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pointerPx = null;
let pickQueued = false;
let dragging = false;
const brainInv = new THREE.Matrix4();

canvas.addEventListener('pointermove', (e) => {
  const r = canvas.getBoundingClientRect();
  pointerPx = [e.clientX - r.left, e.clientY - r.top];
  pointer.set((pointerPx[0] / r.width) * 2 - 1, -(pointerPx[1] / r.height) * 2 + 1);
  if (dragging) return;
  if (!pickQueued) { pickQueued = true; requestAnimationFrame(pick); }
});
canvas.addEventListener('pointerdown', () => { dragging = true; showTip(null); });
window.addEventListener('pointerup', () => { dragging = false; });
canvas.addEventListener('pointerleave', () => { pointerPx = null; showTip(null); });

function isSeeThrough(obj) {
  const u = obj.userData;
  if (u.kind === 'cortex') return cortex.style === 'xray' || cortex.opacity < 0.5;
  if (u.kind === 'structure') return u.group.style === 'xray' || u.group.opacity < 0.5;
  if (u.kind === 'tms') return Boolean(u.glass);
  return false;
}

function pick() {
  pickQueued = false;
  if (!pointerPx || dragging) return;
  raycaster.setFromCamera(pointer, camera);
  const objs = [];
  if (cortex.group.visible) {
    cortex.prepareRaycast();
    objs.push(...cortex.meshes.filter((m) => m.visible));
  }
  for (const g of structureGroups) objs.push(...g.pickables());
  objs.push(...tms.pickables());
  const sliceObjs = slices.pickables();
  objs.push(...sliceObjs);
  raycaster.firstHitOnly = false;
  const hits = raycaster.intersectObjects(objs, false).filter((h) => {
    if (h.object.userData.kind === 'slice') return true;
    return keepPoint(brain.worldToLocal(h.point.clone()));
  });
  let hit = hits.find((h) => !isSeeThrough(h.object)) || hits[0] || null;

  // tracts: nearest visible streamline point near the pointer ray
  brainInv.copy(brain.matrixWorld).invert();
  const ray = raycaster.ray.clone().applyMatrix4(brainInv);
  const pxMm = (t) => (2 * t * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / canvas.clientHeight;
  const tp = tracts.pick(ray, (t) => 5 * pxMm(t), (x, y, z) => keepPoint(new THREE.Vector3(x, y, z)));
  let info = null;
  let tractHover = null, structHover = null, cortexHl = null, tmsHover = null;
  if (tp && (!hit || tp.t < hit.distance + 0.5 || isSeeThrough(hit.object))) {
    tractHover = tp.item;
    info = { ...tracts.describe(tp.item), mni: tp.point };
    if (man.flow.notes[tp.item.id]) info.lines.push(man.flow.notes[tp.item.id]);
  } else if (hit) {
    const u = hit.object.userData;
    const local = brain.worldToLocal(hit.point.clone()).toArray();
    if (u.kind === 'cortex') {
      info = cortex.describe(hit);
      cortexHl = info.highlight || null;
    } else if (u.kind === 'structure') {
      info = { ...u.group.describe(hit), mni: local };
      structHover = info;
    } else if (u.kind === 'slice') {
      info = { ...slices.describe(local), mni: local };
    } else if (u.kind === 'tms') {
      info = tms.describe(hit);
      tmsHover = info.item;
    }
  }
  tracts.setHover(tractHover);
  tms.setHover(tmsHover);
  for (const g of structureGroups) g.setHover(structHover && structHover.group === g ? structHover.item : null);
  cortex.setHighlight(cortexHl);
  showTip(info);
  requestRender();
}

function showTip(info) {
  if (!info || !pointerPx) { tip.hidden = true; return; }
  tip.replaceChildren();
  const t = ui.el('div', 'tip-title');
  if (info.swatch) {
    const sw = ui.el('span', 'swatch');
    sw.style.background = info.swatch;
    t.append(sw);
  }
  t.append(info.title);
  tip.append(t);
  for (const l of info.lines || []) tip.append(ui.el('div', 'tip-line', l));
  if (info.mni) tip.append(ui.el('div', 'tip-coord', `MNI ${info.mni.map((v) => Math.round(v)).join(', ')}`));
  tip.hidden = false;
  const r = canvas.getBoundingClientRect();
  const x = Math.min(pointerPx[0] + r.left + 16, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(pointerPx[1] + r.top + 16, window.innerHeight - tip.offsetHeight - 8);
  tip.style.transform = `translate(${x}px, ${y}px)`;
}

// ---------------------------------------------------------------------------- labels

// Text labels drawn over the canvas at 3D points. A label hides when its point is cut away, faces away from
// the camera (labels with a normal) or sits behind solid cortex (labels marked occlude).
const labelLayer = document.getElementById('labels');
const labelEls = [];
const lp = new THREE.Vector3(), lw = new THREE.Vector3(), ln = new THREE.Vector3(), camLocal = new THREE.Vector3();
const occluder = new THREE.Raycaster();

function updateLabels() {
  const items = [...cortex.labelItems(), ...tms.labelItems()];
  const w = canvas.clientWidth, h = canvas.clientHeight;
  brain.worldToLocal(camLocal.copy(camera.position));
  const cortexHides = cortex.style !== 'xray' && cortex.opacity >= 0.5;
  let meshes = null;
  let n = 0;
  for (const it of items) {
    lp.fromArray(it.pos);
    if (!keepPoint(lp)) continue;
    if (it.normal && ln.fromArray(it.normal).dot(lw.copy(camLocal).sub(lp)) <= 0) continue;
    lw.copy(lp).applyMatrix4(brain.matrixWorld);
    if (it.occlude && cortexHides) {
      if (!meshes) { cortex.prepareRaycast(); meshes = cortex.meshes.filter((m) => m.visible); }
      const d = camera.position.distanceTo(lw);
      occluder.set(camera.position, lw.clone().sub(camera.position).divideScalar(d));
      occluder.far = d - 2;
      occluder.firstHitOnly = clip.axis === 'off';
      const hits = occluder.intersectObjects(meshes, false);
      if (hits.some((x) => keepPoint(brain.worldToLocal(x.point.clone())))) continue;
    }
    lw.project(camera);
    if (lw.z > 1) continue;
    let e = labelEls[n];
    if (!e) labelEls.push(e = labelLayer.appendChild(ui.el('div', 'label3d')));
    n++;
    if (e.textContent !== it.text) e.textContent = it.text;
    e.style.borderColor = it.color || '';
    e.style.transform = `translate(${(lw.x * 0.5 + 0.5) * w}px, ${(-lw.y * 0.5 + 0.5) * h}px) translate(-50%, -50%)`;
    e.hidden = false;
  }
  for (let i = n; i < labelEls.length; i++) labelEls[i].hidden = true;
}

// ---------------------------------------------------------------------------- loop

let lastFrame = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  const moved = controls.update();
  if (flow.group.visible) {
    if (flow.playing) { flow.step(dt); needsRender = true; }
    flow.sync();
  }
  if (moved || needsRender) {
    needsRender = false;
    renderer.render(scene, camera);
    updateLabels();
  }
});

window.brainviewer = {
  THREE, scene, camera, controls, renderer, brain, cortex, aseg, massp, thalamic, context, tracts, slices, volume, tms,
  flow,
  setView, requestRender, clip, applyClip,
};
