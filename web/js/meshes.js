import * as THREE from 'three';
import { fetchArray, fetchPositions } from './data.js';
import { makeSolidMaterial, makeXRayMaterial, setOpacity } from './materials.js';

// A set of labelled surface meshes (aseg structures, nuclei, hulls) loaded from one binary pair.
export class MeshGroup {
  constructor(desc, parent, { style = 'solid', opacity = 1 } = {}) {
    this.desc = desc;
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    this.items = desc.structures.map((s) => ({ ...s, on: s.visible !== false, mesh: null }));
    this.style = style;
    this.opacity = opacity;
    this.loading = null;
  }

  get loaded() { return !!this.items[0]?.mesh; }

  load() {
    if (!this.loading) this.loading = this._load();
    return this.loading;
  }

  async _load() {
    const [P, I] = await Promise.all([fetchPositions(this.desc.positions), fetchArray(this.desc.indices)]);
    for (const s of this.items) {
      const pos = P.slice(s.vStart * 3, (s.vStart + s.vCount) * 3);
      const src = I.subarray(s.iStart, s.iStart + s.iCount);
      const idx = s.vCount < 65536 ? new Uint16Array(src.length) : new Uint32Array(src.length);
      for (let i = 0; i < src.length; i++) idx[i] = src[i] - s.vStart;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
      geo.computeVertexNormals();
      geo.computeBoundingSphere();
      s.solid = makeSolidMaterial({ color: s.color });
      s.xray = makeXRayMaterial({ color: s.color, opacity: 0.6 });
      s.mesh = new THREE.Mesh(geo, s.solid);
      s.mesh.userData = { kind: 'structure', group: this, item: s };
      s.mesh.visible = s.on;
      this.group.add(s.mesh);
    }
    this.setStyle(this.style);
  }

  setVisible(v) {
    this.group.visible = v;
    if (v) this.load();
  }

  setItemVisible(item, v) {
    item.on = v;
    if (item.mesh) item.mesh.visible = v;
  }

  setStyle(style) {
    this.style = style;
    for (const s of this.items) if (s.mesh) s.mesh.material = style === 'xray' ? s.xray : s.solid;
    this.setOpacity(this.opacity);
  }

  setOpacity(o) {
    this.opacity = o;
    for (const s of this.items) {
      if (!s.mesh) continue;
      setOpacity(s.solid, o);
      s.xray.uniforms.uOpacity.value = 0.75 * o;
      s.mesh.renderOrder = this.style === 'xray' || o < 1 ? 5 : 0;
    }
  }

  setHover(item) {
    if (this.hovered === item) return;
    if (this.hovered?.solid) this.hovered.solid.emissive.setRGB(0, 0, 0);
    this.hovered = item;
    if (item?.solid) item.solid.emissive.copy(item.solid.color).multiplyScalar(0.45);
  }

  pickables() {
    if (!this.group.visible) return [];
    return this.items.filter((s) => s.mesh && s.on).map((s) => s.mesh);
  }

  describe(hit) {
    const s = hit.object.userData.item;
    const side = s.hemi === 'L' ? 'Left ' : s.hemi === 'R' ? 'Right ' : '';
    const lines = [this.desc.label + (s.category && s.category !== this.desc.label ? ` · ${s.category}` : '')];
    lines.push(`${s.voxels.toLocaleString()} mm³ in the template`);
    const name = side && /^[A-Z][a-z]/.test(s.name) ? s.name[0].toLowerCase() + s.name.slice(1) : s.name;
    return { title: side + name, lines, swatch: s.color, item: s, group: this };
  }
}
