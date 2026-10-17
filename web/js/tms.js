import * as THREE from 'three';
import { clipPlanes, makeSolidMaterial, makeXRayMaterial } from './materials.js';

const Z = new THREE.Vector3(0, 0, 1);

// Published TMS targets: a dot on the cortex, a ring on the scalp where the coil sits, and a line joining them.
// Deep-TMS targets also get a glass sphere, because those coils reach a broad area rather than a spot.
export class TmsTargets {
  constructor(desc, parent) {
    this.desc = desc;
    this.group = new THREE.Group();
    this.group.visible = false;
    parent.add(this.group);
    this.showLabels = true;
    this.hovered = null;
    this.items = desc.targets.map((t) => ({ ...t, on: true, meshes: [] }));
    for (const it of this.items) {
      const color = new THREE.Color(it.color);
      it.solid = makeSolidMaterial({ color });
      const flat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, clippingPlanes: clipPlanes });
      const lineMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8, clippingPlanes: clipPlanes });
      it.glass = it.broad ? makeXRayMaterial({ color: it.color, opacity: 0.7, power: 1.4 }) : null;
      it.objects = [];
      for (const p of it.points) {
        const c = new THREE.Vector3(...p.cortex), s = new THREE.Vector3(...p.scalp), n = new THREE.Vector3(...p.normal);
        const dot = new THREE.Mesh(new THREE.SphereGeometry(it.broad ? 2.5 : 3.2, 24, 16), it.solid);
        dot.position.copy(c);
        const ring = new THREE.Mesh(new THREE.RingGeometry(7, 10, 48), flat);
        ring.position.copy(s).addScaledVector(n, 1.5);
        ring.quaternion.setFromUnitVectors(Z, n);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([c, s]), lineMat);
        const objs = [dot, ring, line];
        if (it.glass) {
          const area = new THREE.Mesh(new THREE.SphereGeometry(12, 32, 20), it.glass);
          area.position.copy(c);
          area.renderOrder = 6;
          area.userData = { kind: 'tms', item: it, glass: true };
          objs.push(area);
          it.meshes.push(area);
        }
        for (const m of [dot, ring]) {
          m.userData = { kind: 'tms', item: it };
          it.meshes.push(m);
        }
        for (const o of objs) this.group.add(o);
        it.objects.push(...objs);
      }
    }
  }

  setVisible(v) { this.group.visible = v; }

  setItemVisible(item, v) {
    item.on = v;
    for (const o of item.objects) o.visible = v;
  }

  setHover(item) {
    if (this.hovered === item) return;
    if (this.hovered) this.hovered.solid.emissive.setRGB(0, 0, 0);
    this.hovered = item;
    if (item) item.solid.emissive.copy(item.solid.color).multiplyScalar(0.45);
  }

  pickables() {
    if (!this.group.visible) return [];
    return this.items.filter((t) => t.on).flatMap((t) => t.meshes);
  }

  describe(hit) {
    const t = hit.object.userData.item;
    return {
      title: t.name, swatch: t.color, item: t,
      lines: [`${t.group} · ${t.coil}`, t.use, t.how, `Source: ${t.source}`],
      mni: t.mni || t.points[0].cortex,
    };
  }

  // Labels float just outside the scalp ring and show while that side of the head faces the camera.
  labelItems() {
    if (!this.group.visible || !this.showLabels) return [];
    return this.items.filter((t) => t.on).flatMap((t) => t.points.map((p) => ({
      text: t.name, color: t.color,
      pos: p.scalp.map((v, i) => v + p.normal[i] * 16), normal: p.normal,
    })));
  }
}
