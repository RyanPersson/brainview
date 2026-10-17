import * as THREE from 'three';

// One shared clipping-plane list: every clippable material references this array, so toggling the
// cut (push/pop a plane) updates all of them at once.
export const clipPlanes = [];

// "Glass brain" shading: opacity rises toward silhouettes (Fresnel-like), so nested structures show through.
export function makeXRayMaterial({ vertexColors = false, color = '#9fb7d9', opacity = 0.5, power = 2.0 } = {}) {
  const m = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: opacity },
      uPower: { value: power },
    },
    vertexShader: /* glsl */`
      #include <common>
      #include <clipping_planes_pars_vertex>
      varying vec3 vN;
      varying vec3 vV;
      varying vec3 vCol;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal);
        vV = normalize(-mvPosition.xyz);
        #ifdef USE_COLOR
          vCol = color;
        #else
          vCol = vec3(1.0);
        #endif
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <common>
      #include <clipping_planes_pars_fragment>
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uPower;
      varying vec3 vN;
      varying vec3 vV;
      varying vec3 vCol;
      void main() {
        #include <clipping_planes_fragment>
        float ndv = abs(dot(normalize(vN), normalize(vV)));
        float f = pow(1.0 - ndv, uPower);
        vec3 c = vCol * uColor;
        gl_FragColor = vec4(c * (0.55 + 0.6 * f), uOpacity * (0.06 + 0.94 * f));
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexColors,
    clipping: true,
  });
  m.clippingPlanes = clipPlanes;
  return m;
}

export function makeSolidMaterial(opts = {}) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.72, metalness: 0.0, ...opts });
  m.clippingPlanes = clipPlanes;
  return m;
}

// Apply an opacity to a standard material, switching transparency mode only when needed.
export function setOpacity(mat, opacity) {
  const transparent = opacity < 0.999;
  if (mat.transparent !== transparent) {
    mat.transparent = transparent;
    mat.depthWrite = !transparent;
    mat.needsUpdate = true;
  }
  mat.opacity = opacity;
}
