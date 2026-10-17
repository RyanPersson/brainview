// Small colormap helpers (sRGB, 0..1).

const MAPS = {
  // diverging: blue (negative) - light grey - orange/red (positive)
  diverging: [
    [0.0, [0.16, 0.32, 0.70]], [0.25, [0.42, 0.62, 0.86]], [0.5, [0.88, 0.88, 0.86]],
    [0.75, [0.93, 0.62, 0.38]], [1.0, [0.72, 0.18, 0.12]],
  ],
  // perceptually ordered sequential (inferno-like)
  sequential: [
    [0.0, [0.05, 0.03, 0.18]], [0.2, [0.30, 0.06, 0.42]], [0.4, [0.57, 0.15, 0.40]],
    [0.6, [0.83, 0.30, 0.26]], [0.8, [0.97, 0.58, 0.10]], [1.0, [0.99, 0.95, 0.60]],
  ],
};

export function makeLUT(name, n = 256) {
  const stops = MAPS[name];
  const lut = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1];
    const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return lut;
}

export function cssGradient(name) {
  return `linear-gradient(90deg, ${MAPS[name].map(([t, c]) =>
    `rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${t * 100}%`).join(', ')})`;
}
