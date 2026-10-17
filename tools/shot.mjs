// Headless check: load the viewer once, then for each "name::js" argument run the snippet against
// window.brainview (as B) and save out/<name>.png.   usage: node tools/shot.mjs "a::B.setView('Left')" ...
import puppeteer from 'puppeteer-core';
const W = +(process.env.W || 1440), H = +(process.env.H || 900);
const browser = await puppeteer.launch({
  executablePath: `${process.env.HOME}/.cache/puppeteer/chrome-headless-shell/linux-145.0.7632.77/chrome-headless-shell-linux64/chrome-headless-shell`,
  headless: 'shell',
  args: ['--no-sandbox', '--use-angle=gl-egl', '--ignore-gpu-blocklist', `--window-size=${W},${H}`],
});
const page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
page.on('console', (m) => { if (m.type() !== 'debug') console.log(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
const t0 = Date.now();
await page.goto(process.env.URL || 'http://127.0.0.1:8051/', { waitUntil: 'load' });
await page.waitForFunction(() => !document.getElementById('loading') && window.brainview, { timeout: 120000 });
console.log(`[time] loaded in ${Date.now() - t0} ms`);
for (const arg of process.argv.slice(2)) {
  const [name, snippet = '', mouse = ''] = arg.split('::');
  const t1 = Date.now();
  const r = await page.evaluate(`(async () => { const B = window.brainview; ${snippet}; B.requestRender(); })()`);
  if (r !== undefined) console.log(`[result] ${JSON.stringify(r)}`);
  await page.waitForFunction(() => document.getElementById('status').hidden, { timeout: 120000 });
  await new Promise((r) => setTimeout(r, 500));
  if (mouse) {
    const [x, y] = mouse.split(',').map(Number);
    await page.mouse.move(x - 5, y - 5);
    await page.mouse.move(x, y, { steps: 3 });
    await new Promise((r) => setTimeout(r, 400));
    console.log(`[tip] ${await page.$eval('#tip', (e) => (e.hidden ? '(hidden)' : e.innerText.replace(/\n/g, ' | ')))}`);
  }
  await page.screenshot({ path: `out/${name}.png` });
  console.log(`[shot] ${name} (${Date.now() - t1} ms)`);
}
await browser.close();
