// End-to-end check of the scan workflow against the dev server (npm run dev):
// import the sample scan, mark three tape lengths, trace the room, reload from IndexedDB.
// Prints the recovered scale and traced wall lengths; the sample's truth is 1.0417 and
// 5.20, 2.60, 1.80, 1.50, 3.40, 4.10 m. Screenshots go to ./scan-check-*.png.
const { chromium } = require('playwright');
const URL = process.env.URL || 'http://localhost:5173/';
const shot = (name) => `scan-check-${name}.png`;
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1500, height: 900 } });
  const logs = [];
  p.on('console', m => { if (m.type() === 'error') logs.push(m.text()) });
  p.on('pageerror', e => logs.push('PAGEERROR ' + e.message));
  await p.goto(URL);
  await p.evaluate(() => localStorage.clear());
  await p.reload();
  await p.waitForFunction(() => window.__apt);
  // Start from an empty apartment, as with a first real scan.
  await p.click('summary:has-text("File")');
  await p.getByRole('button', { name: 'New empty apartment' }).click();
  await p.click('summary:has-text("File")');
  await p.evaluate(() => window.__apt.useStore.getState().setView('plan'));
  await p.getByRole('button', { name: 'Try a sample' }).click();
  await p.waitForFunction(() => (window.__apt.useStore.getState().doc.scans || []).length === 1, null, { timeout: 30000 });
  await p.waitForTimeout(3000);
  await p.screenshot({ path: shot('1-import') });
  const info = await p.evaluate(() => { const r = window.__apt.useStore.getState().doc.scans[0]; return { up: r.upAxis, floor: r.floor, ceiling: r.ceiling, yawDeg: r.yaw * 180 / Math.PI, x: r.x, y: r.y } });
  console.log(JSON.stringify(info));

  // Map a true room point (meters) to a screen point via the scan's alignment and the SVG's CTM.
  const screenOf = (pt) => p.evaluate((pt) => {
    const { useStore, align, sample } = window.__apt;
    const ref = useStore.getState().doc.scans[0];
    const q = align.scanToPlan(ref, sample.sampleFilePoint(pt));
    const svg = document.querySelector('svg.plan');
    const m = svg.getScreenCTM();
    return { x: m.a * q.x + m.c * q.y + m.e, y: m.b * q.x + m.d * q.y + m.f };
  }, pt);

  // Measure the three sample references.
  const refs = [[{x:0.02,y:0.02},{x:5.18,y:0.03},5.2],[{x:0.03,y:0.02},{x:0.02,y:4.08},4.1],[{x:3.42,y:2.62},{x:3.41,y:4.08},1.5]];
  await p.getByRole('button', { name: 'Mark a measured length' }).click();
  for (let i = 0; i < refs.length; i++) {
    const [a, bb, len] = refs[i];

    const sa = await screenOf(a); const sb = await screenOf(bb);
    await p.mouse.click(sa.x, sa.y); await p.mouse.click(sb.x, sb.y);
    await p.fill('#ref-length', String(len * 100));
    await p.getByRole('button', { name: 'Use this length' }).click();
  }
  await p.keyboard.press('Escape');
  const cal = await p.evaluate(() => { const r = window.__apt.useStore.getState().doc.scans[0]; return { scale: r.scale, truth: window.__apt.sample.SAMPLE_TRUTH.scale, refs: r.references.length } });
  console.log('calibration', JSON.stringify(cal));
  await p.screenshot({ path: shot('2-calibrated') });

  // Trace the L-shaped room by clicking near (but not on) the corners.
  await p.getByRole('button', { name: 'Trace a room' }).click();
  const outline = [[0.03,0.04],[5.17,0.03],[5.16,2.63],[3.37,2.58],[3.43,4.07],[0.04,4.12]];
  for (const [x,y] of outline) { const s = await screenOf({x,y}); await p.mouse.click(s.x, s.y); }
  const s0 = await screenOf({x:0,y:0}); await p.mouse.click(s0.x, s0.y);
  await p.waitForTimeout(500);
  const room = await p.evaluate(() => { const r = window.__apt.useStore.getState().doc.rooms.at(-1); if (!r) return null; const P = r.polygon; return { ceiling: r.ceilingHeight, walls: P.map((a, i) => { const b = P[(i+1)%P.length]; return Math.hypot(b.x-a.x, b.y-a.y).toFixed(3) }) } });
  console.log('room', JSON.stringify(room));
  await p.screenshot({ path: shot('3-traced') });
  await p.click('.left .layout-name:has-text("Sample living room")');
  await p.getByRole('button', { name: 'Top view' }).click();
  await p.waitForTimeout(800);

  await p.screenshot({ path: shot('6-top') });
  // Reload: the scan should come back from IndexedDB.
  await p.reload();
  await p.waitForFunction(() => window.__apt && window.__apt.runtime.useScanRuntime.getState().entries[window.__apt.useStore.getState().doc.scans[0].id]?.status === 'ready', null, { timeout: 30000 });
  console.log('reloaded ok');
  console.log(logs.slice(0, 20).join('\n'));
  await b.close();
})();
