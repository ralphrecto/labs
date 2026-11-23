// Scripted playthrough for the feedback loop (docs/feedback-log.md). With `npm run dev` running:
//   node scripts/playthrough.cjs [outDir]
// Plays the first-run flow and a short furnishing session on a desktop and a phone viewport,
// saving numbered screenshots and a notes.json of console errors and step timings.
const { chromium, devices } = require('playwright');
const fs = require('fs');
const path = require('path');
const URL = process.env.URL || 'http://localhost:5173/';
const OUT = process.argv[2] || 'playthrough';
fs.mkdirSync(OUT, { recursive: true });
const notes = { errors: [], steps: [] };

async function run(label, ctxOpts, journey) {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const ctx = await b.newContext(ctxOpts);
  const p = await ctx.newPage();
  p.on('console', (m) => { if (m.type() === 'error') notes.errors.push(`[${label}] ${m.text()}`) });
  p.on('pageerror', (e) => notes.errors.push(`[${label}] PAGEERROR ${e.message}`));
  await p.goto(URL);
  await p.evaluate(() => { localStorage.clear(); indexedDB.deleteDatabase('apartment-designer-scan-files') });
  await p.reload();
  await p.waitForTimeout(1500);
  let n = 0;
  const shot = async (name) => { n++; const f = path.join(OUT, `${label}-${String(n).padStart(2, '0')}-${name}.png`); await p.screenshot({ path: f }); };
  const step = async (name, fn) => {
    const t = Date.now();
    try { await fn(); await p.waitForTimeout(700); await shot(name); notes.steps.push({ label, name, ok: true, ms: Date.now() - t }) }
    catch (e) { notes.steps.push({ label, name, ok: false, err: String(e.message).split('\n')[0] }); try { await shot(name + '-FAILED') } catch {} }
  };
  await journey(p, step);
  await b.close();
}

const clickText = (p, t) => p.getByRole('button', { name: t }).first().click({ timeout: 5000 });

async function scanJourney(p, step) {
  await step('welcome', async () => {});
  await step('scan-step', () => clickText(p, 'Scan my apartment'));
  await step('sample-imported', async () => { await clickText(p, /Try ours/i); await p.waitForTimeout(1500) });
  await step('sample-measured', async () => { await p.waitForTimeout(3500) });
  await step('detecting', async () => { await clickText(p, /^Next$|That’s enough|Skip for now/).catch(() => {}) });
  await step('detected', async () => { await p.waitForTimeout(6000) });
  await step('welcome-home', async () => { await p.locator('.onb-row.sticky .g-btn.primary').click({ timeout: 4000 }).catch(() => clickText(p, /Trace walls myself/)) });
  await step('start-designing', () => clickText(p, 'Start designing'));
  await play(p, step);
}

async function play(p, step) {
  await step('furnish-drawer', async () => {});
  const tabs = p.locator('.buy-tab');
  const nt = await tabs.count();
  for (let i = 1; i < Math.min(nt, 3); i++) await step(`drawer-tab-${i}`, () => tabs.nth(i).click());
  await step('place-item-1', () => p.locator('.buy-hit').first().click());
  const reopen = async () => { await p.getByRole('button', { name: 'Done' }).first().click({ timeout: 3000 }).catch(() => {}); await p.waitForTimeout(400) };
  await step('place-item-2', async () => { await p.keyboard.press('Escape'); await reopen(); await tabs.nth(0).click(); await p.locator('.buy-hit').nth(2).click() });
  await step('place-item-3', async () => { await p.keyboard.press('Escape'); await reopen(); await p.locator('.buy-hit').nth(3).click() });
  await step('click-canvas-select', async () => {
    const c = await p.locator('canvas').first().boundingBox();
    await p.mouse.click(c.x + c.width / 2, c.y + c.height / 2);
  });
  await step('goals', async () => { await p.keyboard.press('Escape'); await p.locator('[class*="goal"]').first().click({ timeout: 3000 }) });
  await step('goals-closed', async () => { await p.keyboard.press('Escape'); });
  await step('style-picker', async () => { await p.keyboard.press('Escape'); await p.locator('.sty-btn').first().click({ timeout: 3000 }) });
  for (const st of ['Lifelike', 'Clay', 'Scan', 'Cartoon']) await step(`style-${st.toLowerCase()}`, async () => { await p.waitForTimeout(600); if (!(await p.locator('.sty-pop').isVisible().catch(() => false))) { await p.locator('.sty-btn').first().click(); await p.waitForTimeout(400) } await p.locator('.sty-pop [role=menuitemradio], .sty-pop button').filter({ hasText: st }).first().click({ timeout: 3000 }); await p.waitForTimeout(800) });
  await step('build-mode', async () => { await p.keyboard.press('Escape'); await clickText(p, /^Build/) });
  await step('look-mode', () => clickText(p, /^Look/));
  await step('look-walk', async () => { await clickText(p, /walk|eye|inside/i) });
  await step('back-to-furnish', async () => { await p.keyboard.press('Escape'); await clickText(p, /^(Furnish|Buy)/) });
  await step('settings', () => p.getByRole('button', { name: 'Settings' }).click());
}

async function sampleJourney(p, step) {
  await step('welcome', async () => {});
  await step('sample-home', () => clickText(p, /sample home/i));
  await play(p, step);
}

(async () => {
  if (process.env.ONLY !== 'phone') await run('desktop', { viewport: { width: 1440, height: 900 } }, scanJourney);
  if (process.env.ONLY !== 'desktop') await run('phone', { ...devices['iPhone 14'], browserName: undefined, defaultBrowserType: undefined }, sampleJourney);
  fs.writeFileSync(path.join(OUT, 'notes.json'), JSON.stringify(notes, null, 2));
  console.log(JSON.stringify(notes.steps.filter((s) => !s.ok), null, 1));
  console.log('errors:', notes.errors.length, notes.errors.slice(0, 8).join('\n'));
})();
