/*!
 * Kira stream-screen animation engine   v1.0
 *
 * Usage: define CFG on the page, then call  initScene(CFG)
 *        For page-specific extras, define window.pageInit() before calling initScene().
 *
 * CFG contract (all regions as fractions of 1920 × 1080):
 *   snowRegion     { x, y, w, h }   — rectangle snow particles are confined to
 *   crtRegion      { x, y, w, h }   — CRT monitor screen overlay
 *   lightPoints    [[x,y], ...]      — string-light bulb centres
 *   snowCount      number            — particle count
 *   snowSpeedMin/Max, snowSizeMin/Max, snowOpMin/Max, snowDrift
 *   crtBaseGlow    0–1               — base opacity of CRT overlay
 *   crtAmp         0–1               — sine-wave amplitude on top of base
 *   crtPeriod      ms                — full cycle duration
 *   lightRadius    px                — radius of each glow dot
 *   lightPeriodMin/Max  ms           — twinkle cycle range (per-dot randomised)
 *   cardSize       px                — PNG rendered as cardSize × cardSize square
 *   cardLeft/Top   px                — top-left of PNG element on 1920×1080 canvas
 *   textSafeTop/Right/Bottom/Left   — fractions of cardSize (from cards.json)
 */
(function () {
  'use strict';

  const W = 1920, H = 1080;
  let cfg;

  /* ── Snow ──────────────────────────────────────────────────────────────── */
  const flakes = [];
  let snowCanvas, ctx;

  function spawnFlake(scatter) {
    const r  = cfg.snowRegion;
    const x0 = r.x * W, y0 = r.y * H, rw = r.w * W, rh = r.h * H;
    return {
      x:     x0 + Math.random() * rw,
      y:     scatter ? y0 + Math.random() * rh : y0 - 8,
      rad:   cfg.snowSizeMin + Math.random() * (cfg.snowSizeMax - cfg.snowSizeMin),
      vy:    cfg.snowSpeedMin + Math.random() * (cfg.snowSpeedMax - cfg.snowSpeedMin),
      op:    cfg.snowOpMin + Math.random() * (cfg.snowOpMax - cfg.snowOpMin),
      phase: Math.random() * Math.PI * 2,
    };
  }

  function initSnow() {
    snowCanvas = document.getElementById('snow-canvas');
    ctx = snowCanvas.getContext('2d');
    flakes.length = 0;
    for (let i = 0; i < cfg.snowCount; i++) flakes.push(spawnFlake(true));
  }

  function stepSnow(t) {
    const r  = cfg.snowRegion;
    const x0 = r.x * W, y0 = r.y * H, rw = r.w * W, rh = r.h * H;
    ctx.clearRect(0, 0, W, H);
    for (const f of flakes) {
      f.y += f.vy;
      f.x += cfg.snowDrift * Math.sin(t * 0.00055 + f.phase);
      if (f.y > y0 + rh + f.rad) {
        const next = spawnFlake(false);
        next.x = x0 + Math.random() * rw;
        Object.assign(f, next);
      }
      /* Clamp horizontal drift to window region */
      f.x = Math.max(x0 + f.rad, Math.min(x0 + rw - f.rad, f.x));
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.rad, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,' + f.op.toFixed(2) + ')';
      ctx.fill();
    }
  }

  /* ── CRT flicker ───────────────────────────────────────────────────────── */
  let crtEl;

  function initCRT() {
    crtEl = document.getElementById('crt-overlay');
    const r = cfg.crtRegion;
    crtEl.style.left   = (r.x * W).toFixed(1) + 'px';
    crtEl.style.top    = (r.y * H).toFixed(1) + 'px';
    crtEl.style.width  = (r.w * W).toFixed(1) + 'px';
    crtEl.style.height = (r.h * H).toFixed(1) + 'px';
  }

  function stepCRT(t) {
    const op = cfg.crtBaseGlow + cfg.crtAmp * Math.sin((t / cfg.crtPeriod) * Math.PI * 2);
    crtEl.style.opacity = Math.max(0, op).toFixed(3);
  }

  /* ── String lights ─────────────────────────────────────────────────────── */
  const lights = [];

  function initLights() {
    const layer = document.getElementById('lights-layer');
    layer.innerHTML = '';
    lights.length = 0;
    const d = cfg.lightRadius * 2;
    for (const [xf, yf] of cfg.lightPoints) {
      const el = document.createElement('div');
      el.className = 'light-dot';
      el.style.left   = (xf * W).toFixed(1) + 'px';
      el.style.top    = (yf * H).toFixed(1) + 'px';
      el.style.width  = d + 'px';
      el.style.height = d + 'px';
      layer.appendChild(el);
      lights.push({
        el,
        period: cfg.lightPeriodMin + Math.random() * (cfg.lightPeriodMax - cfg.lightPeriodMin),
        phase:  Math.random() * Math.PI * 2,
      });
    }
  }

  function stepLights(t) {
    for (const ld of lights) {
      /* Smooth sine: ranges 0.25 → 0.95 */
      const op = 0.25 + 0.70 * (0.5 + 0.5 * Math.sin((t / ld.period) * Math.PI * 2 + ld.phase));
      ld.el.style.opacity = op.toFixed(3);
    }
  }

  /* ── Banner card ───────────────────────────────────────────────────────── */
  function initBanner() {
    const c  = cfg;
    const s  = c.cardSize;

    const imgEl = document.getElementById('banner-card-img');
    if (imgEl) {
      imgEl.style.width  = s + 'px';
      imgEl.style.height = s + 'px';
      imgEl.style.left   = c.cardLeft + 'px';
      imgEl.style.top    = c.cardTop  + 'px';
    }

    const textEl = document.getElementById('banner-text');
    if (textEl) {
      const tl = c.cardLeft + c.textSafeLeft          * s;
      const tt = c.cardTop  + c.textSafeTop            * s;
      const tr = c.cardLeft + (1 - c.textSafeRight)   * s;
      const tb = c.cardTop  + (1 - c.textSafeBottom)  * s;
      textEl.style.left   = tl.toFixed(0) + 'px';
      textEl.style.top    = tt.toFixed(0) + 'px';
      textEl.style.width  = (tr - tl).toFixed(0) + 'px';
      textEl.style.height = (tb - tt).toFixed(0) + 'px';
    }
  }

  /* ── RAF loop ──────────────────────────────────────────────────────────── */
  function loop(t) {
    stepSnow(t);
    stepCRT(t);
    stepLights(t);
    requestAnimationFrame(loop);
  }

  function run() {
    initSnow();
    initCRT();
    initLights();
    initBanner();
    if (typeof window.pageInit === 'function') window.pageInit();
    requestAnimationFrame(loop);
  }

  /* Public entry point — safe to call before or after DOMContentLoaded */
  window.initScene = function (config) {
    cfg = config;
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', run);
    } else {
      run();
    }
  };

}());
