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
  const flakes       = [];
  const mirrorFlakes = [];
  let snowCanvas, ctx;

  /* fadeTail: fraction of region height where flakes fade to transparent.
     They are NEVER rendered below the region bottom. */
  const FADE_TAIL = 0.09;

  /* ── Occlusion masks ───────────────────────────────────────────────────── */
  /* Built from the bg image once at init. 4× downsampled binary lookup:     */
  /*   1 = sky/glass (draw snow), 0 = occluder (skip — flake keeps falling). */
  let _snowMask   = null;
  let _mirrorMask = null;
  const _MASK_DS  = 4;   /* downsample factor */

  function _buildRegionMask(bgImg, region) {
    if (!bgImg || !bgImg.complete || !bgImg.naturalWidth || !region) return null;
    const imgW = bgImg.naturalWidth, imgH = bgImg.naturalHeight;
    const scale  = Math.max(W / imgW, H / imgH);   /* object-fit: cover */
    const offX   = (W - imgW * scale) / 2;
    const offY   = (H - imgH * scale) / 2;
    const rx = region.x * W, ry = region.y * H;
    const rw = region.w * W, rh = region.h * H;
    const mw = Math.max(1, Math.ceil(rw / _MASK_DS));
    const mh = Math.max(1, Math.ceil(rh / _MASK_DS));
    const oc  = document.createElement('canvas');
    oc.width  = mw; oc.height = mh;
    const oc2 = oc.getContext('2d');
    oc2.drawImage(bgImg,
      Math.max(0, (rx - offX) / scale), Math.max(0, (ry - offY) / scale),
      rw / scale, rh / scale,
      0, 0, mw, mh);
    const pxData = oc2.getImageData(0, 0, mw, mh).data;
    const mask   = new Uint8Array(mw * mh);
    const wbT    = cfg.maskWbThresh !== undefined ? cfg.maskWbThresh : 20;
    const lumLo  = cfg.maskLumMin   !== undefined ? cfg.maskLumMin   : 35;
    const lumHi  = cfg.maskLumMax   !== undefined ? cfg.maskLumMax   : 255;
    for (let mi = 0; mi < mw * mh; mi++) {
      const rr = pxData[mi*4], gg = pxData[mi*4+1], bb = pxData[mi*4+2];
      const lum = (rr*299 + gg*587 + bb*114) / 1000;
      const wb  = rr - bb;
      mask[mi]  = (wb < wbT && lum > lumLo && lum < lumHi) ? 1 : 0;
    }
    return { data: mask, w: mw, h: mh, rx, ry };
  }

  function _maskAllows(mask, fx, fy) {
    if (!mask) return true;
    const mx = Math.floor((fx - mask.rx) / _MASK_DS);
    const my = Math.floor((fy - mask.ry) / _MASK_DS);
    if (mx < 0 || mx >= mask.w || my < 0 || my >= mask.h) return true;
    return mask.data[my * mask.w + mx] === 1;
  }

  function _initMasks() {
    const bgEl = document.querySelector('img.bg');
    if (!bgEl) return;
    function build() {
      _snowMask   = _buildRegionMask(bgEl, cfg.snowRegion);
      _mirrorMask = cfg.mirrorRegion
                    ? _buildRegionMask(bgEl, cfg.mirrorRegion)
                    : null;
    }
    if (bgEl.complete && bgEl.naturalWidth > 0) { build(); }
    else { bgEl.addEventListener('load', build, { once: true }); }
  }

  function _spawnInRegion(r, scatter, opScale, sizeOff) {
    const x0 = r.x * W, y0 = r.y * H, rw = r.w * W, rh = r.h * H;
    const sMin = Math.max(1, cfg.snowSizeMin + (sizeOff || 0));
    const sMax = Math.max(sMin, cfg.snowSizeMax + (sizeOff || 0));
    return {
      x:     x0 + Math.random() * rw,
      y:     scatter ? y0 + Math.random() * rh : y0 - 6,
      rad:   sMin + Math.random() * (sMax - sMin),
      vy:    cfg.snowSpeedMin + Math.random() * (cfg.snowSpeedMax - cfg.snowSpeedMin),
      op:    (cfg.snowOpMin + Math.random() * (cfg.snowOpMax - cfg.snowOpMin)) * (opScale || 1),
      phase: Math.random() * Math.PI * 2,
    };
  }

  function spawnFlake(scatter)       { return _spawnInRegion(cfg.snowRegion,   scatter, 1,    0);   }
  function spawnMirrorFlake(scatter) { return _spawnInRegion(cfg.mirrorRegion, scatter, 0.55, -1);  }

  function initSnow() {
    snowCanvas = document.getElementById('snow-canvas');
    ctx = snowCanvas.getContext('2d');
    flakes.length = 0;
    for (let i = 0; i < cfg.snowCount; i++) flakes.push(spawnFlake(true));
  }

  function initMirrorSnow() {
    if (!cfg.mirrorRegion) return;
    mirrorFlakes.length = 0;
    const n = cfg.mirrorSnowCount != null ? cfg.mirrorSnowCount
                                           : Math.round(cfg.snowCount * 0.22);
    for (let i = 0; i < n; i++) mirrorFlakes.push(spawnMirrorFlake(true));
  }

  /* Draw one pool of flakes. driftSign = +1 for main, -1 for mirror.
   * mask (optional): occlusion lookup — null = no masking. */
  function _stepFlakePool(pool, r, t, driftSign, spawnFn, mask) {
    const x0 = r.x * W, y0 = r.y * H, rw = r.w * W, rh = r.h * H;
    const fadeStart = y0 + rh * (1 - FADE_TAIL);
    for (const f of pool) {
      f.y += f.vy;
      f.x += driftSign * cfg.snowDrift * Math.sin(t * 0.00055 + f.phase);
      /* Hard wrap at region bottom — never render past it */
      if (f.y >= y0 + rh) {
        const next = spawnFn(false);
        next.x = x0 + Math.random() * rw;
        Object.assign(f, next);
        continue;
      }
      /* Clamp horizontal drift to region */
      f.x = Math.max(x0 + f.rad, Math.min(x0 + rw - f.rad, f.x));
      /* Occlusion mask: skip draw if this pixel is a frame / divider / plant.
       * The flake keeps falling and reappears below the occluder naturally. */
      if (!_maskAllows(mask, f.x, f.y)) continue;
      /* Fade opacity in the bottom tail so flakes dissolve before the sill */
      let alpha = f.op;
      if (f.y > fadeStart) {
        alpha *= Math.max(0, (y0 + rh - f.y) / (rh * FADE_TAIL));
      }
      if (alpha < 0.01) continue;
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.rad, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,' + alpha.toFixed(2) + ')';
      ctx.fill();
    }
  }

  function stepSnow(t) {
    ctx.clearRect(0, 0, W, H);
    _stepFlakePool(flakes, cfg.snowRegion, t, +1, spawnFlake, _snowMask);
    if (cfg.mirrorRegion && mirrorFlakes.length > 0) {
      _stepFlakePool(mirrorFlakes, cfg.mirrorRegion, t, -1, spawnMirrorFlake, _mirrorMask);
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
    /* No-op if this scene has no banner card (e.g. kira_cam.html). */
    if (!s) return;

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
    initMirrorSnow();
    initCRT();
    initLights();
    initBanner();
    _initMasks();
    if (typeof window.pageInit === 'function') window.pageInit();
    _maybeConnectScreenWs();
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

  /* ── Live text control via dashboard WebSocket ─────────────────────────── */
  /*
   * Connects to /ws/screens on the control server (port 8766).
   * Requires CFG.screenName to be set ("starting" | "brb" | "ending").
   * Reconnects automatically with exponential backoff — OBS browser sources
   * are long-lived and must survive control server restarts.
   *
   * Public API exposed to page scripts:
   *   window.applyOverride(line1, line2) — called by anim.js when an override arrives
   *   window.clearOverride()             — called by anim.js when override is cleared
   *
   * Hooks pages can define:
   *   window.onOverride()    — called before the fade-out (pause timers etc.)
   *   window.restoreDefaults() — called during fade-in after clear (resume timers etc.)
   */
  let _overrideActive = false;
  let _screenWs       = null;
  let _wsRetryMs      = 1500;
  const _wsRetryMax   = 20000;
  const _CTRL_WS      = 'ws://127.0.0.1:8766/ws/screens';

  function _connectScreenWs() {
    try {
      _screenWs = new WebSocket(_CTRL_WS);
    } catch (e) {
      _scheduleScreenReconnect();
      return;
    }

    _screenWs.onopen = function () {
      _wsRetryMs = 1500; /* reset backoff on successful connect */
    };

    _screenWs.onmessage = function (e) {
      try { _handleScreenMsg(JSON.parse(e.data)); } catch (_) {}
    };

    _screenWs.onerror = function () {};

    _screenWs.onclose = function () {
      _scheduleScreenReconnect();
    };
  }

  function _scheduleScreenReconnect() {
    setTimeout(_connectScreenWs, _wsRetryMs);
    _wsRetryMs = Math.min(_wsRetryMs * 2, _wsRetryMax);
  }

  function _handleScreenMsg(msg) {
    var myScreen = cfg && cfg.screenName;
    if (!myScreen) return;

    if (msg.type === 'overrides' && msg.data) {
      /* Sent immediately on connect — restore previous override if any */
      var ov = msg.data[myScreen];
      if (ov) {
        _applyTextOverride(ov.line1 || '', ov.line2 || '');
      }
      /* else: no override stored → keep default behavior running */
      return;
    }

    if (msg.type === 'screen_text' && msg.screen === myScreen) {
      _applyTextOverride(msg.line1 || '', msg.line2 || '');
      return;
    }

    if (msg.type === 'screen_text_clear' && msg.screen === myScreen) {
      _clearTextOverride();
    }
  }

  /* Fade the banner-text container out, swap text, fade back in */
  function _fadeSwap(fn) {
    var el = document.getElementById('banner-text');
    if (!el) { fn(); return; }
    el.style.opacity = '0';
    setTimeout(function () {
      fn();
      el.style.opacity = '1';
    }, 320);
  }

  function _applyTextOverride(line1, line2) {
    /* Signal the page to pause any timer-driven text updates */
    if (!_overrideActive && typeof window.onOverride === 'function') {
      window.onOverride();
    }
    _overrideActive = true;

    _fadeSwap(function () {
      var container = document.getElementById('banner-text');
      if (!container) return;
      var mainEl = container.querySelector('.banner-main');
      var subEl  = container.querySelector('.banner-sub');
      if (mainEl && line1 !== '') mainEl.textContent = line1;
      if (subEl  && line2 !== '') subEl.textContent  = line2;
    });
  }

  function _clearTextOverride() {
    if (!_overrideActive) return;
    _overrideActive = false;
    _fadeSwap(function () {
      /* Let the page restore its own default text + resume timers */
      if (typeof window.restoreDefaults === 'function') window.restoreDefaults();
    });
  }

  /* Start the screen WS after the scene is up (requires cfg.screenName) */
  function _maybeConnectScreenWs() {
    if (cfg && cfg.screenName) _connectScreenWs();
  }

}());
