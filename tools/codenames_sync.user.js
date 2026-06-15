// ==UserScript==
// @name         Kira Codenames Sync (codenames.game → CodenamesState)
// @namespace    https://github.com/JonnyD/NeuroAI_Bot
// @version      1.0.0
// @description  Reads the REAL board (25 words + true colors + reveals + role) straight out of the codenames.game DOM and pushes it live into Kira's CodenamesState tracker on the local control server. This is ground-truth game DATA, not vision pixels.
// @author       Kira
// @match        https://codenames.game/*
// @match        https://www.codenames.game/*
// @run-at       document-idle
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==
//
// ─────────────────────────────────────────────────────────────────────────────
// HOW IT WORKS
// ─────────────────────────────────────────────────────────────────────────────
// codenames.game is a single-page app that renders every card as an <article>
// whose inline style carries the card's TRUE identity as a CSS variable, e.g.
//     --CardColor: var(--neutral-cardBg)
// The spymaster client receives the full colour key, so for a spymaster ALL 25
// cards expose their real colour in the DOM. A guesser only sees colour on cards
// that have already been flipped (revealed) — exactly what we want.
//
// This script:
//   1. Reads the 25 words + colour tokens (blue/red/neutral/black/green).
//   2. Detects whether YOU are the spymaster (full key visible) or a guesser.
//   3. Detects YOUR team colour (so blue/red map to team/opponent correctly).
//   4. Maps colours → CodenamesState identities (team/opponent/neutral/assassin).
//   5. POSTs the board to the local control server and keeps it LIVE via a
//      MutationObserver — re-syncing automatically as cards are revealed and on
//      every new game, so you never re-run anything by hand.
//
// Requires Tampermonkey / Violentmonkey (uses GM_xmlhttpRequest to reach the
// local server without tripping CORS). The manual paste/click panel in the Kira
// dashboard stays as the fallback for any site this script doesn't cover.
// ─────────────────────────────────────────────────────────────────────────────

(function () {
  'use strict';

  // ── Config ────────────────────────────────────────────────────────────────
  const CONFIG = {
    API_BASE: 'http://127.0.0.1:8766', // Kira control server
    MY_TEAM: 'auto',                   // 'auto' | 'blue' | 'red' (override if auto-detect ever picks wrong)
    DEBOUNCE_MS: 350,                  // settle time after DOM mutations before a sync
    POLL_MS: 2000,                     // safety re-scan even if the observer misses a change
    SCRAPE_CLUES: true,                // best-effort: push the active clue word + number
    SHOW_BADGE: true,                  // tiny on-screen status pill (bottom-left)
    LOG: true,                         // console logging
  };

  const ROOM_RE = /^\/r\//;            // codenames.game room URLs are /r/<id>
  const TAG = '[KiraCN]';

  // ── Sync state (diffing so we don't spam the server) ────────────────────────
  let lastWordsKey = '';               // sorted word set → detects a brand-new game
  let lastRole = '';
  let lastTeam = '';
  let lastClueKey = '';
  const lastIdentity = new Map();      // word → identity last sent
  let teamWarned = false;
  let debounceTimer = null;

  // ── Logging ─────────────────────────────────────────────────────────────────
  const log = (...a) => { if (CONFIG.LOG) console.log(TAG, ...a); };
  const warn = (...a) => { if (CONFIG.LOG) console.warn(TAG, ...a); };

  // ── HTTP (GM_xmlhttpRequest bypasses CORS; fetch is a graceful fallback) ─────
  function post(action, body) {
    const url = `${CONFIG.API_BASE}/cmd/${action}`;
    const data = JSON.stringify(body || {});
    return new Promise((resolve) => {
      if (typeof GM_xmlhttpRequest === 'function') {
        GM_xmlhttpRequest({
          method: 'POST',
          url,
          headers: { 'Content-Type': 'application/json' },
          data,
          timeout: 4000,
          onload: (r) => resolve(safeJson(r.responseText)),
          onerror: () => resolve(null),
          ontimeout: () => resolve(null),
        });
      } else {
        fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: data })
          .then((r) => r.json()).then(resolve).catch(() => resolve(null));
      }
    });
  }
  function safeJson(t) { try { return JSON.parse(t); } catch (e) { return null; } }

  // ── Board reading ───────────────────────────────────────────────────────────

  // Pull a single card's word. Front and back faces both render the word, so the
  // article's textContent is doubled ("ATTICATTIC"). Collecting the UNIQUE leaf
  // texts collapses that safely — and, unlike halving the string, it does not
  // corrupt genuinely doubled words like "MURMUR".
  function cardWord(card) {
    const seen = new Set();
    card.querySelectorAll('*').forEach((e) => {
      if (e.children.length === 0) {
        const t = (e.textContent || '').replace(/\s+/g, ' ').trim();
        if (t) seen.add(t);
      }
    });
    let best = '';
    for (const t of seen) {
      if (/^[A-Za-z][A-Za-z .'\-]*$/.test(t) && t.length > best.length) best = t;
    }
    return best.toUpperCase().trim();
  }

  // A card is "revealed" (already flipped/taken) once the agent illustration is
  // painted onto it. Unrevealed cards carry no /char/ artwork.
  function cardRevealed(card) {
    if (card.querySelector('img[src*="/char/"], img[src*="/agent"], img[src*="agent-"]')) return true;
    // some skins paint the agent as a background-image instead of an <img>
    let hit = false;
    const kids = card.querySelectorAll('*');
    for (let i = 0; i < kids.length && !hit; i++) {
      const bg = getComputedStyle(kids[i]).backgroundImage || '';
      if (bg.includes('/char/') || bg.includes('/agent')) hit = true;
    }
    return hit;
  }

  // Read all cards on the board. Returns null unless a plausible board exists.
  function readCards() {
    const cards = [];
    document.querySelectorAll('article[style*="--CardColor"]').forEach((a) => {
      const st = a.getAttribute('style') || '';
      const m = st.match(/--CardColor:\s*var\(--([a-z]+)-cardBg\)/i);
      if (!m) return;
      const word = cardWord(a);
      if (!word) return;
      cards.push({ word, color: m[1].toLowerCase(), revealed: cardRevealed(a) });
    });
    // De-dup any accidental repeats, keep board order.
    const out = [];
    const used = new Set();
    for (const c of cards) {
      if (used.has(c.word)) continue;
      used.add(c.word);
      out.push(c);
    }
    // A classic/duet board is 25. Accept a small window to tolerate variants.
    if (out.length < 20 || out.length > 25) return null;
    return out;
  }

  // Spymaster perspective = the full colour key is visible. Tell-tales: the
  // assassin (black) card is shown, or both team colours appear among the
  // still-unrevealed cards, or the clue input is present.
  function isSpymasterView(cards) {
    const cols = new Set(cards.filter((c) => !c.revealed).map((c) => c.color));
    if (cols.has('black')) return true;
    if (cols.has('blue') && cols.has('red')) return true;
    if (cols.has('green')) return true; // duet key visible
    if (document.querySelector('input[placeholder*="clue" i]')) return true;
    return false;
  }

  function lsNickname() {
    try { return JSON.parse(localStorage.getItem('cnd-lobby'))?.state?.nickname || ''; }
    catch (e) { return ''; }
  }

  // Read the team colour off an element by walking its ancestors and taking the
  // first decisive signal: codenames.game's layout hooks (data-match-slot like
  // "blueSpy"/"redOp", data-match-column left=blue/right=red) OR the themed
  // classes/vars it paints chips with (shadow-map-blue, --blue-gradient, etc.).
  function colorSignalOf(start) {
    let p = start;
    for (let i = 0; i < 20 && p; i++) {
      const cls = (p.className && p.className.toString) ? p.className.toString() : '';
      const style = (p.getAttribute && p.getAttribute('style')) || '';
      const slot = (p.getAttribute && p.getAttribute('data-match-slot')) || '';
      const col = (p.getAttribute && p.getAttribute('data-match-column')) || '';
      if (slot) { if (/^blue/i.test(slot)) return 'blue'; if (/^red/i.test(slot)) return 'red'; }
      if (col === 'left') return 'blue';
      if (col === 'right') return 'red';
      const hay = `${cls} ${style}`;
      if (/shadow-map-blue|blue-gradient|--blue\b|\bblue\b/i.test(hay)) return 'blue';
      if (/shadow-map-red|red-gradient|--red\b|\bred\b/i.test(hay)) return 'red';
      p = p.parentElement;
    }
    return null;
  }

  // Which colour is MINE? The player nickname appears on several chips (team
  // panel, overlay, spectator list…) so we read the team off EVERY chip and take
  // a majority vote — robust against one oddly-themed element. The result is
  // cached for the game; a stale frame never silently flips it.
  let cachedMyColor = null;
  function detectMyColor(cards) {
    if (CONFIG.MY_TEAM === 'blue' || CONFIG.MY_TEAM === 'red') return CONFIG.MY_TEAM;
    const nick = lsNickname();
    if (nick) {
      let b = 0, r = 0;
      document.querySelectorAll('*').forEach((e) => {
        if (e.children.length <= 1 && (e.textContent || '').trim() === nick) {
          const s = colorSignalOf(e);
          if (s === 'blue') b++; else if (s === 'red') r++;
        }
      });
      if (b || r) { cachedMyColor = b >= r ? 'blue' : 'red'; return cachedMyColor; }
    }
    if (cachedMyColor) return cachedMyColor; // keep last good through a transient frame
    // Last resort: in a full-key view the starting team has one more card.
    if (cards) {
      let b = 0, r = 0;
      cards.forEach((c) => { if (c.color === 'blue') b++; else if (c.color === 'red') r++; });
      if (b || r) {
        if (!teamWarned) { warn('could not detect your team from the lobby — guessing by card count. Set CONFIG.MY_TEAM if wrong.'); teamWarned = true; }
        return b >= r ? 'blue' : 'red';
      }
    }
    return null;
  }

  // Map a codenames.game colour token → a CodenamesState identity.
  function identityFor(color, myColor) {
    switch (color) {
      case 'black': return 'assassin';
      case 'neutral': case 'tan': case 'innocent': case 'bystander': case 'beige': return 'neutral';
      case 'green': return 'team'; // duet co-op "agent" = good
    }
    if (color === 'blue' || color === 'red') {
      if (!myColor) return 'unknown';
      return color === myColor ? 'team' : 'opponent';
    }
    return 'unknown';
  }

  // Best-effort active-clue scrape. A clue word is, by the rules, never one of
  // the 25 board words — that plus a stoplist makes this safe from UI noise.
  const CLUE_STOP = new Set(['ADMIN', 'NEWS', 'RULES', 'SETTINGS', 'SPECTATORS', 'OPERATIVES',
    'SPYMASTERS', 'SPYMASTER', 'OPERATIVE', 'BLUE', 'RED', 'TEAM', 'GAME', 'LOG', 'PASS',
    'CLUE', 'GUESS', 'WAITING', 'TURN']);
  function scrapeClue(boardWords) {
    if (!CONFIG.SCRAPE_CLUES) return null;
    const board = new Set(boardWords);
    let found = null;
    document.querySelectorAll('div,span,p,h1,h2,h3').forEach((e) => {
      if (found || e.children.length > 2) return;
      const t = (e.textContent || '').replace(/\s+/g, ' ').trim();
      // "WORD 3" / "WORD · 3" / "WORD: 3"
      const m = t.match(/^([A-Za-z][A-Za-z'\-]{1,20})\s*[·•:\-]?\s*(\d{1,2})$/);
      if (!m) return;
      const word = m[1].toUpperCase();
      const num = parseInt(m[2], 10);
      if (num > 9) return;                    // counters like "News 25" are not clues
      if (CLUE_STOP.has(word)) return;
      if (board.has(word)) return;            // a clue is never a board word
      found = { clue: m[1], number: num };
    });
    return found;
  }

  // ── Sync ────────────────────────────────────────────────────────────────────
  async function syncBoard() {
    if (!ROOM_RE.test(location.pathname)) return; // only inside a room
    const cards = readCards();
    if (!cards) { setBadge('waiting for board…', '#888'); return; }

    const words = cards.map((c) => c.word);
    const wordsKey = words.slice().sort().join('|');
    const spy = isSpymasterView(cards);
    const role = spy ? 'spymaster' : 'guesser';
    const myColor = detectMyColor(cards);

    // New board / new game → (re)start the tracker fresh.
    if (wordsKey !== lastWordsKey) {
      const res = await post('codenames_start', { words, role, team: myColor || '' });
      if (!res || !res.ok) { setBadge('server offline', '#c0392b'); return; }
      lastWordsKey = wordsKey;
      lastRole = role;
      lastTeam = myColor || '';
      lastIdentity.clear();
      lastClueKey = '';
      cachedMyColor = (CONFIG.MY_TEAM === 'blue' || CONFIG.MY_TEAM === 'red') ? CONFIG.MY_TEAM : null;
      log(`new board → ${words.length} words, role=${role}, team=${myColor || '?'}`);
    } else {
      if (role !== lastRole) { await post('codenames_set_role', { role }); lastRole = role; }
      if ((myColor || '') !== lastTeam) {
        // team flipped (or finally detected) → re-evaluate every identity below
        lastTeam = myColor || '';
        lastIdentity.clear();
      }
    }

    // Desired identity per card:
    //   spymaster  → the true colour of EVERY card (full key).
    //   guesser    → only revealed cards carry a known colour; the rest stay unknown.
    let revealedCount = 0;
    for (const c of cards) {
      let ident;
      if (spy) {
        ident = identityFor(c.color, myColor);
      } else {
        ident = c.revealed ? identityFor(c.color, myColor) : 'unknown';
      }
      if (c.revealed) revealedCount++;

      const prev = lastIdentity.get(c.word);
      // Skip the no-op cases: never re-assert "unknown" that was already unknown.
      if (prev === ident) continue;
      if (prev === undefined && ident === 'unknown') { lastIdentity.set(c.word, 'unknown'); continue; }
      const r = await post('codenames_reveal', { word: c.word, identity: ident });
      if (r && r.ok) lastIdentity.set(c.word, ident);
    }

    // Active clue (best-effort).
    const clue = scrapeClue(words);
    if (clue) {
      const key = `${clue.clue}|${clue.number}`;
      if (key !== lastClueKey) {
        await post('codenames_clue', { clue: clue.clue, number: clue.number, by: 'me' });
        lastClueKey = key;
      }
    }

    const teamTxt = myColor ? `${myColor} ` : '';
    setBadge(`${teamTxt}${role} · ${cards.length} cards · ${revealedCount} revealed`, '#2e8b57');
  }

  // ── Status badge ────────────────────────────────────────────────────────────
  let badgeEl = null;
  function setBadge(text, color) {
    if (!CONFIG.SHOW_BADGE) return;
    if (!badgeEl) {
      badgeEl = document.createElement('div');
      Object.assign(badgeEl.style, {
        position: 'fixed', left: '8px', bottom: '8px', zIndex: 2147483647,
        font: '600 11px/1.4 system-ui, sans-serif', color: '#fff',
        padding: '3px 8px', borderRadius: '6px', background: 'rgba(0,0,0,.55)',
        pointerEvents: 'none', letterSpacing: '.2px', userSelect: 'none',
      });
      document.body.appendChild(badgeEl);
    }
    const t = new Date().toLocaleTimeString();
    badgeEl.textContent = `KIRA ◉ ${text} · ${t}`;
    badgeEl.style.boxShadow = `0 0 0 1px ${color} inset`;
  }

  // ── Scheduling: debounced observer + safety poll + SPA route changes ─────────
  function scheduleSync() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { syncBoard().catch((e) => warn(e)); }, CONFIG.DEBOUNCE_MS);
  }

  function start() {
    const obs = new MutationObserver(scheduleSync);
    obs.observe(document.documentElement, {
      subtree: true, childList: true,
      attributes: true, attributeFilter: ['style', 'class', 'src', 'data-match-slot'],
    });
    setInterval(() => { syncBoard().catch(() => {}); }, CONFIG.POLL_MS);

    // codenames.game is an SPA — patch history so room changes reset our cache.
    const reset = () => { lastWordsKey = ''; lastIdentity.clear(); lastClueKey = ''; cachedMyColor = null; scheduleSync(); };
    const wrap = (fn) => function () { const r = fn.apply(this, arguments); reset(); return r; };
    history.pushState = wrap(history.pushState);
    history.replaceState = wrap(history.replaceState);
    window.addEventListener('popstate', reset);

    // Manual trigger (console / keybind) just in case.
    window.__kiraCodenamesSync = () => syncBoard();
    window.addEventListener('keydown', (e) => {
      if (e.altKey && (e.key === 'k' || e.key === 'K')) syncBoard();
    });

    log(`active on ${location.host} — Tampermonkey ${typeof GM_xmlhttpRequest === 'function' ? 'OK' : 'MISSING (CORS may block POSTs)'}`);
    syncBoard().catch((e) => warn(e));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
