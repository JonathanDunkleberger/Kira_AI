/* ─── Kira caption overlay client ────────────────────────────────────────
   Receives caption frames over websocket and reveals each word at its
   Azure-provided audio offset. No timing estimation — every reveal is
   scheduled from a real synthesizer word-boundary event.

   Protocol (JSON messages from server):
     { type: "caption",
       text: "Full caption string",
       words: [ { word: "Hello", offset_ms: 0   },
                { word: "chat",  offset_ms: 320 }, ... ],
       clear_after_ms: 1500 }      // fade after the last word + this delay

     { type: "clear" }              // immediate clear (interrupt)

     { type: "hello" }              // server greeting, no action
*/

(function () {
    // OBS's embedded CEF browser loads index.html via file://, where
    // location.hostname is empty. Hard-code localhost (resolves more reliably
    // in CEF than 127.0.0.1). Auto-reconnect below handles bot-not-yet-running.
    const WS_URL = `ws://localhost:8765`;
    const captionEl = document.getElementById("caption");
    const statusEl = document.getElementById("status");

    let ws = null;
    let reconnectDelay = 500;
    const MAX_RECONNECT_DELAY = 8000;

    // Token for the active caption — used to cancel pending word reveals or
    // the fade-out when a new caption arrives mid-flight (fast back-to-back lines).
    let activeToken = 0;
    let pendingTimeouts = [];

    function setStatus(state) {
        statusEl.textContent = state;
        statusEl.classList.remove("connected", "disconnected");
        statusEl.classList.add(state === "online" ? "connected" : "disconnected");
    }

    function clearPending() {
        for (const id of pendingTimeouts) clearTimeout(id);
        pendingTimeouts = [];
    }

    function showCaption(frame) {
        // Bump the token so any in-flight timers from a previous caption no-op.
        activeToken += 1;
        const myToken = activeToken;
        clearPending();

        // Build the DOM: one <span class="word"> per word, separated by spaces.
        captionEl.innerHTML = "";
        const wordEls = [];
        for (const w of frame.words) {
            const span = document.createElement("span");
            span.className = "word";
            span.textContent = w.word;
            captionEl.appendChild(span);
            wordEls.push(span);
        }

        // Reveal the container immediately (so the outline appears the moment
        // she starts speaking, even though words are individually hidden).
        captionEl.classList.add("visible");
        captionEl.classList.remove("entering");
        // Force reflow so the animation restarts even on rapid successive captions.
        void captionEl.offsetWidth;
        captionEl.classList.add("entering");

        // Schedule each word reveal at its Azure-provided offset (ms from
        // synthesis start). Network jitter on localhost is sub-ms; we treat
        // "now" as the synthesis start moment.
        let lastOffsetMs = 0;
        for (let i = 0; i < frame.words.length; i++) {
            const w = frame.words[i];
            const offset = Math.max(0, w.offset_ms | 0);
            lastOffsetMs = Math.max(lastOffsetMs, offset);
            const el = wordEls[i];
            const id = setTimeout(() => {
                if (myToken !== activeToken) return;   // superseded
                el.classList.add("revealed");
            }, offset);
            pendingTimeouts.push(id);
        }

        // After the last word + clear_after_ms, fade out.
        const clearAfter = Math.max(0, (frame.clear_after_ms | 0));
        // Add a small tail for the average word duration so the last word
        // doesn't pop off the instant it appears.
        const tailEstimate = 350;
        const fadeAt = lastOffsetMs + tailEstimate + clearAfter;
        const fadeId = setTimeout(() => {
            if (myToken !== activeToken) return;
            captionEl.classList.remove("visible");
            // After fade transition, blank the text so nothing flashes if
            // a new caption arrives later with a different word count.
            const blankId = setTimeout(() => {
                if (myToken !== activeToken) return;
                captionEl.innerHTML = "";
            }, 400);
            pendingTimeouts.push(blankId);
        }, fadeAt);
        pendingTimeouts.push(fadeId);
    }

    function clearCaption() {
        activeToken += 1;
        clearPending();
        captionEl.classList.remove("visible");
        setTimeout(() => { captionEl.innerHTML = ""; }, 400);
    }

    function handleMessage(ev) {
        let msg;
        try { msg = JSON.parse(ev.data); }
        catch { return; }
        if (!msg || !msg.type) return;
        if (msg.type === "caption") {
            if (Array.isArray(msg.words) && msg.words.length > 0) {
                showCaption(msg);
            }
        } else if (msg.type === "clear") {
            clearCaption();
        }
    }

    function connect() {
        try {
            ws = new WebSocket(WS_URL);
        } catch (e) {
            setStatus("offline");
            scheduleReconnect();
            return;
        }
        ws.onopen = () => {
            setStatus("online");
            reconnectDelay = 500;
        };
        ws.onmessage = handleMessage;
        ws.onclose = () => {
            setStatus("offline");
            scheduleReconnect();
        };
        ws.onerror = () => { /* onclose will fire next */ };
    }

    function scheduleReconnect() {
        setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(MAX_RECONNECT_DELAY, Math.round(reconnectDelay * 1.6));
    }

    setStatus("offline");
    connect();
})();
