/**
 * Myooz Labs Analytics & Lead Capture Tracker
 * Features:
 * 1. IP & Geo Location Resolution (ipwho.is with ipify fallback)
 * 2. Page Visit & Time-on-Tool Tracking
 * 3. Protected Downloads with Seamless Lead Capture Modal (Name & Email)
 * 4. Process & Operations Logger (window.MyoozTracker.logProcess)
 * 5. Multi-channel Sync: Hugging Face FastAPI backend + LocalStorage persistence
 */

(function () {
    'use strict';

    const BACKEND_BASE = "https://tinahmbuz-audiostems.hf.space";
    const STORAGE_KEY_LEAD = "myooz_lead_user";
    const STORAGE_KEY_ANALYTICS = "myooz_analytics_local";

    // ── 1. Determine Current Tool ──
    function getPageTool() {
        const path = window.location.pathname.toLowerCase();
        if (path.includes("stemer")) return "stemer";
        if (path.includes("midifier")) return "midifier";
        if (path.includes("sampler")) return "sampler";
        if (path.includes("mixter")) return "mixter";
        if (path.includes("master")) return "master";
        if (path.includes("generator")) return "generator";
        if (path.includes("admin")) return "admin";
        return "home";
    }

    const currentTool = getPageTool();
    if (currentTool === "admin") {
        // Do not track admin page as a standard tool visit
        return;
    }

    // ── 2. Device / Browser Information ──
    function getDeviceSummary() {
        const ua = navigator.userAgent;
        let os = "Desktop";
        if (/android/i.test(ua)) os = "Android";
        else if (/iphone|ipad|ipod/i.test(ua)) os = "iOS";
        else if (/macintosh|mac os x/i.test(ua)) os = "macOS";
        else if (/windows/i.test(ua)) os = "Windows";
        else if (/linux/i.test(ua)) os = "Linux";

        let browser = "Browser";
        if (/chrome|crios/i.test(ua) && !/edge|edg/i.test(ua)) browser = "Chrome";
        else if (/safari/i.test(ua) && !/chrome/i.test(ua)) browser = "Safari";
        else if (/firefox|fxios/i.test(ua)) browser = "Firefox";
        else if (/edge|edg/i.test(ua)) browser = "Edge";

        return `${browser} (${os})`;
    }

    // ── 3. IP & Geo Resolution with Session Cache ──
    let clientGeo = null;

    async function resolveClientGeo() {
        const cached = sessionStorage.getItem("myooz_client_geo");
        if (cached) {
            try {
                clientGeo = JSON.parse(cached);
                return clientGeo;
            } catch (e) {}
        }

        try {
            // Primary: ipwho.is (fast, gives country, city, ISP)
            const resp = await fetch("https://ipwho.is/", { cache: "no-store" });
            const data = await resp.json();
            if (data && data.success !== false) {
                clientGeo = {
                    ip: data.ip || "Unknown",
                    country: data.country || "Desconocido",
                    countryCode: data.country_code || "",
                    city: data.city || ""
                };
                sessionStorage.setItem("myooz_client_geo", JSON.stringify(clientGeo));
                return clientGeo;
            }
        } catch (err) {
            // Fallback: ipify
            try {
                const fResp = await fetch("https://api.ipify.org?format=json");
                const fData = await fResp.json();
                clientGeo = {
                    ip: fData.ip || "Unknown",
                    country: "Global",
                    countryCode: "",
                    city: ""
                };
                sessionStorage.setItem("myooz_client_geo", JSON.stringify(clientGeo));
                return clientGeo;
            } catch (err2) {
                clientGeo = { ip: "127.0.0.1", country: "Local", countryCode: "", city: "" };
            }
        }
        return clientGeo;
    }

    // ── 4. Local Storage Analytics Engine (Resilient Fallback) ──
    function getLocalAnalytics() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY_ANALYTICS);
            if (!raw) return { visits: [], leads: [], processes: [] };
            return JSON.parse(raw);
        } catch (e) {
            return { visits: [], leads: [], processes: [] };
        }
    }

    function saveLocalAnalytics(data) {
        try {
            localStorage.setItem(STORAGE_KEY_ANALYTICS, JSON.stringify(data));
        } catch (e) {}
    }

    function recordLocalEvent(type, payload) {
        const store = getLocalAnalytics();
        const event = {
            id: Date.now() + Math.random().toString(36).substr(2, 4),
            timestamp: new Date().toISOString(),
            ...payload
        };

        if (type === "visit") {
            store.visits.unshift(event);
            if (store.visits.length > 300) store.visits.pop();
        } else if (type === "lead") {
            store.leads.unshift(event);
            if (store.leads.length > 500) store.leads.pop();
        } else if (type === "process") {
            store.processes.unshift(event);
            if (store.processes.length > 300) store.processes.pop();
        }

        saveLocalAnalytics(store);
    }

    // ── 5. Server Synchronizer ──
    async function postToServer(endpoint, payload) {
        try {
            await fetch(`${BACKEND_BASE}${endpoint}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        } catch (err) {
            // Backend offline or waking up; local storage safely preserved the event
        }
    }

    // ── 6. Visit & Time Tracking ──
    let pageStartTime = Date.now();
    let visitRecorded = false;
    let visitId = "v_" + Date.now();

    async function trackPageVisit() {
        const geo = await resolveClientGeo();
        const device = getDeviceSummary();

        const visitData = {
            visitId,
            ip: geo.ip,
            country: geo.country,
            city: geo.city,
            tool: currentTool,
            device: device,
            duration: 0
        };

        recordLocalEvent("visit", visitData);
        postToServer("/api/track_visit", visitData);
        visitRecorded = true;
    }

    function updateSessionDuration() {
        if (!visitRecorded) return;
        const durationSec = Math.max(1, Math.round((Date.now() - pageStartTime) / 1000));
        
        // Update local visit duration
        const store = getLocalAnalytics();
        const v = store.visits.find(item => item.visitId === visitId);
        if (v) {
            v.duration = durationSec;
            saveLocalAnalytics(store);
        }

        if (clientGeo) {
            postToServer("/api/track_visit", {
                visitId,
                ip: clientGeo.ip,
                tool: currentTool,
                duration: durationSec
            });
        }
    }

    // Ping duration every 15 seconds
    setInterval(updateSessionDuration, 15000);
    window.addEventListener("beforeunload", updateSessionDuration);

    // ── 7. Lead Capture Modal Injection & Handler ──
    let pendingDownloadAction = null;

    function injectLeadModal() {
        if (document.getElementById("myoozLeadModal")) return;

        const modalHtml = `
        <div id="myoozLeadModal" class="myooz-lead-overlay" style="display: none;">
            <div class="myooz-lead-card">
                <button type="button" class="myooz-lead-close" id="myoozLeadClose" aria-label="Cerrar">&times;</button>
                <div class="myooz-lead-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                </div>
                <h3 class="myooz-lead-title">Descarga tu Archivo</h3>
                <p class="myooz-lead-subtitle">
                    Ingresa tu nombre y correo electrónico para autorizar la descarga y recibir futuras actualizaciones y mejoras de <strong>Myooz Labs</strong>.
                </p>
                <form id="myoozLeadForm" class="myooz-lead-form" onsubmit="return false;">
                    <div class="myooz-lead-group">
                        <label for="myoozLeadName">Nombre o Productor</label>
                        <input type="text" id="myoozLeadName" placeholder="Tu nombre artístico o alias" required autocomplete="name">
                    </div>
                    <div class="myooz-lead-group">
                        <label for="myoozLeadEmail">Correo Electrónico Válido</label>
                        <input type="email" id="myoozLeadEmail" placeholder="tu@email.com" required autocomplete="email">
                        <div id="myoozLeadError" class="myooz-lead-error" style="display: none;"></div>
                    </div>
                    <div style="font-size: 0.7rem; color: rgba(255,255,255,0.5); margin-bottom: 1.25rem; text-align: left; line-height: 1.4;">
                        Descarga inmediata sin costo. Respetamos tu privacidad, cero spam.
                    </div>
                    <button type="submit" id="myoozLeadSubmitBtn" class="myooz-lead-btn">
                        AUTORIZAR Y DESCARGAR AHORA
                    </button>
                </form>
            </div>
        </div>

        <style>
            .myooz-lead-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.85);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 999999;
                padding: 1.5rem;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.25s ease;
            }
            .myooz-lead-overlay.active {
                opacity: 1;
                pointer-events: auto;
            }
            .myooz-lead-card {
                background: rgba(18, 18, 22, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 24px;
                padding: 2.5rem;
                max-width: 440px;
                width: 100%;
                text-align: center;
                position: relative;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 40px rgba(204, 122, 0, 0.15);
                transform: translateY(20px) scale(0.97);
                transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            }
            .myooz-lead-overlay.active .myooz-lead-card {
                transform: translateY(0) scale(1);
            }
            .myooz-lead-close {
                position: absolute;
                top: 18px;
                right: 20px;
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.5);
                font-size: 1.8rem;
                cursor: pointer;
                line-height: 1;
                transition: color 0.2s;
            }
            .myooz-lead-close:hover {
                color: #fff;
            }
            .myooz-lead-icon {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: rgba(204, 122, 0, 0.15);
                border: 1px solid rgba(204, 122, 0, 0.4);
                color: #fbbf24;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1.25rem;
            }
            .myooz-lead-title {
                font-family: 'Righteous', 'Outfit', sans-serif;
                font-size: 1.5rem;
                color: #fff;
                letter-spacing: 1px;
                margin-bottom: 0.5rem;
            }
            .myooz-lead-subtitle {
                font-size: 0.82rem;
                color: rgba(255, 255, 255, 0.7);
                line-height: 1.5;
                margin-bottom: 1.75rem;
            }
            .myooz-lead-group {
                text-align: left;
                margin-bottom: 1rem;
            }
            .myooz-lead-group label {
                display: block;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: rgba(255, 255, 255, 0.75);
                margin-bottom: 6px;
                font-weight: 700;
            }
            .myooz-lead-group input {
                width: 100%;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                padding: 12px 16px;
                color: #fff;
                font-size: 0.9rem;
                font-family: inherit;
                outline: none;
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            .myooz-lead-group input:focus {
                border-color: #f97316;
                box-shadow: 0 0 15px rgba(249, 115, 22, 0.3);
            }
            .myooz-lead-error {
                color: #f87171;
                font-size: 0.75rem;
                margin-top: 6px;
                font-weight: 600;
            }
            .myooz-lead-btn {
                width: 100%;
                background: linear-gradient(180deg, #f97316 0%, #c2410c 100%);
                color: #000;
                font-weight: 800;
                font-size: 0.85rem;
                letter-spacing: 2px;
                text-transform: uppercase;
                border: none;
                padding: 14px 20px;
                border-radius: 50px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                box-shadow: 0 8px 20px rgba(249, 115, 22, 0.3);
            }
            .myooz-lead-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 25px rgba(249, 115, 22, 0.45);
            }
        </style>
        `;

        document.body.insertAdjacentHTML("beforeend", modalHtml);

        // Bind Close Button
        document.getElementById("myoozLeadClose").addEventListener("click", closeLeadModal);

        // Bind Overlay Click
        document.getElementById("myoozLeadModal").addEventListener("click", (e) => {
            if (e.target.id === "myoozLeadModal") closeLeadModal();
        });

        // Bind Submit
        document.getElementById("myoozLeadForm").addEventListener("submit", handleLeadSubmit);
    }

    function openLeadModal(onSuccessCallback) {
        injectLeadModal();
        pendingDownloadAction = onSuccessCallback;
        const modal = document.getElementById("myoozLeadModal");
        if (modal) {
            modal.style.display = "flex";
            setTimeout(() => modal.classList.add("active"), 10);
            document.getElementById("myoozLeadName").focus();
        }
    }

    function closeLeadModal() {
        const modal = document.getElementById("myoozLeadModal");
        if (modal) {
            modal.classList.remove("active");
            setTimeout(() => { modal.style.display = "none"; }, 250);
        }
        pendingDownloadAction = null;
    }

    async function handleLeadSubmit(e) {
        if (e) e.preventDefault();
        const nameInput = document.getElementById("myoozLeadName");
        const emailInput = document.getElementById("myoozLeadEmail");
        const errorEl = document.getElementById("myoozLeadError");
        const submitBtn = document.getElementById("myoozLeadSubmitBtn");

        const name = (nameInput.value || "").trim();
        const email = (emailInput.value || "").trim().toLowerCase();

        // Email validation RFC 5322 standard
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
        if (!name || name.length < 2) {
            errorEl.innerText = "Por favor ingresa un nombre o alias válido.";
            errorEl.style.display = "block";
            nameInput.focus();
            return;
        }

        if (!email || !emailRegex.test(email)) {
            errorEl.innerText = "Por favor ingresa un correo electrónico válido.";
            errorEl.style.display = "block";
            emailInput.focus();
            return;
        }

        errorEl.style.display = "none";
        submitBtn.disabled = true;
        submitBtn.innerText = "Autorizando descarga...";

        const geo = await resolveClientGeo();
        const leadData = {
            name,
            email,
            tool: currentTool,
            file: (pendingDownloadAction && pendingDownloadAction.filename) ? pendingDownloadAction.filename : "audio_export",
            ip: geo.ip,
            country: geo.country,
            city: geo.city
        };

        // 1. Save user registration locally
        localStorage.setItem(STORAGE_KEY_LEAD, JSON.stringify({
            name,
            email,
            registeredAt: Date.now()
        }));

        // 2. Record lead event locally and post to backend
        recordLocalEvent("lead", leadData);
        postToServer("/api/capture_lead", leadData);

        // 3. Close modal smoothly and trigger download
        setTimeout(() => {
            submitBtn.disabled = false;
            submitBtn.innerText = "AUTORIZAR Y DESCARGAR AHORA";
            const callback = pendingDownloadAction;
            closeLeadModal();

            if (callback && typeof callback.execute === "function") {
                callback.execute();
            }
        }, 400);
    }

    // ── 8. Global Interceptor for Download Buttons ──
    function interceptDownloads() {
        document.addEventListener("click", function (e) {
            // Find closest link or button that triggers downloads
            const target = e.target.closest("a[download], #downloadMidiBtn, #downloadAllBtn, #downloadMasterBtn, #downloadActiveBtn, #downloadZipBtn, #btnDownloadWav, .stem-dl-icon-btn, [data-download]");
            if (!target) return;

            // Check if user already registered
            const registeredUser = localStorage.getItem(STORAGE_KEY_LEAD);
            if (registeredUser) {
                // User already registered: log the download action and proceed
                try {
                    const u = JSON.parse(registeredUser);
                    const filename = target.getAttribute("download") || target.innerText || "download";
                    resolveClientGeo().then(geo => {
                        const procData = {
                            tool: currentTool,
                            action: "download",
                            details: `Archivo: ${filename} | Usuario: ${u.email}`,
                            ip: geo.ip,
                            status: "completed"
                        };
                        recordLocalEvent("process", procData);
                        postToServer("/api/log_process", procData);
                    });
                } catch (err) {}
                return; // Let standard download proceed
            }

            // User NOT registered: intercept and prompt modal
            e.preventDefault();
            e.stopPropagation();

            const href = target.getAttribute("href");
            const filename = target.getAttribute("download") || "archivo_exportado";

            openLeadModal({
                filename: filename,
                execute: function () {
                    // Re-trigger download cleanly
                    if (target.id === "downloadAllBtn") {
                        // Special case: Stemer batch zip button with click listener
                        target.click();
                    } else if (href && href !== "#") {
                        const tempLink = document.createElement("a");
                        tempLink.href = href;
                        tempLink.download = filename;
                        tempLink.style.display = "none";
                        document.body.appendChild(tempLink);
                        tempLink.click();
                        document.body.removeChild(tempLink);
                    } else {
                        target.click();
                    }
                }
            });
        }, true);
    }

    // ── 9. Global Public API: window.MyoozTracker ──
    window.MyoozTracker = {
        getLeadUser: function () {
            try {
                const raw = localStorage.getItem(STORAGE_KEY_LEAD);
                return raw ? JSON.parse(raw) : null;
            } catch (e) {
                return null;
            }
        },
        logProcess: async function (tool, action, details) {
            const geo = await resolveClientGeo();
            const proc = {
                tool: tool || currentTool,
                action: action || "process",
                details: typeof details === "object" ? JSON.stringify(details) : String(details),
                ip: geo.ip,
                status: "completed"
            };
            recordLocalEvent("process", proc);
            postToServer("/api/log_process", proc);
        }
    };

    // ── 10. Bootstrap Tracker on DOM Ready ──
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            trackPageVisit();
            interceptDownloads();
        });
    } else {
        trackPageVisit();
        interceptDownloads();
    }

})();

/* DATATA · presencia en directo (entra / sale) */
/* DATATA · presencia en directo
   Avisa al dashboard cuando alguien entra, sigue conectado y sale (al cerrar la pestaña).
   Cambia APP por la clave de la app en DATATA:
   myooz · reu · stem · acopio_v · acopio_c · indep · dijimu · venezuela */
(function () {
  'use strict';
  var APP = 'stem';
  if (/admin|moderador/i.test(location.pathname)) return; // los paneles internos no cuentan como visita
  if (window.__datataPresence) return;
  window.__datataPresence = true;

  var TOPIC = 'https://ntfy.sh/myoozlabs_live_telemetry_v2_e829fa';
  var K = 'datata_sid_' + APP, sid = '', since = 0, geo = {}, left = false, timer = null, started = false;
  var lastAct = Date.now(), ref = '';

  try { if (document.referrer) ref = new URL(document.referrer).hostname.replace(/^www\./, ''); } catch (e) {}
  try {
    var saved = JSON.parse(sessionStorage.getItem(K) || 'null');
    if (saved && saved.sid) { sid = saved.sid; since = saved.since; }
  } catch (e) {}
  if (!sid) {
    sid = Math.random().toString(36).slice(2, 10); since = Date.now();
    try { sessionStorage.setItem(K, JSON.stringify({ sid: sid, since: since })); } catch (e) {}
  }
  try { var g = JSON.parse(sessionStorage.getItem('datata_geo') || 'null'); if (g) geo = g; } catch (e) {}

  /* actividad real de la persona (distingue "está usando la app" de "dejó la pestaña abierta") */
  var actT = 0;
  function act() { var n = Date.now(); if (n - actT > 1000) { actT = n; lastAct = n; } }
  ['pointerdown', 'pointermove', 'keydown', 'scroll', 'touchstart', 'wheel'].forEach(function (t) {
    window.addEventListener(t, act, { passive: true, capture: true });
  });

  function send(ev, beacon) {
    var body = JSON.stringify({
      event: ev, app: APP, sid: sid, since: since,
      ip: geo.ip || '', city: geo.city || '', country: geo.country || '',
      url: location.pathname, userAgent: navigator.userAgent, ref: ref,
      vis: document.visibilityState || 'visible',
      idle: Math.max(0, Math.round((Date.now() - lastAct) / 1000))
    });
    try {
      if (beacon && navigator.sendBeacon && navigator.sendBeacon(TOPIC, body)) return;
      fetch(TOPIC, { method: 'POST', body: body, keepalive: true }).catch(function () {});
    } catch (e) {}
  }
  function start() {
    if (started && !left) return;
    started = true; left = false;
    send('join');
    clearInterval(timer);
    timer = setInterval(function () { send('heartbeat'); }, 15000);
  }
  function stop() {
    if (left) return;
    left = true; clearInterval(timer);
    send('leave', true);
  }

  /* ubicación: dos proveedores por si uno falla o lo bloquea un adblock; nunca se espera más de 1,5 s */
  function saveGeo(x) {
    geo = { ip: x.ip, city: x.city, country: x.country_name || x.country };
    try { sessionStorage.setItem('datata_geo', JSON.stringify(geo)); } catch (e) {}
  }
  if (geo.ip) start();
  else {
    var t = setTimeout(start, 1500);
    fetch('https://ipapi.co/json/')
      .then(function (r) { return r.json(); })
      .then(function (x) { if (!x || !x.ip) throw 0; saveGeo(x); })
      .catch(function () {
        return fetch('https://ipwho.is/').then(function (r) { return r.json(); }).then(function (x) { if (x && x.ip) saveGeo(x); }).catch(function () {});
      })
      .then(function () { clearTimeout(t); start(); });
  }

  /* aviso inmediato al ocultar / volver a mostrar la pestaña */
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') act();
    if (started && !left) send('heartbeat');
  });

  /* apps de una sola página: avisar también cuando cambia de sección */
  var lastPath = location.pathname, navT = null;
  function onNav() {
    if (location.pathname === lastPath) return;
    lastPath = location.pathname;
    clearTimeout(navT);
    navT = setTimeout(function () { if (started && !left) send('heartbeat'); }, 250);
  }
  ['pushState', 'replaceState'].forEach(function (m) {
    var o = history[m];
    history[m] = function () { var r = o.apply(this, arguments); onNav(); return r; };
  });
  window.addEventListener('popstate', onNav);

  window.addEventListener('pagehide', stop);
  window.addEventListener('pageshow', function (e) { if (e.persisted) { started = false; start(); } });
})();
