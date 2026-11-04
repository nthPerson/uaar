// UAAR wearable console: Web Bluetooth link, live charts, labelled-set recorder, WebSocket
// forwarder + live prediction panel, and a hardware-free simulator. Vanilla ES module.
//
// Data flow: every source (BLE notification or simulator) produces the raw notification
// bytes -> protocol.decodeBatch -> ingest(). ingest() fans one batch out to the stats, the
// chart ring buffer, the recorder (sets by session-flag edges) and the WebSocket forwarder.

import * as P from './protocol.js';

// ---- Small DOM helpers -----------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const logEl = $('log');
function log(msg, isError = false) {
  const line = document.createElement('div');
  if (isError) line.className = 'err';
  line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  logEl.appendChild(line);
  while (logEl.childElementCount > 300) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
  if (isError) console.error(msg);
}

function setPill(el, text, kind) {
  el.textContent = text;
  el.className = `pill pill-${kind}`;
}

/** Local time as YYYY-MM-DDTHH:MM:SS (same shape as Python's isoformat(timespec="seconds")). */
function localIso(d = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** YYYYMMDD-HHMMSS, the default session name used by scripts/record_wearable.py. */
function timestampName(d = new Date()) {
  return localIso(d).replace(/-/g, '').replace('T', '-').replace(/:/g, '');
}

function fmtUptime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h ? `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s` : `${m}m ${String(sec).padStart(2, '0')}s`;
}

// ---- Shared state ----------------------------------------------------------------------------
const state = {
  transport: null, // 'ble' | 'sim' | null
  deviceName: '-',
  firmware: '-',
  sessionId: 0,
  sessionActive: false,
  buttonPressed: false,
  calibrated: false,
  lastSeq: null,
  nSamples: 0,
  nDropped: 0,
  nGaps: 0,
  // rate counters, reset every second by the stats ticker
  pktCount: 0,
  sampleCount: 0,
  lastTick: performance.now(),
};

// ---- Ring buffer of physical-unit samples for the charts -------------------------------------
class Ring {
  constructor(capacity) {
    this.buf = new Array(capacity);
    this.cap = capacity;
    this.head = 0; // next write index
    this.size = 0;
  }
  push(x) {
    this.buf[this.head] = x;
    this.head = (this.head + 1) % this.cap;
    if (this.size < this.cap) this.size++;
  }
  /** Oldest -> newest. */
  forEach(fn) {
    const start = (this.head - this.size + this.cap) % this.cap;
    for (let i = 0; i < this.size; i++) fn(this.buf[(start + i) % this.cap], i);
  }
  last() {
    return this.size ? this.buf[(this.head - 1 + this.cap) % this.cap] : null;
  }
  clear() {
    this.head = 0;
    this.size = 0;
  }
}

const CHART_SECONDS = 10;
const ring = new Ring(CHART_SECONDS * P.SAMPLE_RATE_HZ + P.BATCH_SAMPLES * 2);

// ---- Canvas line chart ------------------------------------------------------------------------
class LineChart {
  /**
   * @param canvas  <canvas>
   * @param series  [{key, color}] keys into the physical sample
   * @param yMin/yMax  initial symmetric range; it grows to fit the data and never shrinks below
   */
  constructor(canvas, series, yMin, yMax) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.series = series;
    this.yMin0 = yMin;
    this.yMax0 = yMax;
  }

  draw(ring) {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(c.clientWidth * dpr));
    const h = Math.max(1, Math.floor(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    const padL = 48 * dpr, padR = 8 * dpr, padT = 8 * dpr, padB = 20 * dpr;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    const last = ring.last();
    const tEnd = last ? last.t : 0;
    const tStart = tEnd - CHART_SECONDS;

    // y range: grow to fit the visible data
    let yMin = this.yMin0, yMax = this.yMax0;
    ring.forEach((s) => {
      if (s.t < tStart) return;
      for (const sp of this.series) {
        const v = s[sp.key];
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    });
    const pad = (yMax - yMin) * 0.05;
    yMin -= pad; yMax += pad;
    const xOf = (t) => padL + ((t - tStart) / CHART_SECONDS) * plotW;
    const yOf = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * plotH;

    // session-active shading (green band where the session flag was set)
    ctx.fillStyle = 'rgba(61, 220, 132, 0.12)';
    let bandStart = null;
    ring.forEach((s) => {
      const on = (s.flags & P.FLAG_SESSION_ACTIVE) !== 0;
      if (on && bandStart === null) bandStart = s.t;
      if (!on && bandStart !== null) {
        ctx.fillRect(xOf(Math.max(bandStart, tStart)), padT, xOf(s.t) - xOf(Math.max(bandStart, tStart)), plotH);
        bandStart = null;
      }
    });
    if (bandStart !== null) {
      ctx.fillRect(xOf(Math.max(bandStart, tStart)), padT, xOf(tEnd) - xOf(Math.max(bandStart, tStart)), plotH);
    }

    // grid + axes
    ctx.strokeStyle = '#2b3441';
    ctx.lineWidth = 1 * dpr;
    ctx.fillStyle = '#8d9aab';
    ctx.font = `${11 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const nTicks = 4;
    for (let i = 0; i <= nTicks; i++) {
      const v = yMin + ((yMax - yMin) * i) / nTicks;
      const y = yOf(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(v.toFixed(Math.abs(yMax - yMin) < 10 ? 2 : 0), padL - 4 * dpr, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= CHART_SECONDS; i++) {
      const x = padL + (i / CHART_SECONDS) * plotW;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      ctx.fillText(`${i - CHART_SECONDS}s`, x, padT + plotH + 4 * dpr);
    }
    // zero line
    if (yMin < 0 && yMax > 0) {
      ctx.strokeStyle = '#4a5666';
      ctx.beginPath(); ctx.moveTo(padL, yOf(0)); ctx.lineTo(w - padR, yOf(0)); ctx.stroke();
    }

    // series
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineJoin = 'round';
    for (const sp of this.series) {
      ctx.strokeStyle = sp.color;
      ctx.beginPath();
      let first = true;
      let prevT = null;
      ring.forEach((s) => {
        if (s.t < tStart) return;
        const x = xOf(s.t), y = yOf(s[sp.key]);
        // break the line across gaps larger than 5 sample periods (dropped data)
        if (first || s.t - prevT > 5 / P.SAMPLE_RATE_HZ) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        first = false;
        prevT = s.t;
      });
      ctx.stroke();
    }
  }
}

const chartAcc = new LineChart($('chart-acc'), [
  { key: 'ax', color: '#4fb3ff' }, { key: 'ay', color: '#3ddc84' }, { key: 'az', color: '#ffb84d' }, { key: 'amag', color: '#ffffff' },
], -2, 2);
const chartGyr = new LineChart($('chart-gyr'), [
  { key: 'gx', color: '#4fb3ff' }, { key: 'gy', color: '#3ddc84' }, { key: 'gz', color: '#ffb84d' },
], -250, 250);

// redraw at ~20 fps using requestAnimationFrame, only when new data arrived
let chartDirty = true;
let lastDraw = 0;
function frame(now) {
  if (chartDirty && now - lastDraw >= 50) {
    chartAcc.draw(ring);
    chartGyr.draw(ring);
    chartDirty = false;
    lastDraw = now;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
window.addEventListener('resize', () => { chartDirty = true; });

// ---- Recorder: keeps samples, derives labelled sets from the session-flag edges ----------------
// Mirrors StreamState._on_sample in scripts/record_wearable.py: a set starts on the rising edge
// of FLAG_SESSION_ACTIVE and gets the label that is in the input at that moment; it is
// complete on the falling edge.
class Recorder {
  constructor() { this.reset(); }

  reset() {
    this.recording = false;
    this.samples = []; // raw sample objects in arrival order
    this.sets = []; // completed sets (events.csv rows)
    this.open = null; // the set in progress, if any
    this.prevActive = false;
    this.nDropped = 0;
    this.nGaps = 0;
    this.startedAt = null;
  }

  start() {
    this.reset();
    this.recording = true;
    this.startedAt = localIso();
  }

  stop() {
    this.recording = false;
    // a set still open when recording stops is closed at its last sample so it is not lost
    if (this.open) {
      this.sets.push(this.open);
      this.open = null;
      log(`recording stopped inside a set; set ${this.sets.length - 1} closed at the last sample`);
    }
    this.prevActive = false;
  }

  onSample(s, sessionId, dropped, label) {
    if (!this.recording) return;
    this.samples.push(s);
    if (dropped) { this.nDropped += dropped; this.nGaps += 1; }
    const active = (s.flags & P.FLAG_SESSION_ACTIVE) !== 0;
    if (active && !this.prevActive) {
      this.open = {
        set_index: this.sets.length,
        label,
        session_id: sessionId,
        t_start_ms: s.tMs,
        t_end_ms: s.tMs,
        seq_start: s.seq,
        seq_end: s.seq,
        n_samples: 0,
        n_dropped: 0,
      };
      log(`set ${this.open.set_index} started, label "${label}"`);
    }
    if (active && this.open) {
      this.open.t_end_ms = s.tMs;
      this.open.seq_end = s.seq;
      this.open.n_samples += 1;
      this.open.n_dropped += dropped;
    }
    if (!active && this.prevActive && this.open) {
      this.sets.push(this.open);
      log(`set ${this.open.set_index} "${this.open.label}" done: ${((this.open.t_end_ms - this.open.t_start_ms) / 1000).toFixed(1)} s, ${this.open.n_samples} samples, ${this.open.n_dropped} dropped`);
      this.open = null;
    }
    this.prevActive = active;
  }

  /** The three files uaar.data.wearable.load_session reads. */
  buildFiles({ subject, session, placement, note }) {
    const via = state.transport === 'sim' ? 'sim' : 'ble';
    const rawLines = [
      `# UAAR fw ${state.firmware} name=${state.deviceName} via=${via}`,
      `# recorded ${this.startedAt ?? localIso()}`,
      P.SERIAL_CSV_HEADER,
    ];
    for (const s of this.samples) rawLines.push(P.sampleToCsvLine(s));
    const raw = rawLines.join('\n') + '\n';

    const csvField = (v) => {
      const str = String(v);
      return /[",\n\r]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    };
    const evLines = [P.EVENTS_HEADER];
    for (const e of this.sets) {
      evLines.push([e.set_index, csvField(e.label), e.session_id, e.t_start_ms, e.t_end_ms, e.seq_start, e.seq_end, e.n_samples, e.n_dropped].join(','));
    }
    const events = evLines.join('\n') + '\n';

    const meta = {
      ...(subject ? { subject } : {}), // omitted when blank so the loader falls back to the folder name
      session,
      started_at: this.startedAt,
      device_name: state.deviceName,
      firmware: state.firmware,
      transport: via,
      sample_rate_hz: P.SAMPLE_RATE_HZ,
      acc_lsb_per_g: P.ACC_LSB_PER_G,
      gyr_lsb_per_dps: P.GYR_LSB_PER_DPS,
      placement,
      note,
      n_samples: this.samples.length,
      n_dropped: this.nDropped,
      n_gaps: this.nGaps,
      sets: this.sets.map((e) => ({ set_index: e.set_index, label: e.label, seconds: (e.t_end_ms - e.t_start_ms) / 1000 })),
    };
    return { raw, events, meta: JSON.stringify(meta, null, 2) + '\n' };
  }
}
const recorder = new Recorder();

function downloadText(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ---- WebSocket forwarder + prediction panel -------------------------------------------------
class WsClient {
  constructor() { this.ws = null; }
  get open() { return this.ws && this.ws.readyState === WebSocket.OPEN; }

  connect(url) {
    this.disconnect();
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      log(`websocket: ${e.message}`, true);
      return;
    }
    this.ws = ws;
    setPill($('ws-state'), 'connecting', 'warn');
    ws.onopen = () => {
      setPill($('ws-state'), 'backend connected', 'ok');
      $('btn-ws').textContent = 'Disconnect';
      $('ws-status').textContent = '';
      log(`websocket connected to ${url}`);
    };
    ws.onclose = (ev) => {
      if (this.ws === ws) {
        this.ws = null;
        setPill($('ws-state'), 'backend not connected', 'muted');
        $('btn-ws').textContent = 'Connect';
        prediction.showEmpty();
        log(`websocket closed (code ${ev.code})`);
      }
    };
    ws.onerror = () => log('websocket error (is the backend running?)', true);
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'prediction') prediction.update(msg);
      else if (msg.type === 'status') $('ws-status').textContent = String(msg.text ?? '');
      // unknown types are ignored
    };
  }

  disconnect() {
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.close();
      setPill($('ws-state'), 'backend not connected', 'muted');
      $('btn-ws').textContent = 'Connect';
      prediction.showEmpty();
    }
  }

  /** Forward one decoded batch as raw integers. */
  sendBatch(sessionId, samples) {
    if (!this.open) return;
    this.ws.send(JSON.stringify({
      type: 'samples',
      session_id: sessionId,
      samples: samples.map((s) => [s.tMs, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq]),
    }));
  }
}
const wsClient = new WsClient();

const LABEL_COLORS = ['#4fb3ff', '#3ddc84', '#ffb84d', '#c792ea', '#ff8fa3', '#72e0d1', '#f5e663', '#9fb3ff'];
function labelColor(label) {
  let h = 0;
  for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return LABEL_COLORS[h % LABEL_COLORS.length];
}

const TIMELINE_WINDOWS = 30;

class PredictionPanel {
  constructor(root) {
    this.root = root;
    this.methods = new Map(); // name -> {card, labelEl, pBar, pTxt, uBar, uTxt, tauEl, winEl, canvas, history}
  }

  showEmpty() {
    this.methods.clear();
    this.root.innerHTML = '<div class="pred-empty" id="pred-empty">backend not connected</div>';
  }

  _card(name) {
    let m = this.methods.get(name);
    if (m) return m;
    const empty = $('pred-empty');
    if (empty) empty.remove();
    const card = document.createElement('div');
    card.className = 'method-card';
    card.innerHTML = `
      <div class="method-name"></div>
      <div class="label">-</div>
      <div class="bar p"><div class="fill"></div><div class="txt"><span>p</span><span class="val">-</span></div></div>
      <div class="bar unc"><div class="fill"></div><div class="tau"></div><div class="txt"><span>uncertainty vs tau</span><span class="val">-</span></div></div>
      <canvas class="timeline"></canvas>
      <div class="window"></div>`;
    card.querySelector('.method-name').textContent = name;
    this.root.appendChild(card);
    m = {
      card,
      labelEl: card.querySelector('.label'),
      pFill: card.querySelector('.bar.p .fill'),
      pVal: card.querySelector('.bar.p .val'),
      uBar: card.querySelector('.bar.unc'),
      uFill: card.querySelector('.bar.unc .fill'),
      uVal: card.querySelector('.bar.unc .val'),
      tauEl: card.querySelector('.bar.unc .tau'),
      winEl: card.querySelector('.window'),
      canvas: card.querySelector('.timeline'),
      history: [],
    };
    this.methods.set(name, m);
    return m;
  }

  update(msg) {
    const methods = msg.methods && typeof msg.methods === 'object' ? msg.methods : {};
    for (const [name, r] of Object.entries(methods)) {
      const m = this._card(name);
      const unknown = Boolean(r.unknown);
      const label = unknown ? 'UNKNOWN' : String(r.label ?? '-');
      m.labelEl.textContent = label;
      m.labelEl.classList.toggle('unknown', unknown);
      const p = Number(r.p);
      m.pFill.style.width = Number.isFinite(p) ? `${Math.max(0, Math.min(1, p)) * 100}%` : '0';
      m.pVal.textContent = Number.isFinite(p) ? p.toFixed(2) : '-';
      // uncertainty bar: scale so tau sits at 50 % of the bar width (score / (2 tau))
      const score = Number(r.score), tau = Number(r.tau);
      if (Number.isFinite(score) && Number.isFinite(tau) && tau > 0) {
        const frac = Math.max(0, Math.min(1, score / (2 * tau)));
        m.uFill.style.width = `${frac * 100}%`;
        m.tauEl.style.left = '50%';
        m.tauEl.style.display = '';
        m.uBar.classList.toggle('over', score > tau);
        m.uVal.textContent = `${score.toFixed(3)} / tau ${tau.toFixed(3)}`;
      } else if (Number.isFinite(score)) {
        m.uFill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
        m.tauEl.style.display = 'none';
        m.uBar.classList.toggle('over', unknown);
        m.uVal.textContent = score.toFixed(3);
      } else {
        m.uFill.style.width = '0';
        m.tauEl.style.display = 'none';
        m.uVal.textContent = '-';
      }
      m.winEl.textContent = Number.isFinite(Number(msg.t_start)) ? `window ${Number(msg.t_start).toFixed(2)} - ${Number(msg.t_end).toFixed(2)} s` : '';
      m.history.push({ label, unknown });
      if (m.history.length > TIMELINE_WINDOWS) m.history.shift();
      this._drawTimeline(m);
    }
  }

  _drawTimeline(m) {
    const c = m.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(c.clientWidth * dpr)), h = Math.max(1, Math.floor(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0f1216';
    ctx.fillRect(0, 0, w, h);
    const cellW = w / TIMELINE_WINDOWS;
    const offset = TIMELINE_WINDOWS - m.history.length; // right-align: newest at the right edge
    m.history.forEach((e, i) => {
      const x = (offset + i) * cellW;
      if (e.unknown) {
        ctx.fillStyle = '#ff5c5c';
        ctx.fillRect(x, 0, cellW, h);
        // hatch
        ctx.save();
        ctx.beginPath(); ctx.rect(x, 0, cellW, h); ctx.clip();
        ctx.strokeStyle = '#0f1216';
        ctx.lineWidth = 1.5 * dpr;
        for (let d = -h; d < cellW + h; d += 5 * dpr) {
          ctx.beginPath(); ctx.moveTo(x + d, h); ctx.lineTo(x + d + h, 0); ctx.stroke();
        }
        ctx.restore();
      } else {
        ctx.fillStyle = labelColor(e.label);
        ctx.fillRect(x, 0, cellW, h);
      }
      ctx.strokeStyle = '#0f1216';
      ctx.lineWidth = 1 * dpr;
      ctx.strokeRect(x, 0, cellW, h);
    });
  }
}
const prediction = new PredictionPanel($('pred-methods'));

// ---- Ingest: one decoded batch from any source ---------------------------------------------
function ingest(sessionId, samples) {
  state.pktCount += 1;
  state.sampleCount += samples.length;
  state.sessionId = sessionId;
  const label = $('rec-label').value.trim();
  for (const s of samples) {
    const dropped = P.seqGap(state.lastSeq, s.seq);
    if (dropped) { state.nDropped += dropped; state.nGaps += 1; }
    state.lastSeq = s.seq;
    state.nSamples += 1;
    ring.push(P.toPhysical(s));
    recorder.onSample(s, sessionId, dropped, label);
  }
  const last = samples[samples.length - 1];
  if (last) {
    state.sessionActive = (last.flags & P.FLAG_SESSION_ACTIVE) !== 0;
    state.buttonPressed = (last.flags & P.FLAG_BUTTON_PRESSED) !== 0;
    state.calibrated = (last.flags & P.FLAG_CALIBRATED) !== 0;
  }
  wsClient.sendBatch(sessionId, samples);
  chartDirty = true;
  renderLiveState();
}

function onStatus(st) {
  state.firmware = st.firmware;
  state.calibrated = st.calibrated;
  state.sessionActive = st.sessionActive;
  state.sessionId = st.sessionId;
  $('st-fw').textContent = st.firmware;
  $('st-uptime').textContent = fmtUptime(st.uptimeS);
  renderLiveState();
}

function renderLiveState() {
  $('st-name').textContent = state.deviceName;
  $('st-cal').textContent = state.calibrated ? 'yes' : 'no';
  $('st-sid').textContent = String(state.sessionId);
  $('st-dropped').textContent = `${state.nDropped} in ${state.nGaps} gaps`;
  setPill($('st-button'), state.buttonPressed ? 'PRESSED' : 'released', state.buttonPressed ? 'warn' : 'muted');
  const ind = $('session-indicator');
  ind.textContent = state.sessionActive ? 'SESSION ACTIVE' : 'SESSION IDLE';
  ind.className = `session-indicator ${state.sessionActive ? 'active' : 'idle'}`;
  renderRecorder();
}

let recRenderPending = false;
function renderRecorder() {
  if (recRenderPending) return;
  recRenderPending = true;
  requestAnimationFrame(() => {
    recRenderPending = false;
    const n = recorder.samples.length;
    $('rec-summary').textContent = n
      ? `${n} samples, ${recorder.nDropped} dropped, ${recorder.sets.length} sets${recorder.open ? ' (+1 in progress)' : ''}`
      : 'no samples';
    $('btn-download').disabled = !(n > 0 && !recorder.recording);
    $('btn-clear').disabled = !(n > 0 && !recorder.recording);
    const tbody = $('sets-table').querySelector('tbody');
    const rows = [...recorder.sets, ...(recorder.open ? [recorder.open] : [])];
    tbody.innerHTML = '';
    for (const e of rows) {
      const tr = document.createElement('tr');
      if (e === recorder.open) tr.className = 'open';
      const secs = ((e.t_end_ms - e.t_start_ms) / 1000).toFixed(1);
      for (const v of [e.set_index, e.label || '(no label)', e.session_id, secs, e.n_samples, e.n_dropped]) {
        const td = document.createElement('td');
        td.textContent = String(v);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  });
}

// rates: once per second
setInterval(() => {
  const now = performance.now();
  const dt = (now - state.lastTick) / 1000;
  state.lastTick = now;
  $('st-pps').textContent = (state.pktCount / dt).toFixed(1);
  $('st-sps').textContent = (state.sampleCount / dt).toFixed(1);
  state.pktCount = 0;
  state.sampleCount = 0;
}, 1000);

// ---- Web Bluetooth link ------------------------------------------------------------------------
class BleLink {
  constructor() {
    this.device = null;
    this.imuChar = null;
    this.controlChar = null;
    this.statusChar = null;
    // bound once so add/removeEventListener pair up and subscriptions never stack on reconnect
    this._onImu = (ev) => this._handleImu(ev.target.value);
    this._onStatus = (ev) => this._handleStatus(ev.target.value);
    this._onDisconnected = () => this._handleDisconnected();
  }

  get connected() { return Boolean(this.device && this.device.gatt && this.device.gatt.connected); }

  /** Pick a device (user gesture required) and connect. */
  async connectNew() {
    if (!navigator.bluetooth) throw new Error('Web Bluetooth is not available in this browser (use Chrome or Edge, served from http://localhost or https)');
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: P.DEVICE_NAME_PREFIX }],
      optionalServices: [P.SERVICE_UUID],
    });
    if (this.device && this.device !== device) {
      this.device.removeEventListener('gattserverdisconnected', this._onDisconnected);
    }
    this.device = device;
    device.removeEventListener('gattserverdisconnected', this._onDisconnected);
    device.addEventListener('gattserverdisconnected', this._onDisconnected);
    await this.connectGatt();
  }

  /** (Re)connect GATT on the current device and (re)subscribe. */
  async connectGatt() {
    const device = this.device;
    if (!device) throw new Error('no device chosen yet');
    setPill($('ble-state'), 'connecting', 'warn');
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(P.SERVICE_UUID);
    const [imu, control, status] = await Promise.all([
      service.getCharacteristic(P.IMU_CHAR_UUID),
      service.getCharacteristic(P.CONTROL_CHAR_UUID),
      service.getCharacteristic(P.STATUS_CHAR_UUID),
    ]);
    // Chrome hands back the same characteristic objects across reconnects: remove before add.
    imu.removeEventListener('characteristicvaluechanged', this._onImu);
    status.removeEventListener('characteristicvaluechanged', this._onStatus);
    imu.addEventListener('characteristicvaluechanged', this._onImu);
    status.addEventListener('characteristicvaluechanged', this._onStatus);
    this.imuChar = imu; this.controlChar = control; this.statusChar = status;

    state.transport = 'ble';
    state.deviceName = device.name || '(unnamed)';
    state.lastSeq = null;
    try {
      const v = await status.readValue();
      onStatus(P.decodeStatus(v));
    } catch (e) {
      log(`status read failed: ${e.message}`, true);
    }
    try { await status.startNotifications(); } catch (e) { log(`status notify: ${e.message}`, true); }
    await imu.startNotifications();
    setPill($('ble-state'), `connected: ${state.deviceName}`, 'ok');
    $('btn-disconnect').disabled = false;
    $('btn-reconnect').hidden = true;
    log(`connected to ${state.deviceName}, fw ${state.firmware}`);
    renderLiveState();
  }

  async disconnect() {
    if (this.device && this.device.gatt.connected) this.device.gatt.disconnect();
    // the gattserverdisconnected event does the UI work
  }

  async write(cmd) {
    if (!this.controlChar || !this.connected) throw new Error('not connected');
    const bytes = Uint8Array.of(cmd);
    if (this.controlChar.writeValueWithoutResponse) await this.controlChar.writeValueWithoutResponse(bytes);
    else await this.controlChar.writeValue(bytes);
  }

  _handleImu(dv) {
    try {
      const { sessionId, samples } = P.decodeBatch(dv);
      ingest(sessionId, samples);
    } catch (e) {
      log(`bad IMU packet (${dv.byteLength} bytes): ${e.message}`, true);
    }
  }

  _handleStatus(dv) {
    try { onStatus(P.decodeStatus(dv)); } catch (e) { log(`bad status packet: ${e.message}`, true); }
  }

  _handleDisconnected() {
    setPill($('ble-state'), 'disconnected', 'bad');
    $('btn-disconnect').disabled = true;
    $('btn-reconnect').hidden = false;
    state.transport = null;
    log(`disconnected from ${state.deviceName}`);
  }
}
const ble = new BleLink();

// ---- Simulator: synthetic 50 Hz stream through the same bytes -> decode -> ingest path --------
class Simulator {
  constructor() {
    this.timer = null;
    this.statusTimer = null;
    this.tMs = 0;
    this.seq = 0;
    this.sessionId = 0;
    this.sessionActive = false;
    this.calibrated = false;
    this.nextToggleMs = 10000;
    this.t0 = 0;
  }

  get running() { return this.timer !== null; }

  start() {
    this.t0 = performance.now();
    state.transport = 'sim';
    state.deviceName = 'UAAR-SIM';
    state.firmware = '0.0.0';
    state.lastSeq = null;
    this.timer = setInterval(() => this.tick(), 1000 / (P.SAMPLE_RATE_HZ / P.BATCH_SAMPLES)); // 100 ms
    this.statusTimer = setInterval(() => this.emitStatus(), 1000);
    this.emitStatus();
    setPill($('ble-state'), 'simulating', 'warn');
    $('btn-simulate').textContent = 'Stop simulation';
    log('simulator started: gravity + 1 Hz curl, session toggles every 10 s, ~1 dropped sample / 7 s');
  }

  stop() {
    clearInterval(this.timer); clearInterval(this.statusTimer);
    this.timer = null; this.statusTimer = null;
    state.transport = null;
    setPill($('ble-state'), 'not connected', 'muted');
    $('btn-simulate').textContent = 'Simulate';
    log('simulator stopped');
  }

  /** Same commands the wearable accepts. */
  command(cmd) {
    switch (cmd) {
      case P.CMD_START_SESSION: this.setSession(true); break;
      case P.CMD_STOP_SESSION: this.setSession(false); break;
      case P.CMD_CALIBRATE: this.calibrated = true; log('sim: calibrated'); break;
      case P.CMD_PING: log('sim: # pong'); break;
      case P.CMD_IDENTIFY: log('sim: # UAAR fw 0.0.0 name=UAAR-SIM imu=sim'); break;
      case P.CMD_LED_KNOWN: log('sim: LED known pattern'); break;
      case P.CMD_LED_UNKNOWN: log('sim: LED unknown pattern'); break;
      default: log(`sim: unknown command 0x${cmd.toString(16)}`);
    }
  }

  setSession(on) {
    if (on && !this.sessionActive) this.sessionId += 1;
    this.sessionActive = on;
    this.buttonUntilMs = this.tMs + 200; // button "pressed" briefly around each toggle
    this.nextToggleMs = this.tMs + 10000;
  }

  tick() {
    const samples = [];
    for (let i = 0; i < P.BATCH_SAMPLES; i++) {
      if (this.tMs >= this.nextToggleMs) this.setSession(!this.sessionActive);
      // ~1 dropped sample every 7 s
      if (this.seq % 350 === 349) this.seq += 1;
      const t = this.tMs / 1000;
      const curl = this.sessionActive ? Math.sin(2 * Math.PI * 1.0 * t) : 0; // 1 Hz curl while active
      const noise = () => (Math.random() - 0.5) * 0.03;
      const ax = 0.15 * curl + noise();
      const ay = -0.9 - 0.25 * Math.max(0, curl) + noise();
      const az = 0.35 * curl + noise();
      const gx = 120 * Math.cos(2 * Math.PI * t) * (this.sessionActive ? 1 : 0) + noise() * 40;
      const gy = 15 * curl + noise() * 40;
      const gz = -40 * curl + noise() * 40;
      let flags = 0;
      if (this.sessionActive) flags |= P.FLAG_SESSION_ACTIVE;
      if (this.tMs < (this.buttonUntilMs ?? 0)) flags |= P.FLAG_BUTTON_PRESSED;
      if (this.calibrated) flags |= P.FLAG_CALIBRATED;
      samples.push({
        tMs: this.tMs,
        ax: Math.round(ax * P.ACC_LSB_PER_G), ay: Math.round(ay * P.ACC_LSB_PER_G), az: Math.round(az * P.ACC_LSB_PER_G),
        gx: Math.round(gx * P.GYR_LSB_PER_DPS), gy: Math.round(gy * P.GYR_LSB_PER_DPS), gz: Math.round(gz * P.GYR_LSB_PER_DPS),
        flags,
        seq: this.seq & 0xffff,
      });
      this.seq += 1;
      this.tMs += 1000 / P.SAMPLE_RATE_HZ;
    }
    // go through the wire format exactly like a BLE notification would
    const bytes = P.encodeBatch(this.sessionId, samples);
    const { sessionId, samples: decoded } = P.decodeBatch(new DataView(bytes.buffer));
    ingest(sessionId, decoded);
  }

  emitStatus() {
    const bytes = P.encodeStatus({
      firmware: '0.0.0',
      sessionId: this.sessionId,
      uptimeS: Math.floor((performance.now() - this.t0) / 1000),
      calibrated: this.calibrated,
      sessionActive: this.sessionActive,
    });
    onStatus(P.decodeStatus(bytes));
  }
}
const sim = new Simulator();

// ---- Commands ------------------------------------------------------------------------------------
const COMMANDS = {
  ping: P.CMD_PING,
  start: P.CMD_START_SESSION,
  stop: P.CMD_STOP_SESSION,
  calibrate: P.CMD_CALIBRATE,
  led_known: P.CMD_LED_KNOWN,
  led_unknown: P.CMD_LED_UNKNOWN,
  identify: P.CMD_IDENTIFY,
};

async function sendCommand(name) {
  const cmd = COMMANDS[name];
  if (sim.running) { sim.command(cmd); return; }
  if (!ble.connected) { log('no wearable connected (connect, or use Simulate)', true); return; }
  try {
    await ble.write(cmd);
    log(`sent ${name} (0x${cmd.toString(16).padStart(2, '0')})`);
  } catch (e) {
    log(`command ${name} failed: ${e.message}`, true);
  }
}

// ---- Wire up the UI ----------------------------------------------------------------------------
$('btn-connect').addEventListener('click', async () => {
  if (sim.running) sim.stop();
  try {
    await ble.connectNew();
  } catch (e) {
    if (e && e.name === 'NotFoundError') log('no device chosen');
    else log(`connect failed: ${e.message}`, true);
    setPill($('ble-state'), ble.connected ? `connected: ${state.deviceName}` : 'not connected', ble.connected ? 'ok' : 'muted');
  }
});
$('btn-reconnect').addEventListener('click', async () => {
  try { await ble.connectGatt(); } catch (e) { log(`reconnect failed: ${e.message}`, true); setPill($('ble-state'), 'disconnected', 'bad'); }
});
$('btn-disconnect').addEventListener('click', () => ble.disconnect());
$('btn-simulate').addEventListener('click', () => {
  if (sim.running) { sim.stop(); return; }
  if (ble.connected) ble.disconnect();
  sim.start();
});
document.querySelectorAll('button.cmd').forEach((b) => b.addEventListener('click', () => sendCommand(b.dataset.cmd)));

// record
let autoSession = timestampName();
$('rec-session').value = autoSession;
$('btn-record').addEventListener('click', () => {
  if (recorder.recording) {
    recorder.stop();
    $('btn-record').textContent = 'Start recording';
    $('btn-record').className = 'primary';
    setPill($('rec-state'), 'off', 'muted');
    log(`recording stopped: ${recorder.samples.length} samples, ${recorder.sets.length} sets`);
  } else {
    if (recorder.samples.length && !confirm('Discard the previous recording?')) return;
    // refresh the default session name unless the user typed their own
    if (!$('rec-session').value.trim() || $('rec-session').value === autoSession) {
      autoSession = timestampName();
      $('rec-session').value = autoSession;
    }
    recorder.start();
    $('btn-record').textContent = 'Stop recording';
    $('btn-record').className = 'danger';
    setPill($('rec-state'), 'REC', 'bad');
    log('recording started; sets follow the session flag (button / Start session)');
  }
  renderRecorder();
});
$('btn-clear').addEventListener('click', () => {
  if (!confirm('Discard the recorded samples?')) return;
  recorder.reset();
  renderRecorder();
});
$('btn-download').addEventListener('click', () => {
  const files = recorder.buildFiles({
    subject: $('rec-subject').value.trim(),
    session: $('rec-session').value.trim() || timestampName(),
    placement: $('rec-placement').value.trim(),
    note: $('rec-note').value.trim(),
  });
  downloadText('raw.csv', files.raw, 'text/csv');
  downloadText('events.csv', files.events, 'text/csv');
  downloadText('meta.json', files.meta, 'application/json');
  log(`downloaded raw.csv (${recorder.samples.length} samples), events.csv (${recorder.sets.length} sets), meta.json; put them in one folder for uaar.data.wearable.load_session`);
});

// websocket
$('btn-ws').addEventListener('click', () => {
  if (wsClient.ws) wsClient.disconnect();
  else wsClient.connect($('ws-url').value.trim());
});
$('ws-url').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') $('btn-ws').click(); });

// startup
if (!navigator.bluetooth) {
  log('Web Bluetooth is not available here: use Chrome or Edge on http://localhost (or https). Simulate still works.', true);
}
log('ready. Connect wearable, or Simulate to exercise the page without hardware.');
renderLiveState();

// expose for debugging and headless tests
window.uaar = { state, recorder, sim, ble, wsClient, prediction, ring, protocol: P };
