// UAAR live demo page: upload a sensor stream (live from the wearable, or a recorded raw.csv),
// send it to the prediction server that served this page, and show each model's answer and
// uncertainty. Vanilla ES module, no dependencies. The recording console is record.html.

import * as P from './protocol.js';

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const pct = (x) => `${Math.round(100 * x)}%`;

const HISTORY = 30; // windows kept per card (one per second)
const TRACE_SECONDS = 10;
const N_COLORS = 8; // --c0 .. --c7 in demo.css; further classes are drawn grey

const state = {
  source: null, // 'ble' | 'file' | null
  nSamples: 0,
  nLost: 0,
  lastSeq: null,
  rateCount: 0,
  lastSampleAt: 0,
  classes: [], // class names in class order: the colour of a class is its index here
  cards: new Map(), // name -> Card
};

// ---- 1 Upload: status line ----------------------------------------------------------------------
function setSource(kind, text) {
  state.source = kind;
  $('source-node').textContent = text;
  $('flow').classList.toggle('active', kind !== null);
}

function banner(html) {
  $('banner').hidden = !html;
  $('banner').innerHTML = html || '';
}

function resetCounters() {
  state.nSamples = 0;
  state.nLost = 0;
  state.lastSeq = null;
  trace.clear();
  server.send({ type: 'reset' });
  for (const card of state.cards.values()) card.clear();
}

setInterval(() => {
  $('st-rate').textContent = fmt(state.rateCount);
  state.rateCount = 0;
  $('st-count').textContent = fmt(state.nSamples);
  $('st-lost').textContent = fmt(state.nLost);
  $('flow').classList.toggle('flowing', performance.now() - state.lastSampleAt < 500);
}, 1000);

/** One batch of raw samples from any source: count it, draw it, upload it. */
function ingest(sessionId, samples) {
  for (const s of samples) {
    state.nLost += P.seqGap(state.lastSeq, s.seq);
    state.lastSeq = s.seq;
    trace.push(P.toPhysical(s));
  }
  state.nSamples += samples.length;
  state.rateCount += samples.length;
  state.lastSampleAt = performance.now();
  $('flow').classList.add('flowing');
  server.send({
    type: 'samples',
    session_id: sessionId,
    samples: samples.map((s) => [s.tMs, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq]),
  });
}

// ---- Signal trace: accelerometer x, y, z, last 10 s ---------------------------------------------
class Trace {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cap = TRACE_SECONDS * P.SAMPLE_RATE_HZ;
    this.buf = [];
    this.range = 1.5; // g, grows and shrinks slowly with the signal
    this.dirty = true;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const loop = () => { if (this.dirty) this.draw(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
  clear() { this.buf = []; this.dirty = true; }
  push(s) {
    this.buf.push(s);
    if (this.buf.length > this.cap) this.buf.shift();
    this.dirty = true;
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.dirty = true;
  }
  draw() {
    this.dirty = false;
    const { ctx, canvas, buf } = this;
    const w = canvas.width, h = canvas.height;
    const css = getComputedStyle(document.documentElement);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = css.getPropertyValue('--hairline');
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    if (buf.length < 2) return;
    let peak = 1.5;
    for (const s of buf) peak = Math.max(peak, Math.abs(s.ax), Math.abs(s.ay), Math.abs(s.az));
    this.range += (peak * 1.1 - this.range) * 0.1;
    const x = (i) => ((i + this.cap - buf.length) / (this.cap - 1)) * w;
    const y = (v) => h / 2 - (v / this.range) * (h / 2 - 4);
    const inks = [['ax', '--ink'], ['ay', '--ink-2'], ['az', '--muted']];
    ctx.lineWidth = Math.max(1.5, w / 900);
    ctx.lineJoin = 'round';
    for (const [key, token] of inks) {
      ctx.strokeStyle = css.getPropertyValue(token);
      ctx.beginPath();
      buf.forEach((s, i) => (i ? ctx.lineTo(x(i), y(s[key])) : ctx.moveTo(x(i), y(s[key]))));
      ctx.stroke();
    }
  }
}
const trace = new Trace($('trace'));

// ---- 2 Prediction cards -----------------------------------------------------------------------
function classColor(label) {
  const i = state.classes.indexOf(label);
  return i >= 0 && i < N_COLORS ? `var(--c${i})` : 'var(--other)';
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

class Card {
  constructor(info) {
    this.root = el('article', 'card');
    const head = el('div');
    head.append(el('h2', '', info.title || info.name), el('p', 'how', info.subtitle || ''));

    this.answer = el('div', 'answer waiting');
    this.chip = el('span', 'chip');
    this.name = el('span', 'name', 'Waiting for data');
    this.answer.append(this.name);

    this.meter = el('div', 'meter');
    const mh = el('div', 'meter-head');
    this.value = el('span', 'value', '');
    mh.append(el('span', '', 'Uncertainty'), this.value);
    const track = el('div', 'track');
    this.fill = el('div', 'fill');
    this.limit = el('div', 'limit');
    this.limitText = el('span', '', 'limit');
    this.limit.append(this.limitText);
    this.limit.hidden = true;
    track.append(this.fill, this.limit);
    this.meter.append(mh, track);

    const hist = el('div');
    hist.append(el('div', 'history-caption', `Last ${HISTORY} seconds`));
    this.history = el('div', 'history');
    this.cells = Array.from({ length: HISTORY }, () => el('span', 'cell'));
    this.history.append(...this.cells);
    hist.append(this.history);

    this.root.append(head, this.answer, this.meter, hist);
    this.items = [];
  }

  clear() {
    this.items = [];
    this.answer.className = 'answer waiting';
    this.answer.replaceChildren(this.name);
    this.name.textContent = 'Waiting for data';
    this.value.textContent = '';
    this.fill.style.width = '0';
    this.meter.classList.remove('over');
    this.renderHistory();
  }

  update(m) {
    const label = m.unknown ? 'Unknown' : m.label;
    this.answer.className = `answer${m.unknown ? ' unknown' : ''}`;
    this.chip.className = `chip${m.unknown ? ' unknown' : ''}`;
    this.chip.style.setProperty('--chip', classColor(m.label));
    this.name.textContent = label;
    this.answer.replaceChildren(this.chip, this.name);

    this.value.textContent = pct(m.u);
    this.fill.style.width = pct(m.u);
    this.meter.classList.toggle('over', m.unknown);
    this.limit.hidden = false;
    this.limit.style.left = pct(m.tau_u);
    this.limit.classList.toggle('near-end', m.tau_u > 0.85);
    this.limit.classList.toggle('near-start', m.tau_u < 0.15);
    this.limitText.textContent = `limit ${pct(m.tau_u)}`;

    this.items.push({ label: m.label, unknown: m.unknown, u: m.u });
    if (this.items.length > HISTORY) this.items.shift();
    this.renderHistory();
  }

  renderHistory() {
    const offset = HISTORY - this.items.length;
    this.cells.forEach((cell, i) => {
      const it = this.items[i - offset];
      cell.className = `cell${it && it.unknown ? ' unknown' : ''}`;
      cell.style.background = it && !it.unknown ? classColor(it.label) : '';
      cell.title = it
        ? `${HISTORY - i - 1} s ago: ${it.unknown ? 'unknown' : it.label}, uncertainty ${pct(it.u)}`
        : '';
    });
  }
}

function buildCards(status) {
  state.classes = status.classes || [];
  const legend = $('legend-classes');
  legend.replaceChildren();
  const shown = state.classes.length <= N_COLORS ? state.classes : [];
  for (const name of shown) {
    const li = el('li');
    const chip = el('span', 'chip');
    chip.style.setProperty('--chip', classColor(name));
    li.append(chip, document.createTextNode(name));
    legend.append(li);
  }
  const li = el('li');
  li.append(el('span', 'chip unknown'), document.createTextNode('Unknown'));
  legend.append(li);

  state.cards.clear();
  const box = $('cards');
  box.replaceChildren();
  for (const info of status.cards || []) {
    const card = new Card(info);
    state.cards.set(info.name, card);
    box.append(card.root);
  }
}

// ---- Prediction server (the one that served this page) ------------------------------------------
class Server {
  constructor() {
    this.ws = null;
    this.replayOffered = false;
    this.connect();
  }
  get open() { return this.ws && this.ws.readyState === WebSocket.OPEN; }
  send(msg) { if (this.open) this.ws.send(JSON.stringify(msg)); }
  connect() {
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => banner('');
    ws.onclose = () => {
      banner('The prediction server is not running. Start it with <code>uv run scripts/demo.py</code>; this page reconnects by itself.');
      setTimeout(() => this.connect(), 1500);
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'status') {
        buildCards(msg);
        if (msg.replay && !this.replayOffered && state.source === null) {
          this.replayOffered = true;
          player.playUrl('replay.csv');
        }
      } else if (msg.type === 'prediction') {
        for (const [name, m] of Object.entries(msg.methods || {})) {
          const card = state.cards.get(name);
          if (card) card.update(m);
        }
      }
    };
  }
}

// ---- Source: a recorded raw.csv, uploaded at the pace it was recorded -----------------------------
class Player {
  constructor() { this.timer = null; }
  stop() { clearInterval(this.timer); this.timer = null; }

  async playUrl(url) {
    const res = await fetch(url);
    if (res.ok) this.play(await res.text(), 'Recorded session');
  }

  play(text, name) {
    const samples = text.split('\n').map(P.parseCsvLine).filter(Boolean);
    if (!samples.length) {
      setSource(null, `${name}: no samples found in this file`);
      return;
    }
    this.stop();
    if (ble.connected) ble.disconnect();
    resetCounters();
    setSource('file', `${name}, ${Math.round(samples.length / P.SAMPLE_RATE_HZ)} s`);
    const speed = Number(new URLSearchParams(location.search).get('speed')) || 1;
    const t0 = samples[0].tMs;
    const start = performance.now();
    let i = 0;
    this.timer = setInterval(() => {
      const due = t0 + (performance.now() - start) * speed;
      while (i < samples.length && samples[i].tMs <= due) {
        const batch = samples.slice(i, i + P.BATCH_SAMPLES);
        i += batch.length;
        ingest(batch[0].flags & P.FLAG_SESSION_ACTIVE ? 1 : 0, batch);
      }
      if (i >= samples.length) {
        this.stop();
        setSource('file', `${name}, finished`);
      }
    }, 50);
  }
}

// ---- Source: the wearable over Web Bluetooth -----------------------------------------------------
class Wearable {
  constructor() {
    this.device = null;
    this.wanted = false; // true while the user wants the link: drops are then reconnected
    this._onImu = (ev) => {
      try {
        const { sessionId, samples } = P.decodeBatch(ev.target.value);
        ingest(sessionId, samples);
      } catch { /* a malformed packet is skipped */ }
    };
    this._onDrop = () => this.dropped();
  }
  get connected() { return Boolean(this.device && this.device.gatt && this.device.gatt.connected); }

  async choose() {
    if (!navigator.bluetooth) {
      setSource(null, 'Bluetooth needs Edge or Chrome on http://localhost');
      return;
    }
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: P.DEVICE_NAME_PREFIX }],
      optionalServices: [P.SERVICE_UUID],
    });
    if (this.device) this.device.removeEventListener('gattserverdisconnected', this._onDrop);
    this.device = device;
    device.addEventListener('gattserverdisconnected', this._onDrop);
    this.wanted = true;
    player.stop();
    await this.open();
  }

  async open() {
    setSource('ble', `${this.device.name || 'Wearable'}: connecting`);
    const server = await this.device.gatt.connect();
    const service = await server.getPrimaryService(P.SERVICE_UUID);
    const imu = await service.getCharacteristic(P.IMU_CHAR_UUID);
    imu.removeEventListener('characteristicvaluechanged', this._onImu);
    imu.addEventListener('characteristicvaluechanged', this._onImu);
    resetCounters();
    await imu.startNotifications();
    setSource('ble', `${this.device.name || 'Wearable'} over Bluetooth`);
    $('btn-connect').textContent = 'Disconnect wearable';
  }

  disconnect() {
    this.wanted = false;
    if (this.connected) this.device.gatt.disconnect();
    $('btn-connect').textContent = 'Connect wearable';
    if (state.source === 'ble') setSource(null, 'Wearable disconnected');
  }

  async dropped() {
    if (!this.wanted) return;
    for (let attempt = 1; attempt <= 10 && this.wanted; attempt++) {
      setSource('ble', `${this.device.name || 'Wearable'}: link lost, reconnecting (${attempt})`);
      try {
        await this.open();
        return;
      } catch {
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
    if (this.wanted) this.disconnect();
  }
}

const player = new Player();
const ble = new Wearable();
const server = new Server();

$('btn-connect').addEventListener('click', async () => {
  if (ble.wanted) { ble.disconnect(); return; }
  try {
    await ble.choose();
  } catch (e) {
    if (!e || e.name !== 'NotFoundError') setSource(null, `Could not connect: ${e.message}`);
  }
});

$('file-input').addEventListener('change', async (ev) => {
  const file = ev.target.files[0];
  if (file) player.play(await file.text(), file.name);
  ev.target.value = ''; // the same file can be chosen again
});

// for headless tests and debugging
window.uaarDemo = { state, player, ble, server, trace };
