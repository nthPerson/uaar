// UAAR recording page: record labelled sets from the wearable, following the checklist of
// configs/wearable.yaml. The server writes every sample to data/wearable/<person>/<session>/
// as it arrives (uaar.live.recorder); this page decides when a set starts and ends.
//
// One person can run it alone: the kit's button starts a set (after a countdown) and stops it
// early; with "stop by itself" a set ends after its seconds and the page moves to the next one.
// Add ?sim to the address to try the page without a kit (Space is then the kit's button).

import * as P from './protocol.js';
import { Simulator, Trace, Wearable } from './link.js';

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const SIM = new URLSearchParams(location.search).has('sim');
const MAX_LOST = 0.01; // a set that lost more than 1 % of its samples is flagged
const MIN_SECONDS = 10; // so is a set shorter than this (fine-tuning needs a few windows of it)

/** Why a kept set should be recorded again, or '' if it is fine. */
function problem(set) {
  if (set.lost_share > MAX_LOST) return `${(100 * set.lost_share).toFixed(1)}% of the samples lost`;
  if (set.seconds < MIN_SECONDS) return `only ${set.seconds.toFixed(1)} s`;
  return '';
}

const state = {
  phase: 'idle', // idle (no session) | ready | countdown | recording
  recording: false, // a session is open on the server
  session: null,
  protocol: [],
  seconds: {}, // label -> seconds, when changed on the page
  sets: [],
  current: null, // label of the next set
  firmware: '',
  calibrated: null,
  flag: false, // the kit's session flag, as of the last sample
  sessionId: 0,
  rate: 0,
  lost: 0,
  lastSeq: null,
  t0: 0, // performance.now() at the start of the countdown or of the set
  timer: null,
  expectFall: false, // we told the kit to stop: its flag falling is not a button press
  lastKept: '', // one line about the set that was just saved
};

// ---- traces --------------------------------------------------------------------------------------
const band = (s) => s.rec;
const traceAcc = new Trace($('trace-acc'), { series: [['ax', '--ink'], ['ay', '--ink-2'], ['az', '--muted']], band });
const traceGyr = new Trace($('trace-gyr'), { series: [['gx', '--ink'], ['gy', '--ink-2'], ['gz', '--muted']], minRange: 100, band });

// ---- sound ---------------------------------------------------------------------------------------
let audio = null;
function beep(hz = 880, ms = 120) {
  if (!$('opt-sound').checked) return;
  try {
    audio = audio || new AudioContext();
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.frequency.value = hz;
    gain.gain.value = 0.15;
    osc.connect(gain).connect(audio.destination);
    osc.start();
    osc.stop(audio.currentTime + ms / 1000);
  } catch { /* no sound is not a reason to stop recording */ }
}

// ---- the wearable --------------------------------------------------------------------------------
const Link = SIM ? Simulator : Wearable;
const kit = new Link({
  onSamples: (sessionId, samples) => {
    state.sessionId = sessionId;
    const rec = state.phase === 'recording';
    for (const s of samples) {
      state.lost += P.seqGap(state.lastSeq, s.seq);
      state.lastSeq = s.seq;
      const p = P.toPhysical(s);
      p.rec = rec;
      traceAcc.push(p);
      traceGyr.push(p);
    }
    state.rate += samples.length;
    server.send({
      type: 'samples',
      session_id: sessionId,
      samples: samples.map((s) => [s.tMs, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq]),
    });
    const flag = (samples[samples.length - 1].flags & P.FLAG_SESSION_ACTIVE) !== 0;
    if (flag !== state.flag) {
      state.flag = flag;
      onFlag(flag);
    }
  },
  onStatus: (st) => {
    state.firmware = st.firmware;
    state.calibrated = st.calibrated;
    renderDevice();
  },
  onState: (kind, text) => {
    if (kind === 'ready') { state.lastSeq = null; state.flag = false; }
    $('device-name').textContent = text;
    $('btn-connect').textContent = kit.wanted ? 'Disconnect wearable' : 'Connect wearable';
    if (kind === 'disconnected' && state.phase !== 'idle') cancelSet('The wearable disconnected');
    render();
  },
});

function renderDevice() {
  const cal = state.calibrated == null ? '' : state.calibrated ? 'gyro calibrated' : 'gyro not calibrated';
  $('device-info').textContent = [state.firmware && `firmware ${state.firmware}`, cal].filter(Boolean).join(', ');
}

async function command(cmd) {
  try { await kit.write(cmd); } catch { /* the kit's button still works */ }
}

// ---- the server ----------------------------------------------------------------------------------
class Server {
  constructor() { this.ws = null; this.connect(); }
  get open() { return this.ws && this.ws.readyState === WebSocket.OPEN; }
  send(msg) { if (this.open) this.ws.send(JSON.stringify(msg)); }
  connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    this.ws = ws;
    ws.onopen = () => showBanner('');
    ws.onclose = () => {
      showBanner('The server is not running, so nothing is being saved. Start it with <code>uv run scripts/demo.py</code>; this page reconnects by itself.');
      if (state.phase === 'countdown' || state.phase === 'recording') cancelSet('The server stopped');
      setTimeout(() => this.connect(), 1500);
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'rec_state') onRecState(msg);
      else if (msg.type === 'set') onSetKept(msg);
      else if (msg.type === 'rec_summary') onSummary(msg);
      else if (msg.type === 'rec_error') setNow(state.phase, msg.text, '');
    };
  }
}

function showBanner(html) {
  $('banner').hidden = !html;
  $('banner').innerHTML = html || '';
}

function onRecState(msg) {
  state.recording = msg.recording;
  state.session = msg.session;
  state.protocol = msg.protocol || [];
  state.sets = msg.sets || [];
  const people = $('subjects');
  for (const s of msg.subjects || []) {
    if (![...people.options].some((o) => o.value === s)) people.append(new Option(s, s));
  }
  if (msg.recording) {
    $('subject').value = msg.subject;
    if (state.phase === 'idle') state.phase = 'ready';
    if (!state.current || isDone(state.current)) state.current = nextLabel();
  } else {
    state.phase = 'idle';
  }
  render();
}

// ---- sets ----------------------------------------------------------------------------------------
const row = (label) => state.protocol.find((p) => p.label === label);
const done = (label) => state.sets.filter((s) => s.label === label).length;
const isDone = (label) => { const r = row(label); return r ? done(label) >= r.sets : false; };
const nextLabel = () => (state.protocol.find((p) => done(p.label) < p.sets) || {}).label || null;
const targetSeconds = (label) => state.seconds[label] ?? (row(label) || {}).seconds ?? 30;

/** The kit's session flag changed: its button was pressed, or a command of ours arrived. */
function onFlag(active) {
  if (active) {
    if (state.phase === 'ready' && state.current) startCountdown();
    return;
  }
  if (state.expectFall) { state.expectFall = false; return; }
  if (state.phase === 'countdown') cancelSet('Cancelled');
  else if (state.phase === 'recording') endSet();
}

function startCountdown() {
  const seconds = Math.max(0, Number($('opt-countdown').value) || 0);
  state.phase = 'countdown';
  state.t0 = performance.now();
  let last = null;
  clearInterval(state.timer);
  const tick = () => {
    const left = seconds - (performance.now() - state.t0) / 1000;
    if (left <= 0) { startSet(); return; }
    const whole = Math.ceil(left);
    if (whole !== last) { last = whole; beep(660, 90); }
    setNow('countdown', `${state.current} starts in ${whole}`, 'Get into position', 1 - left / seconds);
  };
  state.timer = setInterval(tick, 50);
  tick();
  render();
}

function startSet() {
  clearInterval(state.timer);
  state.phase = 'recording';
  state.t0 = performance.now();
  server.send({ type: 'set_start', label: state.current, session_id: state.sessionId });
  beep(1320, 250);
  const label = state.current;
  const target = targetSeconds(label);
  state.timer = setInterval(() => {
    const t = (performance.now() - state.t0) / 1000;
    const auto = $('opt-auto').checked;
    if (auto && t >= target) { endSet(true); return; }
    const detail = auto ? `stops by itself at ${target} s` : 'press the button to stop';
    setNow('recording', `Recording ${label}: ${t.toFixed(0)} s`, detail, auto ? t / target : 0);
  }, 100);
  render();
}

/** Keep the set. `byPage` is true when the page ends it (the kit is then told to stop too). */
function endSet(byPage = false) {
  clearInterval(state.timer);
  server.send({ type: 'set_end' });
  if (byPage && state.flag) { state.expectFall = true; command(P.CMD_STOP_SESSION); }
  beep(440, 350);
  state.phase = 'ready';
  render();
}

function cancelSet(why) {
  clearInterval(state.timer);
  if (state.phase === 'recording') server.send({ type: 'set_cancel' });
  if (state.flag) { state.expectFall = true; command(P.CMD_STOP_SESSION); }
  state.phase = state.recording ? 'ready' : 'idle';
  render();
  if (why) $('now-detail').textContent = `${why}. ${$('now-detail').textContent}`;
}

function onSetKept(set) {
  const bad = problem(set);
  state.lastKept = `Saved ${set.label}: ${set.seconds.toFixed(1)} s, ${fmt(set.n_dropped)} samples lost` + (bad ? ` (${bad}: discard it and record it again)` : '');
  state.sets = state.sets.filter((s) => s.set_index !== set.set_index).concat([set]);
  if (isDone(state.current)) state.current = nextLabel();
  render();
}

function onSummary(msg) {
  const box = $('summary');
  box.hidden = false;
  if (!msg.saved) {
    box.textContent = 'No set was kept, so nothing was saved.';
    return;
  }
  const missing = msg.missing.length
    ? `<p>Still to record for the full checklist:</p><ul>${msg.missing.map((m) => `<li><code>${m.label}</code>: ${m.missing} more</li>`).join('')}</ul>`
    : '<p>The checklist is complete.</p>';
  box.innerHTML = `<strong>Session saved</strong><p><code>${msg.folder}</code></p>`
    + `<p>${fmt(msg.n_samples)} samples, ${fmt(msg.n_dropped)} lost, ${msg.labels.reduce((n, l) => n + l.sets, 0)} sets.</p>${missing}`
    + `<p>Next: <code>git add data/wearable/${msg.subject}</code>, then <code>uv run scripts/finetune.py --run models/k4taught_recofit_edl_fold0_bg0_seed0 --train data/wearable/${msg.subject} --test data/wearable/&lt;the other person&gt;</code></p>`;
}

// ---- drawing -------------------------------------------------------------------------------------
function setNow(phase, title, detail, progress = 0) {
  $('now').className = `now ${phase}`;
  $('now-title').textContent = title;
  $('now-detail').textContent = detail;
  $('now-bar').style.width = `${Math.round(100 * Math.min(1, Math.max(0, progress)))}%`;
}

function render() {
  const connected = kit.connected;
  $('btn-calibrate').disabled = !connected || state.phase === 'countdown' || state.phase === 'recording';
  $('btn-session').disabled = !server.open || (!state.recording && !connected);
  $('btn-session').textContent = state.recording ? 'Finish session' : 'Start session';
  $('btn-set').disabled = !state.recording || !connected || (state.phase === 'ready' && !state.current);
  $('btn-set').textContent = state.phase === 'recording' ? 'Stop set' : state.phase === 'countdown' ? 'Cancel' : 'Start set';
  $('subject').disabled = $('placement').disabled = state.recording;

  if (state.phase === 'idle') {
    setNow('idle', connected ? 'Start a session to begin' : 'Connect the wearable, then start a session',
      connected ? 'Choose the person first. Everything after that can be done with the kit button.' : '');
  } else if (state.phase === 'ready') {
    if (!connected) setNow('idle', 'The wearable is not connected', 'Reconnect it to go on; the session stays open.');
    else if (state.current) {
      const r = row(state.current) || {};
      const n = done(state.current) + 1;
      const of = r.sets ? ` (set ${n} of ${Math.max(r.sets, n)})` : '';
      setNow('ready', `Next: ${state.current}${of}`, `${state.lastKept ? state.lastKept + '. ' : ''}${r.hint ? r.hint + '. ' : ''}Press the kit button or Space to start.`);
    } else setNow('ready', 'Everything on the checklist is recorded', `${state.lastKept ? state.lastKept + '. ' : ''}Finish the session, or click a label to record more of it.`);
  }

  const body = $('checklist').tBodies[0];
  body.replaceChildren(...state.protocol.map((p) => {
    const tr = document.createElement('tr');
    const n = done(p.label);
    tr.className = `${p.label === state.current ? 'current' : ''} ${n >= p.sets ? 'done' : ''}`;
    const cell = (text, cls) => { const td = document.createElement('td'); td.textContent = text; if (cls) td.className = cls; return td; };
    const roleText = { known: 'known exercise', background: 'teaching "unsure"', unknown: 'test: unknown', shifted: 'test: kit worn differently' }[p.role] || '';
    const secs = document.createElement('input');
    secs.type = 'number'; secs.min = 5; secs.max = 600; secs.className = 'seconds';
    secs.value = targetSeconds(p.label);
    secs.addEventListener('click', (ev) => ev.stopPropagation());
    secs.addEventListener('change', () => { state.seconds[p.label] = Math.max(5, Number(secs.value) || p.seconds); });
    const tdSecs = document.createElement('td');
    tdSecs.append(secs);
    tr.append(cell(p.label, 'label'), cell(p.hint), cell(roleText), cell(`${n} of ${p.sets}`, `sets-count${n >= p.sets ? ' done' : ''}`), tdSecs);
    tr.addEventListener('click', () => {
      if (state.phase === 'countdown' || state.phase === 'recording') return;
      state.current = p.label;
      render();
    });
    return tr;
  }));

  const sets = $('sets').tBodies[0];
  $('sets-empty').hidden = state.sets.length > 0;
  sets.replaceChildren(...state.sets.map((s, i) => {
    const tr = document.createElement('tr');
    const cell = (text, cls) => { const td = document.createElement('td'); td.textContent = text; if (cls) td.className = cls; return td; };
    const bad = problem(s);
    const btn = document.createElement('button');
    btn.textContent = 'Discard';
    btn.addEventListener('click', () => server.send({ type: 'set_discard', set_index: s.set_index }));
    const td = document.createElement('td');
    td.append(btn);
    const short = s.seconds < MIN_SECONDS;
    const lossy = s.lost_share > MAX_LOST;
    tr.append(cell(String(i + 1)), cell(s.label, 'label'),
      cell(short ? `${s.seconds.toFixed(1)}: too short` : s.seconds.toFixed(1), short ? 'bad' : ''),
      cell(lossy ? `${fmt(s.n_dropped)} (${(100 * s.lost_share).toFixed(1)}%): record again` : fmt(s.n_dropped), lossy ? 'bad' : ''), td);
    if (bad) tr.title = `${bad}: discard this set and record it again`;
    return tr;
  }));
}

setInterval(() => {
  $('st-rate').textContent = fmt(state.rate);
  state.rate = 0;
  $('st-lost').textContent = fmt(state.lost);
}, 1000);

// ---- controls ------------------------------------------------------------------------------------
const server = new Server();

$('btn-connect').addEventListener('click', async () => {
  if (kit.wanted) { kit.disconnect(); return; }
  try {
    await kit.choose();
  } catch (e) {
    if (!e || e.name !== 'NotFoundError') $('device-name').textContent = `Could not connect: ${e.message}`;
  }
  render();
});

$('btn-calibrate').addEventListener('click', async () => {
  $('device-info').textContent = 'calibrating: keep the kit still for 3 seconds';
  await command(P.CMD_CALIBRATE);
  setTimeout(renderDevice, 4000);
});

$('btn-session').addEventListener('click', () => {
  if (state.recording) {
    if (state.phase === 'countdown' || state.phase === 'recording') cancelSet('');
    server.send({ type: 'rec_stop' });
    state.lastKept = '';
    return;
  }
  $('summary').hidden = true;
  state.lost = 0;
  state.lastKept = '';
  state.current = null;
  if (state.flag) { state.expectFall = true; command(P.CMD_STOP_SESSION); } // start from "not in a set"
  beep(880, 60); // a click also lets the browser play the later beeps
  server.send({
    type: 'rec_start',
    subject: $('subject').value.trim(),
    placement: $('placement').value.trim(),
    device: kit.name,
    firmware: state.firmware,
    transport: SIM ? 'sim' : 'ble',
  });
});

/** The on-screen button and Space do what the kit's button does. */
function pressButton() {
  if (!state.recording || !kit.connected) return;
  if (SIM) { kit.pressButton(); return; }
  command(state.flag ? P.CMD_STOP_SESSION : P.CMD_START_SESSION);
}
$('btn-set').addEventListener('click', pressButton);
// Space is the kit button everywhere except while typing. A focused button would otherwise be
// "clicked" by Space (the browser does that on key-up), which could finish the session.
const typing = (el) => el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && el.type !== 'checkbox');
document.addEventListener('keydown', (ev) => {
  if (ev.code !== 'Space' || typing(ev.target)) return;
  ev.preventDefault();
  if (!ev.repeat) pressButton();
});
document.addEventListener('keyup', (ev) => {
  if (ev.code === 'Space' && !typing(ev.target)) ev.preventDefault();
});
document.addEventListener('click', (ev) => {
  if (ev.target.tagName === 'BUTTON') ev.target.blur();
});

if (SIM) $('btn-connect').textContent = 'Connect simulated wearable';
render();

// for headless tests and debugging
window.uaarRecord = { state, kit, server };
