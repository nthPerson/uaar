// Shared by the demo page and the recording page: the Web Bluetooth link to the wearable, a
// stand-in for testing without hardware, and the scrolling signal trace.

import * as P from './protocol.js';

/**
 * The wearable over Web Bluetooth.
 *   onSamples(sessionId, samples)  one decoded IMU notification (raw integer samples)
 *   onStatus(status)               firmware, calibrated, sessionActive ... (protocol.decodeStatus)
 *   onState(kind, text)            'connecting' | 'ready' | 'connected' | 'reconnecting' |
 *                                  'disconnected'; 'ready' fires just before samples start
 * A dropped link is reopened up to ten times for as long as the user wants the connection.
 */
export class Wearable {
  constructor({ onSamples, onStatus = () => {}, onState = () => {} }) {
    this.onSamples = onSamples;
    this.onStatus = onStatus;
    this.onState = onState;
    this.device = null;
    this.control = null;
    this.wanted = false;
    this._onImu = (ev) => {
      try {
        const { sessionId, samples } = P.decodeBatch(ev.target.value);
        this.onSamples(sessionId, samples);
      } catch { /* a malformed packet is skipped */ }
    };
    this._onStatus = (ev) => {
      try { this.onStatus(P.decodeStatus(ev.target.value)); } catch { /* ignored */ }
    };
    this._onDrop = () => this.dropped();
  }

  get connected() { return Boolean(this.device && this.device.gatt && this.device.gatt.connected); }
  get name() { return (this.device && this.device.name) || 'Wearable'; }

  /** Let the user pick a kit (needs a click) and connect to it. */
  async choose() {
    if (!navigator.bluetooth) throw new Error('Bluetooth needs Edge or Chrome on http://localhost');
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: P.DEVICE_NAME_PREFIX }],
      optionalServices: [P.SERVICE_UUID],
    });
    if (this.device) this.device.removeEventListener('gattserverdisconnected', this._onDrop);
    this.device = device;
    device.addEventListener('gattserverdisconnected', this._onDrop);
    this.wanted = true;
    await this.open();
  }

  async open() {
    this.onState('connecting', `${this.name}: connecting`);
    const server = await this.device.gatt.connect();
    const service = await server.getPrimaryService(P.SERVICE_UUID);
    const imu = await service.getCharacteristic(P.IMU_CHAR_UUID);
    imu.removeEventListener('characteristicvaluechanged', this._onImu);
    imu.addEventListener('characteristicvaluechanged', this._onImu);
    // status and control are extras: the stream works without them
    try {
      this.control = await service.getCharacteristic(P.CONTROL_CHAR_UUID);
      const status = await service.getCharacteristic(P.STATUS_CHAR_UUID);
      status.removeEventListener('characteristicvaluechanged', this._onStatus);
      status.addEventListener('characteristicvaluechanged', this._onStatus);
      this.onStatus(P.decodeStatus(await status.readValue()));
      await status.startNotifications();
    } catch { /* keep going with the sample stream only */ }
    this.onState('ready', `${this.name}: connected`);
    await imu.startNotifications();
    this.onState('connected', `${this.name} over Bluetooth`);
  }

  /** Send a one-byte command (protocol.CMD_*). */
  async write(cmd) {
    if (!this.control || !this.connected) throw new Error('the wearable is not connected');
    const bytes = Uint8Array.of(cmd);
    if (this.control.writeValueWithoutResponse) await this.control.writeValueWithoutResponse(bytes);
    else await this.control.writeValue(bytes);
  }

  disconnect() {
    this.wanted = false;
    if (this.connected) this.device.gatt.disconnect();
    this.onState('disconnected', 'Wearable disconnected');
  }

  async dropped() {
    if (!this.wanted) return;
    for (let attempt = 1; attempt <= 10 && this.wanted; attempt++) {
      this.onState('reconnecting', `${this.name}: link lost, reconnecting (${attempt})`);
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

/**
 * A pretend wearable for testing the pages without hardware (add ?sim to the address): 50 Hz
 * samples, a slow "curl" while its session flag is on. `pressButton()` toggles the flag like
 * the kit's button; the start/stop commands set it.
 */
export class Simulator {
  constructor({ onSamples, onStatus = () => {}, onState = () => {} }) {
    this.onSamples = onSamples;
    this.onStatus = onStatus;
    this.onState = onState;
    this.connected = false;
    this.wanted = false;
    this.name = 'UAAR-SIM';
    this.active = false;
    this.seq = 0;
    this.tMs = 1000;
    this.sessionId = 0;
    this.timer = null;
  }
  async choose() {
    this.connected = this.wanted = true;
    this.onStatus({ firmware: 'sim', calibrated: true, sessionActive: false, sessionId: 0, uptimeS: 0 });
    this.onState('ready', `${this.name}: connected`);
    this.timer = setInterval(() => this.tick(), 1000 / P.SAMPLE_RATE_HZ * P.BATCH_SAMPLES);
    this.onState('connected', `${this.name} (simulated)`);
  }
  tick() {
    const samples = [];
    for (let i = 0; i < P.BATCH_SAMPLES; i++) {
      const t = this.tMs / 1000;
      const move = this.active ? Math.sin(2 * Math.PI * 0.6 * t) : 0;
      const noise = () => Math.round((Math.random() - 0.5) * 60);
      samples.push({
        tMs: this.tMs, seq: this.seq,
        ax: Math.round(2500 * move) + noise(), ay: noise(), az: Math.round(4096 * (1 - 0.3 * Math.abs(move))) + noise(),
        gx: noise(), gy: Math.round(3000 * Math.cos(2 * Math.PI * 0.6 * t) * (this.active ? 1 : 0)) + noise(), gz: noise(),
        flags: (this.active ? P.FLAG_SESSION_ACTIVE : 0) | P.FLAG_CALIBRATED,
      });
      this.tMs += 1000 / P.SAMPLE_RATE_HZ;
      this.seq = (this.seq + 1) % 65536;
    }
    this.onSamples(this.sessionId, samples);
  }
  pressButton() { this.setActive(!this.active); }
  setActive(on) { if (on && !this.active) this.sessionId += 1; this.active = on; }
  async write(cmd) {
    if (cmd === P.CMD_START_SESSION) this.setActive(true);
    if (cmd === P.CMD_STOP_SESSION) this.setActive(false);
  }
  disconnect() {
    clearInterval(this.timer);
    this.connected = this.wanted = false;
    this.onState('disconnected', 'Wearable disconnected');
  }
}

/**
 * Scrolling line trace of the last `seconds` of samples in physical units (protocol.toPhysical).
 * `series` is a list of [key, css colour token]; the vertical range follows the signal and is
 * never smaller than `minRange`. `band(sample)` may return true to shade a sample's column.
 */
export class Trace {
  constructor(canvas, { series, minRange = 1.5, seconds = 10, band = null }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.series = series;
    this.minRange = minRange;
    this.band = band;
    this.cap = seconds * P.SAMPLE_RATE_HZ;
    this.buf = [];
    this.range = minRange;
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
    const x = (i) => ((i + this.cap - buf.length) / (this.cap - 1)) * w;
    if (this.band && buf.length) {
      ctx.fillStyle = css.getPropertyValue('--band');
      const step = w / (this.cap - 1);
      buf.forEach((s, i) => { if (this.band(s)) ctx.fillRect(x(i), 0, step + 1, h); });
    }
    ctx.strokeStyle = css.getPropertyValue('--hairline');
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    if (buf.length < 2) return;
    let peak = this.minRange;
    for (const s of buf) for (const [key] of this.series) peak = Math.max(peak, Math.abs(s[key]));
    this.range += (peak * 1.1 - this.range) * 0.1;
    const y = (v) => h / 2 - (v / this.range) * (h / 2 - 4);
    ctx.lineWidth = Math.max(1.5, w / 900);
    ctx.lineJoin = 'round';
    for (const [key, token] of this.series) {
      ctx.strokeStyle = css.getPropertyValue(token);
      ctx.beginPath();
      buf.forEach((s, i) => (i ? ctx.lineTo(x(i), y(s[key])) : ctx.moveTo(x(i), y(s[key]))));
      ctx.stroke();
    }
  }
}
