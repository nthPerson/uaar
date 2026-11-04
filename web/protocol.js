// UAAR wearable wire protocol, browser side. Mirror of firmware/include/protocol.h and
// src/uaar/data/wearable.py; tests/test_web_protocol.py pins the constants below to the Python
// module, so change all three together.
//
// Plain ES module, no dependencies, usable from the page (app.js) and from node:test.

// ---- Identity --------------------------------------------------------------------------------
export const DEVICE_NAME_PREFIX = 'UAAR-';
export const SERVICE_UUID = '7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001';
export const IMU_CHAR_UUID = '7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001'; // notify: sample batches
export const CONTROL_CHAR_UUID = '7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001'; // write: 1-byte command
export const STATUS_CHAR_UUID = '7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001'; // read/notify: status
export const REQUESTED_MTU = 185; // informational: Web Bluetooth cannot request an MTU

// ---- Sampling --------------------------------------------------------------------------------
export const SAMPLE_RATE_HZ = 50;
export const BATCH_SAMPLES = 5;
export const ACC_LSB_PER_G = 4096;
export const GYR_LSB_PER_DPS = 32.8;

// ---- Commands (host -> wearable, one byte on the control characteristic) ---------------------
export const CMD_LED_KNOWN = 0x01;
export const CMD_LED_UNKNOWN = 0x02;
export const CMD_START_SESSION = 0x03;
export const CMD_STOP_SESSION = 0x04;
export const CMD_CALIBRATE = 0x05;
export const CMD_PING = 0x10;
export const CMD_IDENTIFY = 0x11;

// ---- Sample flags ----------------------------------------------------------------------------
export const FLAG_RESERVED = 1 << 0;
export const FLAG_SESSION_ACTIVE = 1 << 1;
export const FLAG_BUTTON_PRESSED = 1 << 2;
export const FLAG_CALIBRATED = 1 << 3;

// ---- Wire sizes (little-endian, packed) ------------------------------------------------------
// Sample  <I6hBH : t_ms u32, ax ay az gx gy gz i16, flags u8, seq u16      -> 19 bytes
// Header  <HBB   : session_id u16, n u8, reserved u8                        -> 4 bytes
// Status  <BBBHBIBB : fw major/minor/patch u8, session_id u16, reserved u8,
//                     uptime_s u32, calibrated u8, session_active u8        -> 12 bytes
export const SAMPLE_SIZE = 19;
export const BATCH_HEADER_SIZE = 4;
export const MAX_BATCH_SIZE = BATCH_HEADER_SIZE + BATCH_SAMPLES * SAMPLE_SIZE; // 99
export const STATUS_SIZE = 12;

// ---- Serial / CSV ----------------------------------------------------------------------------
export const SERIAL_CSV_HEADER = 't_ms,ax,ay,az,gx,gy,gz,flags,seq';
export const RAW_COLUMNS = SERIAL_CSV_HEADER.split(',');
export const EVENTS_HEADER = 'set_index,label,session_id,t_start_ms,t_end_ms,seq_start,seq_end,n_samples,n_dropped';

/** Accept a DataView, ArrayBuffer or typed array and return a DataView over the same bytes. */
export function toDataView(buf) {
  if (buf instanceof DataView) return buf;
  if (buf instanceof ArrayBuffer) return new DataView(buf);
  if (ArrayBuffer.isView(buf)) return new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  throw new TypeError('expected DataView, ArrayBuffer or typed array');
}

/**
 * One IMU notification -> { sessionId, samples }.
 * Decodes by the header's `n` field only, so any notification length (MTU) works as long as
 * the declared samples are present. Throws RangeError on a bad length.
 */
export function decodeBatch(buf) {
  const dv = toDataView(buf);
  if (dv.byteLength < BATCH_HEADER_SIZE) {
    throw new RangeError(`batch too short: ${dv.byteLength} bytes`);
  }
  const sessionId = dv.getUint16(0, true);
  const n = dv.getUint8(2);
  const expected = BATCH_HEADER_SIZE + n * SAMPLE_SIZE;
  if (n > BATCH_SAMPLES || dv.byteLength < expected) {
    throw new RangeError(`batch declares ${n} samples but has ${dv.byteLength} bytes`);
  }
  const samples = new Array(n);
  for (let i = 0; i < n; i++) {
    const o = BATCH_HEADER_SIZE + i * SAMPLE_SIZE;
    samples[i] = {
      tMs: dv.getUint32(o, true),
      ax: dv.getInt16(o + 4, true),
      ay: dv.getInt16(o + 6, true),
      az: dv.getInt16(o + 8, true),
      gx: dv.getInt16(o + 10, true),
      gy: dv.getInt16(o + 12, true),
      gz: dv.getInt16(o + 14, true),
      flags: dv.getUint8(o + 16),
      seq: dv.getUint16(o + 17, true),
    };
  }
  return { sessionId, samples };
}

/** Inverse of decodeBatch (tests and the simulator). Returns a Uint8Array. */
export function encodeBatch(sessionId, samples) {
  if (samples.length > BATCH_SAMPLES) throw new RangeError('too many samples for one batch');
  const out = new Uint8Array(BATCH_HEADER_SIZE + samples.length * SAMPLE_SIZE);
  const dv = new DataView(out.buffer);
  dv.setUint16(0, sessionId & 0xffff, true);
  dv.setUint8(2, samples.length);
  dv.setUint8(3, 0);
  samples.forEach((s, i) => {
    const o = BATCH_HEADER_SIZE + i * SAMPLE_SIZE;
    dv.setUint32(o, s.tMs >>> 0, true);
    dv.setInt16(o + 4, s.ax, true);
    dv.setInt16(o + 6, s.ay, true);
    dv.setInt16(o + 8, s.az, true);
    dv.setInt16(o + 10, s.gx, true);
    dv.setInt16(o + 12, s.gy, true);
    dv.setInt16(o + 14, s.gz, true);
    dv.setUint8(o + 16, s.flags & 0xff);
    dv.setUint16(o + 17, s.seq & 0xffff, true);
  });
  return out;
}

/** 12-byte status characteristic -> plain object (same keys as the Python decode_status). */
export function decodeStatus(buf) {
  const dv = toDataView(buf);
  if (dv.byteLength < STATUS_SIZE) {
    throw new RangeError(`status too short: ${dv.byteLength} bytes`);
  }
  return {
    firmware: `${dv.getUint8(0)}.${dv.getUint8(1)}.${dv.getUint8(2)}`,
    sessionId: dv.getUint16(3, true),
    uptimeS: dv.getUint32(6, true),
    calibrated: dv.getUint8(10) !== 0,
    sessionActive: dv.getUint8(11) !== 0,
  };
}

/** Inverse of decodeStatus (simulator). */
export function encodeStatus({ firmware = '0.0.0', sessionId = 0, uptimeS = 0, calibrated = false, sessionActive = false }) {
  const [major, minor, patch] = firmware.split('.').map((x) => parseInt(x, 10) || 0);
  const out = new Uint8Array(STATUS_SIZE);
  const dv = new DataView(out.buffer);
  dv.setUint8(0, major);
  dv.setUint8(1, minor);
  dv.setUint8(2, patch);
  dv.setUint16(3, sessionId & 0xffff, true);
  dv.setUint8(5, 0);
  dv.setUint32(6, uptimeS >>> 0, true);
  dv.setUint8(10, calibrated ? 1 : 0);
  dv.setUint8(11, sessionActive ? 1 : 0);
  return out;
}

/** Raw counts -> { t (s), ax..az (g), gx..gz (deg/s), amag (g), flags, seq }. */
export function toPhysical(s) {
  const ax = s.ax / ACC_LSB_PER_G;
  const ay = s.ay / ACC_LSB_PER_G;
  const az = s.az / ACC_LSB_PER_G;
  return {
    t: s.tMs / 1000,
    ax,
    ay,
    az,
    amag: Math.sqrt(ax * ax + ay * ay + az * az),
    gx: s.gx / GYR_LSB_PER_DPS,
    gy: s.gy / GYR_LSB_PER_DPS,
    gz: s.gz / GYR_LSB_PER_DPS,
    flags: s.flags,
    seq: s.seq,
  };
}

/**
 * Dropped samples implied by two consecutive rolling 16-bit seq values.
 * A step of 1 (mod 65536) is normal and returns 0; a step of k > 1 returns k - 1. The step is
 * taken modulo 65536 exactly like uaar.data.wearable.seq_gaps, so a wrap 65535 -> 0 is clean
 * (and, like Python, a backwards jump counts as a large gap). `prev` may be null/undefined for
 * the first sample.
 */
export function seqGap(prev, cur) {
  if (prev == null) return 0;
  const step = (((cur - prev) % 65536) + 65536) % 65536;
  return step > 1 ? step - 1 : 0;
}

/** One sample -> the exact serial CSV line (raw integer counts). */
export function sampleToCsvLine(s) {
  return `${s.tMs},${s.ax},${s.ay},${s.az},${s.gx},${s.gy},${s.gz},${s.flags},${s.seq}`;
}

/** Serial CSV line -> sample object, or null for comments, the header and malformed lines. */
export function parseCsvLine(line) {
  line = line.trim();
  if (!line || line.startsWith('#') || line === SERIAL_CSV_HEADER) return null;
  const parts = line.split(',');
  if (parts.length !== 9) return null;
  const v = parts.map((p) => Number(p));
  if (v.some((x) => !Number.isInteger(x))) return null;
  return { tMs: v[0], ax: v[1], ay: v[2], az: v[3], gx: v[4], gy: v[5], gz: v[6], flags: v[7], seq: v[8] };
}

/** Hex string <-> bytes, for tests and log output. */
export function hexToBytes(hex) {
  const clean = hex.replace(/[^0-9a-fA-F]/g, '');
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(clean.substr(i * 2, 2), 16);
  return out;
}

export function bytesToHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}
