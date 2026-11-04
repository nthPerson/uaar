// node --test web/test   (node >= 18). Also run by tests/test_web_protocol.py, which passes a
// freshly generated golden packet from uaar.data.wearable.encode_batch via UAAR_GOLDEN_JSON.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as P from '../protocol.js';

// Golden batch produced by Python: encode_batch(7, [Sample(1000+20*i, 1408, -235, 3800, -3, 7,
// -2, FLAG_SESSION_ACTIVE|FLAG_CALIBRATED, 65531+i) for i in range(5)]).hex()
const GOLDEN_HEX = '07000500e8030000800515ffd80efdff0700feff0afbfffc030000800515ffd80efdff0700feff0afcff10040000800515ffd80efdff0700feff0afdff24040000800515ffd80efdff0700feff0afeff38040000800515ffd80efdff0700feff0affff';
const GOLDEN = {
  session_id: 7,
  samples: Array.from({ length: 5 }, (_, i) => [1000 + 20 * i, 1408, -235, 3800, -3, 7, -2, 2 | 8, 65531 + i]),
};
// STATUS_STRUCT.pack(0, 1, 0, 3, 0, 4242, 1, 1).hex()
const STATUS_HEX = '000100030000921000000101';

function toObj(row) {
  const [tMs, ax, ay, az, gx, gy, gz, flags, seq] = row;
  return { tMs, ax, ay, az, gx, gy, gz, flags, seq };
}

function checkGolden(golden) {
  const bytes = P.hexToBytes(golden.hex);
  assert.equal(bytes.length, P.BATCH_HEADER_SIZE + golden.samples.length * P.SAMPLE_SIZE);
  const { sessionId, samples } = P.decodeBatch(bytes);
  assert.equal(sessionId, golden.session_id);
  assert.deepEqual(samples, golden.samples.map(toObj));
  // re-encode must reproduce the Python bytes exactly
  assert.equal(P.bytesToHex(P.encodeBatch(sessionId, samples)), golden.hex);
  // decoding works from a DataView and from an ArrayBuffer too
  assert.deepEqual(P.decodeBatch(new DataView(bytes.buffer)).samples, samples);
  assert.deepEqual(P.decodeBatch(bytes.buffer.slice(0)).samples, samples);
}

test('sizes match protocol.h', () => {
  assert.equal(P.SAMPLE_SIZE, 19);
  assert.equal(P.BATCH_HEADER_SIZE, 4);
  assert.equal(P.MAX_BATCH_SIZE, 99);
  assert.equal(P.STATUS_SIZE, 12);
});

test('decodes the embedded golden batch from Python encode_batch', () => {
  checkGolden({ hex: GOLDEN_HEX, ...GOLDEN });
});

test('decodes the golden batch passed by pytest (UAAR_GOLDEN_JSON)', { skip: !process.env.UAAR_GOLDEN_JSON }, () => {
  const golden = JSON.parse(readFileSync(process.env.UAAR_GOLDEN_JSON, 'utf8'));
  checkGolden(golden.batch);
  const st = P.decodeStatus(P.hexToBytes(golden.status.hex));
  assert.deepEqual(st, golden.status.expected);
});

test('decodeBatch rejects short payloads and honours n', () => {
  const bytes = P.hexToBytes(GOLDEN_HEX);
  assert.throws(() => P.decodeBatch(bytes.subarray(0, 30)), RangeError);
  // a shorter notification with n = 2 decodes two samples
  const two = P.encodeBatch(1, P.decodeBatch(bytes).samples.slice(0, 2));
  assert.equal(two.length, 4 + 2 * 19);
  assert.equal(P.decodeBatch(two).samples.length, 2);
  assert.throws(() => P.encodeBatch(1, new Array(6).fill(toObj(GOLDEN.samples[0]))), RangeError);
});

test('decodeStatus matches Python decode_status', () => {
  assert.deepEqual(P.decodeStatus(P.hexToBytes(STATUS_HEX)), {
    firmware: '0.1.0', sessionId: 3, uptimeS: 4242, calibrated: true, sessionActive: true,
  });
  assert.deepEqual(P.decodeStatus(P.encodeStatus({ firmware: '1.2.3', sessionId: 9, uptimeS: 77 })), {
    firmware: '1.2.3', sessionId: 9, uptimeS: 77, calibrated: false, sessionActive: false,
  });
});

test('toPhysical uses the LSB scales', () => {
  const p = P.toPhysical({ tMs: 20, ax: 4096, ay: 0, az: 0, gx: 328, gy: -328, gz: 0, flags: 2, seq: 1 });
  assert.equal(p.t, 0.02);
  assert.equal(p.ax, 1);
  assert.equal(p.amag, 1);
  assert.ok(Math.abs(p.gx - 10) < 1e-9 && Math.abs(p.gy + 10) < 1e-9);
});

test('seqGap mirrors seq_gaps: wrap is fine, k>1 steps count k-1', () => {
  assert.equal(P.seqGap(null, 5), 0);
  assert.equal(P.seqGap(65535, 0), 0);
  assert.equal(P.seqGap(1, 4), 2);
  assert.equal(P.seqGap(5, 9), 3);
  assert.equal(P.seqGap(10, 10), 0);
  // a backwards step wraps modulo 65536 exactly like np.diff(seq) % 65536 in seq_gaps
  assert.equal(P.seqGap(10, 3), 65528);
});

test('sampleToCsvLine / parseCsvLine round trip in the serial format', () => {
  const s = toObj(GOLDEN.samples[1]);
  const line = P.sampleToCsvLine(s);
  assert.equal(line, '1020,1408,-235,3800,-3,7,-2,10,65532');
  assert.deepEqual(P.parseCsvLine(line), s);
  assert.equal(P.parseCsvLine('# UAAR fw 0.1.0'), null);
  assert.equal(P.parseCsvLine(P.SERIAL_CSV_HEADER), null);
  assert.equal(P.parseCsvLine('1,2,3'), null);
});
