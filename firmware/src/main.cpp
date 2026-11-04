// UAAR wearable firmware: a 50 Hz six-axis IMU streamer for the CompE 510 project.
//
//   * BMI160 read at 100 Hz over I2C; pairs averaged to one 50 Hz sample (anti-aliasing)
//   * every sample goes out as a CSV line on USB serial and, when a central is connected,
//     in 5-sample BLE notifications (protocol.h)
//   * button: short press toggles a recording session (session_id increments on start),
//     3 s long press calibrates the gyro bias with the device held still
//   * LED shows the state (ui.h) and can flash "known" / "unknown" on command from the host
//   * no on-device inference, no sleep: the battery slide switch is the power control
#include <Arduino.h>
#include <Preferences.h>
#include <Wire.h>

#include "ble.h"
#include "bmi160.h"
#include "protocol.h"
#include "ui.h"

#ifndef UAAR_FW_VERSION
#define UAAR_FW_VERSION "0.0.0-dev"
#endif

namespace {
constexpr uint8_t PIN_BUTTON = D1;        // GPIO2, tactile switch to GND
constexpr uint8_t PIN_LED = LED_BUILTIN;  // GPIO21, active-low
constexpr uint32_t ODR_PERIOD_US = 1000000UL / UAAR_IMU_ODR_HZ;  // 10 ms
constexpr uint8_t READS_PER_SAMPLE = UAAR_IMU_ODR_HZ / UAAR_SAMPLE_RATE_HZ;  // 2
constexpr uint32_t STATUS_PERIOD_MS = 1000;
constexpr uint32_t CALIB_MS = 3000;

Bmi160 imu;
BleLink ble;
Button button;
StatusLed led;
Preferences prefs;

bool imu_ok = false;
bool session_active = false;
uint16_t session_id = 0;
uint16_t seq = 0;
uint32_t next_read_us = 0;
uint32_t last_status_ms = 0;
uint32_t imu_errors = 0;

// Averaging accumulator for one emitted sample.
int32_t acc_sum[3] = {0, 0, 0}, gyr_sum[3] = {0, 0, 0};
uint8_t reads_in_sample = 0;

// Gyro bias (LSB) measured with the device still; subtracted from every gyro read.
bool calibrated = false;
int32_t gyr_bias[3] = {0, 0, 0};
uint32_t calib_until_ms = 0;
int64_t calib_sum[3];
uint32_t calib_n = 0;

String deviceName() {
  const uint64_t mac = ESP.getEfuseMac();
  char buf[16];
  snprintf(buf, sizeof(buf), "%s%04X", UAAR_DEVICE_NAME_PREFIX, (unsigned)(mac & 0xFFFF));
  return String(buf);
}

void loadCalibration() {
  prefs.begin("uaar", false);
  calibrated = prefs.getBool("cal", false);
  if (calibrated) {
    gyr_bias[0] = prefs.getInt("gbx", 0);
    gyr_bias[1] = prefs.getInt("gby", 0);
    gyr_bias[2] = prefs.getInt("gbz", 0);
  }
  prefs.end();
}

void saveCalibration() {
  prefs.begin("uaar", false);
  prefs.putBool("cal", true);
  prefs.putInt("gbx", gyr_bias[0]);
  prefs.putInt("gby", gyr_bias[1]);
  prefs.putInt("gbz", gyr_bias[2]);
  prefs.end();
}

void refreshLed() {
  if (calib_until_ms) led.setMode(LedMode::Calibrating);
  else if (session_active) led.setMode(LedMode::Session);
  else led.setMode(ble.connected() ? LedMode::Connected : LedMode::Advertising);
}

void printBanner() {
  Serial.printf("# UAAR fw %s name=%s chip=%s rev=%d\n", UAAR_FW_VERSION, deviceName().c_str(),
                ESP.getChipModel(), ESP.getChipRevision());
  Serial.printf("# rate=%u Hz odr=%u Hz acc=%d LSB/g gyr=%.1f LSB/dps sample=%u B batch=%u B\n",
                UAAR_SAMPLE_RATE_HZ, UAAR_IMU_ODR_HZ, UAAR_ACC_LSB_PER_G, UAAR_GYR_LSB_PER_DPS,
                (unsigned)sizeof(UaarSample), (unsigned)sizeof(UaarBatch));
  if (imu_ok) Serial.printf("# bmi160 ok %s\n", imu.describe().c_str());
  else if (imu.address()) Serial.printf("# bmi160 FAILED (%s) %s\n", imu.failure(), imu.describe().c_str());
  else Serial.println("# bmi160 NOT FOUND (no I2C device at 0x68/0x69)");
  Serial.printf("# gyro bias %s (%ld,%ld,%ld) LSB\n", calibrated ? "applied" : "none",
                (long)gyr_bias[0], (long)gyr_bias[1], (long)gyr_bias[2]);
  Serial.printf("# session %s id=%u seq=%u imu_errors=%lu\n", session_active ? "active" : "idle",
                session_id, seq, (unsigned long)imu_errors);
  Serial.println(UAAR_SERIAL_CSV_HEADER);
}

void setSession(bool on) {
  if (on == session_active) return;
  session_active = on;
  if (on) ++session_id;
  Serial.printf("# session %s id=%u t_ms=%lu\n", on ? "start" : "stop", session_id,
                (unsigned long)millis());
  refreshLed();
}

void startCalibration() {
  calib_until_ms = millis() + CALIB_MS;
  calib_n = 0;
  calib_sum[0] = calib_sum[1] = calib_sum[2] = 0;
  Serial.println("# calibrating: hold still for 3 s");
  refreshLed();
}

void finishCalibration() {
  for (int i = 0; i < 3; ++i) gyr_bias[i] = (int32_t)(calib_sum[i] / (int64_t)calib_n);
  calib_until_ms = 0;
  calibrated = true;
  saveCalibration();
  Serial.printf("# calibrated n=%lu bias=(%ld,%ld,%ld) LSB\n", (unsigned long)calib_n,
                (long)gyr_bias[0], (long)gyr_bias[1], (long)gyr_bias[2]);
  refreshLed();
}

void handleCommand(uint8_t cmd) {
  switch (cmd) {
    case UAAR_CMD_LED_KNOWN: led.flash(LedFlash::Known); break;
    case UAAR_CMD_LED_UNKNOWN: led.flash(LedFlash::Unknown); break;
    case UAAR_CMD_START_SESSION: setSession(true); break;
    case UAAR_CMD_STOP_SESSION: setSession(false); break;
    case UAAR_CMD_CALIBRATE: startCalibration(); break;
    case UAAR_CMD_PING: Serial.println("# pong"); break;
    case UAAR_CMD_IDENTIFY: printBanner(); break;
    default: Serial.printf("# unknown command 0x%02X\n", cmd); break;
  }
}

void pollSerialCommands() {
  while (Serial.available()) {
    switch (Serial.read()) {
      case 'K': handleCommand(UAAR_CMD_LED_KNOWN); break;
      case 'U': handleCommand(UAAR_CMD_LED_UNKNOWN); break;
      case 'S': handleCommand(UAAR_CMD_START_SESSION); break;
      case 'X': handleCommand(UAAR_CMD_STOP_SESSION); break;
      case 'C': handleCommand(UAAR_CMD_CALIBRATE); break;
      case 'P': handleCommand(UAAR_CMD_PING); break;
      case 'I': handleCommand(UAAR_CMD_IDENTIFY); break;
      default: break;  // line endings and anything else
    }
  }
}

void publishStatus() {
  UaarStatus st{};
  unsigned a = 0, b = 0, c = 0;
  sscanf(UAAR_FW_VERSION, "%u.%u.%u", &a, &b, &c);
  st.fw_major = a; st.fw_minor = b; st.fw_patch = c;
  st.session_id = session_id;
  st.reserved = 0;
  st.uptime_s = millis() / 1000;
  st.calibrated = calibrated ? 1 : 0;
  st.session_active = session_active ? 1 : 0;
  ble.publishStatus(st);
}

void emitSample(const int16_t a[3], const int16_t g[3]) {
  UaarSample s;
  s.t_ms = millis();
  s.ax = a[0]; s.ay = a[1]; s.az = a[2];
  s.gx = g[0]; s.gy = g[1]; s.gz = g[2];
  s.flags = (session_active ? UAAR_FLAG_SESSION_ACTIVE : 0) |
            (button.isDown() ? UAAR_FLAG_BUTTON_PRESSED : 0) |
            (calibrated ? UAAR_FLAG_CALIBRATED : 0);
  s.seq = seq++;
  ble.push(s, session_id);
  Serial.printf("%lu,%d,%d,%d,%d,%d,%d,%u,%u\n", (unsigned long)s.t_ms, s.ax, s.ay, s.az,
                s.gx, s.gy, s.gz, s.flags, s.seq);
}

void readImu() {
  int16_t a[3], g[3];
  if (!imu.read(a, g)) { ++imu_errors; return; }
  if (calib_until_ms) {
    for (int i = 0; i < 3; ++i) calib_sum[i] += g[i];
    ++calib_n;
    if (millis() >= calib_until_ms && calib_n > 0) finishCalibration();
  }
  for (int i = 0; i < 3; ++i) {
    acc_sum[i] += a[i];
    gyr_sum[i] += g[i] - (calibrated ? gyr_bias[i] : 0);
  }
  if (++reads_in_sample < READS_PER_SAMPLE) return;
  int16_t aa[3], gg[3];
  for (int i = 0; i < 3; ++i) {
    aa[i] = (int16_t)constrain(acc_sum[i] / READS_PER_SAMPLE, -32768, 32767);
    gg[i] = (int16_t)constrain(gyr_sum[i] / READS_PER_SAMPLE, -32768, 32767);
    acc_sum[i] = gyr_sum[i] = 0;
  }
  reads_in_sample = 0;
  emitSample(aa, gg);
}
}  // namespace

void setup() {
  led.begin(PIN_LED, true);
  button.begin(PIN_BUTTON, CALIB_MS);
  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);  // give a monitor a moment to attach

  Wire.begin();  // XIAO ESP32-S3 variant: SDA = D4 (GPIO5), SCL = D5 (GPIO6)
  Wire.setClock(400000);
  const uint8_t addr = Bmi160::probe(Wire);
  imu_ok = addr != 0 && imu.begin(Wire, addr);
  loadCalibration();
  ble.begin(deviceName().c_str(), handleCommand);
  Serial.println();
  printBanner();
  refreshLed();
  next_read_us = micros();
}

void loop() {
  switch (button.poll()) {
    case Button::ShortPress: setSession(!session_active); break;
    case Button::LongPress: startCalibration(); break;
    default: break;
  }
  pollSerialCommands();

  const uint32_t now_us = micros();
  if (imu_ok && (int32_t)(now_us - next_read_us) >= 0) {
    next_read_us += ODR_PERIOD_US;
    if ((int32_t)(now_us - next_read_us) > (int32_t)(5 * ODR_PERIOD_US)) next_read_us = now_us;  // fell behind: resync
    readImu();
  }

  if (millis() - last_status_ms >= STATUS_PERIOD_MS) {
    last_status_ms = millis();
    publishStatus();
    if (!calib_until_ms) refreshLed();  // picks up BLE connect/disconnect
  }
  led.update();
  delay(1);
}
