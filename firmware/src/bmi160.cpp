#include "bmi160.h"

bool Bmi160::writeReg(uint8_t reg, uint8_t val) {
  wire_->beginTransmission(addr_);
  wire_->write(reg);
  wire_->write(val);
  return wire_->endTransmission() == 0;
}

bool Bmi160::readRegs(uint8_t reg, uint8_t* buf, size_t n) {
  wire_->beginTransmission(addr_);
  wire_->write(reg);
  if (wire_->endTransmission(false) != 0) return false;  // repeated start
  if (wire_->requestFrom(addr_, (uint8_t)n) != n) return false;
  for (size_t i = 0; i < n; ++i) buf[i] = (uint8_t)wire_->read();
  return true;
}

uint8_t Bmi160::probe(TwoWire& wire) {
  const uint8_t candidates[] = {0x69, 0x68};
  for (uint8_t a : candidates) {
    wire.beginTransmission(a);
    if (wire.endTransmission() == 0) return a;
  }
  return 0;
}

bool Bmi160::begin(TwoWire& wire, uint8_t address) {
  wire_ = &wire;
  addr_ = address;
  failure_ = "";
  const uint8_t id = readReg(CHIP_ID_REG);
  if (id != CHIP_ID) { failure_ = "chip id"; return false; }
  writeReg(CMD, CMD_SOFT_RESET);
  delay(60);
  // Datasheet "interface idle time": in suspend mode (the state after reset) the chip needs
  // >= 450 us between register accesses or writes are silently lost. Bring both sensors to
  // normal power mode first, then write the configuration with a short delay after each write.
  bool ok = writeReg(CMD, CMD_ACC_PMU_NORMAL);
  delay(5);  // accel start-up (datasheet: 3.8 ms)
  ok &= writeReg(CMD, CMD_GYR_PMU_NORMAL);
  delay(80);  // gyro start-up (datasheet: 80 ms)
  if (!ok) { failure_ = "pmu write"; return false; }
  const uint8_t config[][2] = {
      {ACC_CONF, ACC_CONF_VAL}, {ACC_RANGE, ACC_RANGE_8G}, {GYR_CONF, GYR_CONF_VAL}, {GYR_RANGE, GYR_RANGE_1000DPS}};
  for (const auto& rv : config) {
    ok &= writeReg(rv[0], rv[1]);
    delay(2);
  }
  if (!ok) { failure_ = "config write"; return false; }
  // Verify the configuration landed and both sensors are in normal power mode.
  if (readReg(ACC_CONF) != ACC_CONF_VAL || readReg(ACC_RANGE) != ACC_RANGE_8G) { failure_ = "acc readback"; return false; }
  if (readReg(GYR_CONF) != GYR_CONF_VAL || readReg(GYR_RANGE) != GYR_RANGE_1000DPS) { failure_ = "gyr readback"; return false; }
  if (readReg(ERR_REG) != 0x00) { failure_ = "err_reg"; return false; }
  if ((readReg(PMU_STATUS) & 0x3C) != 0x14) { failure_ = "pmu status"; return false; }
  return true;
}

bool Bmi160::read(int16_t acc[3], int16_t gyr[3]) {
  uint8_t b[12];
  if (!readRegs(DATA_GYR_X_L, b, sizeof(b))) return false;  // 0x0C..0x11 gyro, 0x12..0x17 accel
  for (int i = 0; i < 3; ++i) {
    gyr[i] = (int16_t)(b[2 * i] | (b[2 * i + 1] << 8));
    acc[i] = (int16_t)(b[6 + 2 * i] | (b[6 + 2 * i + 1] << 8));
  }
  return true;
}

String Bmi160::describe() {
  char buf[96];
  snprintf(buf, sizeof(buf), "addr=0x%02X err=0x%02X pmu=0x%02X acc_conf=0x%02X acc_range=0x%02X gyr_conf=0x%02X gyr_range=0x%02X",
           addr_, readReg(ERR_REG), readReg(PMU_STATUS), readReg(ACC_CONF), readReg(ACC_RANGE),
           readReg(GYR_CONF), readReg(GYR_RANGE));
  return String(buf);
}
