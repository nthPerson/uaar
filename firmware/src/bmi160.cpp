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
  if (readReg(CHIP_ID_REG) != CHIP_ID) return false;
  writeReg(CMD, CMD_SOFT_RESET);
  delay(60);
  bool ok = writeReg(ACC_CONF, ACC_CONF_VAL);
  ok &= writeReg(ACC_RANGE, ACC_RANGE_8G);
  ok &= writeReg(GYR_CONF, GYR_CONF_VAL);
  ok &= writeReg(GYR_RANGE, GYR_RANGE_1000DPS);
  ok &= writeReg(CMD, CMD_ACC_PMU_NORMAL);
  delay(5);  // accel start-up (datasheet: 3.8 ms)
  ok &= writeReg(CMD, CMD_GYR_PMU_NORMAL);
  delay(80);  // gyro start-up (datasheet: 80 ms)
  // Verify the configuration landed: the datasheet's ERR_REG must be 0 and both PMUs normal.
  ok &= readReg(ACC_CONF) == ACC_CONF_VAL && readReg(ACC_RANGE) == ACC_RANGE_8G;
  ok &= readReg(GYR_CONF) == GYR_CONF_VAL && readReg(GYR_RANGE) == GYR_RANGE_1000DPS;
  ok &= readReg(ERR_REG) == 0x00;
  ok &= (readReg(PMU_STATUS) & 0x3C) == 0x14;  // gyr_pmu (bits 3:2) = 01, acc_pmu (5:4) = 01
  return ok;
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
