// BMI160 over I2C, register level (Bosch BMI160 datasheet rev 1.0): accel +-8 g, gyro
// +-1000 dps, both at 100 Hz ODR with the normal (OSR... "nrm") filter. Raw LSB out.
#pragma once
#include <Arduino.h>
#include <Wire.h>

class Bmi160 {
 public:
  static constexpr uint8_t CHIP_ID = 0xD1;
  // Probe both addresses (SDO low 0x68, SDO high 0x69); returns 0 when neither answers.
  static uint8_t probe(TwoWire& wire);
  // Reset, configure and power up. Returns false if CHIP_ID is wrong or the configuration
  // does not read back as written. Call describe() afterwards for the register values.
  bool begin(TwoWire& wire, uint8_t address);
  // One 12-byte burst: gyro x,y,z then accel x,y,z. Returns false on an I2C error.
  bool read(int16_t acc[3], int16_t gyr[3]);
  // Human-readable register dump for the boot banner.
  String describe();
  uint8_t address() const { return addr_; }

 private:
  enum Reg : uint8_t {
    CHIP_ID_REG = 0x00, ERR_REG = 0x02, PMU_STATUS = 0x03, DATA_GYR_X_L = 0x0C,
    ACC_CONF = 0x40, ACC_RANGE = 0x41, GYR_CONF = 0x42, GYR_RANGE = 0x43, CMD = 0x7E,
  };
  static constexpr uint8_t CMD_SOFT_RESET = 0xB6, CMD_ACC_PMU_NORMAL = 0x11, CMD_GYR_PMU_NORMAL = 0x15;
  // ODR 100 Hz = 0x08 in bits 3:0, bwp "normal" = 0b010 in bits 6:4 -> 0x28.
  static constexpr uint8_t ACC_CONF_VAL = 0x28, ACC_RANGE_8G = 0x08;
  static constexpr uint8_t GYR_CONF_VAL = 0x28, GYR_RANGE_1000DPS = 0x01;

  bool writeReg(uint8_t reg, uint8_t val);
  bool readRegs(uint8_t reg, uint8_t* buf, size_t n);
  uint8_t readReg(uint8_t reg) { uint8_t v = 0xFF; readRegs(reg, &v, 1); return v; }

  TwoWire* wire_ = nullptr;
  uint8_t addr_ = 0;
};
