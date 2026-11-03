// UAAR wearable wire protocol. Single source of truth on the firmware side; mirrored by
// src/uaar/data/wearable.py and web/protocol.js. Change all three together.
//
// The packet layout is deliberately identical to the FormCoach (BDA 696) IMU stream format so
// one recorder can read either device; the code on both sides is written separately.
#pragma once
#include <stdint.h>

// ---- Identity ------------------------------------------------------------------------------
#define UAAR_DEVICE_NAME_PREFIX "UAAR-"
#define UAAR_SERVICE_UUID "7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001"
#define UAAR_IMU_CHAR_UUID "7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001"      // notify: sample batches
#define UAAR_CONTROL_CHAR_UUID "7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001"  // write: 1-byte commands
#define UAAR_STATUS_CHAR_UUID "7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001"   // read/notify: status
#define UAAR_REQUESTED_MTU 185

// ---- Sampling ------------------------------------------------------------------------------
#define UAAR_IMU_ODR_HZ 100        // BMI160 output data rate
#define UAAR_SAMPLE_RATE_HZ 50     // two ODR reads averaged per emitted sample
#define UAAR_BATCH_SAMPLES 5       // samples per BLE notification (100 ms)
#define UAAR_ACC_LSB_PER_G 4096    // +-8 g
#define UAAR_GYR_LSB_PER_DPS 32.8f // +-1000 dps

// ---- Commands (host -> wearable, control characteristic or one serial letter) ---------------
enum UaarCommand : uint8_t {
  UAAR_CMD_LED_KNOWN = 0x01,     // 'K': brief "recognized" LED pattern
  UAAR_CMD_LED_UNKNOWN = 0x02,   // 'U': brief "unknown movement" LED pattern
  UAAR_CMD_START_SESSION = 0x03, // 'S'
  UAAR_CMD_STOP_SESSION = 0x04,  // 'X'
  UAAR_CMD_CALIBRATE = 0x05,     // 'C': 3 s still -> gyro bias stored in NVS
  UAAR_CMD_PING = 0x10,          // 'P': replies "# pong"
  UAAR_CMD_IDENTIFY = 0x11,      // 'I': reprints the boot banner (fw, name, imu, calibration)
};

// ---- Sample flags ---------------------------------------------------------------------------
enum UaarFlag : uint8_t {
  UAAR_FLAG_RESERVED = 1 << 0,       // always 0 (FormCoach uses this bit for its gate)
  UAAR_FLAG_SESSION_ACTIVE = 1 << 1,
  UAAR_FLAG_BUTTON_PRESSED = 1 << 2,
  UAAR_FLAG_CALIBRATED = 1 << 3,     // gyro bias subtracted on device
};

// ---- Wire layout (little-endian, packed) ---------------------------------------------------
#pragma pack(push, 1)
struct UaarSample {
  uint32_t t_ms;       // device millis() at the sample
  int16_t ax, ay, az;  // raw accel LSB (UAAR_ACC_LSB_PER_G)
  int16_t gx, gy, gz;  // raw gyro LSB (UAAR_GYR_LSB_PER_DPS), bias-subtracted when calibrated
  uint8_t flags;       // UaarFlag bits
  uint16_t seq;        // rolling counter; gaps mean dropped samples
};

struct UaarBatchHeader {
  uint16_t session_id;
  uint8_t n;  // samples that follow (<= UAAR_BATCH_SAMPLES)
  uint8_t reserved;
};

struct UaarBatch {
  UaarBatchHeader header;
  UaarSample samples[UAAR_BATCH_SAMPLES];
};

struct UaarStatus {
  uint8_t fw_major, fw_minor, fw_patch;
  uint16_t session_id;
  uint8_t reserved;      // 0 (FormCoach: gate state)
  uint32_t uptime_s;
  uint8_t calibrated;
  uint8_t session_active;
};
#pragma pack(pop)

#define UAAR_SAMPLE_SIZE 19
#define UAAR_BATCH_HEADER_SIZE 4
#define UAAR_MAX_BATCH_SIZE (UAAR_BATCH_HEADER_SIZE + UAAR_BATCH_SAMPLES * UAAR_SAMPLE_SIZE)  // 99
#define UAAR_STATUS_SIZE 12
static_assert(sizeof(UaarSample) == UAAR_SAMPLE_SIZE, "UaarSample must be 19 packed bytes");
static_assert(sizeof(UaarBatchHeader) == UAAR_BATCH_HEADER_SIZE, "header must be 4 bytes");
static_assert(sizeof(UaarBatch) == UAAR_MAX_BATCH_SIZE, "UaarBatch must be 99 bytes");
static_assert(sizeof(UaarStatus) == UAAR_STATUS_SIZE, "UaarStatus must be 12 bytes");
static_assert(UAAR_MAX_BATCH_SIZE <= UAAR_REQUESTED_MTU - 3, "batch must fit one notification");

// ---- Serial (115200 baud) ------------------------------------------------------------------
// Lines starting with '#' are comments; everything else is one CSV sample per line.
#define UAAR_SERIAL_CSV_HEADER "t_ms,ax,ay,az,gx,gy,gz,flags,seq"
