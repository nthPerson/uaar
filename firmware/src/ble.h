// BLE GATT server (protocol.h): IMU notify batches, control writes, status read/notify.
#pragma once
#include <NimBLEDevice.h>
#include "protocol.h"

class BleLink {
 public:
  using CommandHandler = void (*)(uint8_t cmd);
  void begin(const char* name, CommandHandler handler);
  bool connected() const { return connected_; }
  void push(const UaarSample& s, uint16_t session_id);  // notifies every UAAR_BATCH_SAMPLES
  void publishStatus(const UaarStatus& st);
  // Called from the NimBLE callbacks.
  void onConnection(bool up);
  void onCommand(uint8_t cmd) { if (handler_) handler_(cmd); }

 private:
  NimBLEServer* server_ = nullptr;
  NimBLECharacteristic* imu_ = nullptr;
  NimBLECharacteristic* control_ = nullptr;
  NimBLECharacteristic* status_ = nullptr;
  CommandHandler handler_ = nullptr;
  UaarBatch batch_{};
  bool connected_ = false;
};
