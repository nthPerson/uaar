#include "ble.h"
#include <Arduino.h>

namespace {
BleLink* g_link = nullptr;

struct ServerCallbacks : NimBLEServerCallbacks {
  void onConnect(NimBLEServer* server, NimBLEConnInfo& info) override {
    // 15-30 ms connection interval, no latency, 4 s supervision timeout: 10 notifications/s
    // fit comfortably and a dropped link is noticed quickly.
    server->updateConnParams(info.getConnHandle(), 12, 24, 0, 400);
    if (g_link) g_link->onConnection(true);
  }
  void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int reason) override {
    Serial.printf("# ble disconnect reason=%d\n", reason);
    if (g_link) g_link->onConnection(false);
    NimBLEDevice::startAdvertising();
  }
  void onMTUChange(uint16_t mtu, NimBLEConnInfo&) override { Serial.printf("# ble mtu=%u\n", mtu); }
};

struct ControlCallbacks : NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* chr, NimBLEConnInfo&) override {
    NimBLEAttValue v = chr->getValue();
    if (v.size() >= 1 && g_link) g_link->onCommand(v[0]);
  }
};
}  // namespace

void BleLink::begin(const char* name, CommandHandler handler) {
  g_link = this;
  handler_ = handler;
  NimBLEDevice::init(name);
  NimBLEDevice::setMTU(UAAR_REQUESTED_MTU);
  server_ = NimBLEDevice::createServer();
  static ServerCallbacks server_cb;
  server_->setCallbacks(&server_cb);

  NimBLEService* svc = server_->createService(UAAR_SERVICE_UUID);
  imu_ = svc->createCharacteristic(UAAR_IMU_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);
  control_ = svc->createCharacteristic(UAAR_CONTROL_CHAR_UUID,
                                       NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  static ControlCallbacks control_cb;
  control_->setCallbacks(&control_cb);
  status_ = svc->createCharacteristic(UAAR_STATUS_CHAR_UUID,
                                      NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  svc->start();

  // The 31-byte advertisement holds flags + the 128-bit service UUID; the name does not fit
  // beside them, so it goes in the scan response.
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  NimBLEAdvertisementData adv_data;
  adv_data.setFlags(BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP);
  adv_data.addServiceUUID(UAAR_SERVICE_UUID);
  NimBLEAdvertisementData scan_data;
  scan_data.setName(name);
  adv->setAdvertisementData(adv_data);
  adv->setScanResponseData(scan_data);
  adv->enableScanResponse(true);
  bool started = adv->start();
  Serial.printf("# ble name=%s addr=%s advertising=%d\n", name,
                NimBLEDevice::getAddress().toString().c_str(), started);
  batch_.header.n = 0;
}

void BleLink::onConnection(bool up) {
  connected_ = up;
  batch_.header.n = 0;  // never carry a partial batch across a (re)connect
  Serial.printf("# ble %s\n", up ? "connected" : "disconnected");
}

void BleLink::push(const UaarSample& s, uint16_t session_id) {
  if (!connected_) return;
  batch_.header.session_id = session_id;
  batch_.samples[batch_.header.n++] = s;
  if (batch_.header.n < UAAR_BATCH_SAMPLES) return;
  const size_t len = UAAR_BATCH_HEADER_SIZE + batch_.header.n * UAAR_SAMPLE_SIZE;
  imu_->setValue((const uint8_t*)&batch_, len);
  imu_->notify();
  batch_.header.n = 0;
}

void BleLink::publishStatus(const UaarStatus& st) {
  if (!status_) return;
  status_->setValue((const uint8_t*)&st, sizeof(st));
  if (connected_) status_->notify();
}
