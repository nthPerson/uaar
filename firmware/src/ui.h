// Push button on D1 (to GND, internal pull-up; debounced; short vs 3 s long press) and the
// built-in LED (GPIO21, active-low) with a few blink patterns.
#pragma once
#include <Arduino.h>

class Button {
 public:
  enum Event : uint8_t { None = 0, ShortPress = 1, LongPress = 2 };
  void begin(uint8_t pin, uint32_t long_press_ms = 3000);
  Event poll();                       // call every loop; events fire once
  bool isDown() const { return down_; }

 private:
  uint8_t pin_ = 0;
  uint32_t long_press_ms_ = 3000;
  bool down_ = false, raw_ = false, long_fired_ = false;
  uint32_t changed_at_ = 0, down_at_ = 0;
};

enum class LedMode : uint8_t {
  Off,
  Advertising,  // one short blink every 2 s
  Connected,    // two short blinks every 2 s
  Session,      // 1 Hz, 50 % duty: recording
  Calibrating,  // 4 Hz
};

enum class LedFlash : uint8_t { Known, Unknown };

class StatusLed {
 public:
  void begin(uint8_t pin, bool active_low);
  void setMode(LedMode m) { mode_ = m; }
  void flash(LedFlash f);  // temporary pattern on top of the mode
  void update();

 private:
  void write(bool on) { digitalWrite(pin_, (on != active_low_) ? HIGH : LOW); }
  uint8_t pin_ = 0;
  bool active_low_ = true;
  LedMode mode_ = LedMode::Off;
  LedFlash flash_ = LedFlash::Known;
  uint32_t flash_until_ = 0;
};
