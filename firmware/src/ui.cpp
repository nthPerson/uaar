#include "ui.h"

namespace {
constexpr uint32_t DEBOUNCE_MS = 30;
}

void Button::begin(uint8_t pin, uint32_t long_press_ms) {
  pin_ = pin;
  long_press_ms_ = long_press_ms;
  pinMode(pin_, INPUT_PULLUP);
}

Button::Event Button::poll() {
  const bool raw = digitalRead(pin_) == LOW;
  const uint32_t now = millis();
  Event ev = None;
  if (raw != raw_) {  // edge: start the debounce clock
    raw_ = raw;
    changed_at_ = now;
  } else if (raw != down_ && now - changed_at_ >= DEBOUNCE_MS) {  // stable new state
    down_ = raw;
    if (down_) {
      down_at_ = now;
      long_fired_ = false;
    } else if (!long_fired_) {
      ev = ShortPress;  // released before the long-press threshold
    }
  }
  if (down_ && !long_fired_ && now - down_at_ >= long_press_ms_) {
    long_fired_ = true;
    ev = LongPress;  // fires while still held, once
  }
  return ev;
}

void StatusLed::begin(uint8_t pin, bool active_low) {
  pin_ = pin;
  active_low_ = active_low;
  pinMode(pin_, OUTPUT);
  write(false);
}

void StatusLed::flash(LedFlash f) {
  flash_ = f;
  flash_until_ = millis() + (f == LedFlash::Known ? 400 : 1000);
}

void StatusLed::update() {
  const uint32_t now = millis();
  if (flash_until_) {
    if (now < flash_until_) {
      // Known: solid 400 ms.  Unknown: 10 Hz flicker for 1 s.
      write(flash_ == LedFlash::Known ? true : (now % 100) < 50);
      return;
    }
    flash_until_ = 0;
  }
  const uint32_t p2 = now % 2000, p1 = now % 1000;
  switch (mode_) {
    case LedMode::Off: write(false); break;
    case LedMode::Advertising: write(p2 < 60); break;
    case LedMode::Connected: write(p2 < 60 || (p2 >= 200 && p2 < 260)); break;
    case LedMode::Session: write(p1 < 500); break;
    case LedMode::Calibrating: write((now % 250) < 125); break;
  }
}
