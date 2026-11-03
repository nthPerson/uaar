# firmware/

PlatformIO project for the UAAR wearable: Seeed XIAO ESP32-S3 + GY-BMI160 on I2C (D4/D5),
push button on D1, built-in LED. The firmware does one job: stream six-axis IMU samples at
50 Hz over USB serial and BLE, with a button-driven session marker. There is no on-device
inference and no sleep mode; the battery slide switch is the power control.

| Command (repo root) | What |
|---|---|
| `uv tool run platformio run -d firmware` | compile |
| `uv tool run platformio run -d firmware -t upload --upload-port /dev/ttyACM0` | flash over USB-C |
| `uv tool run platformio device monitor -d firmware -p /dev/ttyACM0` | serial monitor (115200) |

Under WSL2 the XIAO's USB port must first be handed to Linux: `tools/attach-xiao.sh`.

## Behaviour

- **Boot banner** (lines starting with `#`): firmware version, device name `UAAR-XXXX` (last
  four hex digits of the MAC), BMI160 register dump, stored gyro bias, then the CSV header.
- **Samples**: one CSV line per sample, `t_ms,ax,ay,az,gx,gy,gz,flags,seq`, raw sensor counts
  (4096 LSB/g, 32.8 LSB per °/s). `seq` is a rolling 16-bit counter; a gap means a dropped sample.
- **Button**: short press toggles a session (`flags` bit 1; `session_id` increments on start, and
  the device prints `# session start id=N t_ms=...`). Hold 3 s to calibrate the gyro bias (keep
  the device still); the bias is stored in NVS and subtracted from every gyro sample (`flags` bit 3).
- **LED**: one blink every 2 s while advertising, two blinks when a BLE central is connected,
  1 Hz blink during a session, 4 Hz during calibration. Hosts can flash "known" (solid 0.4 s) or
  "unknown" (1 s flicker) with the LED commands.
- **Serial commands** (one letter): `P` ping, `I` reprint the banner, `S`/`X` start/stop session,
  `C` calibrate, `K`/`U` LED known/unknown.
- **BLE**: see `include/protocol.h`. Service `7a0c0001-…`, IMU notify characteristic with
  5-sample batches (99 bytes; MTU 185 requested), control write characteristic (1-byte command
  codes), status read/notify characteristic (12 bytes, once per second).

The wire format is intentionally the same as the FormCoach (BDA 696) device's so one recorder
can read either; the code is written separately for each project.
