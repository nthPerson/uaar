#!/usr/bin/env bash
# attach-xiao.sh — attach the XIAO ESP32S3's USB JTAG/serial port to WSL,
# entirely from inside WSL (no separate Windows terminal needed).
#
# Why this works without an admin shell: usbipd-win >= 4.0 requires elevation
# only for the ONE-TIME `bind`; `attach` runs unelevated. If the device ever
# shows "Not shared" (fresh Windows port, or after a usbipd upgrade), this
# script escalates just the bind via a normal Windows UAC prompt and then
# attaches unelevated as usual.
#
# The XIAO is matched by VID:PID 303a:1001 (Espressif's USB JTAG/serial debug
# unit), NOT by device name — the name embeds a COM number that changes
# between plugs.
#
# ⚠ This WSL does not append the Windows PATH, so Windows executables are
# reached by absolute /mnt/c/... paths.

set -euo pipefail

VIDPID="303a:1001"

USBIPD="$(command -v usbipd.exe || true)"
[ -n "$USBIPD" ] || USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
[ -x "$USBIPD" ] || {
  echo "usbipd.exe not found — install usbipd-win on Windows first" >&2
  echo "(winget install usbipd), then re-run." >&2
  exit 1
}

PWSH="$(command -v powershell.exe || true)"
[ -n "$PWSH" ] || PWSH="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# usbipd is a Windows exe: its output is CRLF-terminated — strip \r or every
# downstream string compare silently fails.
line="$("$USBIPD" list | tr -d '\r' | grep -m1 "$VIDPID" || true)"
if [ -z "$line" ]; then
  echo "XIAO not visible to Windows (no $VIDPID device on the bus)." >&2
  echo "Plug the XIAO's own USB-C (end-wall slot — NOT the TP4056 charge" >&2
  echo "port) and re-run. A different cable/port is fine; bind persists." >&2
  exit 1
fi

busid="$(awk '{print $1}' <<<"$line")"
state="$(grep -oE '(Not shared|Shared \(forced\)|Shared|Attached)$' <<<"$line" || true)"
echo "XIAO at busid $busid — state: ${state:-unknown}"

case "$state" in
  Attached)
    : ;;  # nothing to do; fall through to the port wait
  Shared|"Shared (forced)")
    "$USBIPD" attach --wsl --busid "$busid"
    ;;
  "Not shared")
    echo "One-time bind needed — approve the Windows UAC prompt that pops up..."
    "$PWSH" -NoProfile -Command \
      "Start-Process usbipd -Verb RunAs -Wait -ArgumentList 'bind','--busid','$busid'"
    "$USBIPD" attach --wsl --busid "$busid"
    ;;
  *)
    echo "Unrecognized state '${state:-}' — inspect with: \"$USBIPD\" list" >&2
    exit 1
    ;;
esac

# Enumeration takes a moment — and a deep-sleeping XIAO never enumerates at
# all, which is the #1 reason this wait times out on a battery-powered unit.
for _ in $(seq 1 20); do
  port="$(ls /dev/ttyACM* 2>/dev/null | head -1 || true)"
  if [ -n "$port" ]; then
    echo "Ready: $port  (idf.py -p $port flash monitor)"
    exit 0
  fi
  sleep 0.5
done

echo "Attached, but no /dev/ttyACM* appeared within 10 s." >&2
echo "Most likely the unit is DEEP-ASLEEP (USB never enumerates in deep" >&2
echo "sleep): press and HOLD the button ~1 s — until the LED lights and the" >&2
echo "rising beeps play — then re-run. Fallbacks: hold the XIAO's BOOT" >&2
echo "button while re-plugging USB (forces the ROM bootloader), or reset a" >&2
echo "stale attachment with: \"$USBIPD\" detach --busid $busid  and re-run." >&2
exit 1
