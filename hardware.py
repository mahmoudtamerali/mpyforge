from __future__ import annotations

from .models import LoadedFirmware
from .uf2 import Uf2Image, family_from_id

_DEVICE_LABELS = {
    "esp32": "ESP32",
    "esp32s2": "ESP32-S2",
    "esp32s3": "ESP32-S3",
    "esp32c3": "ESP32-C3",
    "esp32c6": "ESP32-C6",
    "esp32h2": "ESP32-H2",
    "rp2040": "Raspberry Pi Pico / RP2040",
    "rp2350": "Raspberry Pi Pico 2 / RP2350",
    "stm32": "STM32",
    "mimxrt": "NXP i.MX RT",
    "samd": "SAMD",
    "nrf": "Nordic nRF",
}

_DEVICE_PHRASES = (
    (b"Raspberry Pi Pico with RP2040", "rp2040"),
    (b"Raspberry Pi Pico 2 with RP2350", "rp2350"),
    (b"Generic ESP32 module with ESP32", "esp32"),
    (b"ESP32 module with ESP32", "esp32"),
    (b"Generic ESP32-S2 module with ESP32-S2", "esp32s2"),
    (b"Generic ESP32-S3 module with ESP32-S3", "esp32s3"),
    (b"Generic ESP32-C3 module with ESP32-C3", "esp32c3"),
    (b"Generic ESP32-C6 module with ESP32-C6", "esp32c6"),
    (b"Generic ESP32-H2 module with ESP32-H2", "esp32h2"),
    (b"Generic STM32 module with STM32", "stm32"),
    (b"Generic NXP i.MX RT module with MIMXRT", "mimxrt"),
    (b"Generic SAMD module with SAMD", "samd"),
    (b"Generic Nordic nRF module with nRF", "nrf"),

)


def device_label(device_family: str) -> str:
    # Turn the internal family code into the label shown in reports.
    if not device_family or device_family == "unknown":
        return "Unknown hardware"
    if device_family in _DEVICE_LABELS:
        return _DEVICE_LABELS[device_family]
    if device_family.startswith("esp32"):
        return device_family.upper()
    return device_family.replace("_", " ").title()


def detect_device_family(raw_data: bytes | None, uf2_image: Uf2Image | None = None) -> str:
    # Prefer the UF2 family ID when it exists, then fall back to banner phrases.
    if uf2_image is not None:
        family = family_from_id(uf2_image.family_id)
        if family != "unknown":
            return family
    if raw_data is None:
        return "unknown"
    for phrase, family in _DEVICE_PHRASES:
        if phrase in raw_data:
            return family
    return "unknown"


def _hardware_group(device_family: str) -> str:
    # Group related targets so the memmap renderer can pick the right backend.
    if device_family.startswith("esp32"):
        return "esp32"
    if device_family in {"rp2040", "rp2350"}:
        return "raspberry-pi"
    return "other"


def render_memmap(loaded: LoadedFirmware, hardware: str = "auto") -> str:
    # Dispatch to the format-specific memmap renderer once the hardware family is known.
    selected = _hardware_group(loaded.traits.device_family) if hardware == "auto" else hardware
    if loaded.source_format == "uf2":
        from .uf2 import render_memmap as render_uf2_memmap

        return render_uf2_memmap(loaded)
    if selected == "esp32":
        from .esp32 import render_memmap as render_esp32_memmap

        return render_esp32_memmap(loaded)
    raise ValueError(f"memmap rendering is not implemented for {loaded.traits.device_family!r}")