from __future__ import annotations

import re
import struct
from typing import Optional
from pathlib import Path

from .models import FirmwareBanner, LayoutTraits, LoadedFirmware, MemoryMap
from .hardware import detect_device_family
from .uf2 import Uf2Image, build_memory_map as build_uf2_memory_map, is_uf2, parse_uf2
from .esp32 import normalize_shifted_dump, build_memory_map as build_esp32_memory_map


_BANNER_RE = re.compile(
    rb"MicroPython v(?P<version>[0-9A-Za-z._-]+) on (?P<date>\d{4}-\d{2}-\d{2})"
)


def _parse_version(text: str) -> tuple[int, int, int]:
    # Only the numeric prefix matters for choosing the recovery layout.
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return (0, 0, 0)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def derive_traits(version: tuple[int, int, int]) -> LayoutTraits:
    # Map firmware release ranges to the layouts we know how to parse.
    match version:
        case version if version < (1, 19, 0):
            return LayoutTraits(
                label="v5-legacy",
                mpy_format="v5",
                qstr_layout="v5-inline",
                has_sorted_pool_flag=False,
            )
        case version if version < (1, 22, 0):
            return LayoutTraits(
                label="v6-early",
                mpy_format="v6",
                qstr_layout="v6-split",
                has_sorted_pool_flag=False,
            )
        case version if version < (1, 23, 0):
            return LayoutTraits(
                label="v6-sorted",
                mpy_format="v6",
                qstr_layout="v6-split",
                has_sorted_pool_flag=True,
            )
        case _:
            return LayoutTraits(
                label="v6-current",
                mpy_format="v6",
                qstr_layout="v6-split",
                has_sorted_pool_flag=True,
            )


def detect_banner(data: bytes) -> Optional[FirmwareBanner]:
    # Use the first banner match so the earliest firmware banner wins.
    matches = list(_BANNER_RE.finditer(data))
    if not matches:
        return None
    best = matches[0]
    version_text = best.group("version").decode("ascii")
    build_date = best.group("date").decode("ascii")
    raw_end = data.find(b"\x00", best.start())
    if raw_end < 0:
        raw_end = data.find(b"\n", best.start())
    if raw_end < 0:
        raw_end = best.end()
    raw_text = data[best.start():raw_end].decode("ascii", errors="replace")
    return FirmwareBanner(
        raw_text=raw_text,
        version_text=version_text,
        version=_parse_version(version_text),
        build_date=build_date,
    )


def read_firmware(path: str | Path) -> bytes:
    # Read the firmware image once so every later stage works from the same bytes.
    return Path(path).read_bytes()


def build_memory_map(
    data: bytes,
    traits: LayoutTraits,
    source_format: str = "raw",
    uf2_image: Uf2Image | None = None,
) -> MemoryMap:
    # Choose the address-map builder from the image type, not from the firmware banner.
    match source_format:
        case "uf2":
            if uf2_image is None:
                raise ValueError("UF2 source data is missing")
            return build_uf2_memory_map(uf2_image)
        case "raw":
            family = (traits.device_family or "unknown").lower()
            if family.startswith("esp32"):
                return build_esp32_memory_map(data, traits)
            raise ValueError(f"memmap parsing is not implemented for {traits.device_family}")
        case _:
            raise ValueError(f"unsupported firmware source format: {source_format}")


def _load_raw_and_source(path: str | Path) -> tuple[bytes, str, tuple[str, ...], Uf2Image | None]:
    # Keep the raw bytes and the source metadata together so later stages can reuse both.
    raw = read_firmware(path)
    if is_uf2(raw):
        image = parse_uf2(raw)
        return image.raw, "uf2", image.source_info, image
    return raw, "raw", (), None


def load_workspace_firmware(path: str | Path) -> LoadedFirmware:
    # Normalize the image first so every later offset is measured against one shared view.
    raw, source_format, source_info, uf2_image = _load_raw_and_source(path)
    banner = detect_banner(raw)
    if banner is None:
        raise ValueError("unable to detect a MicroPython banner")
    version = banner.version
    traits = derive_traits(version)
    traits.device_family = detect_device_family(raw, uf2_image)
    memory_map = build_memory_map(raw, traits, source_format, uf2_image)

    match source_format:
        case "uf2":
            normalized = raw
        case "raw":
            # Raw ESP32 dumps can start shifted; normalize them before any pointer math.
            match (traits.device_family or "unknown").lower():
                case family if family.startswith("esp32"):
                    normalized, _ = normalize_shifted_dump(raw)
                case _:
                    raise ValueError(
                        f"unsupported or unknown device family for raw firmware: {traits.device_family!r}"
                    )
        case _:
            raise ValueError(f"unsupported firmware source format: {source_format}")

    return LoadedFirmware(normalized, memory_map, banner, traits, source_format, source_info)


def read_u16(data: bytes, offset: int) -> int:
    # Read a little-endian 16-bit value directly from the firmware buffer.
    return struct.unpack_from("<H", data, offset)[0]


def read_u32(data: bytes, offset: int) -> int:
    # Read a little-endian 32-bit value directly from the firmware buffer.
    return struct.unpack_from("<I", data, offset)[0]
