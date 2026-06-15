from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .models import LoadedFirmware, MemoryMap
from .rendering import TextReport

UF2_MAGIC_START0 = 0x0A324655
UF2_MAGIC_START1 = 0x9E5D5157
UF2_MAGIC_END = 0x0AB16F30
UF2_BLOCK_SIZE = 512
UF2_PAYLOAD_OFFSET = 32
UF2_PAYLOAD_MAX = UF2_BLOCK_SIZE - UF2_PAYLOAD_OFFSET - 4
UF2_FLAG_FAMILY_ID = 0x2000


def _load_family_ids() -> dict[int, str]:
    # Load the UF2 family lookup from the bundled JSON file when it is available.
    family_file = Path(__file__).with_name("uf2families.json")
    if not family_file.is_file():
        return {}
    payload = json.loads(family_file.read_text(encoding="utf-8"))
    families: dict[int, str] = {}
    for entry in payload:
        family_id = int(entry["id"], 16)
        families[family_id] = str(entry["short_name"]).strip().lower()
    return families


UF2_FAMILY_IDS = _load_family_ids()


@dataclass
class Uf2Image:
    raw: bytes
    family_id: int | None
    min_addr: int
    max_addr: int
    block_count: int
    payload_size: int
    source_info: tuple[str, ...]


@dataclass
class Uf2MemoryMap(MemoryMap):
    flash_start: int = 0
    flash_end: int = 0


def is_uf2(data: bytes) -> bool:
    # A UF2 image is a stream of fixed-size blocks with matching start and end markers.
    if len(data) < UF2_BLOCK_SIZE or len(data) % UF2_BLOCK_SIZE != 0:
        return False
    magic0 = struct.unpack_from("<I", data, 0)[0]
    magic1 = struct.unpack_from("<I", data, 4)[0]
    magic_end = struct.unpack_from("<I", data, UF2_BLOCK_SIZE - 4)[0]
    return magic0 == UF2_MAGIC_START0 and magic1 == UF2_MAGIC_START1 and magic_end == UF2_MAGIC_END


def family_from_id(family_id: int | None) -> str:
    # Use the mapped UF2 family name when one is available.
    if family_id is None:
        return "unknown"
    return UF2_FAMILY_IDS.get(family_id, "unknown")


def parse_uf2(data: bytes) -> Uf2Image:
    # Reassemble the payload blocks into one linear image and keep the UF2 metadata.
    if not is_uf2(data):
        raise ValueError("not a UF2 image")

    min_addr: int | None = None
    max_addr: int | None = None
    family_id: int | None = None
    payload_size: int | None = None
    blocks: list[tuple[int, bytes]] = []

    # loop over each block and extract the payloads.
    for offset in range(0, len(data), UF2_BLOCK_SIZE):
        block = data[offset : offset + UF2_BLOCK_SIZE]
        magic0, magic1, flags, target_addr, size, block_no, num_blocks = struct.unpack_from("<IIIIIII", block, 0)
        magic_end = struct.unpack_from("<I", block, UF2_BLOCK_SIZE - 4)[0]
        if magic0 != UF2_MAGIC_START0 or magic1 != UF2_MAGIC_START1 or magic_end != UF2_MAGIC_END:
            raise ValueError("invalid UF2 block")
        if size > UF2_PAYLOAD_MAX:
            raise ValueError("invalid UF2 payload size")
        if payload_size is None:
            payload_size = size
        elif payload_size != size:
            payload_size = max(payload_size, size)
        if flags & UF2_FLAG_FAMILY_ID:
            block_family_id = struct.unpack_from("<I", block, 28)[0]
            if family_id is None:
                family_id = block_family_id
            elif family_id != block_family_id:
                raise ValueError("mixed UF2 family IDs are not supported")

        payload = block[UF2_PAYLOAD_OFFSET : UF2_PAYLOAD_OFFSET + size]
        blocks.append((target_addr, payload))
        min_addr = target_addr if min_addr is None else min(min_addr, target_addr)
        max_addr = target_addr + size if max_addr is None else max(max_addr, target_addr + size)

    if not blocks or min_addr is None or max_addr is None:
        raise ValueError("empty UF2 image")

    image = bytearray(b"\xFF" * (max_addr - min_addr))
    for target_addr, payload in blocks:
        start = target_addr - min_addr
        image[start : start + len(payload)] = payload

    info = [
        f"blocks={hex(len(blocks))}",
        f"payload_size={hex(payload_size or 0)}",
        f"range={hex(min_addr)}-{hex(max_addr)}",
    ]
    if family_id is not None:
        info.append(f"family_id={hex(family_id)}")
        info.append(f"family={family_from_id(family_id)}")

    return Uf2Image(bytes(image), family_id, min_addr, max_addr, len(blocks), payload_size or 0, tuple(info))


def build_memory_map(image: Uf2Image) -> MemoryMap:
    # UF2 images already carry the flash base address in the payload addresses.
    return Uf2MemoryMap(
        drom_vaddr=image.min_addr,
        drom_file_offset=0,
        drom_size=len(image.raw),
        flash_start=image.min_addr,
        flash_end=image.max_addr,
    )


def render_memmap(loaded: LoadedFirmware) -> str:
    # Show the UF2 source metadata first, then the reconstructed linear memory view.
    report = TextReport().banner(
        "UF2 firmware memmap",
        loaded.banner.raw_text if loaded.banner is not None else None,
    ).kv([
        ("image format", loaded.source_format),
        ("device family", loaded.traits.device_family),
    ])
    if loaded.source_info:
        report.section("UF2 source").text(*[f"  - {item}" for item in loaded.source_info])

    memory_map = loaded.memory_map
    if memory_map is None:
        report.blank().section("Memory map")
        report.text("UF2 decoding succeeded, but memmap parsing is not available for this image yet.")
        return report.render()

    report.blank().section("Memory map")
    report.kv([
        ("drom_vaddr", hex(memory_map.drom_vaddr)),
        ("drom_file_offset", hex(memory_map.drom_file_offset)),
        ("drom_size", hex(memory_map.drom_size)),
        ("flash_start", hex(getattr(memory_map, "flash_start", 0))),
        ("flash_end", hex(getattr(memory_map, "flash_end", 0))),
    ])
    report.table(
        ["Region", "Start", "End", "File Off", "Size"],
        [[
            "DROM",
            hex(memory_map.drom_vaddr),
            hex(memory_map.drom_end_vaddr),
            hex(memory_map.drom_file_offset),
            hex(memory_map.drom_size),
        ]],
    )
    if getattr(memory_map, "flash_start", 0) or getattr(memory_map, "flash_end", 0):
        report.section("Flash range").table(
            ["Start", "End"],
            [[hex(getattr(memory_map, "flash_start", 0)), hex(getattr(memory_map, "flash_end", 0))]],
        )
    return report.render()