from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

from .hardware import device_label
from .models import LoadedFirmware, LayoutTraits, MemoryMap, Segment
from .rendering import TextReport

PARTITION_MAGIC = 0x50AA
PARTITION_MD5_MAGIC = 0xEBEB
PARTITION_ENTRY_SIZE = 32
PARTITION_TYPE_APP = 0x00
PARTITION_OFFSET = 0x8000
SHIFTED_PARTITION_OFFSET = 0x7000
APP_OFFSET = 0x10000
APP_HEADER_SIZE = 24
DROM_MIN = 0x3F400000
DROM_MAX = 0x3F800000


@dataclass
class PartitionEntry:
    partition_type: int
    subtype: int
    offset: int
    size: int
    label: str
    flags: int


@dataclass
class Esp32MemoryMap(MemoryMap):
    normalize_shift: int = 0
    app_offset: int = 0


def has_partition_table(data: bytes, offset: int) -> bool:
    # Stop at the first valid end marker; that is enough to trust the table header.
    if offset + PARTITION_ENTRY_SIZE > len(data):
        return False
    seen = False
    for cursor in range(offset, min(len(data), offset + 32 * 32), PARTITION_ENTRY_SIZE):
        entry = data[cursor : cursor + PARTITION_ENTRY_SIZE]
        if len(entry) < PARTITION_ENTRY_SIZE:
            return False
        magic = struct.unpack_from("<H", entry, 0)[0]
        if magic == PARTITION_MD5_MAGIC:
            return seen
        if magic == 0xFFFF and all(byte == 0xFF for byte in entry):
            return seen
        if magic != PARTITION_MAGIC:
            return False
        seen = True
    return seen


def normalize_shifted_dump(data: bytes) -> tuple[bytes, int]:
    # Some raw dumps are shifted by one page, so pad them back to the expected base.
    if has_partition_table(data, PARTITION_OFFSET):
        return data, 0
    if has_partition_table(data, SHIFTED_PARTITION_OFFSET):
        return (b"\x00" * 0x1000) + data, 0x1000
    return data, 0


def _read_partition_table(data: bytes) -> tuple[PartitionEntry, ...]:
    # Read only the application partition records; those are what anchor the app image.
    if not has_partition_table(data, PARTITION_OFFSET):
        return ()
    partitions: list[PartitionEntry] = []
    for cursor in range(PARTITION_OFFSET, min(len(data), PARTITION_OFFSET + 32 * 32), PARTITION_ENTRY_SIZE):
        entry = data[cursor : cursor + PARTITION_ENTRY_SIZE]
        if len(entry) < PARTITION_ENTRY_SIZE:
            break
        magic = struct.unpack_from("<H", entry, 0)[0]
        if magic == PARTITION_MD5_MAGIC:
            break
        if magic == 0xFFFF and all(byte == 0xFF for byte in entry):
            break
        if magic != PARTITION_MAGIC:
            break
        partitions.append(
            PartitionEntry(
                partition_type=entry[2],
                subtype=entry[3],
                offset=struct.unpack_from("<I", entry, 4)[0],
                size=struct.unpack_from("<I", entry, 8)[0],
                label=entry[12:28].split(b"\x00", 1)[0].rstrip(b"\xff").decode("ascii", errors="replace"),
                flags=struct.unpack_from("<I", entry, 28)[0],
            )
        )
    return tuple(partitions)


def _application_offsets(partitions: tuple[PartitionEntry, ...]) -> tuple[int, ...]:
    return tuple(partition.offset for partition in partitions if partition.partition_type == PARTITION_TYPE_APP and partition.offset)


def _iter_segments(data: bytes, app_offset: int) -> Iterable[Segment]:
    if app_offset + APP_HEADER_SIZE > len(data):
        return
    if data[app_offset] != 0xE9:
        return
    seg_count = data[app_offset + 1]
    pos = app_offset + APP_HEADER_SIZE
    for _ in range(seg_count):
        if pos + 8 > len(data):
            return
        load_addr, size = struct.unpack_from("<II", data, pos)
        pos += 8
        if size <= 0 or pos + size > len(data):
            return
        yield Segment(
            load_addr=load_addr,
            size=size,
            file_offset=pos,
        )
        pos += size


def _discover_segments(data: bytes) -> tuple[int, tuple[Segment, ...]]:
    # Try the partition table first, then fall back to the default app offset.
    partitions = _read_partition_table(data)
    app_offsets = _application_offsets(partitions)
    segment_offsets = app_offsets or (APP_OFFSET,)
    for candidate in segment_offsets:
        candidate_segments = tuple(_iter_segments(data, candidate))
        if candidate_segments:
            return candidate, candidate_segments
    return segment_offsets[0], ()


def build_memory_map(data: bytes, traits: LayoutTraits | None = None) -> MemoryMap:
    # Pick the DROM segment from the recovered segments and keep the file shift with it.
    normalized, shift = normalize_shifted_dump(data)
    app_offset, segments = _discover_segments(normalized)
    if not segments:
        family = traits.device_family if traits is not None else "unknown"
        raise ValueError(f"unable to parse the application image for {family}")
    drom = next((segment for segment in segments if DROM_MIN <= segment.load_addr < DROM_MAX), None)
    if drom is None:
        family = traits.device_family if traits is not None else "unknown"
        raise ValueError(f"unable to parse the application image for {family}")
    return Esp32MemoryMap(
        normalize_shift=shift,
        app_offset=app_offset,
        drom_vaddr=drom.load_addr,
        drom_file_offset=drom.file_offset,
        drom_size=drom.size,
        segments=segments,
    )


def render_memmap(loaded: LoadedFirmware) -> str:
    memory_map = loaded.memory_map
    report = TextReport().banner(
        f"{device_label(loaded.traits.device_family)} firmware memmap",
        loaded.banner.raw_text if loaded.banner is not None else None,
    ).kv([
        ("image format", loaded.source_format),
    ])
    if loaded.source_format == "uf2" and loaded.source_info:
        report.section("UF2 source").text(*[f"  - {item}" for item in loaded.source_info])
    if memory_map is None:
        report.blank().section("Memory map").text("memmap parsing is not available for this ESP32 firmware yet.")
        return report.render()
    report.blank().section("Memory map")
    # Show the recovered base addresses before listing the segments and partitions.
    report.kv([
        ("normalize_shift", hex(getattr(memory_map, 'normalize_shift', 0))),
        ("app_offset", hex(getattr(memory_map, 'app_offset', 0))),
        ("drom_vaddr", hex(memory_map.drom_vaddr)),
        ("drom_file_offset", hex(memory_map.drom_file_offset)),
        ("drom_size", hex(memory_map.drom_size)),
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
    app_offset, segments = _discover_segments(loaded.raw)
    if segments:
        report.section("Segments")
        rows = []
        for index, segment in enumerate(segments):
            rows.append([
                index,
                hex(segment.load_addr),
                hex(segment.file_offset),
                hex(segment.size),
                hex(segment.load_addr + segment.size),
            ])
        report.kv([
            ("app_offset", hex(app_offset)),
            ("segment_count", len(segments)),
        ])
        report.table(["#", "Load Addr", "File Off", "Size", "End"], rows)
    partitions = _read_partition_table(loaded.raw)
    if partitions:
        report.section("Partitions")
        rows = []
        for index, partition in enumerate(partitions):
            rows.append([
                index,
                partition.label or "<unnamed>",
                hex(partition.partition_type),
                hex(partition.subtype),
                hex(partition.offset),
                hex(partition.size),
                hex(partition.flags),
            ])
        report.table(["#", "Label", "Type", "Subtype", "Offset", "Size", "Flags"], rows)
        report.kv([
            ("partition_count", len(partitions)),
        ])
    return report.render()