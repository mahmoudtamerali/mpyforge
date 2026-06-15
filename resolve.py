from __future__ import annotations

from .firmware import read_u32
from .models import LayoutTraits, MemoryMap, ModuleRecord, NameTable


def _looks_like_v6_module(data: bytes, memmap: MemoryMap, offset: int) -> bool:
    # v6 module records carry qstr, object, and proto pointers in the struct itself.
    qstr_vaddr = read_u32(data, offset)
    obj_vaddr = read_u32(data, offset + 4)
    proto_vaddr = read_u32(data, offset + 8)
    if not memmap.contains_vaddr(proto_vaddr):
        return False
    if qstr_vaddr and not memmap.contains_vaddr(qstr_vaddr):
        return False
    if obj_vaddr and not memmap.contains_vaddr(obj_vaddr):
        return False
    return True


def _looks_like_v5_raw_code(data: bytes, memmap: MemoryMap, offset: int) -> bool:
    # v5 content entries point directly at raw-code nodes, so check the raw-code header.
    kind = read_u32(data, offset)
    fun_vaddr = read_u32(data, offset + 4)
    return kind in {2, 3, 4, 5} and memmap.contains_vaddr(fun_vaddr)


def _looks_like_content_entry(data: bytes, memmap: MemoryMap, offset: int, traits: LayoutTraits) -> bool:
    if traits.mpy_format == "v6":
        return _looks_like_v6_module(data, memmap, offset)
    return _looks_like_v5_raw_code(data, memmap, offset)


def _candidate_runs(
    data: bytes,
    memmap: MemoryMap,
    traits: LayoutTraits,
    start: int,
    end: int,
    count: int,
) -> list[int]:
    # Test each possible content-array start against the recovered name count.
    hits: list[int] = []
    limit = end - count * 4
    if limit < start:
        return hits
    for candidate in range(start, limit + 1, 4):
        ok = True
        for index in range(count):
            entry_vaddr = read_u32(data, candidate + index * 4)
            entry_offset = memmap.vaddr_to_offset(entry_vaddr)
            if entry_offset is None or not _looks_like_content_entry(data, memmap, entry_offset, traits):
                ok = False
                break
        if ok:
            hits.append(candidate)
    return hits


def find_content_array(
    data: bytes,
    memmap: MemoryMap,
    name_table: NameTable,
    traits: LayoutTraits,
) -> int:
    # Bias the search around the name table first, then fall back to the whole image.
    count = len(name_table.names)
    before_start = max(memmap.drom_file_offset, name_table.offset - 0x2000)
    before_hits = _candidate_runs(data, memmap, traits, before_start, name_table.offset, count)
    if before_hits:
        return before_hits[-1]

    after_start = min(memmap.drom_end_file_offset, name_table.end_offset)
    after_hits = _candidate_runs(data, memmap, traits, after_start, memmap.drom_end_file_offset, count)
    if after_hits:
        return after_hits[0]

    all_hits = _candidate_runs(
        data,
        memmap,
        traits,
        memmap.drom_file_offset,
        memmap.drom_end_file_offset,
        count,
    )
    if all_hits:
        return min(all_hits, key=lambda candidate: abs(candidate - name_table.offset))
    raise ValueError("unable to locate the frozen content array near the name table")


def resolve_modules(
    data: bytes,
    memmap: MemoryMap,
    name_table: NameTable,
    traits: LayoutTraits,
) -> tuple[int, tuple[ModuleRecord, ...]]:
    # Map each content-array entry to the module record layout used by that firmware version.
    content_offset = find_content_array(data, memmap, name_table, traits)
    modules: list[ModuleRecord] = []
    for index, name in enumerate(name_table.names):
        ptr_offset = content_offset + index * 4
        content_vaddr = memmap.offset_to_vaddr(ptr_offset)
        if content_vaddr is None:
            raise ValueError(f"content array entry for {name} is outside the memory map")
        struct_vaddr = read_u32(data, ptr_offset)
        struct_offset = memmap.vaddr_to_offset(struct_vaddr)
        if struct_offset is None:
            raise ValueError(f"broken module pointer for {name}")
        if traits.mpy_format == "v6":
            qstr_vaddr = read_u32(data, struct_offset)
            obj_vaddr = read_u32(data, struct_offset + 4)
            proto_vaddr = read_u32(data, struct_offset + 8)
            const_table_vaddr = 0
        else:
            qstr_vaddr = 0
            obj_vaddr = 0
            proto_vaddr = struct_vaddr
            const_table_vaddr = read_u32(data, struct_offset + 8)
        modules.append(
            ModuleRecord(
                name=name,
                content_vaddr=content_vaddr,
                struct_vaddr=struct_vaddr,
                proto_vaddr=proto_vaddr,
                qstr_table_vaddr=qstr_vaddr,
                obj_table_vaddr=obj_vaddr,
                const_table_vaddr=const_table_vaddr,
            )
        )
    return content_offset, tuple(modules)
