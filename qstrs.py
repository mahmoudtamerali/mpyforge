from __future__ import annotations

from .firmware import read_u16, read_u32
from .models import LayoutTraits, MemoryMap, ModuleRecord, QstrPool


def _is_ascii_qstr(text: bytes) -> bool:
    # Qstr text should stay in the printable ASCII range we can safely decode.
    if not text:
        return False
    return all(byte in (9, 10, 13, 27) or 32 <= byte < 127 for byte in text)


def _read_c_string(data: bytes, offset: int, limit: int = 128) -> bytes:
    # Read a NUL-terminated string without walking past the search window.
    chunk = data[offset : offset + limit]
    end = chunk.find(b"\x00")
    if end < 0:
        return b""
    return chunk[:end]


def _looks_like_v5_qstr_entry(data: bytes, memmap: MemoryMap, qstr_vaddr: int) -> bool:
    # A valid v5 entry needs a length byte, ASCII text, and a trailing NUL.
    qstr_offset = memmap.vaddr_to_offset(qstr_vaddr)
    if qstr_offset is None or qstr_offset + 4 > len(data):
        return False
    length = data[qstr_offset + 2]
    if qstr_offset + 3 + length >= len(data):
        return False
    if length == 0:
        return data[qstr_offset + 3] == 0 and any(data[qstr_offset : qstr_offset + 3])
    text = data[qstr_offset + 3 : qstr_offset + 3 + length]
    return _is_ascii_qstr(text) and data[qstr_offset + 3 + length] == 0


def _resolve_v5_qstrs_offset(data: bytes, memmap: MemoryMap, pool_offset: int) -> int | None:
    # v5-inline pools can start at either of the two known qstr table offsets.
    qstrs_offset = 0x10
    second_qstr_vaddr = read_u32(data, pool_offset + qstrs_offset + 4)
    if _looks_like_v5_qstr_entry(data, memmap, second_qstr_vaddr):
        return qstrs_offset

    qstrs_offset = 0x18
    first_qstr_vaddr = read_u32(data, pool_offset + qstrs_offset)
    if _looks_like_v5_qstr_entry(data, memmap, first_qstr_vaddr):
        return qstrs_offset
    return None


def _looks_like_v6_qstr_entry(
    data: bytes,
    memmap: MemoryMap,
    lengths_vaddr: int,
    qstr_vaddr: int,
    index: int,
) -> bool:
    # v6 stores the string bytes separately from the length table, so empty entries
    # fall back to a C-string probe.
    qstr_offset = memmap.vaddr_to_offset(qstr_vaddr)
    lengths_offset = memmap.vaddr_to_offset(lengths_vaddr)
    if qstr_offset is None or lengths_offset is None:
        return False
    if lengths_offset + index >= len(data):
        return False
    length = data[lengths_offset + index]
    if length == 0:
        text = _read_c_string(data, qstr_offset)
    else:
        text = data[qstr_offset : qstr_offset + length]
    return _is_ascii_qstr(text)


def _scan_pool_candidates(data: bytes, memmap: MemoryMap, traits: LayoutTraits) -> dict[int, QstrPool]:
    # Scan 4-byte aligned regions and keep only the ones that look like qstr pool headers.
    candidates: dict[int, QstrPool] = {}
    data_end = len(data)
    for offset in range(0, data_end - 0x20 + 1, 4):
        prev_vaddr = read_u32(data, offset)
        total_prev_len = read_u32(data, offset + 4)
        alloc = read_u32(data, offset + 8)
        length = read_u32(data, offset + 12)
        if alloc != 10 or length == 0 or length > 5000:
            continue
        pool_vaddr = memmap.offset_to_vaddr(offset)
        if pool_vaddr is None:
            continue
        sorted_pool = False
        total_prev = total_prev_len
        if traits.has_sorted_pool_flag:
            sorted_pool = bool(total_prev_len & 0x80000000)
            total_prev &= 0x7FFFFFFF
        if total_prev > 5000:
            continue
        if traits.qstr_layout == "v5-inline":
            qstrs_offset = _resolve_v5_qstrs_offset(data, memmap, offset)
            if qstrs_offset is None:
                continue
            qstr_table_vaddr = memmap.offset_to_vaddr(offset + qstrs_offset)
            if qstr_table_vaddr is None:
                continue
            pool = QstrPool(
                vaddr=pool_vaddr,
                prev_vaddr=prev_vaddr,
                total_prev=total_prev,
                length=length,
                sorted_pool=sorted_pool,
                qstr_table_vaddr=qstr_table_vaddr,
            )
        else:
            hash_vaddr = read_u32(data, offset + 16)
            length_vaddr = read_u32(data, offset + 20)
            qstr_table_vaddr = memmap.offset_to_vaddr(offset + 0x18)
            first_qstr_vaddr = read_u32(data, offset + 24)
            if length_vaddr == 0:
                continue
            if qstr_table_vaddr is None:
                continue
            if memmap.vaddr_to_offset(length_vaddr) is None:
                continue
            relaxed_static = prev_vaddr == 0 and total_prev == 0
            if not relaxed_static and not _looks_like_v6_qstr_entry(data, memmap, length_vaddr, first_qstr_vaddr, 0):
                continue
            if relaxed_static and not memmap.contains_vaddr(first_qstr_vaddr):
                continue
            pool = QstrPool(
                vaddr=pool_vaddr,
                prev_vaddr=prev_vaddr,
                total_prev=total_prev,
                length=length,
                sorted_pool=sorted_pool,
                hash_vaddr=hash_vaddr,
                length_vaddr=length_vaddr,
                qstr_table_vaddr=qstr_table_vaddr,
            )
        candidates[pool.vaddr] = pool
    return candidates


def _find_predecessor(candidates: dict[int, QstrPool], current: QstrPool, used: set[int]) -> QstrPool | None:
    # Prefer the pool whose qstr count lines up exactly; fall back to the stored back-pointer.
    exact_matches = [
        pool
        for pool in candidates.values()
        if pool.vaddr not in used and pool.total_prev + pool.length == current.total_prev
    ]
    if exact_matches:
        return max(exact_matches, key=lambda pool: pool.vaddr)
    pointer_match = candidates.get(current.prev_vaddr)
    if pointer_match is not None and pointer_match.vaddr not in used:
        return pointer_match
    return None


def recover_qstr_pools(data: bytes, memmap: MemoryMap, traits: LayoutTraits) -> tuple[QstrPool, ...]:
    # Walk the linked list from the newest pool back to the oldest one.
    candidates = _scan_pool_candidates(data, memmap, traits)
    if not candidates:
        raise ValueError("unable to locate any qstr pools")
    tail = max(candidates.values(), key=lambda pool: (pool.total_prev + pool.length, pool.vaddr))
    ordered: list[QstrPool] = []
    current = tail
    seen: set[int] = set()
    while current.vaddr not in seen:
        seen.add(current.vaddr)
        ordered.append(current)
        if current.prev_vaddr == 0:
            break
        prev = _find_predecessor(candidates, current, seen)
        if prev is None:
            break
        current = prev
    ordered.reverse()
    return tuple(ordered)


def build_qstr_mapping(data: bytes, memmap: MemoryMap, pools: tuple[QstrPool, ...], traits: LayoutTraits) -> dict[int, str]:
    # Convert the recovered pool chain into a global qstr ID -> string map.
    mapping: dict[int, str] = {}
    for pool in pools:
        table_offset = memmap.vaddr_to_offset(pool.qstr_table_vaddr)
        if table_offset is None:
            continue
        for index in range(pool.length):
            qstr_id = pool.total_prev + index
            if traits.qstr_layout == "v5-inline":
                qstr_vaddr = read_u32(data, table_offset + index * 4)
                entry_offset = memmap.vaddr_to_offset(qstr_vaddr)
                if entry_offset is None:
                    continue
                length = data[entry_offset + 2]
                text = data[entry_offset + 3 : entry_offset + 3 + length]
            else:
                qstr_vaddr = read_u32(data, table_offset + index * 4)
                entry_offset = memmap.vaddr_to_offset(qstr_vaddr)
                length_offset = memmap.vaddr_to_offset(pool.length_vaddr)
                if entry_offset is None or length_offset is None:
                    continue
                length = data[length_offset + index]
                text = _read_c_string(data, entry_offset) if length == 0 else data[entry_offset : entry_offset + length]
            mapping[qstr_id] = text.decode("ascii", errors="replace")
    return mapping


def read_module_qstr_table(
    data: bytes,
    memmap: MemoryMap,
    module: ModuleRecord,
    qstr_count: int,
    limit: int = 1024,
) -> tuple[int, ...]:
    # Stop before the next table so the module qstr list does not run into other data.
    if module.qstr_table_vaddr == 0:
        return ()
    offset = memmap.vaddr_to_offset(module.qstr_table_vaddr)
    if offset is None:
        return ()
    bound_offsets = []
    for next_vaddr in (module.obj_table_vaddr, module.proto_vaddr):
        if next_vaddr and next_vaddr > module.qstr_table_vaddr:
            next_offset = memmap.vaddr_to_offset(next_vaddr)
            if next_offset is not None:
                bound_offsets.append(next_offset)
    if bound_offsets:
        limit = min(limit, max(0, (min(bound_offsets) - offset) // 2))
    items: list[int] = []
    for index in range(limit):
        if offset + index * 2 + 2 > len(data):
            break
        value = read_u16(data, offset + index * 2)
        if value >= qstr_count:
            break
        items.append(value)
    return tuple(items)
