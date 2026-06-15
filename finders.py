from __future__ import annotations

from .models import NameTable

_MODULE_CHARS = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-"
)


def _is_valid_name(name: bytes) -> bool:
    # Frozen module names must look like real paths, not arbitrary byte strings.
    if not name or not name.endswith(b".py"):
        return False
    if b".." in name or name.startswith(b"/") or name.endswith(b"/"):
        return False
    return all(byte in _MODULE_CHARS for byte in name)


def _read_candidate_chain(data: bytes, start: int) -> tuple[tuple[str, ...], int]:
    # Follow the NUL-separated name chain until the table ends or a name stops matching.
    names: list[str] = []
    pos = start
    while pos < len(data):
        end = data.find(b"\x00", pos)
        if end < 0:
            return (), len(data)
        if end == pos:  # Double NUL marks the end of the frozen name chain.
            return (tuple(names), end + 1) if names else ((), end + 1)
        name_bytes = data[pos:end]
        if not _is_valid_name(name_bytes):
            return (), pos + 1
        names.append(name_bytes.decode("ascii"))
        pos = end + 1

    return (), len(data)


def find_frozen_names(data: bytes) -> NameTable:
    # Search for the longest valid chain of module paths and keep that as the table.
    best: NameTable | None = None
    cursor = 0
    needle = b".py\x00"
    while True:
        hit = data.find(needle, cursor)
        if hit < 0:
            break
        start = hit
        while start > 0 and data[start - 1] in _MODULE_CHARS:
            start -= 1
        names, end_pos = _read_candidate_chain(data, start)
        if len(names) >= 2:
            candidate = NameTable(offset=start, names=names)
            if best is None or (len(candidate.names), candidate.end_offset) > (len(best.names), best.end_offset):
                best = candidate
            cursor = end_pos
            continue
        cursor = hit + 1
    if best is None:
        raise ValueError("unable to locate a frozen-name table")
    return best
