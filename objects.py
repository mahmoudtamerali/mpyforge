from __future__ import annotations

import math
import struct

from .firmware import read_u32
from .models import DecodedObject, MemoryMap
from .rendering import unresolved_qstr_label

_MAX_STR_LEN = 1024 * 1024
_MAX_TUPLE_LEN = 256
_MAX_INT_DIGITS = 10000
_MAX_DECODE_DEPTH = 8


def decode_small_int(word: int) -> int | None:
    # Small ints use the low tag bit and keep the signed payload in the rest.
    if word & 1 == 0:
        return None
    value = word >> 1
    if value & (1 << 30):
        value -= 1 << 31
    return value


def _pointer_offset(data: bytes, memmap: MemoryMap, word: int) -> int | None:
    # Only treat the word as a pointer if it lands inside a mapped region.
    if not memmap.contains_vaddr(word):
        return None
    offset = memmap.vaddr_to_offset(word)
    if offset is None or offset + 4 > len(data):
        return None
    return offset


def _read_str_value(data: bytes, memmap: MemoryMap, offset: int) -> str | None:
    # Strings store their length and data pointer in the object header.
    if offset + 16 > len(data):
        return None
    length = read_u32(data, offset + 8)
    data_ptr = read_u32(data, offset + 12)
    if length == 0 or length > _MAX_STR_LEN:
        return None
    data_offset = _pointer_offset(data, memmap, data_ptr)
    if data_offset is None or data_offset + length > len(data):
        return None
    raw = data[data_offset : data_offset + length]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_bytes_value(data: bytes, memmap: MemoryMap, offset: int) -> bytes | None:
    # Bytes objects use the same header shape as strings, but keep the payload raw.
    if offset + 16 > len(data):
        return None
    length = read_u32(data, offset + 8)
    data_ptr = read_u32(data, offset + 12)
    if length == 0 or length > _MAX_STR_LEN:
        return None
    data_offset = _pointer_offset(data, memmap, data_ptr)
    if data_offset is None or data_offset + length > len(data):
        return None
    return bytes(data[data_offset : data_offset + length])


def _read_float_value(data: bytes, offset: int) -> float | None:
    if offset + 8 > len(data):
        return None
    return struct.unpack_from("<f", data, offset + 4)[0]


def _read_complex_value(data: bytes, offset: int) -> complex | None:
    if offset + 12 > len(data):
        return None
    real, imag = struct.unpack_from("<ff", data, offset + 4)
    return complex(real, imag)


def _read_int_value(data: bytes, memmap: MemoryMap, offset: int) -> int | None:
    # Big ints store their digit count and a separate digit array pointer.
    if offset + 16 > len(data):
        return None
    flags = read_u32(data, offset + 4)
    length = read_u32(data, offset + 8)
    digits_ptr = read_u32(data, offset + 12)
    if length > _MAX_INT_DIGITS:
        return None
    if length == 0:
        return 0
    digits_offset = _pointer_offset(data, memmap, digits_ptr)
    if digits_offset is None or digits_offset + length * 2 > len(data):
        return None
    digits = struct.unpack_from(f"<{length}H", data, digits_offset)
    value = 0
    for index in range(length - 1, -1, -1):
        value = (value << 16) | digits[index]
    if flags & 1:
        value = -value
    return value


def _decode_pointer_word(
    data: bytes,
    memmap: MemoryMap,
    word: int,
    qstr_lookup: dict[int, str] | None,
    depth: int,
) -> DecodedObject:
    # Probe the pointed-to object in the order that gives the most readable result.
    offset = _pointer_offset(data, memmap, word)
    if offset is None:
        return DecodedObject("ptr", hex(word))

    text = _read_str_value(data, memmap, offset)
    if text is not None:
        return DecodedObject("str", text)

    raw_bytes = _read_bytes_value(data, memmap, offset)
    if raw_bytes is not None:
        return DecodedObject("bytes", raw_bytes)

    int_value = _read_int_value(data, memmap, offset)
    if int_value is not None:
        return DecodedObject("int", int_value)

    if offset + 8 <= len(data):
        length = read_u32(data, offset + 4)
        if 0 < length <= _MAX_TUPLE_LEN and offset + 8 + length * 4 <= len(data):
            items: list[object] = []
            for index in range(length):
                item_word = read_u32(data, offset + 8 + index * 4)
                item = _decode_object_word(
                    data,
                    memmap,
                    item_word,
                    qstr_lookup=qstr_lookup,
                    depth=depth + 1,
                )
                items.append(item.value)
            return DecodedObject("tuple", tuple(items))

    float_value = _read_float_value(data, offset)
    if float_value is not None and math.isfinite(float_value) and data[offset + 4 : offset + 8] != b"\x00\x00\x00\x00":
        return DecodedObject("float", float_value)

    complex_value = _read_complex_value(data, offset)
    if complex_value is not None:
        return DecodedObject("complex", complex_value)

    return DecodedObject("ptr", hex(word))


def _decode_object_word(
    data: bytes,
    memmap: MemoryMap,
    word: int,
    qstr_lookup: dict[int, str] | None = None,
    depth: int = 0,
) -> DecodedObject:
    # Decode the tag bits first, then fall back to pointer decoding when needed.
    small = decode_small_int(word)
    if small is not None:
        return DecodedObject("int", small)
    if word == 0:
        return DecodedObject("none", None)
    if (word & 0x7) == 0x2:
        qstr_id = word >> 3
        text = qstr_lookup.get(qstr_id, unresolved_qstr_label(qstr_id)) if qstr_lookup is not None else unresolved_qstr_label(qstr_id)
        return DecodedObject("qstr", text)
    if (word & 0x7) == 0x6:
        if word == 0x06:
            return DecodedObject("none", None)
        if word == 0x0E:
            return DecodedObject("bool", False)
        if word == 0x1E:
            return DecodedObject("bool", True)
        return DecodedObject("raw", hex(word))
    if memmap.contains_vaddr(word):
        if depth >= _MAX_DECODE_DEPTH:
            return DecodedObject("ptr", hex(word))
        return _decode_pointer_word(data, memmap, word, qstr_lookup, depth)
    return DecodedObject("raw", hex(word))


def decode_object_table(
    data: bytes,
    memmap: MemoryMap,
    obj_table_vaddr: int,
    count: int,
    *,
    start_index: int = 0,
    qstr_lookup: dict[int, str] | None = None,
) -> tuple[DecodedObject, ...]:
    # Skip the leading slots when the caller knows the table begins after qstr args.
    if obj_table_vaddr == 0 or count <= 0:
        return ()
    offset = memmap.vaddr_to_offset(obj_table_vaddr)
    if offset is None:
        return ()
    return tuple(
        _decode_object_word(
            data,
            memmap,
            read_u32(data, offset + (start_index + index) * 4),
            qstr_lookup=qstr_lookup,
        )
        for index in range(count)
    )
