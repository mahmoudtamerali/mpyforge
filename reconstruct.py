from __future__ import annotations

from .bytecode import (
    CHILD_OPS,
    EXTRA_BYTE_MASK,
    FORMAT_OFFSET,
    FORMAT_QSTR,
    FORMAT_VAR_UINT,
    LOAD_CONST_OBJ,
    OPCODE_FORMAT_TABLE,
    RETURN_VALUE,
    decode_var_uint,
    _decode_opcode,
)
from .models import CodeNode, DecodedObject, LayoutTraits
from .rendering import unresolved_qstr_label

_V5_REWRITE_TRAITS = LayoutTraits("v5", "v5", "v5-inline", False)

MPY_HEADER = bytes((ord("M"), 6, 0, 31))
OBJ_NONE = 1
OBJ_FALSE = 2
OBJ_TRUE = 3
OBJ_ELLIPSIS = 4
OBJ_STR = 5
OBJ_BYTES = 6
OBJ_INT = 7
OBJ_FLOAT = 8
OBJ_COMPLEX = 9
OBJ_TUPLE = 10

def encode_uint(value: int) -> bytes:
    # Encode an integer using MicroPython's 7-bit continuation format.
    buf = bytearray([value & 0x7F])
    value >>= 7
    while value:
        buf.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(buf)


def encode_prelude_size(n_info: int, n_cell: int) -> bytes:
    # Pack prelude metadata into the compact size field used by .mpy files.
    payload = bytearray()
    info = n_info
    cell = n_cell
    while True:
        value = ((info & 0x3F) << 1) | (cell & 1)
        info >>= 6
        cell >>= 1
        if info or cell:
            value |= 0x80
        payload.append(value)
        if not (info or cell):
            break
    return bytes(payload)


def _encode_qstr(text: str | None) -> bytes:
    # Emit the qstr as a normal string entry so the output stays self-contained.
    payload = (text or "").encode("utf-8")
    return encode_uint(len(payload) << 1) + payload + b"\x00"


def encode_object(obj: DecodedObject) -> bytes:
    # Map each recovered object kind back to the object tag used in .mpy output.
    if obj.kind == "none":
        return bytes((OBJ_NONE,))
    if obj.kind == "bool":
        return bytes((OBJ_TRUE if obj.value else OBJ_FALSE,))
    if obj.kind == "ellipsis":
        return bytes((OBJ_ELLIPSIS,))
    if obj.kind == "int":
        payload = str(obj.value).encode("ascii")
        return bytes((OBJ_INT,)) + encode_uint(len(payload)) + payload
    if obj.kind == "float":
        payload = repr(obj.value).encode("ascii")
        return bytes((OBJ_FLOAT,)) + encode_uint(len(payload)) + payload
    if obj.kind == "complex":
        payload = repr(obj.value).encode("ascii")
        return bytes((OBJ_COMPLEX,)) + encode_uint(len(payload)) + payload
    if obj.kind == "tuple":
        items = bytearray((OBJ_TUPLE,))
        value = obj.value if isinstance(obj.value, tuple) else ()
        items.extend(encode_uint(len(value)))
        for item in value:
            items.extend(encode_object(DecodedObject("int", item)))
        return bytes(items)
    if obj.kind == "bytes":
        payload = bytes(obj.value)  # type: ignore[arg-type]
        return bytes((OBJ_BYTES,)) + encode_uint(len(payload)) + payload + b"\x00"
    if obj.kind in {"str", "qstr"}:
        payload = str(obj.value).encode("utf-8")
        return bytes((OBJ_STR,)) + encode_uint(len(payload)) + payload + b"\x00"
    if obj.kind in {"ptr", "raw"}:
        payload = f"<{obj.kind} {obj.value}>".encode("utf-8")
        return bytes((OBJ_STR,)) + encode_uint(len(payload)) + payload + b"\x00"
    raise ValueError(f"unsupported object kind: {obj.kind!r}")


def _encode_raw_code(node: CodeNode) -> bytes:
    # Write one raw-code node, then append any nested child nodes after it.
    payload = bytearray()
    payload.extend(encode_uint((len(node.bytecode) << 3) | ((1 if node.children else 0) << 2)))
    payload.extend(node.bytecode)
    if node.children:
        payload.extend(encode_uint(len(node.children)))
        for child in node.children:
            payload.extend(_encode_raw_code(child))
    return bytes(payload)


def _collect_v5_qstr_ids(root: CodeNode) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []

    def add(qstr_id: int) -> None:
        if qstr_id and qstr_id not in seen:
            seen.add(qstr_id)
            ordered.append(qstr_id)

    def visit(node: CodeNode) -> None:
        add(node.prelude.source_file)
        add(node.prelude.simple_name)
        for qstr_id in node.prelude.arg_names:
            add(qstr_id)
        qstr_ids = node.qstr_ids if node.qstr_ids else tuple(sorted(node.qstr_args))
        for qstr_id in qstr_ids:
            add(qstr_id)
        for child in node.children:
            visit(child)

    visit(root)
    return tuple(ordered)


def collect_v5_qstr_table(root: CodeNode, qstr_lookup: dict[int, str]) -> tuple[str, ...]:
    # Turn the collected v5 qstr IDs into the final string table.
    qstr_ids = _collect_v5_qstr_ids(root)
    return tuple(qstr_lookup.get(qstr_id, unresolved_qstr_label(qstr_id)) for qstr_id in qstr_ids)


def collect_v5_qstr_ids(root: CodeNode) -> tuple[int, ...]:
    # Expose the ordered v5 qstr ID list to callers that need the raw IDs.
    return _collect_v5_qstr_ids(root)


def _rewrite_v5_node(
    node: CodeNode,
    qstr_id_to_local: dict[int, int],
    object_base: int,
) -> int:
    # Rebuild this node in place before recursing into children, because child object
    # indexes depend on how many objects belong to the current node.
    original = node.bytecode
    prelude = node.prelude
    if prelude.sig_size <= 0 or prelude.size_size <= 0 or prelude.n_info <= 0:
        return 0

    source_info_start = prelude.sig_size + prelude.size_size
    source_info_end = source_info_start + prelude.n_info
    closure_end = source_info_end + prelude.n_cells

    source_cursor = source_info_start
    for _ in range(1 + len(prelude.arg_names)):
        _, source_cursor = decode_var_uint(original, source_cursor)
    line_info = original[source_cursor:source_info_end]

    new_source = bytearray()
    new_source.extend(encode_uint(qstr_id_to_local.get(prelude.simple_name, 0)))
    for arg_name in prelude.arg_names:
        new_source.extend(encode_uint(qstr_id_to_local.get(arg_name, 0)))
    new_source.extend(line_info)

    rewritten = bytearray()
    rewritten.extend(original[: prelude.sig_size])
    rewritten.extend(encode_prelude_size(len(new_source), prelude.n_cells))
    rewritten.extend(new_source)
    rewritten.extend(original[source_info_end:closure_end])

    # The opcode stream is rewritten in place so qstr IDs and object references
    # line up with the compact tables we are rebuilding.
    n_args = len(prelude.arg_names)
    node_object_count = max(node.max_obj - n_args + 1, 0) if node.max_obj >= n_args else 0
    opcodes_offset = prelude.opcode_offset
    cursor = opcodes_offset
    while cursor < len(original):
        opcode, size, arg = _decode_opcode(original, cursor, _V5_REWRITE_TRAITS)
        fmt = (OPCODE_FORMAT_TABLE >> (2 * (opcode >> 4))) & 0x03
        # RETURN is the end of the function body; the bytes after it are not code.
        if opcode == RETURN_VALUE:
            rewritten.extend(original[cursor:cursor + size])
            break
        if fmt == FORMAT_QSTR and arg is not None:
            local_idx = qstr_id_to_local.get(arg, 0)
            rewritten.append(opcode)
            rewritten.extend(bytes((local_idx & 0xFF, (local_idx >> 8) & 0xFF)))
        elif opcode == LOAD_CONST_OBJ and arg is not None:
            local_idx = max(arg - n_args, 0)
            rewritten.append(opcode)
            rewritten.extend(encode_uint(object_base + local_idx))
            if opcode & EXTRA_BYTE_MASK == 0 and size > 1:
                rewritten.append(original[cursor + size - 1])
        elif opcode in CHILD_OPS and arg is not None:
            local_idx = max(arg - n_args - node_object_count, 0)
            rewritten.append(opcode)
            rewritten.extend(encode_uint(local_idx))
            if opcode & EXTRA_BYTE_MASK == 0 and size > 1:
                rewritten.append(original[cursor + size - 1])
        else:
            rewritten.extend(original[cursor:cursor + size])
        cursor += size

    node.bytecode = bytes(rewritten)

    total_objects = node_object_count
    running_object_base = object_base + node_object_count
    for child in node.children:
        # Child nodes reuse the next object slot range after the current node.
        child_total = _rewrite_v5_node(child, qstr_id_to_local, running_object_base)
        running_object_base += child_total
        total_objects += child_total
    return total_objects


def build_mpy_v5(
    root: CodeNode,
    objects: tuple[DecodedObject, ...],
    qstr_lookup: dict[int, str],
) -> bytes:
    # Rebase the v5 tree first, then emit the shared .mpy layout.
    qstr_table = collect_v5_qstr_table(root, qstr_lookup)
    qstr_ids = _collect_v5_qstr_ids(root)
    qstr_id_to_local = {qstr_id: index for index, qstr_id in enumerate(qstr_ids)}
    _rewrite_v5_node(root, qstr_id_to_local, 0)
    return build_mpy(root, qstr_table, objects)


def build_mpy(root: CodeNode, qstr_table: tuple[str, ...], objects: tuple[DecodedObject, ...]) -> bytes:
    # Write the header, qstr table, object table, and code payload in order.
    payload = bytearray(MPY_HEADER)
    payload.extend(encode_uint(len(qstr_table)))
    payload.extend(encode_uint(len(objects)))
    for entry in qstr_table:
        payload.extend(_encode_qstr(entry))
    for obj in objects:
        payload.extend(encode_object(obj))
    payload.extend(_encode_raw_code(root))
    return bytes(payload)
