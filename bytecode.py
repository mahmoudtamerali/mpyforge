from __future__ import annotations

from .firmware import read_u16, read_u32
from .models import CodeNode, LayoutTraits, MemoryMap, ModuleRecord, PreludeInfo

FORMAT_QSTR = 1
FORMAT_VAR_UINT = 2
FORMAT_OFFSET = 3
OPCODE_FORMAT_TABLE = 0x000003A4
EXTRA_BYTE_MASK = 0x9E
SIGNED_OFFSET_OPS = {0x40, 0x42, 0x43, 0x44}
CHILD_OPS = {0x20, 0x21, 0x32, 0x33}
RETURN_VALUE = 0x63
LOAD_CONST_OBJ = 0x23
MAX_CODE_BYTES = 0x10000
MAX_DEPTH = 32

# Unconditional control-flow that never falls through to the next opcode.
_FLOW_TERMINATORS = {0x40, 0x42, 0x4A, 0x63, 0x64, 0x65, 0x66}  # UNWIND_JUMP, JUMP, POP_EXCEPT_JUMP, RETURN/RAISE*


def decode_var_uint(data: bytes, offset: int) -> tuple[int, int]:
    # MicroPython stores integers as 7-bit chunks with a continuation bit.
    value = data[offset] & 0x7F
    pos = offset
    while data[pos] & 0x80:
        pos += 1
        value = (value << 7) | (data[pos] & 0x7F)
    return value, pos + 1


def _decode_prelude_sig(data: bytes, offset: int) -> tuple[int, int, int, int, int, int, int]:
    # The first prelude bytes pack stack, exception, and argument counts together.
    pos = offset
    first = data[pos]
    pos += 1
    n_state = ((first >> 3) & 0x0F) + 1
    n_exc = (first >> 2) & 0x01
    scope_flags = 0
    n_pos_args = first & 0x03
    n_kwonly = 0
    n_def_pos = 0
    shift = 0
    while first & 0x80:
        first = data[pos]
        pos += 1
        n_state |= (first & 0x30) << (2 * shift)
        n_exc |= (first & 0x02) << shift
        scope_flags |= ((first & 0x40) >> 6) << shift
        n_pos_args |= (first & 0x04) << shift
        n_kwonly |= ((first & 0x08) >> 3) << shift
        n_def_pos |= (first & 0x01) << shift
        shift += 1
    return n_state, n_exc, scope_flags, n_pos_args, n_kwonly, n_def_pos, pos


def _decode_prelude_size(data: bytes, offset: int) -> tuple[int, int, int]:
    # The size field packs source-info and cell counts into a compact varint-like form.
    pos = offset
    n_info = 0
    n_cell = 0
    shift = 0
    while True:
        value = data[pos]
        pos += 1
        n_info |= ((value & 0x7E) >> 1) << (6 * shift)
        n_cell |= (value & 0x01) << shift
        if not (value & 0x80):
            break
        shift += 1
    return n_info, n_cell, pos


def _parse_v5_prelude(data: bytes) -> PreludeInfo:
    n_state, n_exc, scope_flags, n_pos_args, n_kwonly, n_def_pos, pos = _decode_prelude_sig(data, 0)
    n_info, n_cell, names_pos = _decode_prelude_size(data, pos)
    simple_name = read_u16(data, names_pos)
    source_file = read_u16(data, names_pos + 2)
    opcode_offset = names_pos + n_info + n_cell
    return PreludeInfo(
        opcode_offset=opcode_offset,
        simple_name=simple_name,
        source_file=source_file,
        arg_names=(),
        n_state=n_state,
        n_exc_stack=n_exc,
        n_pos_args=n_pos_args,
        n_kwonly_args=n_kwonly,
        n_def_pos_args=n_def_pos,
        scope_flags=scope_flags,
        n_info=n_info,
        n_cells=n_cell,
        sig_size=pos,
        size_size=names_pos - pos,
    )


def parse_prelude(data: bytes, traits: LayoutTraits | None = None) -> PreludeInfo:
    if traits is not None and traits.mpy_format == "v5":
        return _parse_v5_prelude(data)

    # Newer formats store the simple name and argument names as varuint IDs.
    n_state, n_exc, scope_flags, n_pos_args, n_kwonly, n_def_pos, pos = _decode_prelude_sig(data, 0)
    n_info, n_cell, names_pos = _decode_prelude_size(data, pos)
    simple_name, cursor = decode_var_uint(data, names_pos)
    arg_names: list[int] = []
    for _ in range(n_pos_args + n_kwonly):
        name, cursor = decode_var_uint(data, cursor)
        arg_names.append(name)
    opcode_offset = names_pos + n_info + n_cell
    return PreludeInfo(
        opcode_offset=opcode_offset,
        simple_name=simple_name,
        source_file=0,
        arg_names=tuple(arg_names),
        n_state=n_state,
        n_exc_stack=n_exc,
        n_pos_args=n_pos_args,
        n_kwonly_args=n_kwonly,
        n_def_pos_args=n_def_pos,
        scope_flags=scope_flags,
        n_info=n_info,
        n_cells=n_cell,
        sig_size=pos,
        size_size=names_pos - pos,
    )


def _decode_opcode(data: bytes, offset: int, traits: LayoutTraits | None = None) -> tuple[int, int, int | None]:
    # Decode one opcode and report how many bytes it consumed.
    opcode = data[offset]
    fmt = (OPCODE_FORMAT_TABLE >> (2 * (opcode >> 4))) & 0x03
    pos = offset + 1
    arg: int | None = None
    if fmt in (FORMAT_QSTR, FORMAT_VAR_UINT):
        # v5 keeps qstr operands as fixed 16-bit IDs; newer formats use varuints.
        if fmt == FORMAT_QSTR and traits is not None and traits.mpy_format == "v5":
            arg = data[pos] | (data[pos + 1] << 8)
            pos += 2
        else:
            arg = data[pos] & 0x7F
            if opcode == 0x22 and arg & 0x40:
                arg |= -1 << 7
            while data[pos] & 0x80:
                pos += 1
                arg = (arg << 7) | (data[pos] & 0x7F)
            pos += 1
    elif fmt == FORMAT_OFFSET:
        if data[pos] & 0x80:
            arg = (data[pos] & 0x7F) | (data[pos + 1] << 7)
            pos += 2
            if opcode in SIGNED_OFFSET_OPS:
                arg -= 0x4000
        else:
            arg = data[pos]
            pos += 1
            if opcode in SIGNED_OFFSET_OPS:
                arg -= 0x40
    if opcode & EXTRA_BYTE_MASK == 0 and not (fmt == FORMAT_QSTR and traits is not None and traits.mpy_format == "v5"):
        pos += 1
    return opcode, pos - offset, arg


def _scan_opcode_stream(
    window: bytes,
    opcode_offset: int,
    traits: LayoutTraits,
) -> tuple[int, list[int], set[int], set[int]]:
    # Record every qstr/object/child reference we see while chasing jump targets.
    qstr_scan_ids: list[int] = []
    obj_indexes: set[int] = set()
    child_indexes: set[int] = set()

    # We don't know the bytecode length up-front (frozen firmware doesn't always
    # store it).  Instead, scan forward until we've passed the furthest jump
    # target we have seen, and we hit an opcode that doesn't fall through.
    cursor = opcode_offset
    furthest = opcode_offset
    while 0 <= cursor < len(window):
        try:
            opcode, size, arg = _decode_opcode(window, cursor, traits)
        except IndexError:
            break
        if size <= 0:
            break

        next_cursor = cursor + size
        fmt = (OPCODE_FORMAT_TABLE >> (2 * (opcode >> 4))) & 0x03

        if fmt == FORMAT_QSTR and arg is not None:
            qstr_scan_ids.append(arg)
        elif opcode == LOAD_CONST_OBJ and arg is not None:
            obj_indexes.add(arg)
        elif opcode in CHILD_OPS and arg is not None and arg >= 0:
            child_indexes.add(arg)

        if fmt == FORMAT_OFFSET and arg is not None:
            # Offsets are relative to the ip following the offset encoding.
            # UNWIND_JUMP is special: it has an extra byte, and the VM computes
            # the destination relative to the ip pointing at that extra byte.
            base = next_cursor - 1 if opcode == 0x40 else next_cursor
            target = base + arg
            if opcode_offset <= target < len(window):
                furthest = max(furthest, target)

        cursor = next_cursor
        # Keep scanning while a later jump can still land inside the window.
        if opcode in _FLOW_TERMINATORS and cursor > furthest:
            break

    return cursor, qstr_scan_ids, obj_indexes, child_indexes


def _read_v6_children(data: bytes, memmap: MemoryMap, children_vaddr: int, count: int) -> list[int]:
    # v6 stores child raw-code pointers in a flat array, one pointer per child.
    if children_vaddr == 0 or count == 0:
        return []
    offset = memmap.vaddr_to_offset(children_vaddr)
    if offset is None:
        return []
    children: list[int] = []
    for index in range(count):
        child = read_u32(data, offset + index * 4)
        if memmap.contains_vaddr(child):
            children.append(child)
    return children


def _child_pointers_from_v5(data: bytes, memmap: MemoryMap, raw_offset: int, fun_data_offset: int) -> list[int]:
    # In v5, child pointers sit between the raw-code header and the function body.
    pointers: list[int] = []
    for cursor in range(raw_offset + 0x10, fun_data_offset, 4):
        child_vaddr = read_u32(data, cursor)
        child_offset = memmap.vaddr_to_offset(child_vaddr)
        if child_offset is None:
            continue
        kind = read_u32(data, child_offset)
        next_fun = read_u32(data, child_offset + 4)
        if kind in {2, 3, 4, 5} and memmap.contains_vaddr(next_fun):
            pointers.append(child_vaddr)
    return pointers


def _read_v5_const_table(
    data: bytes,
    memmap: MemoryMap,
    const_table_vaddr: int,
    n_args: int,
    n_obj: int,
    n_children: int,
) -> tuple[tuple[int, ...], tuple[int, ...], list[int]]:
    # v5 stores argument qstrs, objects, and child pointers in one contiguous block.
    if const_table_vaddr == 0:
        return (), (), []
    offset = memmap.vaddr_to_offset(const_table_vaddr)
    if offset is None:
        return (), (), []
    qstr_ids = tuple(
        read_u32(data, offset + index * 4) >> 3
        for index in range(n_args)
    )
    obj_words = tuple(
        read_u32(data, offset + (n_args + index) * 4)
        for index in range(n_obj)
    )
    children_offset = offset + (n_args + n_obj) * 4
    children: list[int] = []
    for index in range(n_children):
        child_vaddr = read_u32(data, children_offset + index * 4)
        if memmap.contains_vaddr(child_vaddr):
            children.append(child_vaddr)
    return qstr_ids, obj_words, children


def _extract_node(
    data: bytes,
    memmap: MemoryMap,
    traits: LayoutTraits,
    module_name: str,
    proto_vaddr: int,
    depth: int,
) -> CodeNode:
    if depth > MAX_DEPTH:
        raise ValueError("bytecode tree exceeded the safety depth")
    children_vaddrs: list[int] = []
    raw_struct_vaddr = 0
    const_table_vaddr = 0
    if traits.mpy_format == "v6":
        # Current and older v6 layouts share the same caller path, but not the same header shape.
        proto_offset = memmap.vaddr_to_offset(proto_vaddr)
        if proto_offset is None:
            raise ValueError("bad v6 proto pointer")
        if traits.label == "v6-current":
            # Current v6 builds can point either at the raw-code header or straight at fun_data.
            if data[proto_offset] == 0 and data[proto_offset + 1] == 0:
                kind = data[proto_offset + 2] & 0x07
                if kind not in {2, 3, 4, 5}:
                    raise ValueError("unexpected v6-current raw-code kind")
                fun_data_vaddr = read_u32(data, proto_offset + 4)
                fun_data_offset = memmap.vaddr_to_offset(fun_data_vaddr)
                if fun_data_offset is None:
                    raise ValueError("bad v6 fun_data pointer")
                raw_struct_vaddr = proto_vaddr
                children_table_vaddr = read_u32(data, proto_offset + 8)
            else:
                fun_data_vaddr = proto_vaddr
                children_table_vaddr = 0
        elif traits.label in {"v6-early", "v6-sorted"}:
            first_word = read_u32(data, proto_offset)
            kind = first_word & 0x07
            if kind not in {2, 3, 4, 5}:
                raise ValueError("unexpected v6-legacy raw-code kind")
            fun_data_vaddr = read_u32(data, proto_offset + 4)
            fun_data_offset = memmap.vaddr_to_offset(fun_data_vaddr)
            if fun_data_offset is None:
                raise ValueError("bad v6 fun_data pointer")
            raw_struct_vaddr = proto_vaddr
            children_table_vaddr = read_u32(data, proto_offset + 8)
        else:
            raise ValueError(f"unsupported v6 raw-code layout label: {traits.label!r}")
    else:
        # v5 modules point at the raw-code struct, which carries the const table pointer.
        raw_struct_vaddr = proto_vaddr
        raw_offset = memmap.vaddr_to_offset(proto_vaddr)
        if raw_offset is None:
            raise ValueError("bad v5 raw-code pointer")
        fun_data_vaddr = read_u32(data, raw_offset + 4)
        const_table_vaddr = read_u32(data, raw_offset + 8)
        fun_data_offset = memmap.vaddr_to_offset(fun_data_vaddr)
        if fun_data_offset is None:
            raise ValueError("bad v5 fun_data pointer")
        children_table_vaddr = 0
    fun_data_offset = memmap.vaddr_to_offset(fun_data_vaddr)
    if fun_data_offset is None:
        raise ValueError("fun_data is outside DROM")
    window = data[fun_data_offset : min(fun_data_offset + MAX_CODE_BYTES, memmap.drom_end_file_offset)]
    prelude = parse_prelude(window, traits)
    cursor = prelude.opcode_offset
    qstr_args = {prelude.simple_name}
    if prelude.source_file:
        qstr_args.add(prelude.source_file)
    cursor, qstr_scan_ids, obj_indexes, child_indexes = _scan_opcode_stream(window, cursor, traits)

    obj_words: tuple[int, ...] = ()
    ordered_qstrs: list[int] = []

    # v6 bytecode qstr operands are indices into the module qstr table (local indices).
    if traits.mpy_format == "v6":
        qstr_args.update(qstr_scan_ids)

    if traits.mpy_format == "v6" and raw_struct_vaddr:
        # v6 child indexes are resolved through the separate child pointer table.
        children_vaddrs = _read_v6_children(data, memmap, children_table_vaddr, max(child_indexes, default=-1) + 1)
    elif traits.mpy_format == "v5":
        # v5 stores qstr args, objects, and children in the shared const table.
        n_args = prelude.n_pos_args + prelude.n_kwonly_args
        max_obj_idx = max(obj_indexes, default=-1)
        max_child_idx = max(child_indexes, default=-1)
        n_obj = max(max_obj_idx - n_args + 1, 0) if max_obj_idx >= n_args else 0
        n_children = max(max_child_idx - n_args - n_obj + 1, 0) if max_child_idx >= n_args + n_obj else 0
        arg_qstrs, obj_words, children_vaddrs = _read_v5_const_table(data, memmap, const_table_vaddr, n_args, n_obj, n_children)
        ordered_qstrs: list[int] = []
        seen_qstrs: set[int] = set()
        for qstr_id in (prelude.source_file, prelude.simple_name, *arg_qstrs, *qstr_scan_ids):
            if qstr_id not in seen_qstrs:
                seen_qstrs.add(qstr_id)
                ordered_qstrs.append(qstr_id)
        prelude = PreludeInfo(
            opcode_offset=prelude.opcode_offset,
            simple_name=prelude.simple_name,
            source_file=prelude.source_file,
            arg_names=tuple(arg_qstrs),
            n_state=prelude.n_state,
            n_exc_stack=prelude.n_exc_stack,
            n_pos_args=prelude.n_pos_args,
            n_kwonly_args=prelude.n_kwonly_args,
            n_def_pos_args=prelude.n_def_pos_args,
            scope_flags=prelude.scope_flags,
            n_info=prelude.n_info,
            n_cells=prelude.n_cells,
            sig_size=prelude.sig_size,
            size_size=prelude.size_size,
        )
        qstr_args.update(ordered_qstrs)
    else:
        const_table_vaddr = 0
        children_vaddrs = []
        obj_words = ()
    child_nodes = [
        _extract_node(data, memmap, traits, module_name, child_vaddr, depth + 1)
        for child_vaddr in children_vaddrs[: max(child_indexes, default=-1) + 1]
    ]
    qstr_ids = tuple(ordered_qstrs) if traits.mpy_format == "v5" else tuple(sorted(qstr_args))
    return CodeNode(
        source=module_name,
        fun_data_vaddr=fun_data_vaddr,
        raw_struct_vaddr=raw_struct_vaddr,
        const_table_vaddr=const_table_vaddr,
        bytecode=bytes(window[:cursor]),
        prelude=prelude,
        qstr_ids=qstr_ids,
        qstr_args=qstr_args,
        obj_words=obj_words if traits.mpy_format == "v5" else (),
        obj_indexes=obj_indexes,
        children=child_nodes,
    )


def extract_module_code(
    data: bytes,
    memmap: MemoryMap,
    module: ModuleRecord,
    traits: LayoutTraits,
) -> CodeNode:
    # The module record already gives us the entry point for the root raw-code node.
    proto_vaddr = module.proto_vaddr
    return _extract_node(data, memmap, traits, module.name, proto_vaddr, 0)


def walk_code_tree(root: CodeNode) -> list[CodeNode]:
    # Flatten the tree so callers can compute totals without recursing themselves.
    nodes = [root]
    for child in root.children:
        nodes.extend(walk_code_tree(child))
    return nodes
