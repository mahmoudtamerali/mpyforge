from __future__ import annotations

from pathlib import Path
from typing import cast

from .bytecode import extract_module_code, walk_code_tree
from .firmware import load_workspace_firmware
from .finders import find_frozen_names
from .hardware import device_label, render_memmap
from .models import CodeNode, DecodedObject, LoadedFirmware, MemoryMap, RecoveryLog
from .objects import decode_object_table
from .pipeline import _pad_objects
from .qstrs import build_qstr_mapping, read_module_qstr_table, recover_qstr_pools
from .reconstruct import build_mpy, build_mpy_v5, collect_v5_qstr_ids, collect_v5_qstr_table
from .rendering import (
    banner_lines,
    describe_object,
    escape_text,
    format_code_tree,
    key_value_lines,
    materialize_qstr_table,
    materialize_v5_qstr_table,
    select_modules,
    table_lines,
)
from .resolve import resolve_modules


def _load_module_context(
    firmware: Path,
) -> tuple[LoadedFirmware, bytes, MemoryMap, tuple, tuple, dict[int, str]]:
    # Recover everything the reports need once, then reuse the shared results.
    loaded = load_workspace_firmware(firmware)
    data = loaded.raw
    memory_map = cast(MemoryMap, loaded.memory_map)
    name_table = find_frozen_names(data)
    _, modules = resolve_modules(data, memory_map, name_table, loaded.traits)
    pools = recover_qstr_pools(data, memory_map, loaded.traits)
    qstr_map = build_qstr_mapping(data, memory_map, pools, loaded.traits)
    return loaded, data, memory_map, modules, pools, qstr_map


def _module_artifacts(
    loaded: LoadedFirmware,
    data: bytes,
    memory_map: MemoryMap,
    modules: tuple,
    qstr_map: dict[int, str],
    module_name: str | None,
):
    # Build the qstr table, object table, and code tree from the same module pass.
    selected_modules = select_modules(modules, module_name)
    max_qstr_id = max(qstr_map, default=-1) + 1 if qstr_map else 0
    for module in selected_modules:
        root = extract_module_code(data, memory_map, module, loaded.traits)
        nodes = walk_code_tree(root)
        max_qstr = max((node.max_qstr for node in nodes), default=-1)
        max_obj = max((node.max_obj for node in nodes), default=-1)
        if loaded.traits.mpy_format == "v6":
            qstr_ids = read_module_qstr_table(data, memory_map, module, max(max_qstr_id, max_qstr + 1))
            qstr_table = materialize_qstr_table(qstr_map, qstr_ids, max_qstr)
            lookup = {index: value for index, value in enumerate(qstr_table) if value}
            objects = _pad_objects(
                decode_object_table(data, memory_map, module.obj_table_vaddr, max_obj + 1, qstr_lookup=qstr_map),
                max_obj + 1,
            )
        else:
            qstr_table = collect_v5_qstr_table(root, qstr_map)
            lookup = qstr_map
            object_chunks: list[DecodedObject] = []
            for node in nodes:
                if node.const_table_vaddr == 0:
                    continue
                arg_count = node.prelude.n_pos_args + node.prelude.n_kwonly_args
                object_count = max(node.max_obj - arg_count + 1, 0) if node.max_obj >= arg_count else 0
                if object_count == 0:
                    continue
                node_objects = decode_object_table(
                    data,
                    memory_map,
                    node.const_table_vaddr,
                    object_count,
                    start_index=arg_count,
                    qstr_lookup=qstr_map,
                )
                object_chunks.extend(_pad_objects(node_objects, object_count))
            objects = tuple(object_chunks)
            yield module, root, qstr_table, objects, lookup
            continue
        yield module, root, qstr_table, objects, lookup


def _module_display_name(module_name: str) -> str:
    # Show package names the way Python imports them, not as file paths.
    path = Path(module_name)
    if path.name == "__init__.py":
        parent = path.parent.as_posix()
        return path.stem if parent == "." else parent.replace("/", ".")
    return path.with_suffix("").as_posix().replace("/", ".")


def _module_file_name(module_name: str) -> str:
    # The report only needs the leaf file name here.
    return Path(module_name).name


def _module_offset(memory_map: MemoryMap, vaddr: int) -> str:
    # Print both the address and the backing file offset when the address maps cleanly.
    if vaddr == 0:
        return "0x000000"
    offset = memory_map.vaddr_to_offset(vaddr)
    return hex(offset) if offset is not None else "?"


def _name_offsets(data: bytes, start: int, names: tuple[str, ...]) -> tuple[int, ...]:
    # Recover the starting byte offset for each frozen module name.
    offsets: list[int] = []
    pos = start
    for _ in names:
        offsets.append(pos)
        end = data.find(b"\x00", pos)
        if end < 0:
            break
        pos = end + 1
    return tuple(offsets)


def _code_tree_depth(root: CodeNode) -> int:
    # The depth is the longest chain of nested code objects.
    if not root.children:
        return 0
    return 1 + max(_code_tree_depth(child) for child in root.children)


def _header_lines(loaded: LoadedFirmware) -> list[str]:
    # Every report starts with the same firmware banner.
    lines = banner_lines(
        f"{device_label(loaded.traits.device_family)} firmware",
        loaded.banner.raw_text if loaded.banner is not None else None,
    )
    return lines


def _format_qstr_entries(qstr_table: tuple[str, ...], qstr_ids: tuple[int, ...]) -> list[str]:
    # Pair each table slot with the qstr ID it came from.
    lines: list[str] = []
    for slot, value in enumerate(qstr_table):
        qstr_index = qstr_ids[slot] if slot < len(qstr_ids) else slot
        text = escape_text(value) if value else "<missing>"
        lines.append(f"  - slot={slot} qstr_index={qstr_index}: {text}")
    return lines


def _format_legacy_qstr_entries(qstr_map: dict[int, str]) -> list[str]:
    # v5 reports already carry global qstr IDs, so format them directly.
    return [f"  - qstr_index={qstr_index}: {escape_text(value) if value else '<missing>'}" for qstr_index, value in sorted(qstr_map.items())]


def emit_memmap_report(firmware: Path) -> int:
    # Show the recovered memory map for the selected firmware image.
    loaded = load_workspace_firmware(firmware)
    print(render_memmap(loaded))
    return 0


def emit_find_report(firmware: Path) -> int:
    # Stage 1 finds the frozen module name table and prints the raw offsets.
    loaded = load_workspace_firmware(firmware)
    data = loaded.raw
    table = find_frozen_names(data)
    memory_map = cast(MemoryMap, loaded.memory_map)
    lines = _header_lines(loaded)
    lines.extend(["", "Frozen Module Finder (Stage 1)", ""])
    lines.extend(key_value_lines([
        ("device", device_label(loaded.traits.device_family)),
        ("version", loaded.banner.raw_text if loaded.banner is not None else "unknown"),
        ("modules", len(table.names)),
        ("table_offset", hex(table.offset)),
        ("table_range", f"{hex(table.offset)} - {hex(table.end_offset)}"),
    ]))
    lines.append("")
    rows = []
    name_offsets = _name_offsets(data, table.offset, table.names)
    for index, module_name in enumerate(table.names):
        file_name = _module_file_name(module_name)
        rows.append([
            index,
            _module_display_name(module_name),
            file_name,
            hex(name_offsets[index]) if index < len(name_offsets) else hex(table.offset),
            "[package]" if file_name == "__init__.py" else "",
        ])
    lines.extend(table_lines(["#", "Module Path", "File Name", "Offset", "Kind"], rows))
    print("\n".join(lines))
    return 0


def emit_resolve_report(firmware: Path) -> int:
    # Stage 2 turns the content array into concrete module records.
    loaded = load_workspace_firmware(firmware)
    data = loaded.raw
    table = find_frozen_names(data)
    memory_map = cast(MemoryMap, loaded.memory_map)
    content_offset, modules = resolve_modules(data, memory_map, table, loaded.traits)
    lines = _header_lines(loaded)
    lines.extend(["", "MicroPython Frozen Module Resolver (Stage 2)", ""])
    lines.extend(key_value_lines([
        ("mp_frozen_names", hex(table.offset + memory_map.drom_vaddr)),
        ("mp_frozen_mpy_content", f"{hex(content_offset + memory_map.drom_vaddr)} ({len(modules)} entries)"),
        ("modules", len(modules)),
    ]))
    lines.append("")
    rows = []
    for index, module in enumerate(modules):
        # Resolve output keeps both the module struct and the function entry visible.
        rows.append([
            index,
            module.name,
            f"{hex(module.struct_vaddr)} ({_module_offset(memory_map, module.struct_vaddr)})",
            f"{hex(module.qstr_table_vaddr)} ({_module_offset(memory_map, module.qstr_table_vaddr)})" if module.qstr_table_vaddr else "0x00000000 (0x000000)",
            f"{hex(module.obj_table_vaddr)} ({_module_offset(memory_map, module.obj_table_vaddr)})" if module.obj_table_vaddr else "0x00000000 (0x000000)",
            f"{hex(module.proto_vaddr)} ({_module_offset(memory_map, module.proto_vaddr)})",
        ])
    lines.extend(table_lines(["#", "Module", "struct_VA", "qstr_tbl", "obj_tbl", "proto_fun"], rows))
    print("\n".join(lines))
    return 0


def emit_qstr_report(firmware: Path, module_name: str | None) -> int:
    # Stage 3 rebuilds the qstr tables for either one module or all modules.
    loaded, data, memory_map, modules, pools, qstr_map = _load_module_context(firmware)
    lines = _header_lines(loaded)
    lines.extend(["", "MicroPython QStr Pool Recovery (Stage 3)", ""])
    lines.extend(key_value_lines([
        ("device", device_label(loaded.traits.device_family)),
        ("version", loaded.banner.raw_text if loaded.banner is not None else "unknown"),
        ("pool_count", len(pools)),
        ("qstr_count_max", max((pool.total_prev + pool.length for pool in pools), default=0)),
    ]))
    lines.append("")
    pool_rows = []
    for index, pool in enumerate(pools):
        start_id = pool.total_prev
        end_id = pool.total_prev + pool.length - 1 if pool.length else pool.total_prev
        pool_rows.append([
            index,
            f"{hex(pool.vaddr)} ({_module_offset(cast(MemoryMap, loaded.memory_map), pool.vaddr)})",
            pool.length,
            f"{start_id}..{end_id}",
            "yes" if pool.sorted_pool else "no",
        ])
    lines.extend(["--- QStr Pool Chain ---", ""])
    lines.extend(table_lines(["#", "Pool VAddr", "Entries", "IDs", "Sorted"], pool_rows))
    lines.append("")
    max_qstr_id = max((pool.total_prev + pool.length for pool in pools), default=0)
    for index, module in enumerate(select_modules(modules, module_name)):
        if index:
            lines.append("")
        root = extract_module_code(data, memory_map, module, loaded.traits)
        max_qstr = max((node.max_qstr for node in walk_code_tree(root)), default=-1)
        # v5 reports use the tree walk; v6 reports use the module-local table.
        if loaded.traits.mpy_format == "v5":
            qstr_ids = collect_v5_qstr_ids(root)
            qstr_table = tuple(qstr_map.get(qstr_id, "") for qstr_id in qstr_ids)
        else:
            qstr_ids = read_module_qstr_table(data, memory_map, module, max(max_qstr_id, max_qstr + 1))
            qstr_table = materialize_qstr_table(qstr_map, qstr_ids, max_qstr)
        lines.extend(["", f"--- Module QStr Table: {module.name} ---", ""])
        rows = []
        if loaded.traits.mpy_format == "v5":
            for local_index, qstr_id in enumerate(qstr_ids):
                value = qstr_table[local_index] if local_index < len(qstr_table) else ""
                rows.append([local_index, qstr_id, escape_text(value) if value else "<missing>"])
        else:
            for slot, value in enumerate(qstr_table):
                qstr_index = qstr_ids[slot] if slot < len(qstr_ids) else slot
                rows.append([slot, qstr_index, escape_text(value) if value else "<missing>"])
        lines.extend(table_lines(["Idx", "QStr ID", "String"], rows))
    print("\n".join(lines))
    return 0


def emit_bytecode_report(firmware: Path, module_name: str | None) -> int:
    # Stage 4 prints the recovered code tree for each selected module.
    loaded, data, memory_map, modules, _, qstr_map = _load_module_context(firmware)
    artifacts = list(_module_artifacts(loaded, data, memory_map, modules, qstr_map, module_name))
    if not artifacts:
        raise SystemExit("no module was resolved")
    lines = _header_lines(loaded)
    lines.extend(["", "MicroPython Bytecode Extraction (Stage 4)", ""])
    for index, (module, root, _, _, lookup) in enumerate(artifacts):
        if index:
            lines.append("")
        # Print each module separately so the tree stays readable in multi-module images.
        nodes = walk_code_tree(root)
        lines.extend(["", f"Module: {module.name}"])
        lines.extend(key_value_lines([
            ("code_objects", len(nodes)),
            ("bytecode_bytes", sum(len(node.bytecode) for node in nodes)),
            ("max_depth", _code_tree_depth(root)),
        ]))
        lines.append("")
        lines.extend(format_code_tree(root, lookup))
    print("\n".join(lines))
    return 0


def emit_objtable_report(firmware: Path, module_name: str | None) -> int:
    # Stage 5 decodes the recovered object table with the matching version rules.
    loaded, data, memory_map, modules, _, qstr_map = _load_module_context(firmware)
    lines = _header_lines(loaded)
    for index, module in enumerate(select_modules(modules, module_name)):
        if index:
            lines.append("")
        root = extract_module_code(data, memory_map, module, loaded.traits)
        max_obj = max((node.max_obj for node in walk_code_tree(root)), default=-1)
        # v5 objects start after the positional arguments; v6 objects live in a separate table.
        if loaded.traits.mpy_format == "v5":
            arg_count = root.prelude.n_pos_args + root.prelude.n_kwonly_args
            objects = decode_object_table(
                data,
                memory_map,
                module.const_table_vaddr,
                max(max_obj - arg_count + 1, 0),
                start_index=arg_count,
                qstr_lookup=qstr_map,
            )
        else:
            objects = decode_object_table(data, memory_map, module.obj_table_vaddr, max_obj + 1, qstr_lookup=qstr_map)
        lines.extend(["", "MicroPython Object Table Recovery (Stage 5)", "", f"Module: {module.name}"])
        lines.extend(key_value_lines([
            ("object_count", len(objects)),
            ("qstr_lookup", "enabled" if loaded.traits.mpy_format == "v6" else "legacy"),
        ]))
        lines.append("")
        rows = [[index, obj.kind, describe_object(obj)] for index, obj in enumerate(objects)]
        lines.extend(table_lines(["Idx", "Kind", "Value"], rows))
    print("\n".join(lines))
    return 0


def emit_reconstruct_report(firmware: Path, output_dir: Path, module_name: str | None) -> int:
    # Stage 6 writes the reconstructed .mpy files back out to disk.
    loaded, data, memory_map, modules, _, qstr_map = _load_module_context(firmware)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    rows = []
    for module, root, qstr_table, objects, _ in _module_artifacts(loaded, data, memory_map, modules, qstr_map, module_name):
        # Write one .mpy file per recovered module, using the matching version writer.
        if loaded.traits.mpy_format == "v5":
            blob = build_mpy_v5(root, objects, qstr_map)
        else:
            blob = build_mpy(root, qstr_table, objects)
        output_path = output_dir / Path(module.name).with_suffix(".mpy")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(blob)
        written += 1
        rows.append([module.name, len(blob), len(qstr_table), len(objects), len(walk_code_tree(root))])
    lines = _header_lines(loaded)
    lines.extend(["", "MicroPython .mpy Reconstruction (Stage 6)", ""])
    lines.extend(key_value_lines([
        ("modules", written),
        ("output_dir", output_dir),
    ]))
    lines.append("")
    lines.extend(table_lines(["Module", "Bytes", "QStrs", "Objects", "Code"], rows))
    print("\n".join(lines))
    return 0


