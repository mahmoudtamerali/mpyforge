from __future__ import annotations

from pathlib import Path
from typing import cast

from .bytecode import extract_module_code, walk_code_tree
from .firmware import load_workspace_firmware
from .finders import find_frozen_names
from .models import CodeNode, DecodedObject, MemoryMap, RecoveryLog
from .objects import decode_object_table
from .qstrs import build_qstr_mapping, read_module_qstr_table, recover_qstr_pools
from .reconstruct import build_mpy, build_mpy_v5, collect_v5_qstr_table
from .rendering import materialize_qstr_lookup, materialize_qstr_table
from .resolve import resolve_modules

def _flatten_metrics(root: CodeNode) -> tuple[int, int]:
    # Collapse the tree to the largest qstr and object indexes referenced anywhere.
    nodes = walk_code_tree(root)
    max_qstr = max((node.max_qstr for node in nodes), default=-1)
    max_obj = max((node.max_obj for node in nodes), default=-1)
    return max_qstr, max_obj


def _pad_objects(objects: tuple[DecodedObject, ...], target_size: int) -> tuple[DecodedObject, ...]:
    # Keep object indexes stable by filling missing slots with explicit none values.
    if target_size <= len(objects):
        return objects
    padding = tuple(DecodedObject("none", None) for _ in range(target_size - len(objects)))
    return objects + padding


def analyze_firmware(path: str | Path, output_dir: str | Path) -> dict[str, object]:
    # Run the same recovery stages used by the CLI, but keep the intermediate results in one report.
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = RecoveryLog()
    errors: list[str] = []
    total_mpy_bytes = 0
    best_recon: tuple[str, bytes] | None = None
    banner = None
    traits = None
    names = None
    max_qstr_id = 0

    try:
        loaded = load_workspace_firmware(path)
        data = loaded.raw
        memory_map = cast(MemoryMap, loaded.memory_map)
        banner = loaded.banner
        traits = loaded.traits
        log.add(
            "firmware",
            f"device_family={traits.device_family}",
            f"version={(banner.raw_text if banner else 'unknown')}",
            *([f"normalize_shift={hex(getattr(memory_map, 'normalize_shift'))}"] if hasattr(memory_map, 'normalize_shift') else []),
            *([f"app_offset={hex(getattr(memory_map, 'app_offset'))}"] if hasattr(memory_map, 'app_offset') else []),
            *( [f"source_format={loaded.source_format}"] if loaded.source_format != "raw" else [] ),
        )

        names = find_frozen_names(data)
        log.add(
            "find",
            f"names_offset={hex(names.offset)}",
            f"modules_found={len(names.names)}",
            *[f"name={name}" for name in names.names[:8]],
        )

        content_offset, modules = resolve_modules(data, memory_map, names, traits)
        # The report keeps the first few recovered modules so the reader can see what was found.
        log.add(
            "resolve",
            f"content_offset={hex(content_offset)}",
            f"resolved_modules={len(modules)}",
            *[
                f"{module.name}: struct={hex(module.struct_vaddr)} proto={hex(module.proto_vaddr)}"
                for module in modules[:6]
            ],
        )

        pools = recover_qstr_pools(data, memory_map, traits)
        qstr_map = build_qstr_mapping(data, memory_map, pools, traits)
        max_qstr_id = max((pool.total_prev + pool.length for pool in pools), default=0)
        log.add(
            "qstr",
            f"pool_count={len(pools)}",
            *[
                f"pool={hex(pool.vaddr)} total_prev={pool.total_prev} len={pool.length} sorted={pool.sorted_pool}"
                for pool in pools
            ],
            f"mapped_qstr_entries={len(qstr_map)}",
        )

        bytecode_lines: list[str] = []
        obj_lines: list[str] = []
        for module in modules:
            try:
                root = extract_module_code(data, memory_map, module, traits)
                nodes = walk_code_tree(root)
                max_qstr, max_obj = _flatten_metrics(root)
                # v6 keeps qstrs and objects in separate tables; v5 stores them in the raw-code tree.
                if traits.mpy_format == "v6":
                    qstr_ids = read_module_qstr_table(data, memory_map, module, max_qstr_id)
                    qstr_table = materialize_qstr_table(qstr_map, qstr_ids, max_qstr)
                    local_lookup = materialize_qstr_lookup(qstr_map, qstr_ids)
                    objects = _pad_objects(
                        decode_object_table(data, memory_map, module.obj_table_vaddr, max_obj + 1, qstr_lookup=qstr_map),
                        max_obj + 1,
                    )
                else:
                    qstr_table = collect_v5_qstr_table(root, qstr_map)
                    local_lookup = qstr_map
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

                # Rebuild each module with the matching writer so the output stays version-correct.
                if traits.mpy_format == "v5":
                    blob = build_mpy_v5(root, objects, qstr_map)
                else:
                    blob = build_mpy(root, qstr_table, objects)
                total_mpy_bytes += len(blob)
                if best_recon is None or len(blob) > len(best_recon[1]):
                    best_recon = (module.name, blob)

                bytecode_lines.append(
                    f"{module.name}: code_bytes={len(root.bytecode)} qstr_max={max_qstr} obj_max={max_obj} children={len(root.children)}"
                )
                obj_lines.append(
                    f"{module.name}: obj_entries={len(objects)} kinds={[obj.kind for obj in objects[:4]]}"
                )
            except Exception as exc:  # pragma: no cover - exercised via analysis
                errors.append(f"{module.name}: {exc}")
                bytecode_lines.append(f"{module.name}: error={exc}")
                obj_lines.append(f"{module.name}: skipped={exc}")

        log.extend("bytecode", bytecode_lines[:20])
        log.extend("objtable", obj_lines[:20])
        log.add(
            "reconstruct",
            f"total_mpy_bytes={total_mpy_bytes}",
            f"representative_recon={(best_recon[0] if best_recon else 'none')}",
        )
    except Exception as exc:  # pragma: no cover - exercised via analysis
        errors.append(str(exc))
        log.add("fatal", f"error={exc}")

    stem = path.stem
    log_path = output_dir / f"{stem}.txt"
    log_path.write_text(log.render(), encoding="utf-8")
    if best_recon is not None:
        (output_dir / f"{stem}-recon.mpy").write_bytes(best_recon[1])

    status = "ok"
    if errors:
        status = "warning" if names is not None else "error"
    report = {
        "filename": path.name,
        "device_family": traits.device_family if traits is not None else "unknown",
        "version": banner.raw_text if banner is not None else None,
        "status": status,
        "modules_found": len(names.names) if names is not None else 0,
        "qstr_count_max": max_qstr_id,
        "total_mpy_bytes": total_mpy_bytes,
        "errors": errors,
    }
    return report
