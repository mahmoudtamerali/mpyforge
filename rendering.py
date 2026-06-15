from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, cast

from .models import CodeNode, DecodedObject, ModuleRecord


def select_modules(modules: Sequence[ModuleRecord], module_name: str | None) -> tuple[ModuleRecord, ...]:
    if module_name is None:
        return tuple(modules)
    selected = tuple(module for module in modules if module.name == module_name)
    if not selected:
        raise SystemExit(f"module not found: {module_name}")
    return selected


def materialize_qstr_table(qstr_map: dict[int, str], qstr_ids: tuple[int, ...], max_index: int) -> tuple[str, ...]:
    if max_index < 0:
        return ()
    entries: list[str] = []
    for index in range(max_index + 1):
        if index < len(qstr_ids):
            entries.append(qstr_map.get(qstr_ids[index], ""))
        else:
            entries.append("")
    return tuple(entries)


def materialize_qstr_lookup(qstr_map: dict[int, str], qstr_ids: tuple[int, ...]) -> dict[int, str]:
    return {
        qstr_id: qstr_map.get(qstr_id, "")
        for qstr_id in qstr_ids
        if qstr_id in qstr_map
    }


def materialize_v5_qstr_table(qstr_map: dict[int, str], max_index: int) -> tuple[str, ...]:
    return tuple(qstr_map.get(index, "") for index in range(max_index + 1)) if max_index >= 0 else ()


def unresolved_qstr_label(qstr_id: int) -> str:
    return f"<unresolved qstr {qstr_id}>"


def escape_text(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def banner_lines(title: str, subtitle: str | None = None, width: int = 60) -> list[str]:
    lines = [title]
    if subtitle:
        lines.append(subtitle)
    inner_width = max((len(line) for line in lines), default=0)
    width = max(width, inner_width + 4)
    border = "=" * width
    formatted = [border]
    for line in lines:
        formatted.append(line.center(width))
    formatted.append(border)
    return formatted


def key_value_lines(items: Sequence[tuple[str, object]], indent: str = "  ") -> list[str]:
    if not items:
        return []
    width = max(len(label) for label, _ in items)
    return [f"{indent}{label.ljust(width)} : {value}" for label, value in items]


def table_lines(headers: Sequence[str], rows: Sequence[Sequence[object]], indent: str = "  ") -> list[str]:
    if not headers:
        return []
    columns = len(headers)
    widths = [len(str(header)) for header in headers]
    normalized_rows = [tuple(str(cell) for cell in row) for row in rows]
    for row in normalized_rows:
        for index in range(columns):
            cell = row[index] if index < len(row) else ""
            widths[index] = max(widths[index], len(cell))
    header_line = indent + "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    separator = indent + "  ".join("-" * widths[index] for index in range(columns))
    lines = [header_line, separator]
    for row in normalized_rows:
        cells = [row[index] if index < len(row) else "" for index in range(columns)]
        lines.append(indent + "  ".join(cells[index].ljust(widths[index]) for index in range(columns)))
    return lines


def describe_object(obj: DecodedObject) -> str:
    if obj.kind == "tuple":
        items = cast(tuple[object, ...], obj.value)
        return "(" + ", ".join(str(item) for item in items) + ")"
    if obj.kind == "qstr":
        return escape_text(str(obj.value))
    if obj.kind == "none":
        return "none"
    if obj.kind == "bool":
        return "true" if bool(obj.value) else "false"
    if obj.kind in {"ptr", "raw"}:
        return str(obj.value)
    return escape_text(str(obj.value))


def format_code_tree(root: CodeNode, qstr_lookup: dict[int, str]) -> list[str]:
    lines: list[str] = []

    def visit(node: CodeNode, depth: int) -> None:
        indent = "  " * depth
        name = escape_text(qstr_lookup.get(node.prelude.simple_name, "<module>"))
        kind = "module" if depth == 0 else "function"
        lines.append(f"{indent}- {name} [{kind}, {len(node.bytecode)} bytes, {len(node.children)} children]")
        lines.append(
            f"{indent}  prelude: state={node.prelude.n_state} exc={node.prelude.n_exc_stack} pos={node.prelude.n_pos_args} kwonly={node.prelude.n_kwonly_args} def_pos={node.prelude.n_def_pos_args} scope=0x{node.prelude.scope_flags:02x}"
        )
        if node.bytecode:
            preview = " ".join(f"{byte:02x}" for byte in node.bytecode[:32])
            suffix = " ..." if len(node.bytecode) > 32 else ""
            lines.append(f"{indent}  bytes: {preview}{suffix}")
        for child in node.children:
            visit(child, depth + 1)

    visit(root, 0)
    return lines


@dataclass
class TextReport:
    lines: list[str] = field(default_factory=list)

    def banner(self, title: str, subtitle: str | None = None, width: int = 60) -> TextReport:
        self.lines.extend(banner_lines(title, subtitle, width))
        return self

    def section(self, title: str) -> TextReport:
        if self.lines and self.lines[-1] != "":
            self.lines.append("")
        self.lines.append(title)
        return self

    def blank(self) -> TextReport:
        if not self.lines or self.lines[-1] != "":
            self.lines.append("")
        return self

    def text(self, *items: str) -> TextReport:
        self.lines.extend(items)
        return self

    def kv(self, items: Sequence[tuple[str, object]], indent: str = "  ") -> TextReport:
        self.lines.extend(key_value_lines(items, indent))
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]], indent: str = "  ") -> TextReport:
        self.lines.extend(table_lines(headers, rows, indent))
        return self

    def render(self) -> str:
        return "\n".join(self.lines)
