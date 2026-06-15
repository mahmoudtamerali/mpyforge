from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class FirmwareBanner:
    raw_text: str
    version_text: str
    version: tuple[int, int, int]
    build_date: str


@dataclass
class LoadedFirmware:
    raw: bytes
    memory_map: "MemoryMap | None"
    banner: FirmwareBanner | None
    traits: LayoutTraits
    source_format: str = "raw"
    source_info: tuple[str, ...] = ()


@dataclass
class LayoutTraits:
    label: str
    mpy_format: str
    qstr_layout: str
    has_sorted_pool_flag: bool
    device_family: str = "unknown"


@dataclass
class Segment:
    load_addr: int
    size: int
    file_offset: int


@dataclass
class MemoryMap:
    drom_vaddr: int
    drom_file_offset: int
    drom_size: int
    segments: tuple["Segment", ...] = field(default_factory=tuple)

    @property
    def drom_end_vaddr(self) -> int:
        return self.drom_vaddr + self.drom_size

    @property
    def drom_end_file_offset(self) -> int:
        return self.drom_file_offset + self.drom_size

    def contains_vaddr(self, vaddr: int) -> bool:
        if self.drom_vaddr <= vaddr < self.drom_end_vaddr:
            return True
        return any(segment.load_addr <= vaddr < segment.load_addr + segment.size for segment in self.segments)

    def contains_offset(self, file_offset: int) -> bool:
        if self.drom_file_offset <= file_offset < self.drom_end_file_offset:
            return True
        return any(
            segment.file_offset <= file_offset < segment.file_offset + segment.size
            for segment in self.segments
        )

    def vaddr_to_offset(self, vaddr: int) -> Optional[int]:
        if self.drom_vaddr <= vaddr < self.drom_end_vaddr:
            return self.drom_file_offset + (vaddr - self.drom_vaddr)
        for segment in self.segments:
            if segment.load_addr <= vaddr < segment.load_addr + segment.size:
                return segment.file_offset + (vaddr - segment.load_addr)
        return None

    def offset_to_vaddr(self, file_offset: int) -> Optional[int]:
        if self.drom_file_offset <= file_offset < self.drom_end_file_offset:
            return self.drom_vaddr + (file_offset - self.drom_file_offset)
        for segment in self.segments:
            if segment.file_offset <= file_offset < segment.file_offset + segment.size:
                return segment.load_addr + (file_offset - segment.file_offset)
        return None


@dataclass
class NameTable:
    offset: int
    names: tuple[str, ...]

    @property
    def end_offset(self) -> int:
        size = sum(len(name.encode("ascii")) + 1 for name in self.names) + 1
        return self.offset + size


@dataclass
class ModuleRecord:
    name: str
    content_vaddr: int
    struct_vaddr: int
    proto_vaddr: int
    qstr_table_vaddr: int = 0
    obj_table_vaddr: int = 0
    const_table_vaddr: int = 0


@dataclass
class QstrPool:
    vaddr: int
    prev_vaddr: int
    total_prev: int
    length: int
    sorted_pool: bool
    hash_vaddr: int = 0
    length_vaddr: int = 0
    qstr_table_vaddr: int = 0


@dataclass
class PreludeInfo:
    opcode_offset: int
    simple_name: int
    source_file: int
    arg_names: tuple[int, ...]
    n_state: int
    n_exc_stack: int
    n_pos_args: int
    n_kwonly_args: int
    n_def_pos_args: int
    scope_flags: int
    n_info: int = 0
    n_cells: int = 0
    sig_size: int = 0
    size_size: int = 0


@dataclass
class CodeNode:
    source: str
    fun_data_vaddr: int
    raw_struct_vaddr: int
    bytecode: bytes
    prelude: PreludeInfo
    const_table_vaddr: int = 0
    qstr_ids: tuple[int, ...] = field(default_factory=tuple)
    qstr_args: set[int] = field(default_factory=set)
    obj_words: tuple[int, ...] = field(default_factory=tuple)
    obj_indexes: set[int] = field(default_factory=set)
    children: list["CodeNode"] = field(default_factory=list)
    kind: str = "bytecode"

    @property
    def max_qstr(self) -> int:
        values = list(self.qstr_ids) if self.qstr_ids else list(self.qstr_args)
        values.append(self.prelude.simple_name)
        values.append(self.prelude.source_file)
        values.extend(self.prelude.arg_names)
        return max(values, default=-1)

    @property
    def max_obj(self) -> int:
        return max(self.obj_indexes, default=-1)


@dataclass
class DecodedObject:
    kind: str
    value: object


@dataclass
class RecoveryLog:
    sections: list[tuple[str, list[str]]] = field(default_factory=list)

    def add(self, title: str, *lines: str) -> None:
        self.sections.append((title, [line for line in lines if line]))

    def extend(self, title: str, lines: Iterable[str]) -> None:
        self.sections.append((title, [line for line in lines if line]))

    def render(self) -> str:
        chunks: list[str] = []
        for title, lines in self.sections:
            chunks.append(f"[{title}]")
            if lines:
                chunks.extend(lines)
            else:
                chunks.append("(no details)")
            chunks.append("")
        return "\n".join(chunks).rstrip() + "\n"
