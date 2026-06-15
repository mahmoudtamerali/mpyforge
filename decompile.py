#!/usr/bin/env python3
"""
Stage 7: .mpy → .py Decompilation
===================================

Reads ``.mpy`` files (produced by Stage 6 or any MicroPython compiler)
and reconstructs approximate Python source code by:

1. Parsing the ``.mpy`` v6 binary format (header, qstr table, obj table,
   raw code tree).
2. Decoding the bytecode instruction stream for each code object.
3. Simulating the operand stack to rebuild Python expressions.
4. Recognising control-flow patterns (``if``/``elif``/``else``, ``while``,
   ``for``, ``try``/``except``/``finally``, ``with``) from jump instructions.
5. Emitting formatted Python source with proper indentation.

Limitations (inherent to bytecode decompilation):
- Comments are lost at compile time and cannot be recovered.
- Variable / function names survive only when stored as qstrs.
- Some complex expressions may be wrapped in extra parentheses.
- Decorator syntax is not preserved (translated to explicit calls).
- Walrus operator, match/case (Python 3.10+), and other constructs that
  MicroPython doesn't support are never generated.

Usage:
    python -m mpyforge decompile module.mpy
    python -m mpyforge decompile module.mpy -o module.py
    python -m mpyforge decompile output_dir/        # batch: all .mpy → .py

The module also exposes reusable helpers such as ``decompile_mpy_file`` and
``decompile_mpy_dir`` for programmatic use.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .bytecode import decode_var_uint as _decode_var_uint


def decode_var_uint(data: bytes, offset: int) -> tuple[int, int]:
    value, next_offset = _decode_var_uint(data, offset)
    return value, next_offset - offset


# MicroPython "static" qstr IDs used by .mpy files.
#
# The .mpy format stores some qstr entries as compact static IDs which are
# resolved against this fixed table.
#
# This list is taken from the MicroPython toolchain (same table used by
# `tools/mpy-tool.py`) and is kept small on purpose: it only includes the
# built-in/static qstrs.
STATIC_QSTR_LIST: tuple[str, ...] = (
    "",
    "__dir__",
    "\n",
    " ",
    "*",
    "/",
    "<module>",
    "_",
    "__call__",
    "__class__",
    "__delitem__",
    "__enter__",
    "__exit__",
    "__getattr__",
    "__getitem__",
    "__hash__",
    "__init__",
    "__int__",
    "__iter__",
    "__len__",
    "__main__",
    "__module__",
    "__name__",
    "__new__",
    "__next__",
    "__qualname__",
    "__repr__",
    "__setitem__",
    "__str__",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "EOFError",
    "Ellipsis",
    "Exception",
    "GeneratorExit",
    "ImportError",
    "IndentationError",
    "IndexError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "NameError",
    "NoneType",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "SyntaxError",
    "SystemExit",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "append",
    "args",
    "bool",
    "builtins",
    "bytearray",
    "bytecode",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "clear",
    "close",
    "const",
    "copy",
    "count",
    "dict",
    "dir",
    "divmod",
    "end",
    "endswith",
    "eval",
    "exec",
    "extend",
    "find",
    "format",
    "from_bytes",
    "get",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "id",
    "index",
    "insert",
    "int",
    "isalpha",
    "isdigit",
    "isinstance",
    "islower",
    "isspace",
    "issubclass",
    "isupper",
    "items",
    "iter",
    "join",
    "key",
    "keys",
    "len",
    "list",
    "little",
    "locals",
    "lower",
    "lstrip",
    "main",
    "map",
    "micropython",
    "next",
    "object",
    "open",
    "ord",
    "pop",
    "popitem",
    "pow",
    "print",
    "range",
    "read",
    "readinto",
    "readline",
    "remove",
    "replace",
    "repr",
    "reverse",
    "rfind",
    "rindex",
    "round",
    "rsplit",
    "rstrip",
    "self",
    "send",
    "sep",
    "set",
    "setattr",
    "setdefault",
    "sort",
    "sorted",
    "split",
    "start",
    "startswith",
    "staticmethod",
    "step",
    "stop",
    "str",
    "strip",
    "sum",
    "super",
    "throw",
    "to_bytes",
    "tuple",
    "type",
    "update",
    "upper",
    "utf-8",
    "value",
    "values",
    "write",
    "zip",
)


def _static_qstr_label(static_id: int) -> str:
    return f"<static_qstr_{static_id}>"


# ============================================================================
# .mpy v6 format constants
# ============================================================================

MPY_MAGIC = 0x4D  # 'M'
MPY_VERSION = 6

MP_CODE_BYTECODE = 2

# ============================================================================
# Opcode constants (from py/bc0.h and tools/mpy-tool.py)
# ============================================================================

# Base groups
_BASE_RESERVED = 0x00
_BASE_QSTR_O = 0x10
_BASE_VINT_E = 0x20
_BASE_VINT_O = 0x30
_BASE_JUMP_E = 0x40
_BASE_BYTE_O = 0x50
_BASE_BYTE_E = 0x60

# Multi-byte ranges
_LOAD_CONST_SMALL_INT_MULTI = 0x70
_LOAD_FAST_MULTI = 0xB0
_STORE_FAST_MULTI = 0xC0
_UNARY_OP_MULTI = 0xD0
_BINARY_OP_MULTI = 0xD7

_LOAD_CONST_SMALL_INT_MULTI_NUM = 64
_LOAD_CONST_SMALL_INT_MULTI_EXCESS = 16
_LOAD_FAST_MULTI_NUM = 16
_STORE_FAST_MULTI_NUM = 16
_UNARY_OP_MULTI_NUM = 4
_BINARY_OP_MULTI_NUM = 35

# Named opcodes
OP_LOAD_CONST_FALSE = _BASE_BYTE_O + 0x00          # 0x50
OP_LOAD_CONST_NONE = _BASE_BYTE_O + 0x01           # 0x51
OP_LOAD_CONST_TRUE = _BASE_BYTE_O + 0x02           # 0x52
OP_LOAD_CONST_SMALL_INT = _BASE_VINT_E + 0x02      # 0x22
OP_LOAD_CONST_STRING = _BASE_QSTR_O + 0x00         # 0x10
OP_LOAD_CONST_OBJ = _BASE_VINT_E + 0x03            # 0x23
OP_LOAD_NULL = _BASE_BYTE_O + 0x03                  # 0x53

OP_LOAD_FAST_N = _BASE_VINT_E + 0x04               # 0x24
OP_LOAD_DEREF = _BASE_VINT_E + 0x05                # 0x25
OP_LOAD_NAME = _BASE_QSTR_O + 0x01                 # 0x11
OP_LOAD_GLOBAL = _BASE_QSTR_O + 0x02               # 0x12
OP_LOAD_ATTR = _BASE_QSTR_O + 0x03                 # 0x13
OP_LOAD_METHOD = _BASE_QSTR_O + 0x04               # 0x14
OP_LOAD_SUPER_METHOD = _BASE_QSTR_O + 0x05         # 0x15
OP_LOAD_BUILD_CLASS = _BASE_BYTE_O + 0x04           # 0x54
OP_LOAD_SUBSCR = _BASE_BYTE_O + 0x05               # 0x55

OP_STORE_FAST_N = _BASE_VINT_E + 0x06              # 0x26
OP_STORE_DEREF = _BASE_VINT_E + 0x07               # 0x27
OP_STORE_NAME = _BASE_QSTR_O + 0x06                # 0x16
OP_STORE_GLOBAL = _BASE_QSTR_O + 0x07              # 0x17
OP_STORE_ATTR = _BASE_QSTR_O + 0x08                # 0x18
OP_STORE_SUBSCR = _BASE_BYTE_O + 0x06              # 0x56

OP_DELETE_FAST = _BASE_VINT_E + 0x08               # 0x28
OP_DELETE_DEREF = _BASE_VINT_E + 0x09              # 0x29
OP_DELETE_NAME = _BASE_QSTR_O + 0x09               # 0x19
OP_DELETE_GLOBAL = _BASE_QSTR_O + 0x0A             # 0x1A

OP_DUP_TOP = _BASE_BYTE_O + 0x07                   # 0x57
OP_DUP_TOP_TWO = _BASE_BYTE_O + 0x08               # 0x58
OP_POP_TOP = _BASE_BYTE_O + 0x09                   # 0x59
OP_ROT_TWO = _BASE_BYTE_O + 0x0A                   # 0x5A
OP_ROT_THREE = _BASE_BYTE_O + 0x0B                 # 0x5B

OP_UNWIND_JUMP = _BASE_JUMP_E + 0x00               # 0x40
OP_JUMP = _BASE_JUMP_E + 0x02                      # 0x42
OP_POP_JUMP_IF_TRUE = _BASE_JUMP_E + 0x03          # 0x43
OP_POP_JUMP_IF_FALSE = _BASE_JUMP_E + 0x04         # 0x44
OP_JUMP_IF_TRUE_OR_POP = _BASE_JUMP_E + 0x05       # 0x45
OP_JUMP_IF_FALSE_OR_POP = _BASE_JUMP_E + 0x06      # 0x46
OP_SETUP_WITH = _BASE_JUMP_E + 0x07                # 0x47
OP_SETUP_EXCEPT = _BASE_JUMP_E + 0x08              # 0x48
OP_SETUP_FINALLY = _BASE_JUMP_E + 0x09             # 0x49
OP_POP_EXCEPT_JUMP = _BASE_JUMP_E + 0x0A           # 0x4A
OP_FOR_ITER = _BASE_JUMP_E + 0x0B                  # 0x4B
OP_WITH_CLEANUP = _BASE_BYTE_O + 0x0C              # 0x5C
OP_END_FINALLY = _BASE_BYTE_O + 0x0D               # 0x5D
OP_GET_ITER = _BASE_BYTE_O + 0x0E                  # 0x5E
OP_GET_ITER_STACK = _BASE_BYTE_O + 0x0F            # 0x5F

OP_BUILD_TUPLE = _BASE_VINT_E + 0x0A               # 0x2A
OP_BUILD_LIST = _BASE_VINT_E + 0x0B                # 0x2B
OP_BUILD_MAP = _BASE_VINT_E + 0x0C                 # 0x2C
OP_STORE_MAP = _BASE_BYTE_E + 0x02                 # 0x62
OP_BUILD_SET = _BASE_VINT_E + 0x0D                 # 0x2D
OP_BUILD_SLICE = _BASE_VINT_E + 0x0E               # 0x2E
OP_STORE_COMP = _BASE_VINT_E + 0x0F                # 0x2F
OP_UNPACK_SEQUENCE = _BASE_VINT_O + 0x00           # 0x30
OP_UNPACK_EX = _BASE_VINT_O + 0x01                 # 0x31

OP_RETURN_VALUE = _BASE_BYTE_E + 0x03              # 0x63
OP_RAISE_LAST = _BASE_BYTE_E + 0x04                # 0x64
OP_RAISE_OBJ = _BASE_BYTE_E + 0x05                 # 0x65
OP_RAISE_FROM = _BASE_BYTE_E + 0x06                # 0x66
OP_YIELD_VALUE = _BASE_BYTE_E + 0x07               # 0x67
OP_YIELD_FROM = _BASE_BYTE_E + 0x08                # 0x68

OP_MAKE_FUNCTION = _BASE_VINT_O + 0x02             # 0x32
OP_MAKE_FUNCTION_DEFARGS = _BASE_VINT_O + 0x03     # 0x33
OP_MAKE_CLOSURE = _BASE_VINT_E + 0x00              # 0x20
OP_MAKE_CLOSURE_DEFARGS = _BASE_VINT_E + 0x01      # 0x21
OP_CALL_FUNCTION = _BASE_VINT_O + 0x04             # 0x34
OP_CALL_FUNCTION_VAR_KW = _BASE_VINT_O + 0x05      # 0x35
OP_CALL_METHOD = _BASE_VINT_O + 0x06               # 0x36
OP_CALL_METHOD_VAR_KW = _BASE_VINT_O + 0x07        # 0x37

OP_IMPORT_NAME = _BASE_QSTR_O + 0x0B               # 0x1B
OP_IMPORT_FROM = _BASE_QSTR_O + 0x0C               # 0x1C
OP_IMPORT_STAR = _BASE_BYTE_E + 0x09               # 0x69

# Scope flags
MP_SCOPE_FLAG_GENERATOR = 0x01
MP_SCOPE_FLAG_VARKEYWORDS = 0x02
MP_SCOPE_FLAG_VARARGS = 0x04
MP_SCOPE_FLAG_DEFKWARGS = 0x08

# Instruction format codes
FORMAT_BYTE = 0
FORMAT_QSTR = 1
FORMAT_VAR_UINT = 2
FORMAT_OFFSET = 3

_OPCODE_FORMAT_TABLE = 0x000003A4
_EXTRA_BYTE_MASK = 0x9E
_SIGNED_OFFSET_OPCODES = frozenset({0x40, 0x42, 0x43, 0x44})

# Binary op symbols (indexed by MP_BINARY_OP_*)
_BINARY_OP_SYMBOLS = [
    "<",       # 0  LESS
    ">",       # 1  MORE
    "==",      # 2  EQUAL
    "<=",      # 3  LESS_EQUAL
    ">=",      # 4  MORE_EQUAL
    "!=",      # 5  NOT_EQUAL
    "in",      # 6  IN
    "is",      # 7  IS
    None,      # 8  EXCEPTION_MATCH (not emitted as Python operator)
    "|=",      # 9  INPLACE_OR
    "^=",      # 10 INPLACE_XOR
    "&=",      # 11 INPLACE_AND
    "<<=",     # 12 INPLACE_LSHIFT
    ">>=",     # 13 INPLACE_RSHIFT
    "+=",      # 14 INPLACE_ADD
    "-=",      # 15 INPLACE_SUBTRACT
    "*=",      # 16 INPLACE_MULTIPLY
    "@=",      # 17 INPLACE_MAT_MULTIPLY
    "//=",     # 18 INPLACE_FLOOR_DIVIDE
    "/=",      # 19 INPLACE_TRUE_DIVIDE
    "%=",      # 20 INPLACE_MODULO
    "**=",     # 21 INPLACE_POWER
    "|",       # 22 OR
    "^",       # 23 XOR
    "&",       # 24 AND
    "<<",      # 25 LSHIFT
    ">>",      # 26 RSHIFT
    "+",       # 27 ADD
    "-",       # 28 SUBTRACT
    "*",       # 29 MULTIPLY
    "@",       # 30 MAT_MULTIPLY
    "//",      # 31 FLOOR_DIVIDE
    "/",       # 32 TRUE_DIVIDE
    "%",       # 33 MODULO
    "**",      # 34 POWER
]

# Inplace binary ops (indices 9-21) map to augmented assignment
_INPLACE_OPS = set(range(9, 22))

# Normal binary ops that produce expressions (indices 22-34)
_NORMAL_BINARY_OPS = set(range(22, 35))

# Comparison ops (indices 0-7)
_COMPARISON_OPS = set(range(0, 8))

# Python operator precedence for binary ops (higher number = tighter binding).
# Matches Python language reference precedence levels.
_BINARY_OP_PRECEDENCE = {
    22: 4,   # |   OR
    23: 5,   # ^   XOR
    24: 6,   # &   AND
    25: 7,   # <<  LSHIFT
    26: 7,   # >>  RSHIFT
    27: 8,   # +   ADD
    28: 8,   # -   SUBTRACT
    29: 9,   # *   MULTIPLY
    30: 9,   # @   MAT_MULTIPLY
    31: 9,   # //  FLOOR_DIVIDE
    32: 9,   # /   TRUE_DIVIDE
    33: 9,   # %   MODULO
    34: 10,  # **  POWER
}


def _paren_if_needed(expr: str, parent_prec: int, is_rhs: bool = False) -> str:
    """Wrap *expr* in parentheses if it was produced by a lower-precedence op.

    We tag binary-op results with a precedence marker ``\\x00P<n>\\x00`` at the
    end so we can detect them cheaply without parsing.  Only a **suffix** tag
    (at the very end of the string) counts as this expression's own precedence;
    tags embedded inside function calls or subscripts are from sub-expressions
    and must be ignored here.
    """
    if not expr.endswith("\x00"):
        return expr
    tag_start = expr.rindex("\x00P")
    tag_end = expr.index("\x00", tag_start + 2)
    # Only treat as this expression's tag if it ends at the very end
    if tag_end != len(expr) - 1:
        return expr
    child_prec = int(expr[tag_start + 2:tag_end])
    clean = expr[:tag_start]
    if child_prec < parent_prec or (is_rhs and child_prec == parent_prec):
        return f"({clean})"
    return clean


def _tag_prec(expr: str, prec: int) -> str:
    """Append a hidden precedence tag to a binary-op expression."""
    return f"{expr}\x00P{prec}\x00"


def _strip_prec(expr: str) -> str:
    """Remove all precedence tags from an expression (for final output)."""
    while "\x00P" in expr:
        tag_start = expr.index("\x00P")
        tag_end = expr.index("\x00", tag_start + 2)
        expr = expr[:tag_start] + expr[tag_end + 1:]
    return expr

# Unary op symbols (indexed by MP_UNARY_OP_*)
_UNARY_OP_SYMBOLS = [
    "+",       # 0 POSITIVE
    "-",       # 1 NEGATIVE
    "~",       # 2 INVERT
    "not ",    # 3 NOT
]





# ============================================================================
# Instruction decoding
# ============================================================================


@dataclass
class Instruction:
    """A single decoded bytecode instruction."""

    offset: int          # byte offset within fun_data (from opcode start)
    opcode: int          # raw opcode byte
    size: int            # total instruction size in bytes
    arg: Any = None
    has_arg: bool = False
    extra_arg: Optional[int] = None

    @property
    def next_offset(self) -> int:
        return self.offset + self.size


def decode_instruction(bc: bytes, ip: int) -> Instruction:
    """Decode one bytecode instruction at ``bc[ip]``."""
    opcode = bc[ip]
    ip_start = ip
    fmt = (_OPCODE_FORMAT_TABLE >> (2 * (opcode >> 4))) & 3
    ip += 1
    arg = None
    has_arg = False
    extra_arg = None

    if fmt in (FORMAT_QSTR, FORMAT_VAR_UINT):
        has_arg = True
        arg = bc[ip] & 0x7F
        if opcode == OP_LOAD_CONST_SMALL_INT and (arg & 0x40):
            arg |= -1 << 7  # sign extend
        while bc[ip] & 0x80:
            ip += 1
            arg = (arg << 7) | (bc[ip] & 0x7F)
        ip += 1
    elif fmt == FORMAT_OFFSET:
        has_arg = True
        if bc[ip] & 0x80 == 0:
            arg = bc[ip]
            ip += 1
            if opcode in _SIGNED_OFFSET_OPCODES:
                arg -= 0x40
        else:
            arg = (bc[ip] & 0x7F) | (bc[ip + 1] << 7)
            ip += 2
            if opcode in _SIGNED_OFFSET_OPCODES:
                arg -= 0x4000

    if opcode & _EXTRA_BYTE_MASK == 0:
        extra_arg = bc[ip]
        ip += 1

    return Instruction(
        offset=ip_start,
        opcode=opcode,
        size=ip - ip_start,
        arg=arg,
        has_arg=has_arg,
        extra_arg=extra_arg,
    )


def decode_all_instructions(bc: bytes, start: int, end: int) -> List[Instruction]:
    """Decode all instructions in bc[start:end]."""
    instructions: List[Instruction] = []
    ip = start
    while ip < end:
        insn = decode_instruction(bc, ip)
        instructions.append(insn)
        ip += insn.size
    return instructions


# ============================================================================
# Prelude decoding
# ============================================================================


@dataclass
class Prelude:
    """Decoded bytecode prelude."""

    n_state: int = 0
    n_exc_stack: int = 0
    scope_flags: int = 0
    n_pos_args: int = 0
    n_kwonly_args: int = 0
    n_def_pos_args: int = 0

    # Qstr indices for simple_name, source_file, arg names
    simple_name_idx: int = 0
    arg_name_indices: List[int] = field(default_factory=list)

    # Byte offsets within fun_data
    offset_source_info: int = 0
    offset_line_info: int = 0
    offset_opcodes: int = 0

    @property
    def is_generator(self) -> bool:
        return bool(self.scope_flags & MP_SCOPE_FLAG_GENERATOR)

    @property
    def has_varargs(self) -> bool:
        return bool(self.scope_flags & MP_SCOPE_FLAG_VARARGS)

    @property
    def has_varkw(self) -> bool:
        return bool(self.scope_flags & MP_SCOPE_FLAG_VARKEYWORDS)


def decode_prelude(bc: bytes, ip: int = 0) -> Prelude:
    """Decode the full prelude from bytecode.

    Matches the encoding from py/bc.h:
      - Signature: ``S:4 E:1 A:2`` per byte (bit 7 = continuation),
        continuation bytes: ``xFSSKAED``
      - Size: ``xIIIIIIC`` per byte (6 bits I, 1 bit C, bit 7 = continuation)
      - Source info: var-uint qstr indices (simple_name, arg_names)
    """
    p = Prelude()

    # --- Signature (matches mpy-tool.py read_prelude_sig) ---
    z = bc[ip]; ip += 1
    S = (z >> 3) & 0xF
    E = (z >> 2) & 0x1
    A = z & 0x3
    F = 0
    K = 0
    D = 0
    n = 0
    while z & 0x80:
        z = bc[ip]; ip += 1
        # Continuation byte format: xFSSKAED
        S |= (z & 0x30) << (2 * n)
        E |= (z & 0x02) << n
        F |= ((z & 0x40) >> 6) << n
        A |= (z & 0x4) << n
        K |= ((z & 0x08) >> 3) << n
        D |= (z & 0x1) << n
        n += 1
    S += 1

    p.n_state = S
    p.n_exc_stack = E
    p.scope_flags = F
    p.n_pos_args = A
    p.n_kwonly_args = K
    p.n_def_pos_args = D

    # --- Size (matches mpy-tool.py read_prelude_size) ---
    # Each byte: xIIIIIIC (6 bits for n_info, 1 bit for n_cell, bit 7 = continuation)
    n_info = 0
    n_cell = 0
    n = 0
    while True:
        z = bc[ip]; ip += 1
        n_info |= ((z & 0x7E) >> 1) << (6 * n)
        n_cell |= (z & 1) << n
        if not (z & 0x80):
            break
        n += 1

    p.offset_source_info = ip

    # --- Source info: read qstr indices ---
    si_ip = ip
    # simple_name
    val, consumed = decode_var_uint(bc, si_ip)
    p.simple_name_idx = val
    si_ip += consumed

    # arg names (n_pos + n_kwonly, matching mpy-tool.py extract_prelude)
    total_args = p.n_pos_args + p.n_kwonly_args
    p.arg_name_indices = []
    for _ in range(total_args):
        val, consumed = decode_var_uint(bc, si_ip)
        p.arg_name_indices.append(val)
        si_ip += consumed

    p.offset_line_info = si_ip
    p.offset_opcodes = p.offset_source_info + n_info + n_cell

    return p


# ============================================================================
# .mpy file reader
# ============================================================================


@dataclass
class MpyCodeObject:
    """A code object parsed from an .mpy file."""

    fun_data: bytes = b""
    prelude: Optional[Prelude] = None
    children: List["MpyCodeObject"] = field(default_factory=list)

    # Resolved names from prelude + qstr_table
    simple_name: str = ""
    arg_names: List[str] = field(default_factory=list)


@dataclass
class MpyModule:
    """A complete parsed .mpy file."""

    source_file: str = ""
    header: bytes = b""
    small_int_bits: int = 31
    qstr_table: List[str] = field(default_factory=list)
    obj_table: List[Any] = field(default_factory=list)
    root_code: Optional[MpyCodeObject] = None


def read_mpy_file(path: str) -> MpyModule:
    """Parse an .mpy v6 file and return an MpyModule."""
    with open(path, "rb") as f:
        data = f.read()
    return read_mpy_bytes(data)


def read_mpy_bytes(data: bytes) -> MpyModule:
    """Parse .mpy v6 from raw bytes."""
    mod = MpyModule()
    pos = 0

    # Header (4 bytes)
    if len(data) < 4 or data[0] != MPY_MAGIC:
        raise ValueError("not a valid .mpy file")
    if data[1] != MPY_VERSION:
        raise ValueError(f"unsupported .mpy version {data[1]} (expected {MPY_VERSION})")

    mod.header = data[0:4]
    feature_byte = data[2]
    mod.small_int_bits = data[3]
    pos = 4

    native_arch = (feature_byte >> 2) & 0x2F
    # Read arch_flags if present
    if native_arch != 0 and (feature_byte & 0x40):
        _, consumed = decode_var_uint(data, pos)
        pos += consumed

    # n_qstr, n_obj
    n_qstr, consumed = decode_var_uint(data, pos)
    pos += consumed
    n_obj, consumed = decode_var_uint(data, pos)
    pos += consumed

    # qstr_table
    for _ in range(n_qstr):
        val, consumed = decode_var_uint(data, pos)
        pos += consumed
        if val & 1:
            # Static qstr
            static_id = val >> 1
            if 1 <= static_id <= len(STATIC_QSTR_LIST):
                mod.qstr_table.append(STATIC_QSTR_LIST[static_id - 1])
            else:
                mod.qstr_table.append(_static_qstr_label(static_id))
        else:
            # Inline string
            length = val >> 1
            s = data[pos:pos + length].decode("utf-8", errors="replace")
            pos += length
            if pos < len(data) and data[pos] == 0:
                pos += 1  # skip NUL
            mod.qstr_table.append(s)

    # obj_table
    for _ in range(n_obj):
        obj, consumed = _read_persistent_obj(data, pos)
        pos += consumed
        mod.obj_table.append(obj)

    # raw_code tree
    mod.root_code, consumed = _read_raw_code(data, pos, mod.qstr_table)
    pos += consumed

    if mod.qstr_table:
        mod.source_file = mod.qstr_table[0]

    return mod


# Persistent object type tags
_OBJ_FUN_TABLE = 0
_OBJ_NONE = 1
_OBJ_FALSE = 2
_OBJ_TRUE = 3
_OBJ_ELLIPSIS = 4
_OBJ_STR = 5
_OBJ_BYTES = 6
_OBJ_INT = 7
_OBJ_FLOAT = 8
_OBJ_COMPLEX = 9
_OBJ_TUPLE = 10


def _read_persistent_obj(data: bytes, pos: int) -> Tuple[Any, int]:
    """Read one persistent object. Returns (object, bytes_consumed)."""
    start = pos
    tag = data[pos]
    pos += 1

    if tag == _OBJ_FUN_TABLE:
        return "<fun_table>", pos - start
    if tag == _OBJ_NONE:
        return None, pos - start
    if tag == _OBJ_FALSE:
        return False, pos - start
    if tag == _OBJ_TRUE:
        return True, pos - start
    if tag == _OBJ_ELLIPSIS:
        return Ellipsis, pos - start

    if tag in (_OBJ_STR, _OBJ_BYTES):
        length, consumed = decode_var_uint(data, pos)
        pos += consumed
        raw = data[pos:pos + length]
        pos += length
        if pos < len(data) and data[pos] == 0:
            pos += 1  # NUL
        if tag == _OBJ_STR:
            return raw.decode("utf-8", errors="replace"), pos - start
        return raw, pos - start

    if tag in (_OBJ_INT, _OBJ_FLOAT, _OBJ_COMPLEX):
        length, consumed = decode_var_uint(data, pos)
        pos += consumed
        ascii_repr = data[pos:pos + length].decode("ascii")
        pos += length
        if tag == _OBJ_INT:
            return int(ascii_repr), pos - start
        if tag == _OBJ_FLOAT:
            return float(ascii_repr), pos - start
        return complex(ascii_repr), pos - start

    if tag == _OBJ_TUPLE:
        n, consumed = decode_var_uint(data, pos)
        pos += consumed
        items = []
        for _ in range(n):
            item, item_consumed = _read_persistent_obj(data, pos)
            pos += item_consumed
            items.append(item)
        return tuple(items), pos - start

    raise ValueError(f"unknown persistent object tag: {tag}")


def _read_raw_code(
    data: bytes, pos: int, qstr_table: List[str]
) -> Tuple[MpyCodeObject, int]:
    """Read a raw_code tree recursively. Returns (MpyCodeObject, bytes_consumed)."""
    start = pos
    kind_len, consumed = decode_var_uint(data, pos)
    pos += consumed

    kind = (kind_len & 3) + MP_CODE_BYTECODE
    has_children = (kind_len >> 2) & 1
    fun_data_len = kind_len >> 3

    co = MpyCodeObject()
    co.fun_data = data[pos:pos + fun_data_len]
    pos += fun_data_len

    if kind == MP_CODE_BYTECODE:
        co.prelude = decode_prelude(co.fun_data)
        # Resolve names from qstr table
        if co.prelude.simple_name_idx < len(qstr_table):
            co.simple_name = qstr_table[co.prelude.simple_name_idx]
        co.arg_names = []
        for idx in co.prelude.arg_name_indices:
            if idx < len(qstr_table):
                co.arg_names.append(qstr_table[idx])
            else:
                co.arg_names.append(f"_arg{idx}")
    else:
        # Skip native code extra data
        # (simplified — native code decompilation not supported)
        pass

    if has_children:
        n_children, consumed = decode_var_uint(data, pos)
        pos += consumed
        for _ in range(n_children):
            child, consumed = _read_raw_code(data, pos, qstr_table)
            pos += consumed
            co.children.append(child)

    return co, pos - start


# ============================================================================
# Decompiler core
# ============================================================================


def _repr_obj(obj: Any) -> str:
    """Produce a Python-source representation of a constant object."""
    if obj is None:
        return "None"
    if obj is True:
        return "True"
    if obj is False:
        return "False"
    if obj is Ellipsis:
        return "..."
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, bytes):
        return repr(obj)
    if isinstance(obj, int):
        return repr(obj)
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, complex):
        return repr(obj)
    if isinstance(obj, tuple):
        if len(obj) == 1:
            return f"({_repr_obj(obj[0])},)"
        inner = ", ".join(_repr_obj(x) for x in obj)
        return f"({inner})"
    return repr(obj)


def _needs_parens(expr: str, context: str = "") -> bool:
    """Heuristic: does *expr* need wrapping in parens for *context*?"""
    # Already parenthesised
    if expr.startswith("(") and expr.endswith(")"):
        return False
    # Simple names / literals never need parens
    if expr.isidentifier() or expr.isdigit():
        return False
    return False  # err on the side of fewer parens


def _wrap(expr: str) -> str:
    """Wrap expression in parens if it contains spaces and isn't already wrapped."""
    if not expr:
        return expr
    # Already wrapped or is a simple token
    if (expr.startswith("(") and expr.endswith(")")) or \
       (expr.startswith("[") and expr.endswith("]")) or \
       (expr.startswith("{") and expr.endswith("}")):
        return expr
    if expr.isidentifier():
        return expr
    # Check if this is a simple attribute chain like a.b.c or a call like f(x)
    depth = 0
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and ch == " ":
            # Has spaces at top level — might need parens in some contexts
            return f"({expr})"
    return expr


class _DecompileError(Exception):
    """Non-fatal error during decompilation of a single code block."""
    pass


class Decompiler:
    """
    Stack-simulation decompiler for MicroPython bytecodes.

    Each public method returns a list of source lines (without trailing
    newlines and without leading indentation — the caller adds indentation).
    """

    def __init__(
        self,
        module: MpyModule,
        verbose: bool = False,
    ):
        self.module = module
        self.qstr_table = module.qstr_table
        self.obj_table = module.obj_table
        self.verbose = verbose
        # Mapping: id(child_code_object) -> list of closed-over variable names
        self._closure_names: Dict[int, List[str]] = {}

        # MicroPython represents `async def` coroutines using generator machinery.
        # We track coroutine-like code objects with a heuristic so we can render
        # `async def` and `await` in output.
        self._async_code_ids: set[int] = set()

    def _is_coroutine_like(self, co: MpyCodeObject) -> bool:
        """Heuristically detect MicroPython coroutine code objects.

        MicroPython bytecode does not provide a dedicated "coroutine" flag in the
        prelude: both generators and coroutines set the generator bit.

        Heuristic used here:
        - generator flag is set
        - contains YIELD_FROM but not YIELD_VALUE
        - contains the await-like sequence GET_ITER, LOAD_CONST_NONE, YIELD_FROM
        """
        if co.prelude is None or not co.fun_data:
            return False
        if not co.prelude.is_generator:
            return False

        bc = co.fun_data
        insns = decode_all_instructions(bc, co.prelude.offset_opcodes, len(bc))
        if not insns:
            return False

        if any(i.opcode == OP_YIELD_VALUE for i in insns):
            return False
        if not any(i.opcode == OP_YIELD_FROM for i in insns):
            return False

        for i in range(len(insns) - 2):
            if (insns[i].opcode == OP_GET_ITER
                    and insns[i + 1].opcode == OP_LOAD_CONST_NONE
                    and insns[i + 2].opcode == OP_YIELD_FROM):
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompile(self) -> str:
        """Decompile the entire module to Python source."""
        if self.module.root_code is None:
            return "# empty module\n"

        lines = self._decompile_code_object(self.module.root_code, is_module=True)
        # Strip any residual precedence tags from the output
        return "\n".join(_strip_prec(ln) for ln in lines) + "\n"

    # ------------------------------------------------------------------
    # Per-code-object decompilation
    # ------------------------------------------------------------------

    def _decompile_code_object(
        self,
        co: MpyCodeObject,
        is_module: bool = False,
    ) -> List[str]:
        """Decompile a single code object into source lines."""
        if co.prelude is None or not co.fun_data:
            return ["# <no bytecode>"]

        bc = co.fun_data
        prelude = co.prelude
        start = prelude.offset_opcodes
        end = len(bc)

        instructions = decode_all_instructions(bc, start, end)
        if not instructions:
            return ["pass"]

        # Mark coroutine-like code objects so _decompile_block can emit `await`.
        if self._is_coroutine_like(co):
            self._async_code_ids.add(id(co))

        try:
            lines = self._decompile_block(
                instructions, 0, len(instructions), co, is_module=is_module,
            )
        except _DecompileError as e:
            lines = [f"# decompilation error: {e}"]
            # Fallback: emit disassembly as comments
            lines.extend(self._disassemble_fallback(instructions, co))

        # Filter out trailing 'return None' at module level and in regular functions
        if is_module:
            lines = self._strip_module_return(lines)
        else:
            # Strip trailing bare 'return' (implicit return None)
            while lines and lines[-1] in ("return", "return None"):
                lines.pop()
            # Strip trailing 'return _local0' in class body functions:
            # Class bodies compiled as closures end with LOAD_FAST(0) + RETURN.
            # Detect: first arg is '*' (closure cell) and last line is 'return _local0'
            if (co.arg_names and co.arg_names[0] == "*"
                    and lines and lines[-1] == "return _local0"):
                lines.pop()

        return lines if lines else ["pass"]

    # ------------------------------------------------------------------
    # Block-level decompilation — handles control flow
    # ------------------------------------------------------------------

    def _decompile_block(
        self,
        insns: List[Instruction],
        start_idx: int,
        end_idx: int,
        co: MpyCodeObject,
        is_module: bool = False,
    ) -> List[str]:
        """
        Decompile a range of instructions producing source lines.

        Uses a stack to accumulate expressions and emits statements
        at appropriate boundaries.
        """
        lines: List[str] = []
        is_async = id(co) in self._async_code_ids
        stack: List[str] = []
        idx = start_idx

        # Build offset → index map (include end_idx+1 so forward JUMPs to block
        # boundary are resolvable, enabling elif chain detection)
        off2idx: Dict[int, int] = {}
        for i in range(start_idx, min(end_idx + 1, len(insns))):
            off2idx[insns[i].offset] = i

        # Pre-scan: detect while True loops (bare backward JUMP at end)
        while_true_starts: Dict[int, int] = {}  # {target_offset: jump_insn_idx}
        for i in range(start_idx, min(end_idx, len(insns))):
            ci = insns[i]
            if ci.opcode == OP_JUMP and ci.arg < 0:
                target = ci.offset + ci.size + ci.arg
                # Only register if target is within our block
                if target in off2idx and off2idx[target] >= start_idx:
                    t_idx = off2idx[target]
                    # Don't register if target is FOR_ITER or GET_ITER_STACK
                    # (those are for-loop back-edges, not while True)
                    if insns[t_idx].opcode not in (OP_FOR_ITER, OP_GET_ITER_STACK):
                        while_true_starts[target] = i

        while idx < end_idx:
            insn = insns[idx]
            op = insn.opcode
            arg = insn.arg if insn.arg is not None else 0

            # Check if this instruction starts a while True loop
            if insn.offset in while_true_starts:
                jump_idx = while_true_starts.pop(insn.offset)
                body_lines = self._decompile_block(
                    insns, idx, jump_idx, co, is_module=is_module
                )
                lines.append("while True:")
                lines.extend(self._indent(body_lines if body_lines else ["pass"]))
                # Skip past the backward JUMP to the instruction after it
                idx = jump_idx + 1
                continue

            # ----- Multi-byte opcodes -----
            if _LOAD_CONST_SMALL_INT_MULTI <= op < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM:
                val = op - _LOAD_CONST_SMALL_INT_MULTI - _LOAD_CONST_SMALL_INT_MULTI_EXCESS
                stack.append(repr(val))
                idx += 1
                continue

            if _LOAD_FAST_MULTI <= op < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
                local_idx = op - _LOAD_FAST_MULTI
                name = self._local_name(co, local_idx)
                stack.append(name)
                idx += 1
                continue

            if _STORE_FAST_MULTI <= op < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                local_idx = op - _STORE_FAST_MULTI
                name = self._local_name(co, local_idx)
                val = stack.pop() if stack else "<?>"
                # Suppress redundant store after augmented assignment (var += x → var = var)
                if val == name and lines and any(
                    lines[-1].startswith(f"{name} ") and "=" in lines[-1][len(name):]
                    for _ in [1]
                ):
                    idx += 1
                    continue
                lines.append(f"{name} = {val}")
                idx += 1
                continue

            if _UNARY_OP_MULTI <= op < _UNARY_OP_MULTI + _UNARY_OP_MULTI_NUM:
                op_idx = op - _UNARY_OP_MULTI
                sym = _UNARY_OP_SYMBOLS[op_idx] if op_idx < len(_UNARY_OP_SYMBOLS) else "<?op>"
                operand = stack.pop() if stack else "<?>"
                if op_idx == 3:  # not
                    stack.append(f"not {_wrap(operand)}")
                else:
                    stack.append(f"{sym}{_wrap(operand)}")
                idx += 1
                continue

            if _BINARY_OP_MULTI <= op < _BINARY_OP_MULTI + _BINARY_OP_MULTI_NUM:
                op_idx = op - _BINARY_OP_MULTI
                sym = _BINARY_OP_SYMBOLS[op_idx] if op_idx < len(_BINARY_OP_SYMBOLS) else "<?op>"
                rhs = stack.pop() if stack else "<?>"
                lhs = stack.pop() if stack else "<?>"

                if op_idx in _INPLACE_OPS:
                    # Augmented assignment: lhs <op>= rhs
                    base_sym = sym  # e.g., "+="
                    lines.append(f"{_strip_prec(lhs)} {base_sym} {_strip_prec(rhs)}")
                    # Check for ROT_THREE + STORE_SUBSCR pattern (subscript augmented assign)
                    # The VM pattern: DUP_TOP_TWO, LOAD_SUBSCR, <val>, BINARY_OP(inplace),
                    #   ROT_THREE, STORE_SUBSCR
                    # We already emitted the augmented assign, so consume the trailing ops
                    next_i = idx + 1
                    if (next_i < end_idx and insns[next_i].opcode == OP_ROT_THREE
                            and next_i + 1 < end_idx
                            and insns[next_i + 1].opcode == OP_STORE_SUBSCR):
                        # Consume ROT_THREE + STORE_SUBSCR + pop the duplicated obj,key
                        if len(stack) >= 2:
                            stack.pop()  # key (from DUP_TOP_TWO)
                            stack.pop()  # obj (from DUP_TOP_TWO)
                        idx += 3  # skip BINARY_OP + ROT_THREE + STORE_SUBSCR
                        continue
                    # Check for ROT_TWO + STORE_ATTR pattern (attribute augmented assign)
                    # The VM pattern: obj, DUP_TOP, LOAD_ATTR, <val>, BINARY_OP(inplace),
                    #   ROT_TWO, STORE_ATTR
                    if (next_i < end_idx and insns[next_i].opcode == OP_ROT_TWO
                            and next_i + 1 < end_idx
                            and insns[next_i + 1].opcode == OP_STORE_ATTR):
                        # Consume ROT_TWO + STORE_ATTR + pop the duplicated obj
                        if stack:
                            stack.pop()  # obj (from DUP_TOP)
                        idx += 3  # skip BINARY_OP + ROT_TWO + STORE_ATTR
                        continue
                    # For simple variable augmented assign, push result for STORE_FAST
                    stack.append(lhs)
                elif op_idx in _COMPARISON_OPS:
                    stack.append(f"{_strip_prec(lhs)} {sym} {_strip_prec(rhs)}")
                else:
                    # Normal binary op — apply precedence-aware parenthesization
                    prec = _BINARY_OP_PRECEDENCE.get(op_idx, 0)
                    clean_lhs = _paren_if_needed(lhs, prec)
                    clean_rhs = _paren_if_needed(rhs, prec, is_rhs=True)
                    stack.append(_tag_prec(f"{clean_lhs} {sym} {clean_rhs}", prec))
                idx += 1
                continue

            # ----- Named opcodes -----

            # LOAD_CONST_*
            if op == OP_LOAD_CONST_FALSE:
                stack.append("False")
            elif op == OP_LOAD_CONST_NONE:
                stack.append("None")
            elif op == OP_LOAD_CONST_TRUE:
                stack.append("True")
            elif op == OP_LOAD_CONST_SMALL_INT:
                stack.append(repr(arg))
            elif op == OP_LOAD_CONST_STRING:
                s = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                stack.append(repr(s))
            elif op == OP_LOAD_CONST_OBJ:
                obj = self.obj_table[arg] if arg < len(self.obj_table) else f"<obj_{arg}>"
                stack.append(_repr_obj(obj))
            elif op == OP_LOAD_NULL:
                stack.append("__null__")

            # LOAD_FAST_N / LOAD_DEREF
            elif op == OP_LOAD_FAST_N:
                stack.append(self._local_name(co, arg))
            elif op == OP_LOAD_DEREF:
                stack.append(self._local_name(co, arg))

            # LOAD_NAME / LOAD_GLOBAL / LOAD_ATTR / LOAD_METHOD
            elif op == OP_LOAD_NAME:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                stack.append(name)
            elif op == OP_LOAD_GLOBAL:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                stack.append(name)
            elif op == OP_LOAD_ATTR:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                obj = stack.pop() if stack else "<?>"
                stack.append(f"{_wrap(obj)}.{name}")
            elif op == OP_LOAD_METHOD:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                obj = stack.pop() if stack else "<?>"
                # LOAD_METHOD pushes the method and the self object
                stack.append(f"{_wrap(obj)}.{name}")
                stack.append("__method_self__")
            elif op == OP_LOAD_SUPER_METHOD:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                obj = stack.pop() if stack else "<?>"
                stack.append(f"super().{name}")
                stack.append("__method_self__")
            elif op == OP_LOAD_BUILD_CLASS:
                stack.append("__build_class__")
            elif op == OP_LOAD_SUBSCR:
                key = stack.pop() if stack else "<?>"
                obj = stack.pop() if stack else "<?>"
                stack.append(f"{_wrap(obj)}[{key}]")

            # STORE_*
            elif op == OP_STORE_FAST_N:
                name = self._local_name(co, arg)
                val = stack.pop() if stack else "<?>"
                # Suppress redundant store after augmented assignment
                if val == name and lines and lines[-1].startswith(f"{name} "):
                    idx += 1
                    continue
                lines.append(f"{name} = {val}")
            elif op == OP_STORE_DEREF:
                name = self._local_name(co, arg)
                val = stack.pop() if stack else "<?>"
                lines.append(f"{name} = {val}")
            elif op == OP_STORE_NAME:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                val = stack.pop() if stack else "<?>"
                # Detect class body: __module__ = __name__ and __qualname__ = "..."
                if name == "__module__" and val == "__name__":
                    idx += 1
                    continue
                if name == "__qualname__":
                    idx += 1
                    continue
                # Suppress func = func after function definition
                if val == name:
                    idx += 1
                    continue
                lines.append(f"{name} = {val}")
            elif op == OP_STORE_GLOBAL:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                val = stack.pop() if stack else "<?>"
                lines.append(f"{name} = {val}")
            elif op == OP_STORE_ATTR:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                obj = stack.pop() if stack else "<?>"
                val = stack.pop() if stack else "<?>"
                lines.append(f"{_wrap(obj)}.{name} = {val}")
            elif op == OP_STORE_SUBSCR:
                key = stack.pop() if stack else "<?>"
                obj = stack.pop() if stack else "<?>"
                val = stack.pop() if stack else "<?>"
                lines.append(f"{_wrap(obj)}[{key}] = {val}")
            elif op == OP_STORE_MAP:
                key = stack.pop() if stack else "<?>"
                val = stack.pop() if stack else "<?>"
                # BUILD_MAP + STORE_MAP pattern — we track dict building
                if stack and stack[-1].startswith("{"):
                    d = stack.pop()
                    if d == "{}":
                        stack.append(f"{{{key}: {val}}}")
                    else:
                        # Append to existing dict literal
                        inner = d[1:-1]
                        stack.append(f"{{{inner}, {key}: {val}}}")
                else:
                    stack.append(f"{{{key}: {val}}}")

            # DELETE_*
            elif op == OP_DELETE_FAST:
                name = self._local_name(co, arg)
                lines.append(f"del {name}")
            elif op == OP_DELETE_DEREF:
                name = self._local_name(co, arg)
                lines.append(f"del {name}")
            elif op == OP_DELETE_NAME:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                lines.append(f"del {name}")
            elif op == OP_DELETE_GLOBAL:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                lines.append(f"del {name}")

            # Stack manipulation
            elif op == OP_DUP_TOP:
                if stack:
                    stack.append(stack[-1])
                else:
                    stack.append("<?>")
            elif op == OP_DUP_TOP_TWO:
                if len(stack) >= 2:
                    stack.append(stack[-2])
                    stack.append(stack[-2])  # original -1 is now at -3
                else:
                    stack.extend(["<?>", "<?>"])
            elif op == OP_POP_TOP:
                val = stack.pop() if stack else None
                if val and val not in ("__null__", "__method_self__", "<?>", "None"):
                    lines.append(val)
            elif op == OP_ROT_TWO:
                if len(stack) >= 2:
                    stack[-1], stack[-2] = stack[-2], stack[-1]
            elif op == OP_ROT_THREE:
                if len(stack) >= 3:
                    stack[-1], stack[-2], stack[-3] = stack[-3], stack[-1], stack[-2]

            # BUILD_*
            elif op == OP_BUILD_TUPLE:
                n = arg
                items = self._pop_n(stack, n)
                if n == 0:
                    stack.append("()")
                elif n == 1:
                    stack.append(f"({items[0]},)")
                else:
                    stack.append(f"({', '.join(items)})")
            elif op == OP_BUILD_LIST:
                n = arg
                items = self._pop_n(stack, n)
                stack.append(f"[{', '.join(items)}]")
            elif op == OP_BUILD_MAP:
                stack.append("{}")
            elif op == OP_BUILD_SET:
                n = arg
                items = self._pop_n(stack, n)
                if n == 0:
                    stack.append("set()")
                else:
                    stack.append(f"{{{', '.join(items)}}}")
            elif op == OP_BUILD_SLICE:
                n = arg
                items = self._pop_n(stack, n)
                stack.append(f"slice({', '.join(items)})")

            # UNPACK_SEQUENCE / UNPACK_EX
            elif op == OP_UNPACK_SEQUENCE:
                n = arg
                seq = stack.pop() if stack else "<?>"
                # Push placeholders in reverse — the VM stores them
                # in subsequent STORE instructions
                for i in range(n):
                    stack.append(f"__unpack_{n - 1 - i}__")
                # We'll handle the assignments when we see STORE_*
                # Better approach: look ahead for stores and emit multi-assign
                result = self._handle_unpack(insns, idx, n, co, seq)
                if result is not None:
                    new_lines, skip = result
                    # Remove the placeholders we just pushed
                    for _ in range(n):
                        if stack:
                            stack.pop()
                    lines.extend(new_lines)
                    idx += skip
                    continue

            elif op == OP_UNPACK_EX:
                n_left = arg & 0xFF
                n_right = (arg >> 8) & 0xFF
                seq = stack.pop() if stack else "<?>"
                total = n_left + 1 + n_right
                for i in range(total):
                    stack.append(f"__unpack_{total - 1 - i}__")
                # Look ahead for stores and emit extended unpack
                result = self._handle_unpack_ex(
                    insns, idx, n_left, n_right, co, seq)
                if result is not None:
                    new_lines, skip = result
                    for _ in range(total):
                        if stack:
                            stack.pop()
                    lines.extend(new_lines)
                    idx += skip
                    continue

            # STORE_COMP (comprehension)
            elif op == OP_STORE_COMP:
                val = stack.pop() if stack else "<?>"
                # This is used internally for list/dict/set comprehensions
                # The value is accumulated by the VM
                lines.append(f"# store_comp {arg}: {val}")

            # RETURN / RAISE / YIELD
            elif op == OP_RETURN_VALUE:
                val = stack.pop() if stack else "None"
                if is_module and val == "None":
                    pass  # Skip implicit return None at module level
                elif not is_module and val == "None":
                    lines.append("return")
                else:
                    lines.append(f"return {val}")
            elif op == OP_RAISE_LAST:
                lines.append("raise")
            elif op == OP_RAISE_OBJ:
                exc = stack.pop() if stack else "<?>"
                lines.append(f"raise {exc}")
            elif op == OP_RAISE_FROM:
                cause = stack.pop() if stack else "<?>"
                exc = stack.pop() if stack else "<?>"
                lines.append(f"raise {exc} from {cause}")
            elif op == OP_YIELD_VALUE:
                val = stack.pop() if stack else "<?>"
                stack.append(f"yield {val}")
            elif op == OP_YIELD_FROM:
                sent_val = stack.pop() if stack else "<?>"
                iterator = stack.pop() if stack else "<?>"
                stack.append(f"yield from {iterator}")

            # IMPORT
            elif op == OP_IMPORT_NAME:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                fromlist = stack.pop() if stack else "None"
                level = stack.pop() if stack else "0"
                import_lines, skip = self._handle_import(
                    insns, idx, name, fromlist, level, co
                )
                lines.extend(import_lines)
                idx += skip
                continue
            elif op == OP_IMPORT_FROM:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<qstr_{arg}>"
                stack.append(name)
            elif op == OP_IMPORT_STAR:
                mod_name = stack.pop() if stack else "<?>"
                lines.append(f"# from {mod_name} import *")

            # GET_ITER / FOR_ITER
            elif op == OP_GET_ITER:
                # In coroutine-like code objects, MicroPython encodes `await expr` as:
                #   <expr>
                #   GET_ITER
                #   LOAD_CONST_NONE
                #   YIELD_FROM
                #   [POP_TOP]    (if used as a statement)
                if (is_async
                        and idx + 2 < end_idx
                        and insns[idx + 1].opcode == OP_LOAD_CONST_NONE
                        and insns[idx + 2].opcode == OP_YIELD_FROM):
                    awaited = stack.pop() if stack else "<?>"

                    # Statement form: await <expr>
                    if (idx + 3 < end_idx and insns[idx + 3].opcode == OP_POP_TOP):
                        lines.append(f"await {awaited}")
                        idx += 4
                        continue

                    # Expression form: push await <expr> back on the stack
                    stack.append(f"await {awaited}")
                    idx += 3
                    continue

                pass  # iter() is implicit in for-loops; leave TOS as-is
            elif op == OP_GET_ITER_STACK:
                pass
            elif op == OP_FOR_ITER:
                # for-loop pattern
                target_off = insn.offset + insn.size + arg
                result = self._handle_for_loop(
                    insns, idx, target_off, co, stack, off2idx
                )
                if result is not None:
                    new_lines, new_idx, stack_update = result
                    lines.extend(new_lines)
                    stack = stack_update
                    idx = new_idx
                    continue
                # Fallback
                stack.append(f"__for_iter__")

            # JUMP / CONDITIONAL JUMPS
            elif op == OP_JUMP:
                target_off = insn.offset + insn.size + arg
                # Forward jump past end of block → break (exits a loop)
                if arg > 0 and target_off not in off2idx:
                    lines.append("break")
                # Forward jump within block: check for optimized range() for-loop
                elif arg > 0 and target_off in off2idx:
                    range_result = self._handle_range_for_loop(
                        insns, idx, target_off, co, stack, off2idx, end_idx, is_module
                    )
                    if range_result is not None:
                        new_lines, new_idx, stack_update = range_result
                        lines.extend(new_lines)
                        stack = stack_update
                        idx = new_idx
                        continue
                    # Check for bottom-tested while loop
                    while_result = self._handle_bottom_while(
                        insns, idx, target_off, co, stack, off2idx, end_idx, is_module
                    )
                    if while_result is not None:
                        new_lines, new_idx, stack_update = while_result
                        lines.extend(new_lines)
                        stack = stack_update
                        idx = new_idx
                        continue
                # Backward jump should be handled by while True pre-scan
                # Forward jumps within block are handled by if/elif/else

            elif op == OP_POP_JUMP_IF_FALSE:
                target_off = insn.offset + insn.size + arg
                cond = stack.pop() if stack else "<?>"
                # Try ternary expression first
                ternary_end = self._try_ternary(
                    insns, idx, target_off, cond, co, stack, off2idx
                )
                if ternary_end is not None:
                    idx = ternary_end
                    continue
                result = self._handle_if(
                    insns, idx, target_off, cond, False, co, off2idx, is_module
                )
                if result is not None:
                    new_lines, new_idx = result
                    lines.extend(new_lines)
                    idx = new_idx
                    continue
                lines.append(f"# pop_jump_if_false {target_off}: {cond}")

            elif op == OP_POP_JUMP_IF_TRUE:
                target_off = insn.offset + insn.size + arg
                cond = stack.pop() if stack else "<?>"
                result = self._handle_if(
                    insns, idx, target_off, cond, True, co, off2idx, is_module
                )
                if result is not None:
                    new_lines, new_idx = result
                    lines.extend(new_lines)
                    idx = new_idx
                    continue
                lines.append(f"# pop_jump_if_true {target_off}: {cond}")

            elif op == OP_JUMP_IF_TRUE_OR_POP:
                target_off = insn.offset + insn.size + arg
                # short-circuit 'or'
                cond = stack[-1] if stack else "<?>"
                # Pattern: TOS stays if true, else pop and continue
                # This translates to `x or y`
                self._handle_short_circuit_or(stack, insns, idx, target_off, off2idx)

            elif op == OP_JUMP_IF_FALSE_OR_POP:
                target_off = insn.offset + insn.size + arg
                # short-circuit 'and'
                self._handle_short_circuit_and(stack, insns, idx, target_off, off2idx)

            elif op == OP_UNWIND_JUMP:
                # In a while loop, UNWIND_JUMP is 'continue' (unwinding exception handlers)
                # It can also be 'break' if it jumps past the loop end
                lines.append("continue")

            # SETUP_WITH / WITH_CLEANUP / SETUP_EXCEPT / SETUP_FINALLY
            elif op == OP_SETUP_WITH:
                target_off = insn.offset + insn.size + arg
                result = self._handle_with(insns, idx, target_off, co, stack, off2idx, is_module)
                if result is not None:
                    new_lines, new_idx, stack_update = result
                    lines.extend(new_lines)
                    stack = stack_update
                    idx = new_idx
                    continue
                lines.append(f"# setup_with target={target_off}")

            elif op == OP_SETUP_EXCEPT:
                target_off = insn.offset + insn.size + arg
                result = self._handle_try_except(
                    insns, idx, target_off, co, stack, off2idx, is_module
                )
                if result is not None:
                    new_lines, new_idx, stack_update = result
                    lines.extend(new_lines)
                    stack = stack_update
                    idx = new_idx
                    continue
                lines.append(f"# setup_except target={target_off}")

            elif op == OP_SETUP_FINALLY:
                target_off = insn.offset + insn.size + arg
                result = self._handle_try_finally(
                    insns, idx, target_off, co, stack, off2idx, is_module
                )
                if result is not None:
                    new_lines, new_idx, stack_update = result
                    lines.extend(new_lines)
                    stack = stack_update
                    idx = new_idx
                    continue
                lines.append(f"# setup_finally target={target_off}")

            elif op == OP_POP_EXCEPT_JUMP:
                pass  # Handled within try/except
            elif op == OP_WITH_CLEANUP:
                pass  # Handled within with
            elif op == OP_END_FINALLY:
                pass  # Handled within try/finally

            # CALL_FUNCTION / CALL_METHOD
            elif op in (OP_CALL_FUNCTION, OP_CALL_FUNCTION_VAR_KW):
                n_pos = arg & 0xFF
                n_kw = (arg >> 8) & 0xFF
                is_var_kw = (op == OP_CALL_FUNCTION_VAR_KW)
                # Inlined comprehension: comp_str(iterable) → substitute
                if (n_pos == 1 and n_kw == 0 and not is_var_kw
                        and len(stack) >= 2
                        and "__comp_iter__" in stack[-2]):
                    iterable_arg = stack.pop()
                    comp_str = stack.pop()
                    stack.append(
                        comp_str.replace("__comp_iter__", iterable_arg))
                else:
                    call_str = self._build_call(stack, n_pos, n_kw, is_var_kw, is_method=False)
                    stack.append(call_str)

            elif op in (OP_CALL_METHOD, OP_CALL_METHOD_VAR_KW):
                n_pos = arg & 0xFF
                n_kw = (arg >> 8) & 0xFF
                is_var_kw = (op == OP_CALL_METHOD_VAR_KW)
                call_str = self._build_call(stack, n_pos, n_kw, is_var_kw, is_method=True)
                stack.append(call_str)

            # MAKE_FUNCTION / MAKE_CLOSURE
            elif op in (OP_MAKE_FUNCTION, OP_MAKE_FUNCTION_DEFARGS):
                child_idx = arg
                has_defargs = (op == OP_MAKE_FUNCTION_DEFARGS)
                result = self._handle_make_function(
                    co, child_idx, has_defargs, stack, lines, is_module
                )
                if result is not None:
                    new_lines, func_name = result
                    lines.extend(new_lines)
                    stack.append(func_name)
                else:
                    stack.append(f"<function_{child_idx}>")

            elif op in (OP_MAKE_CLOSURE, OP_MAKE_CLOSURE_DEFARGS):
                child_idx = arg
                n_closed = insn.extra_arg if insn.extra_arg is not None else 0
                has_defargs = (op == OP_MAKE_CLOSURE_DEFARGS)
                # Pop closure variables from stack and record their names
                closed_names: List[str] = []
                for _ in range(n_closed):
                    closed_names.append(stack.pop() if stack else "<?>")
                closed_names.reverse()  # restore push order
                # Store the mapping so _local_name can resolve '*' slots
                if child_idx < len(co.children):
                    self._closure_names[id(co.children[child_idx])] = closed_names
                result = self._handle_make_function(
                    co, child_idx, has_defargs, stack, lines, is_module
                )
                if result is not None:
                    new_lines, func_name = result
                    lines.extend(new_lines)
                    stack.append(func_name)
                else:
                    stack.append(f"<closure_{child_idx}>")

            else:
                # Unknown opcode — emit as comment
                if self.verbose:
                    lines.append(f"# unknown opcode 0x{op:02X} arg={arg}")

            idx += 1

        return lines

    # ------------------------------------------------------------------
    # Call building
    # ------------------------------------------------------------------

    def _build_call(
        self,
        stack: List[str],
        n_pos: int,
        n_kw: int,
        is_var_kw: bool,
        is_method: bool,
    ) -> str:
        """Build a function/method call expression from stack frames."""
        args_parts: List[str] = []

        # For VAR_KW calls, pop the star_flags bitmap from the stack top
        star_flags = 0
        if is_var_kw:
            flag_str = stack.pop() if stack else "0"
            try:
                star_flags = int(flag_str)
            except (ValueError, TypeError):
                star_flags = 0

        # Pop keyword args (name, value pairs) in reverse
        kw_pairs: List[Tuple[str, str]] = []
        for _ in range(n_kw):
            val = stack.pop() if stack else "<?>"
            name = stack.pop() if stack else "<?>"
            # name is usually a string constant
            if name.startswith("'") and name.endswith("'"):
                name = name[1:-1]
            elif name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            kw_pairs.append((name, val))
        kw_pairs.reverse()

        # Pop positional args
        pos_args = self._pop_n(stack, n_pos)

        # Pop function / method
        if is_method:
            # Remove __method_self__ sentinel
            if stack and stack[-1] == "__method_self__":
                stack.pop()
            func = stack.pop() if stack else "<?>"
        else:
            func = stack.pop() if stack else "<?>"

        # Detect __build_class__ call
        if func == "__build_class__" and len(pos_args) >= 2:
            return self._build_class_call(pos_args, kw_pairs)

        # Detect comprehension call: the function expression contains __comp_iter__
        # Replace __comp_iter__ with the actual iterable argument
        if "__comp_iter__" in func and len(pos_args) >= 1:
            return func.replace("__comp_iter__", pos_args[0])

        # Build argument list
        for i, a in enumerate(pos_args):
            if is_var_kw and a != "__null__":
                # In VAR_KW calls, check if this positional arg is *-expanded
                # The last positional arg(s) are * expanded when star_flags is set
                pos_from_end = n_pos - 1 - i
                if pos_from_end < 1 and (star_flags & 1):
                    # This is the starred arg
                    args_parts.append(f"*{a}")
                    continue
            if a != "__null__":
                args_parts.append(a)

        for name, val in kw_pairs:
            if name in ("__null__", "None") and is_var_kw:
                # NULL key means **kwargs expansion
                args_parts.append(f"**{val}")
            else:
                args_parts.append(f"{name}={val}")

        return f"{func}({', '.join(args_parts)})"

    def _build_class_call(
        self,
        pos_args: List[str],
        kw_pairs: List[Tuple[str, str]],
    ) -> str:
        """
        Handle __build_class__(body_func, name, *bases).

        Returns a placeholder that we'll detect in STORE_NAME to emit
        a proper class definition.
        """
        func_name = pos_args[0] if pos_args else "<?>"
        class_name = pos_args[1] if len(pos_args) > 1 else "<?>"
        bases = pos_args[2:] if len(pos_args) > 2 else []

        # Clean up class name
        if class_name.startswith("'") and class_name.endswith("'"):
            class_name = class_name[1:-1]
        elif class_name.startswith('"') and class_name.endswith('"'):
            class_name = class_name[1:-1]

        # Include keyword arguments (e.g. metaclass=Meta) in the base list
        bases_and_kw: List[str] = list(bases)
        for kn, kv in kw_pairs:
            if kn.startswith("'") and kn.endswith("'"):
                kn = kn[1:-1]
            elif kn.startswith('"') and kn.endswith('"'):
                kn = kn[1:-1]
            bases_and_kw.append(f"{kn}={kv}")

        return f"__class__:{class_name}:{func_name}:{','.join(bases_and_kw)}"

    # ------------------------------------------------------------------
    # Import handling
    # ------------------------------------------------------------------

    def _handle_import(
        self,
        insns: List[Instruction],
        idx: int,
        name: str,
        fromlist: str,
        level: str,
        co: MpyCodeObject,
    ) -> Tuple[List[str], int]:
        """
        Handle IMPORT_NAME and subsequent opcodes.

        Returns (lines, skip_count) where skip_count includes the
        IMPORT_NAME itself.
        """
        lines: List[str] = []
        skip = 1  # IMPORT_NAME consumed

        # Determine import level (relative imports)
        try:
            level_int = int(level)
        except (ValueError, TypeError):
            level_int = 0

        dots = "." * level_int

        # Check what follows
        next_idx = idx + 1
        if next_idx < len(insns):
            next_op = insns[next_idx].opcode

            if next_op == OP_IMPORT_STAR:
                lines.append(f"from {dots}{name} import *")
                skip += 1
                return lines, skip

            # from X import a, b, c pattern:
            # IMPORT_NAME X → IMPORT_FROM a → STORE_NAME a → IMPORT_FROM b → ...→ POP_TOP
            if next_op == OP_IMPORT_FROM:
                imports: List[str] = []
                j = next_idx
                while j < len(insns):
                    if insns[j].opcode == OP_IMPORT_FROM:
                        from_name_idx = insns[j].arg
                        from_name = (
                            self.qstr_table[from_name_idx]
                            if from_name_idx < len(self.qstr_table)
                            else f"<qstr_{from_name_idx}>"
                        )
                        skip += 1
                        j += 1
                        # Expect STORE_NAME or STORE_FAST or STORE_GLOBAL
                        if j < len(insns) and insns[j].opcode in (
                            OP_STORE_NAME, OP_STORE_FAST_N, OP_STORE_GLOBAL,
                        ):
                            store_name_idx = insns[j].arg
                            if insns[j].opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                                store_name = (
                                    self.qstr_table[store_name_idx]
                                    if store_name_idx < len(self.qstr_table)
                                    else from_name
                                )
                            else:
                                store_name = self._local_name(co, store_name_idx)
                            if store_name != from_name:
                                imports.append(f"{from_name} as {store_name}")
                            else:
                                imports.append(from_name)
                            skip += 1
                            j += 1
                        elif j < len(insns) and (
                            _STORE_FAST_MULTI <= insns[j].opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM
                        ):
                            local_idx = insns[j].opcode - _STORE_FAST_MULTI
                            store_name = self._local_name(co, local_idx)
                            if store_name != from_name:
                                imports.append(f"{from_name} as {store_name}")
                            else:
                                imports.append(from_name)
                            skip += 1
                            j += 1
                        else:
                            imports.append(from_name)
                    elif insns[j].opcode == OP_POP_TOP:
                        skip += 1
                        j += 1
                        break
                    else:
                        break

                if imports:
                    lines.append(f"from {dots}{name} import {', '.join(imports)}")
                    return lines, skip

            # Simple import: IMPORT_NAME X → STORE_NAME X
            if next_op in (OP_STORE_NAME, OP_STORE_GLOBAL, OP_STORE_FAST_N):
                if next_op in (OP_STORE_NAME, OP_STORE_GLOBAL):
                    store_idx = insns[next_idx].arg
                    store_name = (
                        self.qstr_table[store_idx]
                        if store_idx < len(self.qstr_table)
                        else name
                    )
                else:
                    store_name = self._local_name(co, insns[next_idx].arg)

                skip += 1
                # import X as Y
                top_name = name.split(".")[0]
                if store_name != top_name:
                    lines.append(f"import {dots}{name} as {store_name}")
                else:
                    lines.append(f"import {dots}{name}")
                return lines, skip

            # STORE_FAST multi
            if _STORE_FAST_MULTI <= next_op < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                local_idx = next_op - _STORE_FAST_MULTI
                store_name = self._local_name(co, local_idx)
                skip += 1
                top_name = name.split(".")[0]
                if store_name != top_name:
                    lines.append(f"import {dots}{name} as {store_name}")
                else:
                    lines.append(f"import {dots}{name}")
                return lines, skip

        # Fallback: simple import
        lines.append(f"import {dots}{name}")
        return lines, skip

    # ------------------------------------------------------------------
    # Control flow: ternary conditional expression
    # ------------------------------------------------------------------

    def _try_ternary(
        self,
        insns: List[Instruction],
        idx: int,
        target_off: int,
        cond: str,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
    ) -> Optional[int]:
        """
        Detect ``true_val if cond else false_val`` and push result onto stack.

        Returns the new instruction index past the ternary, or None.
        """
        false_idx = off2idx.get(target_off)
        if false_idx is None or false_idx < idx + 2:
            return None

        # Instruction before false branch must be JUMP (end of true branch)
        jump_insn = insns[false_idx - 1]
        if jump_insn.opcode != OP_JUMP:
            return None
        # JUMP must be forward (to end of ternary)
        if jump_insn.arg < 0:
            return None

        jump_arg = jump_insn.arg if jump_insn.arg is not None else 0
        end_off = jump_insn.offset + jump_insn.size + jump_arg
        end_idx_ternary = off2idx.get(end_off)
        if end_idx_ternary is None:
            return None

        true_start = idx + 1
        true_end = false_idx - 1  # exclusive (JUMP excluded)
        false_start = false_idx
        false_end = end_idx_ternary  # exclusive

        # Reject branches that contain statements (STORE, POP_TOP, RETURN, flow)
        _STMT_OPS = frozenset((
            OP_POP_TOP, OP_STORE_NAME, OP_STORE_GLOBAL, OP_STORE_FAST_N,
            OP_STORE_ATTR, OP_RETURN_VALUE, OP_FOR_ITER,
            OP_SETUP_EXCEPT, OP_SETUP_FINALLY, OP_SETUP_WITH,
            OP_YIELD_VALUE, OP_YIELD_FROM,
        ))
        for i in range(true_start, true_end):
            o = insns[i].opcode
            if o in _STMT_OPS:
                return None
            if _STORE_FAST_MULTI <= o < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                return None
        for i in range(false_start, false_end):
            o = insns[i].opcode
            if o in _STMT_OPS:
                return None
            if _STORE_FAST_MULTI <= o < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                return None

        # Mini-evaluate each branch
        true_expr = self._eval_expr_branch(insns, true_start, true_end, co)
        if true_expr is None:
            return None
        false_expr = self._eval_expr_branch(insns, false_start, false_end, co)
        if false_expr is None:
            return None

        ternary = (
            f"{_strip_prec(true_expr)} if {_strip_prec(cond)} "
            f"else {_strip_prec(false_expr)}"
        )
        stack.append(ternary)
        return end_idx_ternary

    def _eval_expr_branch(
        self,
        insns: List[Instruction],
        start_idx: int,
        end_idx: int,
        co: MpyCodeObject,
    ) -> Optional[str]:
        """Evaluate a short expression-only branch and return the expression."""
        stk: List[str] = []
        idx = start_idx
        while idx < end_idx:
            insn = insns[idx]
            op = insn.opcode
            arg = insn.arg if insn.arg is not None else 0

            if (_LOAD_CONST_SMALL_INT_MULTI <= op
                    < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM):
                val = op - _LOAD_CONST_SMALL_INT_MULTI - _LOAD_CONST_SMALL_INT_MULTI_EXCESS
                stk.append(repr(val))
            elif _LOAD_FAST_MULTI <= op < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
                stk.append(self._local_name(co, op - _LOAD_FAST_MULTI))
            elif op == OP_LOAD_CONST_SMALL_INT:
                stk.append(repr(arg))
            elif op == OP_LOAD_CONST_OBJ:
                stk.append(_repr_obj(self.obj_table[arg]) if arg < len(self.obj_table) else "<?>")
            elif op == OP_LOAD_CONST_STRING:
                ni = arg
                s = self.qstr_table[ni] if ni < len(self.qstr_table) else f"_s{ni}"
                stk.append(repr(s))
            elif op == OP_LOAD_CONST_NONE:
                stk.append("None")
            elif op == OP_LOAD_CONST_TRUE:
                stk.append("True")
            elif op == OP_LOAD_CONST_FALSE:
                stk.append("False")
            elif op == OP_LOAD_FAST_N:
                stk.append(self._local_name(co, arg))
            elif op == OP_LOAD_DEREF:
                stk.append(self._local_name(co, arg))
            elif op in (OP_LOAD_NAME, OP_LOAD_GLOBAL):
                ni = arg
                stk.append(
                    self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
                )
            elif op == OP_BUILD_LIST:
                n = arg
                items = stk[-n:] if n else []
                if n:
                    del stk[len(stk) - n:]
                stk.append("[" + ", ".join(items) + "]")
            elif op == OP_BUILD_TUPLE:
                n = arg
                items = stk[-n:] if n else []
                if n:
                    del stk[len(stk) - n:]
                if n == 1:
                    stk.append("(" + items[0] + ",)")
                else:
                    stk.append("(" + ", ".join(items) + ")")
            elif op == OP_LOAD_NULL:
                stk.append("__null__")
            elif _BINARY_OP_MULTI <= op < _BINARY_OP_MULTI + _BINARY_OP_MULTI_NUM:
                rhs = stk.pop() if stk else "<?>"
                lhs = stk.pop() if stk else "<?>"
                op_idx = op - _BINARY_OP_MULTI
                op_sym = _BINARY_OP_SYMBOLS[op_idx] if op_idx < len(_BINARY_OP_SYMBOLS) else f"<binop{op_idx}>"
                stk.append(f"{_strip_prec(lhs)} {op_sym} {_strip_prec(rhs)}")
            elif op in (OP_CALL_FUNCTION, OP_CALL_FUNCTION_VAR_KW):
                n_pos = arg & 0xFF
                n_kw = (arg >> 8) & 0xFF
                is_var_kw = (op == OP_CALL_FUNCTION_VAR_KW)
                call_str = self._build_call(
                    stk, n_pos, n_kw, is_var_kw=is_var_kw, is_method=False
                )
                stk.append(call_str)
            elif op in (OP_MAKE_FUNCTION, OP_MAKE_FUNCTION_DEFARGS):
                child_idx = arg
                child = co.children[child_idx] if child_idx < len(co.children) else None
                if child is None:
                    return None
                child_name = child.simple_name or ""
                if child_name in ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"):
                    inner = self._try_inline_comprehension(child, child_name)
                    if inner is None:
                        return None
                    stk.append(inner)
                else:
                    return None
            elif op in (OP_MAKE_CLOSURE, OP_MAKE_CLOSURE_DEFARGS):
                child_idx = arg
                n_closed = insn.extra_arg if insn.extra_arg is not None else 0
                closed_names: List[str] = []
                for _ in range(n_closed):
                    closed_names.append(stk.pop() if stk else "<?>")
                closed_names.reverse()
                child = co.children[child_idx] if child_idx < len(co.children) else None
                if child is None:
                    return None
                self._closure_names[id(child)] = closed_names
                child_name = child.simple_name or ""
                if child_name in ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"):
                    inner = self._try_inline_comprehension(child, child_name)
                    if inner is None:
                        return None
                    stk.append(inner)
                else:
                    return None
            elif op == OP_LOAD_ATTR:
                ni = arg
                attr = self.qstr_table[ni] if ni < len(self.qstr_table) else f"_a{ni}"
                obj = stk.pop() if stk else "<?>"
                stk.append(f"{_strip_prec(obj)}.{attr}")
            else:
                return None  # unsupported opcode
            idx += 1

        if len(stk) != 1:
            return None
        return stk[0]

    # ------------------------------------------------------------------
    # Control flow: if / elif / else
    # ------------------------------------------------------------------

    def _handle_if(
        self,
        insns: List[Instruction],
        idx: int,
        target_off: int,
        cond: str,
        jump_if_true: bool,
        co: MpyCodeObject,
        off2idx: Dict[int, int],
        is_module: bool,
    ) -> Optional[Tuple[List[str], int]]:
        """Handle POP_JUMP_IF_FALSE/TRUE → if/elif/else."""

        body_start_idx = idx + 1
        target_idx = off2idx.get(target_off)
        if target_idx is None:
            return None

        if target_off <= insns[idx].offset:
            return None

        # Detect 'or' short-circuit pattern for POP_JUMP_IF_TRUE:
        # <eval cond1> POP_JUMP_IF_TRUE → body_start
        # <eval cond2> POP_JUMP_IF_FALSE → past_body
        # This means: if cond1 or cond2: <body>
        if jump_if_true:
            # Check if between idx+1 and target_idx there's a POP_JUMP_IF_FALSE
            # whose body starts at target_idx (meaning our target is the if-body)
            for check_i in range(body_start_idx, min(target_idx + 1, len(insns))):
                ci = insns[check_i]
                if ci.opcode == OP_POP_JUMP_IF_FALSE:
                    # This POP_JUMP_IF_FALSE starts the actual if-body at target_idx
                    # Decompile the second condition from between idx+1 and check_i
                    cond2_stack: List[str] = []
                    for c2_idx in range(body_start_idx, check_i):
                        c2_insn = insns[c2_idx]
                        c2_op = c2_insn.opcode
                        # Simplified expression evaluation for condition 2
                        if _LOAD_FAST_MULTI <= c2_op < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
                            cond2_stack.append(self._local_name(co, c2_op - _LOAD_FAST_MULTI))
                        elif c2_op == OP_LOAD_FAST_N:
                            cond2_stack.append(self._local_name(co, c2_insn.arg))
                        elif c2_op == OP_LOAD_GLOBAL:
                            c2_arg = c2_insn.arg if c2_insn.arg is not None else 0
                            cond2_stack.append(self.qstr_table[c2_arg] if c2_arg < len(self.qstr_table) else f"<g{c2_arg}>")
                        elif c2_op == OP_LOAD_CONST_NONE:
                            cond2_stack.append("None")
                        elif _LOAD_CONST_SMALL_INT_MULTI <= c2_op < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM:
                            cond2_stack.append(repr(c2_op - _LOAD_CONST_SMALL_INT_MULTI - _LOAD_CONST_SMALL_INT_MULTI_EXCESS))
                        elif _BINARY_OP_MULTI <= c2_op < _BINARY_OP_MULTI + _BINARY_OP_MULTI_NUM:
                            op_i = c2_op - _BINARY_OP_MULTI
                            sym = _BINARY_OP_SYMBOLS[op_i] if op_i < len(_BINARY_OP_SYMBOLS) else "?"
                            r = cond2_stack.pop() if cond2_stack else "?"
                            l = cond2_stack.pop() if cond2_stack else "?"
                            cond2_stack.append(f"{l} {sym} {r}")
                        elif _UNARY_OP_MULTI <= c2_op < _UNARY_OP_MULTI + _UNARY_OP_MULTI_NUM:
                            op_i = c2_op - _UNARY_OP_MULTI
                            sym = _UNARY_OP_SYMBOLS[op_i] if op_i < len(_UNARY_OP_SYMBOLS) else "?"
                            operand = cond2_stack.pop() if cond2_stack else "?"
                            cond2_stack.append(f"{sym}{operand}" if op_i != 3 else f"not {operand}")
                        elif c2_op == OP_LOAD_CONST_STRING:
                            c2_arg = c2_insn.arg if c2_insn.arg is not None else 0
                            s = self.qstr_table[c2_arg] if c2_arg < len(self.qstr_table) else "?"
                            cond2_stack.append(repr(s))

                    cond2 = cond2_stack[-1] if cond2_stack else "<?>"
                    combined_cond = f"{cond} or {cond2}"

                    # The POP_JUMP_IF_FALSE target is the else/exit
                    pjif_target = ci.offset + ci.size + ci.arg
                    return self._handle_if(
                        insns, check_i, pjif_target, combined_cond, False,
                        co, off2idx, is_module
                    )
                    break

            # Fallback: negate condition for POP_JUMP_IF_TRUE
            if cond.startswith("not "):
                cond = cond[4:]
            else:
                cond = f"not {_wrap(cond)}"

        # Determine where the if-body ends: either at target_idx or at a JUMP
        # that precedes target_idx (indicating an else branch)
        body_end_idx = target_idx

        # Check if the instruction just before target_idx is a JUMP forward
        # (indicating else branch)
        has_else = False
        else_start_idx = target_idx
        else_end_idx = target_idx

        if target_idx > 0:
            prev_insn = insns[target_idx - 1]
            if prev_insn.opcode == OP_JUMP:
                prev_arg = prev_insn.arg if prev_insn.arg is not None else 0
                jump_target = prev_insn.offset + prev_insn.size + prev_arg
                jump_target_idx = off2idx.get(jump_target)
                if jump_target_idx is not None and jump_target_idx > target_idx:
                    has_else = True
                    body_end_idx = target_idx - 1  # exclude the JUMP
                    else_start_idx = target_idx
                    else_end_idx = jump_target_idx

        # Check for while-loop pattern: back-edge jump at end of body
        # The pattern is: POP_JUMP_IF_FALSE exit_target ... JUMP/UNWIND_JUMP back_to_condition
        if body_end_idx > body_start_idx:
            last_body_insn = insns[body_end_idx - 1] if not has_else else insns[body_end_idx]
            # Check for UNWIND_JUMP back to before the conditional
            for check_idx in range(max(body_start_idx, body_end_idx - 3), body_end_idx + 1):
                if check_idx >= len(insns):
                    break
                ci = insns[check_idx]
                if ci.opcode in (OP_JUMP, OP_UNWIND_JUMP):
                    back_target = ci.offset + ci.size + ci.arg
                    # If jump goes backward to before the condition check, it's a while loop
                    # But NOT if the target is a FOR_ITER (that's a for-loop back-edge)
                    if back_target <= insns[idx].offset:
                        # Check in the global instruction list if target is FOR_ITER
                        is_for_back_edge = False
                        for scan_insn in insns:
                            if scan_insn.offset == back_target:
                                if scan_insn.opcode in (OP_FOR_ITER, OP_GET_ITER_STACK):
                                    is_for_back_edge = True
                                break
                        if is_for_back_edge:
                            break  # for-loop back-edge, not while
                        return self._handle_while_loop(
                            insns, idx, body_start_idx, target_idx,
                            cond, co, off2idx, is_module,
                        )

        # Decompile if-body
        body_lines = self._decompile_block(
            insns, body_start_idx, body_end_idx, co, is_module=is_module
        )

        lines: List[str] = []
        lines.append(f"if {cond}:")
        lines.extend(self._indent(body_lines if body_lines else ["pass"]))

        new_idx = else_end_idx if has_else else target_idx

        if has_else:
            # Check if else starts with another POP_JUMP_IF_FALSE (elif)
            if else_start_idx < else_end_idx:
                first_else = insns[else_start_idx]
                if first_else.opcode == OP_POP_JUMP_IF_FALSE:
                    # elif chain — recurse
                    elif_target = first_else.offset + first_else.size + first_else.arg
                    elif_cond = None
                    # We need to decompile the condition expression before the jump
                    # This is tricky — the condition is computed by instructions before
                    # the POP_JUMP_IF_FALSE... but we've already consumed those.
                    # For elif, we just decompile the else block as normal.
                    pass

            else_lines = self._decompile_block(
                insns, else_start_idx, else_end_idx, co, is_module=is_module
            )
            if else_lines:
                # Check if the else block is a single if (elif pattern)
                if len(else_lines) >= 1 and else_lines[0].startswith("if "):
                    lines.append("el" + else_lines[0])
                    rest = else_lines[1:]
                    # Detect trailing code after elif chain → wrap as final else
                    trailing_start = len(rest)
                    for ri, rl in enumerate(rest):
                        if not rl:
                            continue
                        # Lines that are part of elif chain: indented body, elif, else
                        if rl.startswith(("    ", "elif ", "else:")):
                            continue
                        # Bare top-level code = trailing code after elif chain
                        trailing_start = ri
                        break
                    lines.extend(rest[:trailing_start])
                    trailing = rest[trailing_start:]
                    if trailing:
                        lines.append("else:")
                        lines.extend(self._indent(trailing))
                else:
                    lines.append("else:")
                    lines.extend(self._indent(else_lines))

        return lines, new_idx

    # ------------------------------------------------------------------
    # Control flow: while
    # ------------------------------------------------------------------

    def _handle_while_loop(
        self,
        insns: List[Instruction],
        cond_jump_idx: int,
        body_start_idx: int,
        exit_idx: int,
        cond: str,
        co: MpyCodeObject,
        off2idx: Dict[int, int],
        is_module: bool,
    ) -> Optional[Tuple[List[str], int]]:
        """Handle while-loop pattern."""
        # Find the back-edge jump to determine body end
        body_end_idx = exit_idx
        for i in range(body_start_idx, exit_idx):
            ci = insns[i]
            if ci.opcode in (OP_JUMP, OP_UNWIND_JUMP):
                back_target = ci.offset + ci.size + ci.arg
                if back_target <= insns[cond_jump_idx].offset:
                    body_end_idx = i
                    break

        body_lines = self._decompile_block(
            insns, body_start_idx, body_end_idx, co, is_module=is_module
        )

        lines: List[str] = []
        lines.append(f"while {cond}:")
        lines.extend(self._indent(body_lines if body_lines else ["pass"]))

        return lines, exit_idx

    def _handle_bottom_while(
        self,
        insns: List[Instruction],
        jump_idx: int,
        cond_check_off: int,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
        end_idx: int,
        is_module: bool,
    ) -> Optional[Tuple[List[str], int, List[str]]]:
        """Handle bottom-tested while loop.

        MicroPython compiles ``while cond:`` as:

            JUMP →cond_check
            body_start:
              <body>
            cond_check:
              <compute cond>
              POP_JUMP_IF_TRUE →body_start
        """
        cond_idx = off2idx.get(cond_check_off)
        if cond_idx is None:
            return None

        body_start_idx = jump_idx + 1
        if body_start_idx >= cond_idx:
            return None

        # Find POP_JUMP_IF_TRUE/FALSE with backward jump at/after cond_idx
        pjit_idx = None
        jump_if_true = True
        for i in range(cond_idx, min(cond_idx + 10, end_idx)):
            ci = insns[i]
            if ci.opcode == OP_POP_JUMP_IF_TRUE:
                back_target = ci.offset + ci.size + ci.arg
                if back_target <= insns[body_start_idx].offset:
                    pjit_idx = i
                    jump_if_true = True
                    break
            elif ci.opcode == OP_POP_JUMP_IF_FALSE:
                back_target = ci.offset + ci.size + ci.arg
                if back_target <= insns[body_start_idx].offset:
                    pjit_idx = i
                    jump_if_true = False
                    break
        if pjit_idx is None:
            return None

        # Extract condition by evaluating instructions from cond_idx to pjit_idx
        cond_stack: List[str] = []
        for ci in range(cond_idx, pjit_idx):
            c = insns[ci]
            c_op = c.opcode
            if _LOAD_FAST_MULTI <= c_op < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
                cond_stack.append(self._local_name(co, c_op - _LOAD_FAST_MULTI))
            elif c_op == OP_LOAD_FAST_N:
                cond_stack.append(self._local_name(co, c.arg))
            elif c_op in (OP_LOAD_GLOBAL, OP_LOAD_NAME):
                cond_stack.append(
                    self.qstr_table[c.arg]
                    if c.arg < len(self.qstr_table) else f"<g{c.arg}>")
            elif c_op == OP_LOAD_CONST_NONE:
                cond_stack.append("None")
            elif c_op == OP_LOAD_CONST_TRUE:
                cond_stack.append("True")
            elif c_op == OP_LOAD_CONST_FALSE:
                cond_stack.append("False")
            elif (_LOAD_CONST_SMALL_INT_MULTI <= c_op
                  < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM):
                cond_stack.append(repr(
                    c_op - _LOAD_CONST_SMALL_INT_MULTI
                    - _LOAD_CONST_SMALL_INT_MULTI_EXCESS))
            elif c_op == OP_LOAD_CONST_SMALL_INT:
                cond_stack.append(repr(c.arg))
            elif _BINARY_OP_MULTI <= c_op < _BINARY_OP_MULTI + _BINARY_OP_MULTI_NUM:
                op_i = c_op - _BINARY_OP_MULTI
                sym = (_BINARY_OP_SYMBOLS[op_i]
                       if op_i < len(_BINARY_OP_SYMBOLS) else "?")
                r = cond_stack.pop() if cond_stack else "?"
                l = cond_stack.pop() if cond_stack else "?"
                cond_stack.append(f"{l} {sym} {r}")
            elif _UNARY_OP_MULTI <= c_op < _UNARY_OP_MULTI + _UNARY_OP_MULTI_NUM:
                op_i = c_op - _UNARY_OP_MULTI
                sym = (_UNARY_OP_SYMBOLS[op_i]
                       if op_i < len(_UNARY_OP_SYMBOLS) else "?")
                operand = cond_stack.pop() if cond_stack else "?"
                cond_stack.append(
                    f"{sym}{operand}" if op_i != 3 else f"not {operand}")
            elif c_op == OP_LOAD_ATTR:
                name = (self.qstr_table[c.arg]
                        if c.arg < len(self.qstr_table) else f"<a{c.arg}>")
                obj_e = cond_stack.pop() if cond_stack else "?"
                cond_stack.append(f"{_wrap(obj_e)}.{name}")

        cond_expr = cond_stack[-1] if cond_stack else "<?>"
        # POP_JUMP_IF_TRUE jumps back when true → while <cond>:
        # POP_JUMP_IF_FALSE jumps back when false → while not <cond>:
        if not jump_if_true:
            if cond_expr.startswith("not "):
                cond_expr = cond_expr[4:]
            else:
                cond_expr = f"not {_wrap(cond_expr)}"

        body_lines = self._decompile_block(
            insns, body_start_idx, cond_idx, co, is_module=is_module
        )

        result_lines: List[str] = []
        result_lines.append(f"while {cond_expr}:")
        result_lines.extend(self._indent(body_lines if body_lines else ["pass"]))

        # Skip past the POP_JUMP_IF_TRUE and any trailing POP_TOPs
        after_idx = pjit_idx + 1
        while (after_idx < end_idx
               and insns[after_idx].opcode == OP_POP_TOP):
            after_idx += 1

        return result_lines, after_idx, []

    # ------------------------------------------------------------------
    # Control flow: for
    # ------------------------------------------------------------------

    def _handle_for_loop(
        self,
        insns: List[Instruction],
        for_iter_idx: int,
        exit_off: int,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
    ) -> Optional[Tuple[List[str], int, List[str]]]:
        """Handle FOR_ITER → for-loop."""
        iterable = stack.pop() if stack else "<?>"

        exit_idx = off2idx.get(exit_off)
        if exit_idx is None:
            return None

        body_start_idx = for_iter_idx + 1

        # First instruction after FOR_ITER should be STORE_* (loop variable)
        if body_start_idx >= len(insns):
            return None

        store_insn = insns[body_start_idx]
        target_name = None
        unpack_count = 0

        if store_insn.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
            name_idx = store_insn.arg
            target_name = (
                self.qstr_table[name_idx]
                if name_idx < len(self.qstr_table)
                else f"<qstr_{name_idx}>"
            )
            body_start_idx += 1
        elif store_insn.opcode == OP_STORE_FAST_N:
            target_name = self._local_name(co, store_insn.arg)
            body_start_idx += 1
        elif _STORE_FAST_MULTI <= store_insn.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
            local_idx = store_insn.opcode - _STORE_FAST_MULTI
            target_name = self._local_name(co, local_idx)
            body_start_idx += 1
        elif store_insn.opcode == OP_UNPACK_SEQUENCE:
            # Tuple unpacking: for a, b in ... or for a, (b, c) in ...
            unpack_count = store_insn.arg
            names = []
            j = body_start_idx + 1
            for _ in range(unpack_count):
                if j >= len(insns):
                    break
                si = insns[j]
                if si.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                    ni = si.arg
                    names.append(
                        self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
                    )
                    j += 1
                elif si.opcode == OP_STORE_FAST_N:
                    names.append(self._local_name(co, si.arg))
                    j += 1
                elif _STORE_FAST_MULTI <= si.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                    names.append(self._local_name(co, si.opcode - _STORE_FAST_MULTI))
                    j += 1
                elif si.opcode == OP_UNPACK_SEQUENCE:
                    # Nested unpack: for a, (b, c) in ...
                    inner_count = si.arg
                    inner_names = []
                    j += 1
                    for _ in range(inner_count):
                        if j >= len(insns):
                            break
                        si2 = insns[j]
                        if si2.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                            ni = si2.arg
                            inner_names.append(
                                self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
                            )
                        elif si2.opcode == OP_STORE_FAST_N:
                            inner_names.append(self._local_name(co, si2.arg))
                        elif _STORE_FAST_MULTI <= si2.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                            inner_names.append(self._local_name(co, si2.opcode - _STORE_FAST_MULTI))
                        else:
                            inner_names.append("_")
                        j += 1
                    names.append(f"({', '.join(inner_names)})")
                else:
                    names.append("_")
                    j += 1
            target_name = ", ".join(names)
            body_start_idx = j
        else:
            target_name = "_"

        # Find back-edge jump to determine body end
        body_end_idx = exit_idx
        for i in range(body_start_idx, exit_idx):
            ci = insns[i]
            if ci.opcode == OP_JUMP:
                jt = ci.offset + ci.size + ci.arg
                if jt <= insns[for_iter_idx].offset:
                    body_end_idx = i
                    break

        body_lines = self._decompile_block(
            insns, body_start_idx, body_end_idx, co
        )

        lines: List[str] = []
        lines.append(f"for {target_name} in {iterable}:")
        lines.extend(self._indent(body_lines if body_lines else ["pass"]))

        return lines, exit_idx, []

    # ------------------------------------------------------------------
    # Control flow: optimized range() for-loop
    # ------------------------------------------------------------------

    def _handle_range_for_loop(
        self,
        insns: List[Instruction],
        jump_idx: int,
        jump_target_off: int,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
        end_idx: int,
        is_module: bool,
    ) -> Optional[Tuple[List[str], int, List[str]]]:
        """
        Detect and handle MicroPython's optimized ``for x in range(N)`` pattern.

        MicroPython compiles ``for x in range(N)`` without GET_ITER/FOR_ITER:

        Pattern A (variable stop):
            LOAD stop; LOAD start(0); JUMP →cond;
            body_start: DUP_TOP; STORE var; ...body...;
            LOAD 1; BINARY_OP(14);  ← increment
            cond: DUP_TOP_TWO; ROT_TWO; BINARY_OP(0); POP_JUMP_IF_TRUE →body_start;
            POP_TOP; POP_TOP  ← cleanup

        Pattern B (constant stop):
            LOAD start(0); JUMP →cond;
            body_start: DUP_TOP; STORE var; ...body...;
            LOAD 1; BINARY_OP(14);  ← increment
            cond: DUP_TOP; LOAD stop_const; BINARY_OP(0); POP_JUMP_IF_TRUE →body_start;
            POP_TOP  ← cleanup
        """
        target_idx = off2idx.get(jump_target_off)
        if target_idx is None:
            return None

        # Check body start (instruction after the JUMP)
        body_start_idx = jump_idx + 1
        if body_start_idx >= end_idx:
            return None

        # Body should start with DUP_TOP + STORE_* (loop variable)
        if insns[body_start_idx].opcode != OP_DUP_TOP:
            return None
        if body_start_idx + 1 >= end_idx:
            return None

        store_insn = insns[body_start_idx + 1]
        loop_var = None
        if store_insn.opcode == OP_STORE_FAST_N:
            loop_var = self._local_name(co, store_insn.arg)
        elif _STORE_FAST_MULTI <= store_insn.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
            loop_var = self._local_name(co, store_insn.opcode - _STORE_FAST_MULTI)
        elif store_insn.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
            ni = store_insn.arg
            loop_var = self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
        if loop_var is None:
            return None

        real_body_start = body_start_idx + 2  # after DUP_TOP + STORE

        # Detect Pattern A vs B by looking at condition check target
        cond_insn = insns[target_idx]
        stop_expr = None
        start_expr = "0"
        end_of_loop_idx = None

        if cond_insn.opcode == OP_DUP_TOP_TWO:
            # Pattern A: DUP_TOP_TWO + ROT_TWO + BINARY_OP(0) + POP_JUMP_IF_TRUE
            if (target_idx + 3 < len(insns)
                    and insns[target_idx + 1].opcode == OP_ROT_TWO
                    and insns[target_idx + 2].opcode == _BINARY_OP_MULTI + 0  # LESS
                    and insns[target_idx + 3].opcode == OP_POP_JUMP_IF_TRUE):
                # Stop is on stack (pushed before counter)
                start_expr = stack.pop() if stack else "0"  # pop start (counter)
                if stack:
                    stop_expr = stack.pop()  # pop stop
                else:
                    stop_expr = "<?>"
                # Find cleanup: POP_TOP + POP_TOP after POP_JUMP_IF_TRUE
                cleanup_idx = target_idx + 4
                # Skip POP_TOPs
                while cleanup_idx < end_idx and insns[cleanup_idx].opcode == OP_POP_TOP:
                    cleanup_idx += 1
                end_of_loop_idx = cleanup_idx
        elif cond_insn.opcode == OP_DUP_TOP:
            # Pattern B: DUP_TOP + LOAD_CONST/LOAD_FAST + BINARY_OP(0) + POP_JUMP_IF_TRUE
            if (target_idx + 3 < len(insns)
                    and insns[target_idx + 2].opcode == _BINARY_OP_MULTI + 0  # LESS
                    and insns[target_idx + 3].opcode == OP_POP_JUMP_IF_TRUE):
                # Stop is loaded inline in the condition
                stop_load = insns[target_idx + 1]
                if _LOAD_CONST_SMALL_INT_MULTI <= stop_load.opcode < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM:
                    stop_expr = repr(stop_load.opcode - _LOAD_CONST_SMALL_INT_MULTI - _LOAD_CONST_SMALL_INT_MULTI_EXCESS)
                elif stop_load.opcode == OP_LOAD_CONST_SMALL_INT:
                    stop_expr = repr(stop_load.arg)
                elif stop_load.opcode == OP_LOAD_FAST_N:
                    stop_expr = self._local_name(co, stop_load.arg)
                elif _LOAD_FAST_MULTI <= stop_load.opcode < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
                    stop_expr = self._local_name(co, stop_load.opcode - _LOAD_FAST_MULTI)
                elif stop_load.opcode in (OP_LOAD_NAME, OP_LOAD_GLOBAL):
                    stop_expr = self.qstr_table[stop_load.arg] if stop_load.arg < len(self.qstr_table) else f"<?>"
                else:
                    return None

                # Pop start (counter) from stack
                start_expr = stack.pop() if stack else "0"  # start value

                cleanup_idx = target_idx + 4
                while cleanup_idx < end_idx and insns[cleanup_idx].opcode == OP_POP_TOP:
                    cleanup_idx += 1
                end_of_loop_idx = cleanup_idx

        if stop_expr is None or end_of_loop_idx is None:
            return None

        # Find the body end: the instruction before the condition check that
        # does the counter increment (LOAD 1 + BINARY_OP(14))
        # The body runs from real_body_start to just before the increment
        body_end_idx = target_idx
        # Walk backward from target_idx to find BINARY_OP(14) preceded by LOAD 1
        for i in range(target_idx - 1, real_body_start - 1, -1):
            ci = insns[i]
            if ci.opcode == _BINARY_OP_MULTI + 14:  # INPLACE_ADD
                body_end_idx = i - 1  # exclude the LOAD_CONST(1) before it
                break

        body_lines = self._decompile_block(
            insns, real_body_start, body_end_idx, co, is_module=is_module
        )

        result_lines: List[str] = []
        stop_str = _strip_prec(stop_expr)
        start_str = _strip_prec(start_expr)
        if start_str != "0":
            result_lines.append(f"for {loop_var} in range({start_str}, {stop_str}):")
        else:
            result_lines.append(f"for {loop_var} in range({stop_str}):")
        result_lines.extend(self._indent(body_lines if body_lines else ["pass"]))

        return result_lines, end_of_loop_idx, []

    # ------------------------------------------------------------------
    # Control flow: with
    # ------------------------------------------------------------------

    def _handle_with(
        self,
        insns: List[Instruction],
        idx: int,
        cleanup_off: int,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
        is_module: bool,
    ) -> Optional[Tuple[List[str], int, List[str]]]:
        """Handle SETUP_WITH → with statement."""
        ctx_mgr = stack.pop() if stack else "<?>"

        cleanup_idx = off2idx.get(cleanup_off)
        if cleanup_idx is None:
            return None

        body_start_idx = idx + 1

        # Check if next instruction is STORE_* (with X as Y)
        as_name = None
        if body_start_idx < len(insns):
            store_insn = insns[body_start_idx]
            if store_insn.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                ni = store_insn.arg
                as_name = (
                    self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
                )
                body_start_idx += 1
            elif store_insn.opcode == OP_STORE_FAST_N:
                as_name = self._local_name(co, store_insn.arg)
                body_start_idx += 1
            elif _STORE_FAST_MULTI <= store_insn.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                as_name = self._local_name(co, store_insn.opcode - _STORE_FAST_MULTI)
                body_start_idx += 1
            elif store_insn.opcode == OP_POP_TOP:
                body_start_idx += 1  # with X: (no as)

        body_end_idx = cleanup_idx

        body_lines = self._decompile_block(
            insns, body_start_idx, body_end_idx, co, is_module=is_module
        )

        lines: List[str] = []
        if as_name:
            lines.append(f"with {ctx_mgr} as {as_name}:")
        else:
            lines.append(f"with {ctx_mgr}:")
        lines.extend(self._indent(body_lines if body_lines else ["pass"]))

        # Skip WITH_CLEANUP + END_FINALLY
        end_idx = cleanup_idx
        while end_idx < len(insns) and insns[end_idx].opcode in (
            OP_WITH_CLEANUP, OP_END_FINALLY, OP_POP_TOP
        ):
            end_idx += 1

        return lines, end_idx, []

    # ------------------------------------------------------------------
    # Control flow: try/except
    # ------------------------------------------------------------------

    def _handle_try_except(
        self,
        insns: List[Instruction],
        idx: int,
        handler_off: int,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
        is_module: bool,
    ) -> Optional[Tuple[List[str], int, List[str]]]:
        """Handle SETUP_EXCEPT → try/except."""
        handler_idx = off2idx.get(handler_off)
        if handler_idx is None:
            return None

        try_body_start = idx + 1
        try_body_end = handler_idx

        # The try body might end with POP_EXCEPT_JUMP
        real_try_end = try_body_end
        for i in range(try_body_start, try_body_end):
            if insns[i].opcode == OP_POP_EXCEPT_JUMP:
                real_try_end = i
                break

        try_lines = self._decompile_block(
            insns, try_body_start, real_try_end, co, is_module=is_module
        )

        lines: List[str] = []
        lines.append("try:")
        lines.extend(self._indent(try_lines if try_lines else ["pass"]))

        # Find END_FINALLY to know where handlers end
        final_end = len(insns)
        for i in range(handler_idx, len(insns)):
            if insns[i].opcode == OP_END_FINALLY:
                final_end = i + 1
                break

        # Parse exception handlers starting at handler_idx
        # Pattern: DUP_TOP + LOAD_GLOBAL/NAME <ExcType> + BINARY_OP(8) + POP_JUMP_IF_FALSE
        #          POP_TOP + <except body> + POP_EXCEPT_JUMP/UNWIND_JUMP
        h_idx = handler_idx
        except_clauses: List[Tuple[Optional[str], List[str]]] = []

        while h_idx < final_end:
            insn_h = insns[h_idx]
            if insn_h.opcode == OP_END_FINALLY:
                h_idx += 1
                break

            # Check for typed except: DUP_TOP + LOAD_* + BINARY_OP(8=EXCEPTION_MATCH)
            if (insn_h.opcode == OP_DUP_TOP
                    and h_idx + 3 < final_end
                    and insns[h_idx + 2].opcode in (
                        _BINARY_OP_MULTI + 8,  # BINARY_OP EXCEPTION_MATCH
                    )):
                # Extract exception type name
                type_insn = insns[h_idx + 1]
                exc_type = None
                if type_insn.opcode in (OP_LOAD_GLOBAL, OP_LOAD_NAME):
                    exc_type = self.qstr_table[type_insn.arg] if type_insn.arg < len(self.qstr_table) else f"<exc_{type_insn.arg}>"

                # POP_JUMP_IF_FALSE follows the BINARY_OP(8)
                jump_insn = insns[h_idx + 3]
                if jump_insn.opcode == OP_POP_JUMP_IF_FALSE:
                    # After POP_JUMP_IF_FALSE: POP_TOP (bare except) or
                    # STORE_FAST/NAME (except ... as e)
                    body_start = h_idx + 4
                    as_name = None
                    if body_start < final_end:
                        next_insn = insns[body_start]
                        if next_insn.opcode == OP_POP_TOP:
                            body_start += 1
                        elif next_insn.opcode == OP_STORE_FAST_N:
                            as_name = self._local_name(co, next_insn.arg)
                            body_start += 1
                        elif (_STORE_FAST_MULTI <= next_insn.opcode
                              < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM):
                            as_name = self._local_name(
                                co, next_insn.opcode - _STORE_FAST_MULTI)
                            body_start += 1
                        elif next_insn.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                            ni = next_insn.arg
                            as_name = (self.qstr_table[ni]
                                       if ni < len(self.qstr_table)
                                       else f"_v{ni}")
                            body_start += 1

                    # For 'as e', skip the SETUP_FINALLY cleanup wrapper
                    # and use the cleanup target as body end boundary
                    cleanup_body_end = None
                    if (as_name and body_start < final_end
                            and insns[body_start].opcode == OP_SETUP_FINALLY):
                        cleanup_off = (insns[body_start].offset
                                       + insns[body_start].size
                                       + insns[body_start].arg)
                        body_start += 1  # skip SETUP_FINALLY
                        # Body end is just before the cleanup block
                        for ci in range(body_start, final_end):
                            if insns[ci].offset >= cleanup_off:
                                cleanup_body_end = ci
                                break

                    # Find except body end: POP_EXCEPT_JUMP or END_FINALLY
                    # Include UNWIND_JUMP in body so it's decompiled as 'continue'
                    if cleanup_body_end is not None:
                        body_end = cleanup_body_end
                    else:
                        body_end = final_end
                        for bi in range(body_start, final_end):
                            if insns[bi].opcode in (OP_POP_EXCEPT_JUMP, OP_END_FINALLY):
                                body_end = bi
                                break
                            if insns[bi].opcode == OP_UNWIND_JUMP:
                                body_end = bi + 1  # include UNWIND_JUMP in body
                                break

                    except_body = self._decompile_block(
                        insns, body_start, body_end, co, is_module=is_module
                    )
                    exc_label = (f"{exc_type} as {as_name}"
                                 if as_name else exc_type)
                    except_clauses.append((exc_label, except_body))

                    # Skip to after the except body + POP_EXCEPT_JUMP/UNWIND_JUMP
                    h_idx = body_end
                    if h_idx < final_end and insns[h_idx].opcode in (
                        OP_POP_EXCEPT_JUMP, OP_UNWIND_JUMP
                    ):
                        h_idx += 1
                    # For 'as e', skip the cleanup block
                    # (LOAD_CONST None, STORE e, DELETE e, END_FINALLY)
                    if as_name:
                        while (h_idx < final_end
                               and insns[h_idx].opcode != OP_END_FINALLY):
                            h_idx += 1
                        if h_idx < final_end:
                            h_idx += 1  # skip END_FINALLY
                    # Check for next handler indicated by POP_JUMP_IF_FALSE target
                    continue
                else:
                    h_idx += 1
                    continue
            else:
                # Bare except (no type check) or unrecognized pattern
                body_end = final_end
                for bi in range(h_idx, final_end):
                    if insns[bi].opcode == OP_END_FINALLY:
                        body_end = bi
                        break
                except_body = self._decompile_block(
                    insns, h_idx, body_end, co, is_module=is_module
                )
                except_body = [l for l in except_body if not l.startswith("# pop_except")]
                except_clauses.append((None, except_body))
                h_idx = body_end
                if h_idx < final_end and insns[h_idx].opcode == OP_END_FINALLY:
                    h_idx += 1
                break

        for exc_type, exc_body in except_clauses:
            if exc_type:
                lines.append(f"except {exc_type}:")
            else:
                lines.append("except:")
            lines.extend(self._indent(exc_body if exc_body else ["pass"]))

        if not except_clauses:
            lines.append("except:")
            lines.extend(self._indent(["pass"]))

        return lines, final_end, []

    # ------------------------------------------------------------------
    # Control flow: try/finally
    # ------------------------------------------------------------------

    def _handle_try_finally(
        self,
        insns: List[Instruction],
        idx: int,
        finally_off: int,
        co: MpyCodeObject,
        stack: List[str],
        off2idx: Dict[int, int],
        is_module: bool,
    ) -> Optional[Tuple[List[str], int, List[str]]]:
        """Handle SETUP_FINALLY → try/finally."""
        finally_idx = off2idx.get(finally_off)
        if finally_idx is None:
            return None

        try_body_start = idx + 1
        try_body_end = finally_idx

        try_lines = self._decompile_block(
            insns, try_body_start, try_body_end, co, is_module=is_module
        )

        # Find END_FINALLY to delimit finally body
        finally_end = len(insns)
        for i in range(finally_idx, len(insns)):
            if insns[i].opcode == OP_END_FINALLY:
                finally_end = i + 1
                break

        finally_lines = self._decompile_block(
            insns, finally_idx, finally_end, co, is_module=is_module
        )

        lines: List[str] = []

        # MicroPython compiles try/except/finally as nested
        # try/finally wrapping try/except.  Detect and merge.
        if (try_lines
                and try_lines[0] == "try:"
                and any(ln.startswith("except") for ln in try_lines)
                and not any(ln.startswith("finally:") for ln in try_lines)):
            lines.extend(try_lines)
        else:
            lines.append("try:")
            lines.extend(self._indent(try_lines if try_lines else ["pass"]))
        lines.append("finally:")
        lines.extend(self._indent(finally_lines if finally_lines else ["pass"]))

        return lines, finally_end, []

    # ------------------------------------------------------------------
    # Function / class definition
    # ------------------------------------------------------------------

    def _handle_make_function(
        self,
        co: MpyCodeObject,
        child_idx: int,
        has_defargs: bool,
        stack: List[str],
        lines: List[str],
        is_module: bool,
    ) -> Optional[Tuple[List[str], str]]:
        """
        Handle MAKE_FUNCTION: decompile the child code object as a
        function definition and return (lines_to_add, func_name).
        """
        if child_idx >= len(co.children):
            return None

        child = co.children[child_idx]
        if child.prelude is None:
            return None

        func_name = child.simple_name or f"_func_{child_idx}"
        is_generator = child.prelude.is_generator
        is_coroutine = self._is_coroutine_like(child)
        if is_coroutine:
            self._async_code_ids.add(id(child))

        # Build argument signature
        sig = self._build_function_signature(child, has_defargs, stack)

        # Decompile the function body
        body_lines = self._decompile_code_object(child)

        # Lambda detection: single-expression function named <lambda>
        if func_name == "<lambda>":
            # Try to extract a simple return expression
            expr = None
            for bl in body_lines:
                stripped = bl.strip()
                if stripped.startswith("return "):
                    expr = stripped[7:]
                    break
            if expr:
                lambda_str = f"lambda {sig}: {expr}" if sig else f"lambda: {expr}"
                return [], lambda_str

        # List/dict/set comprehension detection
        if func_name in ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"):
            comp_result = self._try_inline_comprehension(child, func_name)
            if comp_result is not None:
                return [], comp_result

        new_lines: List[str] = []
        keyword = "async def" if is_coroutine else "def"

        new_lines.append(f"{keyword} {func_name}({sig}):")
        new_lines.extend(self._indent(body_lines if body_lines else ["pass"]))
        new_lines.append("")  # blank line after function definition

        return new_lines, func_name

    def _build_function_signature(
        self,
        co: MpyCodeObject,
        has_defargs: bool,
        stack: List[str],
    ) -> str:
        """Build the function argument signature string."""
        prelude = co.prelude
        if prelude is None:
            return ""

        parts: List[str] = []
        arg_names = co.arg_names
        n_pos = prelude.n_pos_args
        n_kw = prelude.n_kwonly_args
        n_def = prelude.n_def_pos_args

        # Skip hidden closure variables: in MicroPython closures, the first
        # arg(s) named '*' are internal closure cell references, not real params.
        closure_skip = 0
        for name in arg_names:
            if name == "*":
                closure_skip += 1
            else:
                break

        # Pop default values from stack if present
        # Stack layout for MAKE_*_DEFARGS: [def_tuple, def_dict] on top
        # def_dict is TOS, def_tuple is below it
        defaults: List[str] = []
        kw_defaults: List[str] = []
        if has_defargs:
            # Pop def_dict (keyword defaults — usually __null__ for no kw defaults)
            def_dict = stack.pop() if stack else None
            # Pop def_tuple (positional defaults tuple)
            val = stack.pop() if stack else None
            if val and val.startswith("(") and val.endswith(")"):
                inner = val[1:-1]
                if inner:
                    defaults = [x.strip() for x in inner.split(",") if x.strip()]

        arg_idx = closure_skip  # start after closure vars
        real_n_pos = n_pos - closure_skip  # visible positional args

        # Positional args
        for i in range(real_n_pos):
            name = arg_names[arg_idx] if arg_idx < len(arg_names) else f"_arg{i}"
            arg_idx += 1
            # Check if this arg has a default
            pos_from_end = real_n_pos - 1 - i
            def_idx = len(defaults) - 1 - pos_from_end
            if def_idx >= 0 and def_idx < len(defaults):
                parts.append(f"{name}={defaults[def_idx]}")
            else:
                parts.append(name)

        # *args
        if prelude.has_varargs:
            name = arg_names[arg_idx] if arg_idx < len(arg_names) else "args"
            arg_idx += 1
            parts.append(f"*{name}")
        elif n_kw > 0:
            parts.append("*")

        # keyword-only args
        for i in range(n_kw):
            name = arg_names[arg_idx] if arg_idx < len(arg_names) else f"_kw{i}"
            arg_idx += 1
            parts.append(name)

        # **kwargs
        if prelude.has_varkw:
            name = arg_names[arg_idx] if arg_idx < len(arg_names) else "kwargs"
            arg_idx += 1
            parts.append(f"**{name}")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Comprehension inlining
    # ------------------------------------------------------------------

    def _try_inline_comprehension(
        self,
        co: MpyCodeObject,
        func_name: str,
    ) -> Optional[str]:
        """
        Try to inline a <listcomp>/<dictcomp>/<setcomp>/<genexpr> code object
        into a Python comprehension expression.

        The bytecode pattern is:
            BUILD_LIST 0 (or BUILD_MAP/BUILD_SET)
            LOAD_FAST(N)   ← iterable arg
            GET_ITER / GET_ITER_STACK
            FOR_ITER <exit>
            STORE_FAST(M)  ← loop variable
            ... expression instructions ...
            STORE_COMP <kind>
            JUMP <back to FOR_ITER>
            RETURN_VALUE
        """
        if co.prelude is None:
            return None

        bc = co.fun_data
        insns = decode_all_instructions(bc, co.prelude.offset_opcodes, len(bc))
        if len(insns) < 5:
            return None

        # Find FOR_ITER
        for_iter_idx = None
        for i, ins in enumerate(insns):
            if ins.opcode == OP_FOR_ITER:
                for_iter_idx = i
                break
        if for_iter_idx is None:
            return None

        # Get loop variable name from STORE after FOR_ITER
        store_idx = for_iter_idx + 1
        if store_idx >= len(insns):
            return None
        store_insn = insns[store_idx]
        loop_var = None
        if store_insn.opcode == OP_STORE_FAST_N:
            loop_var = self._local_name(co, store_insn.arg)
        elif store_insn.opcode == OP_STORE_DEREF:
            loop_var = self._local_name(co, store_insn.arg)
        elif _STORE_FAST_MULTI <= store_insn.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
            loop_var = self._local_name(co, store_insn.opcode - _STORE_FAST_MULTI)
        elif store_insn.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
            ni = store_insn.arg
            loop_var = self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
        if loop_var is None:
            return None

        # Find STORE_COMP or YIELD_VALUE (for genexpr) to extract the expression
        store_comp_idx = None
        for i in range(store_idx + 1, len(insns)):
            if insns[i].opcode == OP_STORE_COMP:
                store_comp_idx = i
                break
            if func_name == "<genexpr>" and insns[i].opcode == OP_YIELD_VALUE:
                store_comp_idx = i
                break
        if store_comp_idx is None:
            return None

        # Decompile the expression between store_idx+1 and store_comp_idx
        expr_stack: List[str] = []
        filter_cond: Optional[str] = None
        extra_for_clauses: List[str] = []
        i = store_idx + 1
        while i < store_comp_idx:
            ins = insns[i]
            op = ins.opcode
            arg = ins.arg

            if _LOAD_CONST_SMALL_INT_MULTI <= op < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM:
                expr_stack.append(repr(op - _LOAD_CONST_SMALL_INT_MULTI - _LOAD_CONST_SMALL_INT_MULTI_EXCESS))
            elif op == OP_LOAD_CONST_SMALL_INT:
                expr_stack.append(repr(arg))
            elif op == OP_LOAD_CONST_STRING:
                s = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<q{arg}>"
                expr_stack.append(repr(s))
            elif op == OP_LOAD_CONST_OBJ:
                obj = self.obj_table[arg] if arg < len(self.obj_table) else f"<o{arg}>"
                expr_stack.append(_repr_obj(obj))
            elif _LOAD_FAST_MULTI <= op < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
                expr_stack.append(self._local_name(co, op - _LOAD_FAST_MULTI))
            elif op == OP_LOAD_FAST_N:
                expr_stack.append(self._local_name(co, arg))
            elif op == OP_LOAD_DEREF:
                expr_stack.append(self._local_name(co, arg))
            elif op in (OP_LOAD_NAME, OP_LOAD_GLOBAL):
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<q{arg}>"
                expr_stack.append(name)
            elif op == OP_LOAD_ATTR:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<q{arg}>"
                obj_e = expr_stack.pop() if expr_stack else "<?>"
                expr_stack.append(f"{_wrap(obj_e)}.{name}")
            elif op == OP_LOAD_METHOD:
                name = self.qstr_table[arg] if arg < len(self.qstr_table) else f"<q{arg}>"
                obj_e = expr_stack.pop() if expr_stack else "<?>"
                expr_stack.append(f"{_wrap(obj_e)}.{name}")
                expr_stack.append("__method_self__")
            elif op in (OP_CALL_FUNCTION, OP_CALL_FUNCTION_VAR_KW):
                n_p = arg & 0xFF
                n_k = (arg >> 8) & 0xFF
                # Nested inlined comprehension: calling comp_str(iterable)
                # Replace __comp_iter__ placeholder with actual iterable
                if (n_p == 1 and n_k == 0
                        and len(expr_stack) >= 2
                        and "__comp_iter__" in expr_stack[-2]):
                    iterable_arg = expr_stack.pop()
                    comp_str = expr_stack.pop()
                    expr_stack.append(
                        comp_str.replace("__comp_iter__", iterable_arg))
                else:
                    call_str = self._build_call(expr_stack, n_p, n_k, op == OP_CALL_FUNCTION_VAR_KW, False)
                    expr_stack.append(call_str)
            elif op in (OP_CALL_METHOD, OP_CALL_METHOD_VAR_KW):
                n_p = arg & 0xFF
                n_k = (arg >> 8) & 0xFF
                call_str = self._build_call(expr_stack, n_p, n_k, op == OP_CALL_METHOD_VAR_KW, True)
                expr_stack.append(call_str)
            elif _BINARY_OP_MULTI <= op < _BINARY_OP_MULTI + _BINARY_OP_MULTI_NUM:
                op_i = op - _BINARY_OP_MULTI
                sym = _BINARY_OP_SYMBOLS[op_i] if op_i < len(_BINARY_OP_SYMBOLS) else "?"
                r = expr_stack.pop() if expr_stack else "?"
                l = expr_stack.pop() if expr_stack else "?"
                expr_stack.append(f"{l} {sym} {r}")
            elif _UNARY_OP_MULTI <= op < _UNARY_OP_MULTI + _UNARY_OP_MULTI_NUM:
                op_i = op - _UNARY_OP_MULTI
                sym = _UNARY_OP_SYMBOLS[op_i] if op_i < len(_UNARY_OP_SYMBOLS) else "?"
                operand = expr_stack.pop() if expr_stack else "?"
                expr_stack.append(f"{sym}{operand}" if op_i != 3 else f"not {operand}")
            elif op == OP_BUILD_TUPLE:
                items = self._pop_n(expr_stack, arg)
                if arg == 1:
                    expr_stack.append(f"({items[0]},)")
                else:
                    expr_stack.append(f"({', '.join(items)})")
            elif op == OP_BUILD_LIST:
                items = self._pop_n(expr_stack, arg)
                expr_stack.append(f"[{', '.join(items)}]")
            elif op == OP_LOAD_SUBSCR:
                key = expr_stack.pop() if expr_stack else "?"
                obj_e = expr_stack.pop() if expr_stack else "?"
                expr_stack.append(f"{_wrap(obj_e)}[{key}]")
            elif op == OP_LOAD_CONST_NONE:
                expr_stack.append("None")
            elif op == OP_LOAD_CONST_TRUE:
                expr_stack.append("True")
            elif op == OP_LOAD_CONST_FALSE:
                expr_stack.append("False")
            elif op == OP_POP_JUMP_IF_FALSE:
                # Comprehension 'if' filter
                filter_cond = expr_stack.pop() if expr_stack else None
            elif op == OP_POP_JUMP_IF_TRUE:
                # Negated filter: if not <cond>
                c = expr_stack.pop() if expr_stack else None
                if c is not None:
                    filter_cond = f"not {_wrap(c)}"
            elif op in (OP_MAKE_FUNCTION, OP_MAKE_FUNCTION_DEFARGS):
                # Nested comprehension: inline the child code object
                child_idx = arg
                child_co = (co.children[child_idx]
                            if child_idx < len(co.children) else None)
                if child_co:
                    child_name = child_co.simple_name or ""
                    inner = self._try_inline_comprehension(child_co, child_name)
                    if inner is not None:
                        expr_stack.append(inner)
                    else:
                        expr_stack.append(f"<{child_name}_{child_idx}>")
                else:
                    expr_stack.append(f"<function_{child_idx}>")
            elif op in (OP_MAKE_CLOSURE, OP_MAKE_CLOSURE_DEFARGS):
                child_idx = arg
                n_closed = ins.extra_arg if ins.extra_arg is not None else 0
                closed_names: List[str] = []
                for _ in range(n_closed):
                    closed_names.append(
                        expr_stack.pop() if expr_stack else "<?>")
                closed_names.reverse()
                child_co = (co.children[child_idx]
                            if child_idx < len(co.children) else None)
                if child_co:
                    self._closure_names[id(child_co)] = closed_names
                    child_name = child_co.simple_name or ""
                    inner = self._try_inline_comprehension(child_co, child_name)
                    if inner is not None:
                        expr_stack.append(inner)
                    else:
                        expr_stack.append(f"<{child_name}_{child_idx}>")
                else:
                    expr_stack.append(f"<closure_{child_idx}>")
            elif op in (OP_GET_ITER, OP_GET_ITER_STACK):
                # Nested for-loop in comprehension:
                # <iterable on stack>, GET_ITER_STACK, FOR_ITER, STORE → inner clause
                if (i + 2 < store_comp_idx
                        and insns[i + 1].opcode == OP_FOR_ITER):
                    inner_store = insns[i + 2]
                    inner_var = None
                    if inner_store.opcode == OP_STORE_FAST_N:
                        inner_var = self._local_name(co, inner_store.arg)
                    elif (_STORE_FAST_MULTI <= inner_store.opcode
                            < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM):
                        inner_var = self._local_name(
                            co, inner_store.opcode - _STORE_FAST_MULTI)
                    elif inner_store.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                        ni = inner_store.arg
                        inner_var = (
                            self.qstr_table[ni]
                            if ni < len(self.qstr_table) else f"_v{ni}")
                    if inner_var is not None:
                        iterable = expr_stack.pop() if expr_stack else "<?>"
                        extra_for_clauses.append(
                            f"for {inner_var} in {iterable}")
                        i += 3  # skip GET_ITER, FOR_ITER, STORE
                        continue            # Skip other opcodes silently
            i += 1

        # Build all for clauses
        all_for_str = f"for {loop_var} in __comp_iter__"
        for fc in extra_for_clauses:
            all_for_str += f" {fc}"
        filter_str = f" if {filter_cond}" if filter_cond else ""

        # Build the comprehension string
        if func_name == "<listcomp>":
            expr = expr_stack[-1] if expr_stack else "<?>"
            return f"[{expr} {all_for_str}{filter_str}]"
        elif func_name == "<setcomp>":
            expr = expr_stack[-1] if expr_stack else "<?>"
            return f"{{{expr} {all_for_str}{filter_str}}}"
        elif func_name == "<genexpr>":
            expr = expr_stack[-1] if expr_stack else "<?>"
            return f"({expr} {all_for_str}{filter_str})"
        elif func_name == "<dictcomp>":
            # For dict comp, stack has [value, key] before STORE_COMP
            key = expr_stack.pop() if expr_stack else "<?>"
            value = expr_stack.pop() if expr_stack else "<?>"
            return f"{{{key}: {value} {all_for_str}{filter_str}}}"
        return None

    # ------------------------------------------------------------------
    # Unpack handling
    # ------------------------------------------------------------------

    def _handle_unpack(
        self,
        insns: List[Instruction],
        idx: int,
        n: int,
        co: MpyCodeObject,
        seq_expr: str,
    ) -> Optional[Tuple[List[str], int]]:
        """Handle UNPACK_SEQUENCE followed by N stores."""
        names: List[str] = []
        j = idx + 1
        for _ in range(n):
            if j >= len(insns):
                return None
            si = insns[j]
            if si.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                ni = si.arg
                names.append(
                    self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
                )
            elif si.opcode == OP_STORE_FAST_N:
                names.append(self._local_name(co, si.arg))
            elif _STORE_FAST_MULTI <= si.opcode < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
                names.append(self._local_name(co, si.opcode - _STORE_FAST_MULTI))
            elif si.opcode == OP_STORE_DEREF:
                names.append(self._local_name(co, si.arg))
            else:
                return None
            j += 1

        target = ", ".join(names)
        return [f"{target} = {seq_expr}"], j - idx

    def _handle_unpack_ex(
        self,
        insns: List[Instruction],
        idx: int,
        n_left: int,
        n_right: int,
        co: MpyCodeObject,
        seq_expr: str,
    ) -> Optional[Tuple[List[str], int]]:
        """Handle UNPACK_EX followed by stores: a, *b, c = expr."""
        total = n_left + 1 + n_right
        names: List[str] = []
        j = idx + 1
        for _ in range(total):
            if j >= len(insns):
                return None
            si = insns[j]
            if si.opcode in (OP_STORE_NAME, OP_STORE_GLOBAL):
                ni = si.arg
                names.append(
                    self.qstr_table[ni] if ni < len(self.qstr_table) else f"_v{ni}"
                )
            elif si.opcode == OP_STORE_FAST_N:
                names.append(self._local_name(co, si.arg))
            elif (_STORE_FAST_MULTI <= si.opcode
                  < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM):
                names.append(
                    self._local_name(co, si.opcode - _STORE_FAST_MULTI))
            elif si.opcode == OP_STORE_DEREF:
                names.append(self._local_name(co, si.arg))
            else:
                return None
            j += 1

        # Mark the starred variable
        if n_left < len(names):
            names[n_left] = f"*{names[n_left]}"
        target = ", ".join(names)
        return [f"{target} = {seq_expr}"], j - idx

    # ------------------------------------------------------------------
    # Short-circuit boolean operators
    # ------------------------------------------------------------------

    def _handle_short_circuit_or(
        self,
        stack: List[str],
        insns: List[Instruction],
        idx: int,
        target_off: int,
        off2idx: Dict[int, int],
    ) -> None:
        """JUMP_IF_TRUE_OR_POP → `x or y`."""
        # TOS is left if true (skip evaluation of y); popped if false.
        # We convert: `x or y` where x is current TOS and y will be computed.
        lhs = stack.pop() if stack else "<?>"
        stack.append(f"{_wrap(lhs)} or __pending__")
        # The __pending__ part will be resolved when the target is reached
        # and the next expression is computed. For simplicity, we leave it
        # as is — it gets combined on subsequent pushes.

    def _handle_short_circuit_and(
        self,
        stack: List[str],
        insns: List[Instruction],
        idx: int,
        target_off: int,
        off2idx: Dict[int, int],
    ) -> None:
        """JUMP_IF_FALSE_OR_POP → `x and y`."""
        lhs = stack.pop() if stack else "<?>"
        stack.append(f"{_wrap(lhs)} and __pending__")

    # ------------------------------------------------------------------
    # Store handling for class definitions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _local_name(self, co: MpyCodeObject, idx: Optional[int]) -> str:
        """Get the name of local variable at index *idx*."""
        if idx is None:
            return "_local?"
        if co.prelude is None:
            return f"_local{idx}"
        # Index 0 = first argument, then subsequent locals
        # Argument names are in co.arg_names (n_pos + n_kwonly entries)
        if idx < len(co.arg_names):
            name = co.arg_names[idx]
            # '*' is a hidden closure variable placeholder — resolve from
            # the parent's MAKE_CLOSURE if we captured the closed-over names
            if name == "*":
                cnames = self._closure_names.get(id(co))
                if cnames is not None and idx < len(cnames):
                    return cnames[idx]
                return f"_local{idx}"
            return name
        # *args and **kwargs are NOT in arg_names but occupy slots right after
        # the named args: slot n_pos+n_kw is *args, slot n_pos+n_kw+has_varargs
        # is **kwargs.
        n_pos = co.prelude.n_pos_args
        n_kw = co.prelude.n_kwonly_args
        varargs_slot = n_pos + n_kw
        if co.prelude.has_varargs and idx == varargs_slot:
            return "args"
        varkw_slot = varargs_slot + (1 if co.prelude.has_varargs else 0)
        if co.prelude.has_varkw and idx == varkw_slot:
            return "kwargs"
        return f"_local{idx}"

    def _pop_n(self, stack: List[str], n: int) -> List[str]:
        """Pop *n* items from the stack, returning them in original push order."""
        items: List[str] = []
        for _ in range(n):
            items.append(stack.pop() if stack else "<?>")
        items.reverse()
        return items

    @staticmethod
    def _indent(lines: List[str], level: int = 1) -> List[str]:
        """Add *level* × 4-space indentation to each line."""
        prefix = "    " * level
        return [prefix + line for line in lines]

    def _strip_module_return(self, lines: List[str]) -> List[str]:
        """Remove trailing `return None` from module-level code."""
        while lines and lines[-1] in ("return", "return None"):
            lines.pop()
        return lines

    def _disassemble_fallback(
        self, insns: List[Instruction], co: MpyCodeObject
    ) -> List[str]:
        """Emit bytecode disassembly as comments (fallback)."""
        lines: List[str] = []
        for insn in insns:
            op = insn.opcode
            name = self._opcode_name(op)
            if insn.has_arg:
                lines.append(f"#   {insn.offset:4d}  {name} {insn.arg}")
            else:
                lines.append(f"#   {insn.offset:4d}  {name}")
        return lines

    @staticmethod
    def _opcode_name(op: int) -> str:
        """Human-readable opcode name."""
        if _LOAD_CONST_SMALL_INT_MULTI <= op < _LOAD_CONST_SMALL_INT_MULTI + _LOAD_CONST_SMALL_INT_MULTI_NUM:
            val = op - _LOAD_CONST_SMALL_INT_MULTI - _LOAD_CONST_SMALL_INT_MULTI_EXCESS
            return f"LOAD_CONST_SMALL_INT({val})"
        if _LOAD_FAST_MULTI <= op < _LOAD_FAST_MULTI + _LOAD_FAST_MULTI_NUM:
            return f"LOAD_FAST({op - _LOAD_FAST_MULTI})"
        if _STORE_FAST_MULTI <= op < _STORE_FAST_MULTI + _STORE_FAST_MULTI_NUM:
            return f"STORE_FAST({op - _STORE_FAST_MULTI})"
        if _UNARY_OP_MULTI <= op < _UNARY_OP_MULTI + _UNARY_OP_MULTI_NUM:
            return f"UNARY_OP({op - _UNARY_OP_MULTI})"
        if _BINARY_OP_MULTI <= op < _BINARY_OP_MULTI + _BINARY_OP_MULTI_NUM:
            return f"BINARY_OP({op - _BINARY_OP_MULTI})"

        _OPCODE_NAMES = {
            OP_LOAD_CONST_FALSE: "LOAD_CONST_FALSE",
            OP_LOAD_CONST_NONE: "LOAD_CONST_NONE",
            OP_LOAD_CONST_TRUE: "LOAD_CONST_TRUE",
            OP_LOAD_CONST_SMALL_INT: "LOAD_CONST_SMALL_INT",
            OP_LOAD_CONST_STRING: "LOAD_CONST_STRING",
            OP_LOAD_CONST_OBJ: "LOAD_CONST_OBJ",
            OP_LOAD_NULL: "LOAD_NULL",
            OP_LOAD_FAST_N: "LOAD_FAST_N",
            OP_LOAD_DEREF: "LOAD_DEREF",
            OP_LOAD_NAME: "LOAD_NAME",
            OP_LOAD_GLOBAL: "LOAD_GLOBAL",
            OP_LOAD_ATTR: "LOAD_ATTR",
            OP_LOAD_METHOD: "LOAD_METHOD",
            OP_LOAD_SUPER_METHOD: "LOAD_SUPER_METHOD",
            OP_LOAD_BUILD_CLASS: "LOAD_BUILD_CLASS",
            OP_LOAD_SUBSCR: "LOAD_SUBSCR",
            OP_STORE_FAST_N: "STORE_FAST_N",
            OP_STORE_DEREF: "STORE_DEREF",
            OP_STORE_NAME: "STORE_NAME",
            OP_STORE_GLOBAL: "STORE_GLOBAL",
            OP_STORE_ATTR: "STORE_ATTR",
            OP_STORE_SUBSCR: "STORE_SUBSCR",
            OP_STORE_MAP: "STORE_MAP",
            OP_DELETE_FAST: "DELETE_FAST",
            OP_DELETE_DEREF: "DELETE_DEREF",
            OP_DELETE_NAME: "DELETE_NAME",
            OP_DELETE_GLOBAL: "DELETE_GLOBAL",
            OP_DUP_TOP: "DUP_TOP",
            OP_DUP_TOP_TWO: "DUP_TOP_TWO",
            OP_POP_TOP: "POP_TOP",
            OP_ROT_TWO: "ROT_TWO",
            OP_ROT_THREE: "ROT_THREE",
            OP_JUMP: "JUMP",
            OP_POP_JUMP_IF_TRUE: "POP_JUMP_IF_TRUE",
            OP_POP_JUMP_IF_FALSE: "POP_JUMP_IF_FALSE",
            OP_JUMP_IF_TRUE_OR_POP: "JUMP_IF_TRUE_OR_POP",
            OP_JUMP_IF_FALSE_OR_POP: "JUMP_IF_FALSE_OR_POP",
            OP_UNWIND_JUMP: "UNWIND_JUMP",
            OP_SETUP_WITH: "SETUP_WITH",
            OP_SETUP_EXCEPT: "SETUP_EXCEPT",
            OP_SETUP_FINALLY: "SETUP_FINALLY",
            OP_POP_EXCEPT_JUMP: "POP_EXCEPT_JUMP",
            OP_FOR_ITER: "FOR_ITER",
            OP_WITH_CLEANUP: "WITH_CLEANUP",
            OP_END_FINALLY: "END_FINALLY",
            OP_GET_ITER: "GET_ITER",
            OP_GET_ITER_STACK: "GET_ITER_STACK",
            OP_BUILD_TUPLE: "BUILD_TUPLE",
            OP_BUILD_LIST: "BUILD_LIST",
            OP_BUILD_MAP: "BUILD_MAP",
            OP_BUILD_SET: "BUILD_SET",
            OP_BUILD_SLICE: "BUILD_SLICE",
            OP_STORE_COMP: "STORE_COMP",
            OP_UNPACK_SEQUENCE: "UNPACK_SEQUENCE",
            OP_UNPACK_EX: "UNPACK_EX",
            OP_RETURN_VALUE: "RETURN_VALUE",
            OP_RAISE_LAST: "RAISE_LAST",
            OP_RAISE_OBJ: "RAISE_OBJ",
            OP_RAISE_FROM: "RAISE_FROM",
            OP_YIELD_VALUE: "YIELD_VALUE",
            OP_YIELD_FROM: "YIELD_FROM",
            OP_MAKE_FUNCTION: "MAKE_FUNCTION",
            OP_MAKE_FUNCTION_DEFARGS: "MAKE_FUNCTION_DEFARGS",
            OP_MAKE_CLOSURE: "MAKE_CLOSURE",
            OP_MAKE_CLOSURE_DEFARGS: "MAKE_CLOSURE_DEFARGS",
            OP_CALL_FUNCTION: "CALL_FUNCTION",
            OP_CALL_FUNCTION_VAR_KW: "CALL_FUNCTION_VAR_KW",
            OP_CALL_METHOD: "CALL_METHOD",
            OP_CALL_METHOD_VAR_KW: "CALL_METHOD_VAR_KW",
            OP_IMPORT_NAME: "IMPORT_NAME",
            OP_IMPORT_FROM: "IMPORT_FROM",
            OP_IMPORT_STAR: "IMPORT_STAR",
        }
        return _OPCODE_NAMES.get(op, f"OP_0x{op:02X}")


# ============================================================================
# Post-processing: class definition cleanup
# ============================================================================


def _postprocess_class_defs(source: str) -> str:
    """
    Transform __build_class__() patterns into proper class definitions.

    The decompiler produces:
        def ClassName():
            ...
        ClassName = __class__:ClassName:ClassName:BaseClass

    This transforms it into:
        class ClassName(BaseClass):
            ...
    """
    lines = source.split("\n")

    # 1. Find which functions are class bodies by scanning for __class__: assignments
    class_body_funcs: Dict[str, Tuple[str, str]] = {}  # func_ref -> (class_name, bases)
    class_decorators: Dict[str, str] = {}  # func_ref -> decorator expression
    for line in lines:
        stripped = line.lstrip()
        if " = __class__:" in stripped:
            parts = stripped.split(" = __class__:", 1)
            rest_parts = parts[1].split(":")
            if len(rest_parts) >= 2:
                func_ref = rest_parts[1]
                class_name = rest_parts[0]
                bases_str = rest_parts[2] if len(rest_parts) > 2 else ""
                class_body_funcs[func_ref] = (class_name, bases_str)
        elif "__class__:" in stripped:
            # Decorated class: Name = decorator(__class__:Name:FuncRef:bases)
            m = re.match(
                r"^(\w+)\s*=\s*(.+?)\(__class__:(\w+):(\w+):(.*?)\)$",
                stripped,
            )
            if m:
                class_name = m.group(1)
                decorator = m.group(2)
                func_ref = m.group(4)
                bases_str = m.group(5)
                class_body_funcs[func_ref] = (class_name, bases_str)
                class_decorators[func_ref] = decorator

    # 2. Collect function bodies for class body functions AND find their line ranges
    func_bodies: Dict[str, List[str]] = {}
    skip_lines: set = set()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            indent = len(line) - len(stripped)
            func_name = stripped[4:].split("(")[0]
            if func_name in class_body_funcs:
                start = i
                body_lines: List[str] = []
                j = i + 1
                body_indent = indent + 4
                while j < len(lines):
                    if lines[j].strip() == "":
                        body_lines.append("")
                        j += 1
                        continue
                    cur_indent = len(lines[j]) - len(lines[j].lstrip())
                    if cur_indent >= body_indent:
                        body_lines.append(lines[j][body_indent:])
                        j += 1
                    else:
                        break
                # Strip trailing blank lines, then 'return _localN' (class body artifact)
                while body_lines and body_lines[-1].strip() == "":
                    body_lines.pop()
                while body_lines and re.match(r"^return _local\d+$", body_lines[-1].strip()):
                    body_lines.pop()
                while body_lines and body_lines[-1].strip() == "":
                    body_lines.pop()
                func_bodies[func_name] = body_lines
                # Mark all lines of this function def for skipping
                for li in range(start, j):
                    skip_lines.add(li)
                # Also skip trailing blank line
                if j < len(lines) and lines[j].strip() == "":
                    skip_lines.add(j)
        i += 1

    # 3. Build result: skip class body function defs, replace __class__ assignments
    result: List[str] = []
    i = 0
    while i < len(lines):
        if i in skip_lines:
            i += 1
            continue
        line = lines[i]
        stripped = line.lstrip()

        if " = __class__:" in stripped:
            parts = stripped.split(" = __class__:", 1)
            class_name = parts[0].strip()
            rest_parts = parts[1].split(":")
            func_ref = rest_parts[1] if len(rest_parts) > 1 else ""
            bases = rest_parts[2] if len(rest_parts) > 2 else ""
            indent = len(line) - len(stripped)
            prefix = " " * indent

            decorator = class_decorators.get(func_ref)
            if decorator:
                result.append(f"{prefix}@{decorator}")
            if bases:
                result.append(f"{prefix}class {class_name}({bases}):")
            else:
                result.append(f"{prefix}class {class_name}:")

            body = func_bodies.get(func_ref, [])
            if body:
                # Recursively process nested class defs in the body
                body_src = _postprocess_class_defs("\n".join(body))
                for bl in body_src.split("\n"):
                    if bl == "":
                        result.append("")
                    else:
                        result.append(f"{prefix}    {bl}")
            else:
                result.append(f"{prefix}    pass")
            result.append("")
            i += 1
            continue

        # Decorated class: Name = decorator(__class__:Name:FuncRef:bases)
        dec_m = re.match(
            r"^(\w+)\s*=\s*(.+?)\(__class__:(\w+):(\w+):(.*?)\)$",
            stripped,
        )
        if dec_m and dec_m.group(4) in class_body_funcs:
            class_name = dec_m.group(1)
            decorator = dec_m.group(2)
            func_ref = dec_m.group(4)
            bases = dec_m.group(5)
            indent = len(line) - len(stripped)
            prefix = " " * indent

            result.append(f"{prefix}@{decorator}")
            if bases:
                result.append(f"{prefix}class {class_name}({bases}):")
            else:
                result.append(f"{prefix}class {class_name}:")

            body = func_bodies.get(func_ref, [])
            if body:
                body_src = _postprocess_class_defs("\n".join(body))
                for bl in body_src.split("\n"):
                    if bl == "":
                        result.append("")
                    else:
                        result.append(f"{prefix}    {bl}")
            else:
                result.append(f"{prefix}    pass")
            result.append("")
            i += 1
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


# ============================================================================
# Pipeline integration (firmware → .py)
# ============================================================================


def decompile_mpy_file(path: str, verbose: bool = False) -> str:
    """Decompile a single .mpy file to Python source."""
    module = read_mpy_file(path)
    decomp = Decompiler(module, verbose=verbose)
    source = decomp.decompile()
    source = _postprocess_class_defs(source)
    return source


def decompile_mpy_bytes(data: bytes, verbose: bool = False) -> str:
    """Decompile .mpy bytes to Python source."""
    module = read_mpy_bytes(data)
    decomp = Decompiler(module, verbose=verbose)
    source = decomp.decompile()
    source = _postprocess_class_defs(source)
    return source


def decompile_mpy_dir(
    input_dir: str,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Decompile all .mpy files in a directory.

    Returns list of (mpy_path, py_source, output_path_or_None).
    """
    results: List[Tuple[str, str, Optional[str]]] = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for root, _dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.endswith(".mpy"):
                continue
            mpy_path = os.path.join(root, fname)
            rel_path = os.path.relpath(mpy_path, input_dir)

            try:
                source = decompile_mpy_file(mpy_path, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  [decompile] {rel_path}: error: {e}", file=sys.stderr)
                source = f"# decompilation failed: {e}\n"

            out_path = None
            if output_dir:
                py_rel = rel_path[:-4] + ".py" if rel_path.endswith(".mpy") else rel_path + ".py"
                out_path = os.path.join(output_dir, py_rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(source)
                if verbose:
                    print(f"  [decompile] {rel_path} → {out_path}", file=sys.stderr)

            results.append((mpy_path, source, out_path))

    return results
