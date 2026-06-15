from __future__ import annotations

import argparse
from pathlib import Path

from .cli_reports import (
    emit_bytecode_report,
    emit_find_report,
    emit_memmap_report,
    emit_objtable_report,
    emit_qstr_report,
    emit_reconstruct_report,
    emit_resolve_report,
)
from .decompile import decompile_mpy_dir, decompile_mpy_file


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _cmd_memmap(args: argparse.Namespace) -> int:
    return emit_memmap_report(args.firmware)


def _cmd_find(args: argparse.Namespace) -> int:
    return emit_find_report(args.firmware)


def _cmd_resolve(args: argparse.Namespace) -> int:
    return emit_resolve_report(args.firmware)


def _cmd_qstr(args: argparse.Namespace) -> int:
    return emit_qstr_report(args.firmware, args.module)


def _cmd_bytecode(args: argparse.Namespace) -> int:
    return emit_bytecode_report(args.firmware, args.module)


def _cmd_objtable(args: argparse.Namespace) -> int:
    return emit_objtable_report(args.firmware, args.module)


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    return emit_reconstruct_report(args.firmware, args.output_dir, args.module)


def _cmd_decompile(args: argparse.Namespace) -> int:
    input_path = args.input

    if input_path.is_file():
        source = decompile_mpy_file(input_path, verbose=args.verbose)
        output_path = args.output
        if output_path is None:
            print(source, end="")
            return 0
        if output_path.suffix.lower() != ".py":
            output_path = output_path / (input_path.stem + ".py")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(source, encoding="utf-8")
        return 0

    if input_path.is_dir():
        output_dir = args.output or Path("version_outputs") / "decompiled"
        decompile_mpy_dir(input_path, output_dir, verbose=args.verbose)
        return 0

    raise ValueError(f"input path does not exist: {input_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mpyforge",
        description="Recover frozen MicroPython modules from firmware images.",
        epilog=(
            "Examples:\n"
            "  python -m mpyforge memmap firmware.bin\n"
            "  python -m mpyforge find firmware.bin\n"
            "  python -m mpyforge qstr firmware.bin -m apa106.py\n"
            "  python -m mpyforge reconstruct firmware.bin -o version_outputs\n"
            "  python -m mpyforge decompile version_outputs\\complex.mpy -o complex.py"
        ),
        formatter_class=_HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    memmap = subparsers.add_parser(
        "memmap",
        help="Inspect the firmware memory map.",
        description="Inspect the firmware memory map and show the decoded regions.",
        formatter_class=_HelpFormatter,
    )
    memmap.add_argument("firmware", type=Path, help="Path to the firmware image")
    memmap.set_defaults(func=_cmd_memmap)

    find = subparsers.add_parser(
        "find",
        help="Locate the frozen module name table.",
        description="Locate the frozen module name table and list the discovered module paths.",
        formatter_class=_HelpFormatter,
    )
    find.add_argument("firmware", type=Path, help="Path to the firmware image")
    find.set_defaults(func=_cmd_find)

    resolve = subparsers.add_parser(
        "resolve",
        help="Resolve frozen module records.",
        description="Resolve frozen module records and show the content, qstr, object, and proto pointers.",
        formatter_class=_HelpFormatter,
    )
    resolve.add_argument("firmware", type=Path, help="Path to the firmware image")
    resolve.set_defaults(func=_cmd_resolve)

    qstr = subparsers.add_parser(
        "qstr",
        help="Show recovered qstr tables.",
        description="Show recovered qstr pools and the qstr table for each module.",
        formatter_class=_HelpFormatter,
    )
    qstr.add_argument("firmware", type=Path, help="Path to the firmware image")
    qstr.add_argument("-m", "--module", help="Limit output to a single module")
    qstr.set_defaults(func=_cmd_qstr)

    bytecode = subparsers.add_parser(
        "bytecode",
        help="Walk the recovered bytecode tree.",
        description="Walk the recovered bytecode tree for a module and print its structure.",
        formatter_class=_HelpFormatter,
    )
    bytecode.add_argument("firmware", type=Path, help="Path to the firmware image")
    bytecode.add_argument("-m", "--module", help="Limit output to a single module")
    bytecode.set_defaults(func=_cmd_bytecode)

    objtable = subparsers.add_parser(
        "objtable",
        help="Decode recovered object tables.",
        description="Decode recovered object tables and show the object kinds and values.",
        formatter_class=_HelpFormatter,
    )
    objtable.add_argument("firmware", type=Path, help="Path to the firmware image")
    objtable.add_argument("-m", "--module", help="Limit output to a single module")
    objtable.set_defaults(func=_cmd_objtable)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="Rebuild .mpy files from recovered data.",
        description="Rebuild .mpy files from recovered qstrs, objects, and bytecode.",
        formatter_class=_HelpFormatter,
    )
    reconstruct.add_argument("firmware", type=Path, help="Path to the firmware image")
    reconstruct.add_argument("-m", "--module", help="Limit output to a single module")
    reconstruct.add_argument("-o", "--output-dir", type=Path, default=Path("version_outputs"), help="Directory for output .mpy files")
    reconstruct.set_defaults(func=_cmd_reconstruct)

    decompile = subparsers.add_parser(
        "decompile",
        help="Decompile .mpy files into Python source.",
        description="Decompile a .mpy file or directory of .mpy files into approximate Python source.",
        formatter_class=_HelpFormatter,
    )
    decompile.add_argument("input", type=Path, help="Path to a .mpy file or directory")
    decompile.add_argument("-o", "--output", type=Path, default=None, help="Output .py file or directory")
    decompile.add_argument("-v", "--verbose", action="store_true", help="Show progress messages")
    decompile.set_defaults(func=_cmd_decompile)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
