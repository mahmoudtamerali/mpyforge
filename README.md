# mpyforge

`mpyforge` is a local recovery toolkit for frozen MicroPython modules. It reads a MicroPython firmware image, finds the embedded frozen-module metadata, recovers strings and constants where possible, walks bytecode trees, and can rebuild recovered modules as `.mpy` files.

The current implementation focuses on raw ESP32-family firmware images and UF2 firmware images such as RP2040/RP2350 builds.

This project is inspired from DEFCON 32 conference talk by Wesley McGrew

## Features

- Detects MicroPython firmware banners and derives layout rules from the firmware version.
- Detects supported hardware families from UF2 family IDs or firmware banner text.
- Parses raw ESP32 flash layouts, including shifted dumps, DROM regions, app segments, and partition tables.
- Parses UF2 images into a contiguous flash view for analysis.
- Finds frozen module name tables and resolves module records.
- Recovers qstr pools and module-local qstr tables.
- Walks recovered raw-code and bytecode trees.
- Decodes object tables for supported constant types.
- Reconstructs `.mpy` files from recovered bytecode, qstrs, and objects.
- Decompiles `.mpy` files or directories into approximate Python source.

## Supported Inputs

`mpyforge` expects firmware that contains a recognizable MicroPython banner, for example:

```text
MicroPython v1.28.0 on 2026-03-07
```

Supported source formats:

- Raw ESP32-family firmware dumps (`.bin`)
- UF2 firmware images (`.uf2`)

Micropython frozen modules layout in binary files changes with version.
Supported MicroPython versions:

- legacy layouts for firmware before MicroPython 1.19
- layouts for MicroPython 1.19 up to 1.23
- latest layout for firmware after MicroPython 1.23

## Tested Against

These builds were tested against ESP32 / WROOM Espressif firmware and RP Pico firmware:

| Version | Date       | ESP32 / WROOM Espressif | RP Pico |
| ------- | ---------- | ----------------------- | ------- |
| v1.28.0 | 2026-04-06 | Tested                  | Tested  |
| v1.27.0 | 2025-12-09 | Tested                  | Tested  |
| v1.26.1 | 2025-09-11 | Tested                  | Tested  |
| v1.26.0 | 2025-08-09 | Tested                  | Tested  |
| v1.25.0 | 2025-04-15 | Tested                  | Tested  |
| v1.24.1 | 2024-11-29 | Tested                  | Tested  |
| v1.24.0 | 2024-10-25 | Tested                  | Tested  |
| v1.23.0 | 2024-06-02 | Tested                  | Tested  |
| v1.22.2 | 2024-02-22 | Tested                  | Tested  |
| v1.22.1 | 2024-01-05 | Tested                  | Tested  |
| v1.22.0 | 2023-12-27 | Tested                  | Tested  |
| v1.21.0 | 2023-10-05 | Tested                  | Tested  |
| v1.20.0 | 2023-04-26 | Tested                  | Tested  |
| v1.19.1 | 2022-06-18 | Tested                  | Tested  |
| v1.18   | 2022-01-17 | Tested                  | Tested  |
| v1.17   | 2021-09-02 | Tested                  | Tested  |
| v1.16   | 2021-06-23 | Tested                  | Tested  |
| v1.15   | 2021-04-18 | Tested                  | Tested  |
| v1.14   | 2021-02-02 | Tested                  | Tested  |
| v1.13   | 2020-09-02 | Tested                  | Tested  |
| v1.12   | 2019-12-20 | Tested                  | Tested  |

## Usage

Run the CLI as a Python module from the parent directory of this package, or after installing the project in an environment where `mpyforge` is importable:

```powershell
python -m mpyforge --help
```

General command shape:

```powershell
python -m mpyforge <command> <firmware-or-input> [options]
```

## Commands

### `memmap`

Prints the decoded memory map for a firmware image.

```powershell
python -m mpyforge memmap firmware.bin
python -m mpyforge memmap firmware.uf2
```

For raw ESP32 images, this includes the detected app offset, DROM region, segments, and partition summary when available. For UF2 images, it shows the reconstructed flash range, DROM mapping, and UF2 source details.

### `find`

Finds the frozen module name table and lists discovered module paths.

```powershell
python -m mpyforge find firmware.bin
```

### `resolve`

Resolves frozen module records and prints the recovered module structure, qstr table, object table, and proto/function pointers.

```powershell
python -m mpyforge resolve firmware.bin
```

### `qstr`

Recovers qstr pools and prints module qstr tables.

```powershell
python -m mpyforge qstr firmware.bin
python -m mpyforge qstr firmware.bin -m apa106.py
```

### `bytecode`

Walks recovered bytecode trees and prints code-object structure.

```powershell
python -m mpyforge bytecode firmware.bin
python -m mpyforge bytecode firmware.bin -m apa106.py
```

### `objtable`

Decodes recovered object tables and shows object kinds and values.

```powershell
python -m mpyforge objtable firmware.bin
python -m mpyforge objtable firmware.bin -m apa106.py
```

### `reconstruct`

Writes reconstructed `.mpy` files to an output directory. By default, output goes to `version_outputs`.

```powershell
python -m mpyforge reconstruct firmware.bin
python -m mpyforge reconstruct firmware.bin -o version_outputs
python -m mpyforge reconstruct firmware.bin -m apa106.py -o version_outputs
```

Recovered package paths are preserved where possible, so a module such as `asyncio/core.py` is written under the matching output subdirectory.

### `decompile`

Decompiles a `.mpy` file or a directory of `.mpy` files into approximate Python source.

```powershell
python -m mpyforge decompile version_outputs\apa106.mpy
python -m mpyforge decompile version_outputs\apa106.mpy -o apa106.py
python -m mpyforge decompile version_outputs -o version_outputs\decompiled
```

Use `-v` or `--verbose` to show progress messages during decompilation.

## Programmatic API

The pipeline can also be called from Python:

```python
from mpyforge.pipeline import analyze_firmware

report = analyze_firmware("firmware.bin", "version_outputs")
print(report["status"], report["modules_found"])
```

`analyze_firmware` writes a text recovery log and a representative reconstructed `.mpy` file to the selected output directory.
