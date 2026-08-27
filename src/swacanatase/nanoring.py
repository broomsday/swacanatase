from __future__ import annotations

import argparse
from pathlib import Path

import biotite.structure as struc
from tuber import generate_nanotube
from tuber.structure import build_atom_array
from tuber.writers import write_structure


def generate_armchair_nanoring(
    n: int,
    units: float | int = 1,
    hydrogen_terminate: bool = False,
    center_z: bool = True,
) -> struc.AtomArray:
    """Generate an armchair (n,n) carbon nanohoop scaffold with tuber."""
    geometry = generate_nanotube(
        n=n,
        m=n,
        units=units,
        hydrogen_terminate=hydrogen_terminate,
        center_z=center_z,
    )
    return build_atom_array(geometry.coordinates, geometry.elements)


def write_armchair_nanoring(
    n: int,
    output_path: str | Path,
    units: float | int = 1,
    file_format: str | None = None,
    hydrogen_terminate: bool = False,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    if file_format is None:
        file_format = output_path.suffix.lower().lstrip(".")
    atom_array = generate_armchair_nanoring(
        n=n,
        units=units,
        hydrogen_terminate=hydrogen_terminate,
    )
    return write_structure(
        atom_array=atom_array,
        output_path=output_path,
        file_format=file_format,
        overwrite=overwrite,
    )


def _units_argument(raw: str) -> float | int:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid units value: {raw!r}")
    return int(value) if value.is_integer() else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an armchair (n,n) nanohoop scaffold with tuber."
    )
    parser.add_argument("--n", type=int, required=True, help="Armchair index n for (n,n).")
    parser.add_argument("--units", type=_units_argument, default=1)
    parser.add_argument("--format", choices=["pdb", "cif"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hydrogen-terminate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output_path = write_armchair_nanoring(
        n=args.n,
        units=args.units,
        output_path=args.output,
        file_format=args.format,
        hydrogen_terminate=args.hydrogen_terminate,
        overwrite=args.overwrite,
    )
    print(f"Wrote armchair ({args.n},{args.n}) scaffold: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
