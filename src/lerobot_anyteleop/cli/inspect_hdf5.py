"""``anyteleop-inspect`` — print the structure of a recorded HDF5 episode."""

from __future__ import annotations

import argparse

import h5py


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="episode_XXXXXX.hdf5")
    args = p.parse_args(argv)

    with h5py.File(args.path, "r") as f:
        print(f"# {args.path}")
        if f.attrs:
            print("## attrs")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
        print("## datasets")

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  /{name:<34} shape={obj.shape} dtype={obj.dtype} "
                      f"compression={obj.compression}")

        f.visititems(_visit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
