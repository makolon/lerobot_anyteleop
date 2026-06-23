"""``anyteleop`` — run the SO-101 -> xArm7 teleoperation loop."""

from __future__ import annotations

import argparse

from ..config import TeleopConfig
from ..factory import build_system
from ..teleop import TeleopController


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    p.add_argument("--record", action="store_true", help="Record an HDF5 episode.")
    p.add_argument("--max-steps", type=int, default=None, help="Override loop.max_steps.")
    p.add_argument("--rate", type=float, default=None, help="Override loop.rate_hz.")
    args = p.parse_args(argv)

    cfg = TeleopConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.loop.max_steps = args.max_steps
    if args.rate is not None:
        cfg.loop.rate_hz = args.rate

    print(f"[anyteleop] leader={cfg.leader.backend} follower={cfg.follower.backend} "
          f"cameras={[c.name for c in cfg.cameras]} rate={cfg.loop.rate_hz}Hz record={args.record}")

    system = build_system(cfg)
    controller = TeleopController(system, cfg)
    n = controller.run(record=args.record)

    print(f"[anyteleop] finished after {n} steps.")
    if args.record and system.recorder.path is not None:
        print(f"[anyteleop] recorded -> {system.recorder.path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
