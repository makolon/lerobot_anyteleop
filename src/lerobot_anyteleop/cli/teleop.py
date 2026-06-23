"""``anyteleop`` — run the SO-101 -> follower teleoperation loop.

When recording, each episode is tagged with a **language instruction** (stored in
the HDF5 metadata as ``task``), which the LeRobot converter turns into the
per-frame task. Provide it with ``--task`` (same for all episodes) or let the
tool prompt you before each episode. Use ``--episodes N`` to collect several.
"""

from __future__ import annotations

import argparse

from ..config import TeleopConfig
from ..factory import build_system
from ..teleop import TeleopController


def _resolve_instruction(args, cfg: TeleopConfig, i: int, n: int) -> str:
    if args.task:
        return args.task
    if args.no_prompt:
        return cfg.task
    try:
        s = input(f"\n[episode {i + 1}/{n}] language instruction "
                  f"(Enter = {cfg.task!r}): ").strip()
    except EOFError:
        s = ""
    return s or cfg.task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    p.add_argument("--record", action="store_true", help="Record HDF5 episode(s).")
    p.add_argument("--task", default=None,
                   help="Language instruction for the episode(s) (skips the prompt).")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Don't prompt; use the config's `task` as the instruction.")
    p.add_argument("--max-steps", type=int, default=None, help="Override loop.max_steps.")
    p.add_argument("--rate", type=float, default=None, help="Override loop.rate_hz.")
    args = p.parse_args(argv)

    cfg = TeleopConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.loop.max_steps = args.max_steps
    if args.rate is not None:
        cfg.loop.rate_hz = args.rate

    print(f"[anyteleop] leader={cfg.leader.robot} follower={cfg.follower.robot} "
          f"cameras={[c.name for c in cfg.cameras]} rate={cfg.loop.rate_hz}Hz record={args.record}")

    system = build_system(cfg)
    controller = TeleopController(system, cfg)

    if not args.record:
        n = controller.run(record=False)
        print(f"[anyteleop] finished after {n} steps.")
        return 0

    controller.setup()
    try:
        for i in range(args.episodes):
            instruction = _resolve_instruction(args, cfg, i, args.episodes)
            print(f"[rec] episode {i + 1}/{args.episodes}: {instruction!r} "
                  f"(Ctrl-C to end this episode)")
            n = controller.record_episode(instruction, cfg.loop.max_steps)
            print(f"[rec] saved {system.recorder.path} ({n} steps)")
    except KeyboardInterrupt:
        print("\n[anyteleop] interrupted.")
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
