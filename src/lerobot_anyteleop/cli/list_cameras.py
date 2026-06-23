"""``anyteleop-list-cameras`` — enumerate connected RealSense devices."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    try:
        from ..devices.camera.realsense import list_realsense_devices
    except Exception as e:  # pragma: no cover
        print(f"pyrealsense2 not available: {e}")
        return 1

    devices = list_realsense_devices()
    if not devices:
        print("No RealSense devices found.")
        return 0
    print(f"Found {len(devices)} RealSense device(s):")
    for d in devices:
        print(f"  serial={d['serial']}  name={d['name']}  fw={d['firmware']}")
    print("\nUse the serial numbers in your config's `cameras[].serial`.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
