"""Manual GPU check — run by hand, not part of CI.

`python backend/tests/smoke_gpu.py` on a real NVIDIA box. Exits non-zero if
torch or Paddle can't see CUDA.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ok = True

    # --- torch ---
    import torch

    from app.utils.torch_device import select_torch_device

    dev = select_torch_device()
    print(f"[smoke] torch device       : {dev}")
    if torch.cuda.is_available():
        x = torch.ones(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print(f"[smoke] torch CUDA matmul  : {y:.0f} (ok)")
    else:
        print("[smoke] torch.cuda.is_available() == False")
        ok = False

    # --- paddle ---
    try:
        import paddle

        compiled = paddle.device.is_compiled_with_cuda()
        count = paddle.device.cuda.device_count() if compiled else 0
        print(f"[smoke] paddle cuda build  : {compiled}, devices: {count}")
        if not (compiled and count > 0):
            ok = False
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[smoke] paddle import FAILED: {exc}")
        ok = False

    # --- tier selection ---
    try:
        import app.services.layout_detection as ld

        tier = ld.select_tier(use_gpu=True)
        print(f"[smoke] selected tier      : device={tier['device']} tier={tier['tier']}")
        print(f"[smoke] reason             : {tier['reason']}")
        if tier["device"] != "gpu":
            ok = False
    except Exception as exc:  # pragma: no cover
        print(f"[smoke] select_tier FAILED : {exc}")
        ok = False

    print("[smoke] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
