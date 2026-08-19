"""Unit-test the device/tier selection branches in layout_detection without a
real Paddle install or GPU — Paddle and resource probing are monkeypatched."""

import app.services.layout_detection as ld


def _resources(**overrides):
    base = {
        "warnings": [],
        "gpu_available": False,
        "gpu_vram_gb": 0.0,
        "gpu_vram_free_gb": 0.0,
        "available_ram_gb": 16.0,
        "total_ram_gb": 32.0,
    }
    base.update(overrides)
    return base


def _patch(monkeypatch, **resource_overrides):
    monkeypatch.setattr(ld, "_ensure_paddle", lambda: None)
    monkeypatch.setattr(
        ld, "check_system_resources", lambda use_gpu: _resources(**resource_overrides)
    )


def test_gpu_server_tier(monkeypatch):
    _patch(monkeypatch, gpu_available=True, gpu_vram_free_gb=10.0)
    tier = ld.select_tier(use_gpu=True)
    assert tier["device"] == "gpu"
    assert tier["tier"] == "server"
    assert tier["layout_model"] == ld.SERVER_MODELS["layout"]


def test_gpu_mobile_tier(monkeypatch):
    # Between MOBILE (2) and SERVER (6) free VRAM → mobile on GPU.
    _patch(monkeypatch, gpu_available=True, gpu_vram_free_gb=3.0)
    tier = ld.select_tier(use_gpu=True)
    assert tier["device"] == "gpu"
    assert tier["tier"] == "mobile"


def test_low_vram_falls_back_to_cpu(monkeypatch):
    # Below MOBILE_MIN_VRAM → CPU; ample RAM → server models on CPU.
    _patch(monkeypatch, gpu_available=True, gpu_vram_free_gb=1.0, available_ram_gb=16.0)
    tier = ld.select_tier(use_gpu=True)
    assert tier["device"] == "cpu"
    assert tier["tier"] == "server"


def test_cpu_low_ram_uses_mobile(monkeypatch):
    _patch(monkeypatch, gpu_available=False, available_ram_gb=2.0)
    tier = ld.select_tier(use_gpu=False)
    assert tier["device"] == "cpu"
    assert tier["tier"] == "mobile"


def test_check_resources_without_paddle_reports_no_gpu(monkeypatch):
    # Simulate paddle not yet imported → no GPU detected, no crash.
    monkeypatch.setattr(ld, "_paddle", None)
    info = ld.check_system_resources(use_gpu=True)
    assert info["gpu_available"] is False
    assert info["gpu_vram_gb"] == 0.0
    assert any("GPU not available" in w for w in info["warnings"])
