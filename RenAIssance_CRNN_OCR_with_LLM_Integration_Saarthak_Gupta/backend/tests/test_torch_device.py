"""Device selection must always land on a real device, cuda > mps > cpu."""

from app.utils.torch_device import select_torch_device


def test_select_returns_known_device():
    assert select_torch_device() in ("cuda", "mps", "cpu")


def test_cuda_preferred_when_available(monkeypatch):
    import app.utils.torch_device as td

    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: True)
    assert td.select_torch_device() == "cuda"


def test_mps_used_when_no_cuda(monkeypatch):
    import app.utils.torch_device as td

    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: False)

    class _FakeMps:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(td.torch.backends, "mps", _FakeMps, raising=False)
    assert td.select_torch_device() == "mps"
