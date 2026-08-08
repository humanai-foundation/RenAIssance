from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


def resolve_path(raw_path: str | None) -> str:
    if raw_path is None:
        return "<not set>"
    return str((PROJECT_ROOT / raw_path).resolve())


def path_exists(raw_path: str | None) -> bool | None:
    if raw_path is None:
        return None
    return (PROJECT_ROOT / raw_path).exists()


def iter_config_paths(config: dict) -> list[tuple[str, str | None, bool]]:
    return [
        ("SSL.dataset 1", config["SSL"].get("dataset 1"), True),
        ("SSL.dataset 2", config["SSL"].get("dataset 2"), False),
        ("SSL.dataset 3", config["SSL"].get("dataset 3"), False),
        ("SSL.saved Encoder path", config["SSL"].get("saved Encoder path"), False),
        ("fine-tuning.dataset 1", config["fine-tuning"].get("dataset 1"), True),
        ("fine-tuning.dataset 1 csv", config["fine-tuning"].get("dataset 1 csv"), True),
        ("fine-tuning.dataset 2", config["fine-tuning"].get("dataset 2"), False),
        ("fine-tuning.dataset 2 csv", config["fine-tuning"].get("dataset 2 csv"), False),
        ("fine-tuning.dataset 3", config["fine-tuning"].get("dataset 3"), False),
        ("fine-tuning.dataset 3 csv", config["fine-tuning"].get("dataset 3 csv"), False),
        ("fine-tuning.test dataset", config["fine-tuning"].get("test dataset"), True),
        (
            "fine-tuning.Encoder path for fine-tuning",
            config["fine-tuning"].get("Encoder path for fine-tuning"),
            False,
        ),
        (
            "fine-tuning.Decoder path for fine-tuning",
            config["fine-tuning"].get("Decoder path for fine-tuning"),
            False,
        ),
        ("fine-tuning.char to token", config["fine-tuning"].get("char to token"), True),
        ("fine-tuning.token to char", config["fine-tuning"].get("token to char"), True),
        ("fine-tuning.saved Encoder path", config["fine-tuning"].get("saved Encoder path"), False),
        ("fine-tuning.saved Decoder path", config["fine-tuning"].get("saved Decoder path"), False),
    ]


def main() -> int:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    print(f"Checking paths in {CONFIG_PATH}")
    print()

    missing_required = False
    for label, raw_path, must_exist in iter_config_paths(config):
        exists = path_exists(raw_path)
        absolute_path = resolve_path(raw_path)
        if exists is None:
            status = "OPTIONAL"
        elif exists:
            status = "OK"
        elif not must_exist:
            status = "OPTIONAL"
        else:
            status = "MISSING"
            missing_required = True
        print(f"[{status:<8}] {label}: {absolute_path}")

    print()
    if missing_required:
        print("Some configured paths are missing. Update config.json or place your data/models in the expected folders.")
        return 1

    print("All configured paths exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
