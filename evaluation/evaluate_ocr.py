import argparse
from jiwer import wer, cer


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(
        description="Unified OCR Evaluation (CER / WER)"
    )
    parser.add_argument(
        "--gt", required=True, help="Path to ground truth text file"
    )
    parser.add_argument(
        "--pred", required=True, help="Path to predicted text file"
    )
    args = parser.parse_args()

    gt_text = load_text(args.gt)
    pred_text = load_text(args.pred)

    print(f"WER: {wer(gt_text, pred_text):.4f}")
    print(f"CER: {cer(gt_text, pred_text):.4f}")


if __name__ == "__main__":
    main()
