import csv
import re
from generating_synthetic_noise import SpanishNoiseSynthesizer, Config

def merge_hyphenated_words(text):
    # Remove hyphens at line breaks (e.g., "examp-\nle" -> "example")
    return re.sub(r'-\s*\n\s*', '', text)

def split_sentences(text):
    # Split at periods, keep the period with the sentence, strip whitespace
    sentences = [s.strip() for s in re.split(r'(?<=\.)', text) if s.strip()]
    return sentences

def main():
    input_path = r"C:\Users\prana\Downloads\Synthetic_Noise\Synthetic_Noisy_Data_GSoC25\nobleza.txt"
    output_path = "noisy_sentence_pairs.tsv"
    seed = 42

    # Read and preprocess text
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    text = merge_hyphenated_words(text)
    sentences = split_sentences(text)

    synthesizer = SpanishNoiseSynthesizer(seed=seed)
    enabled_rules = {rule: True for rule in Config.RULE_PROBABILITIES if rule != "identity"}

    pairs = []
    for sent in sentences:
        clean_sent = sent.strip()
        if not clean_sent:
            continue
        # Choose random number of edits (3, 4, or 5)
        num_edits = synthesizer.rng.choice([5,6,7,8,9,10])
        noisy = clean_sent
        applied_rules = []
        for _ in range(num_edits):
            noisy, rules = synthesizer.generate_noisy_variant(noisy, enabled_rules)
            applied_rules.extend(rules)
        noisy = noisy.strip()
        if noisy and noisy != clean_sent:
            pairs.append((noisy, clean_sent))

    # Write TSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Input", "Target"])
        for noisy, target in pairs:
            writer.writerow([noisy, target])

    print(f"Wrote {len(pairs)} noisy sentence pairs to {output_path}")

if __name__ == "__main__":
    main()