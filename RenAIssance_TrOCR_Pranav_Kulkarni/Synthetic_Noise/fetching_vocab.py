# pip install requests wordfreq

# pip install wordfreq[cjk]

# # Download Spanish Wiktionary dump
# !pip install wiktextract

# !python -m wiktextract --language-code es eswiktionary-latest-pages-articles.xml.bz2 > eswiktionary.json


#!/usr/bin/env python3
"""
Spanish Word Lexicon Builder

Combines words from:
1. Hunspell es_ES dictionary
2. Wiktionary Spanish words (via Kaikki.org dumps)
3. wordfreq Spanish word list

Processing:
- Normalize to NFC
- Convert to lowercase
- Keep only alphabetic Spanish characters (a-z, áéíóúü, ñ)
- Filter words longer than 4 characters
- Deduplicate and combine with frequency data
"""

import re
import requests
import zipfile
import gzip
import json
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, Tuple
import tempfile
import os

# Spanish character pattern (alphabetic only)
SPANISH_CHARS = re.compile(r'^[a-záéíóúüñ]+$')

def normalize_word(word: str) -> str:
    """Normalize word to NFC and lowercase."""
    return unicodedata.normalize('NFC', word.lower().strip())

def is_valid_spanish_word(word: str) -> bool:
    """Check if word contains only valid Spanish characters and is > 4 chars."""
    return len(word) > 3 and SPANISH_CHARS.match(word) is not None

def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL to filepath."""
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_hunspell_words(data_dir: Path) -> Set[str]:
    """Extract words from Hunspell es_ES dictionary files."""
    print("Processing Hunspell es_ES dictionary...")

    hunspell_dir = data_dir / "hunspell"
    hunspell_dir.mkdir(exist_ok=True)

    # Download Hunspell es_ES files
    hunspell_urls = {
        "es_ES.dic": "https://cgit.freedesktop.org/libreoffice/dictionaries/plain/es/es_ES.dic",
        "es_ES.aff": "https://cgit.freedesktop.org/libreoffice/dictionaries/plain/es/es_ES.aff"
    }

    words = set()

    # Try to download dictionary file
    dic_file = hunspell_dir / "es_ES.dic"
    if not dic_file.exists():
        success = download_file(hunspell_urls["es_ES.dic"], dic_file)
        if not success:
            print("Could not download Hunspell dictionary. Skipping Hunspell source.")
            return words

    # Parse .dic file (format: word/flags)
    try:
        with open(dic_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # First line is count, skip it
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove affixes (everything after /)
                    word = line.split('/')[0].strip()
                    if word:
                        normalized = normalize_word(word)
                        if is_valid_spanish_word(normalized):
                            words.add(normalized)

        print(f"Extracted {len(words)} words from Hunspell")
    except Exception as e:
        print(f"Error processing Hunspell dictionary: {e}")

    return words

def extract_wiktionary_words(data_dir: Path) -> Set[str]:
    """Extract Spanish words from Wiktionary dump via Kaikki.org."""
    print("Processing Wiktionary words from Kaikki.org...")

    wikt_dir = data_dir / "wiktionary"
    wikt_dir.mkdir(exist_ok=True)

    # Kaikki.org Spanish dump URL
    wikt_url = "https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.json"
    wikt_file = wikt_dir / "kaikki-spanish.json"

    words = set()

    if not wikt_file.exists():
        success = download_file(wikt_url, wikt_file)
        if not success:
            print("Could not download Wiktionary dump. Skipping Wiktionary source.")
            return words

    try:
        with open(wikt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} Wiktionary entries...")

                try:
                    entry = json.loads(line.strip())

                    # Extract word (lemma)
                    if 'word' in entry:
                        word = normalize_word(entry['word'])
                        if is_valid_spanish_word(word):
                            words.add(word)

                    # Extract inflected forms
                    if 'forms' in entry:
                        for form in entry['forms']:
                            if 'form' in form:
                                word = normalize_word(form['form'])
                                if is_valid_spanish_word(word):
                                    words.add(word)

                except json.JSONDecodeError:
                    continue

        print(f"Extracted {len(words)} words from Wiktionary")
    except Exception as e:
        print(f"Error processing Wiktionary dump: {e}")

    return words

def extract_wordfreq_words(data_dir: Path) -> Dict[str, float]:
    """Extract Spanish words with frequencies from wordfreq."""
    print("Processing wordfreq Spanish words...")

    try:
        from wordfreq import word_frequency, available_languages

        if 'es' not in available_languages():
            print("Spanish not available in wordfreq. Install with: pip install wordfreq[cjk]")
            return {}

        # Get top Spanish words (wordfreq works with frequency queries)
        # We'll generate a comprehensive list by trying common words

        words_freq = {}

        # Start with some seed words and expand
        from wordfreq import top_n_list
        spanish_words = top_n_list('es', 50000)  # Top 50k words

        for word in spanish_words:
            normalized = normalize_word(word)
            if is_valid_spanish_word(normalized):
                freq = word_frequency(word, 'es')
                if freq > 0:
                    words_freq[normalized] = freq

        print(f"Extracted {len(words_freq)} words with frequencies from wordfreq")
        return words_freq

    except ImportError:
        print("wordfreq not installed. Install with: pip install wordfreq")
        print("Skipping wordfreq source.")
        return {}
    except Exception as e:
        print(f"Error processing wordfreq: {e}")
        return {}

def combine_and_save_lexicon(hunspell_words: Set[str],
                           wikt_words: Set[str],
                           freq_words: Dict[str, float],
                           output_dir: Path):
    """Combine all sources and save the final lexicon."""
    print("Combining all sources...")

    # Combine all words
    all_words = hunspell_words | wikt_words | set(freq_words.keys())

    print(f"Total unique words: {len(all_words)}")
    print(f"- From Hunspell: {len(hunspell_words)}")
    print(f"- From Wiktionary: {len(wikt_words)}")
    print(f"- From wordfreq: {len(freq_words)}")

    # Sort words alphabetically
    sorted_words = sorted(all_words)

    # Save plain word list
    lexicon_file = output_dir / "spanish_lexicon.txt"
    with open(lexicon_file, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(f"{word}\n")

    print(f"Saved word list to {lexicon_file}")

    # Save word list with frequencies (TSV format)
    freq_file = output_dir / "spanish_lexicon_with_freq.tsv"
    with open(freq_file, 'w', encoding='utf-8') as f:
        f.write("word\tfrequency\n")
        for word in sorted_words:
            freq = freq_words.get(word, 0.0)
            f.write(f"{word}\t{freq:.2e}\n")

    print(f"Saved word list with frequencies to {freq_file}")

    # Generate statistics
    stats_file = output_dir / "lexicon_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Spanish Lexicon Statistics\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total unique words: {len(all_words)}\n")
        f.write(f"Words from Hunspell: {len(hunspell_words)}\n")
        f.write(f"Words from Wiktionary: {len(wikt_words)}\n")
        f.write(f"Words from wordfreq: {len(freq_words)}\n")
        f.write(f"Words with frequency data: {len([w for w in all_words if w in freq_words])}\n")
        f.write(f"Average word length: {sum(len(w) for w in all_words) / len(all_words):.1f}\n")
        f.write(f"Min word length: {min(len(w) for w in all_words)}\n")
        f.write(f"Max word length: {max(len(w) for w in all_words)}\n")

    print(f"Saved statistics to {stats_file}")

def main():
    """Main function to build Spanish lexicon."""
    print("Building Spanish Word Lexicon")
    print("=" * 40)

    # Setup directories
    base_dir = Path("spanish_lexicon_data")
    base_dir.mkdir(exist_ok=True)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Extract words from each source
    hunspell_words = extract_hunspell_words(base_dir)
    wikt_words = extract_wiktionary_words(base_dir)
    freq_words = extract_wordfreq_words(base_dir)

    # Combine and save
    combine_and_save_lexicon(hunspell_words, wikt_words, freq_words, output_dir)

    print("\nLexicon building complete!")
    print(f"Output files saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

