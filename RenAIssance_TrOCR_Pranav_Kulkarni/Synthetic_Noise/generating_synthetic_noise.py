#!/usr/bin/env python3
"""
Spanish Word-Noise Synthesizer
Generates synthetic noisy pairs for OCR error correction and historical text normalization.
Targets 16th-17th century orthographic variants and modern OCR errors.
"""

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import csv
import sys
from pathlib import Path

# Configuration - Rule probabilities and mapping tables
class Config:
    # Rule probabilities (can sum > 1.0 as multiple rules can apply)
    RULE_PROBABILITIES = {
        'identity': 0.10,
        'diacritic': 0.28,
        'historical': 0.20,
        'ocr_confusion': 0.25,
        'typo': 0.10,
        'phonetic': 0.14,
        'merge_split': 0.03,
        'punctuation': 0.05,
        'unicode_corruption': 0.02
    }
    
    # OCR confusion mappings (character -> list of confusions)
    OCR_CONFUSIONS = {
        'i': ['l', '1', 'í', 'ï'],
        'l': ['i', '1', 'í', 'ï', 't'],
        '1': ['i', 'l', 'í'],
        'o': ['0', 'ó', 'ò'],
        '0': ['o', 'ó'],
        'rn': ['m'],
        'cl': ['d'],
        'ſ': ['s', 'f'],
        's': ['ſ', 'f', '5'],
        'c': ['e', 'o'],
        't': ['f', 'l'],
        'f': ['t', 's', 'ſ'],
        'e': ['c', 'é'],
        'a': ['á', 'à'],
        'u': ['ú', 'ü'],
        'n': ['ñ']
    }
    
    # Historical orthography mappings
    HISTORICAL_MAPPINGS = {
        'x': ['j'],
        'j': ['i', 'x'],
        'v': ['u', 'b'],
        'u': ['v'],
        'i': ['j', 'y'],
        'ç': ['c', 'z', 's'],
        'c': ['ç'],
        'z': ['ç', 'c', 's'],
        'y': ['ll', 'i'],
        'll': ['y'],
        'gu': ['qu'],
        'qu': ['gu'],
        'f': ['h'],  # facienda -> hacienda
        'h': ['f'],
        'ni': ['ñ'],
        'ñ': ['ni', 'n']
    }
    
    # Phonetic confusions
    PHONETIC_MAPPINGS = {
        'b': ['v'],
        'v': ['b'],
        'g': ['j'],
        'j': ['g'],
        'c': ['s', 'z'],
        's': ['c', 'z'],
        'z': ['s', 'c'],
        'll': ['y'],
        'y': ['ll'],
        'h': ['']  # h deletion
    }
    
    # Spanish keyboard layout adjacency
    KEYBOARD_ADJACENCY = {
        'q': ['w', 'a', 's'],
        'w': ['q', 'e', 's', 'a'],
        'e': ['w', 'r', 'd', 's'],
        'r': ['e', 't', 'f', 'd'],
        't': ['r', 'y', 'g', 'f'],
        'y': ['t', 'u', 'h', 'g'],
        'u': ['y', 'i', 'j', 'h'],
        'i': ['u', 'o', 'k', 'j'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'ñ', 'l'],
        'a': ['q', 's', 'z'],
        's': ['a', 'd', 'x', 'z', 'w'],
        'd': ['s', 'f', 'c', 'x', 'e'],
        'f': ['d', 'g', 'v', 'c', 'r'],
        'g': ['f', 'h', 'b', 'v', 't'],
        'h': ['g', 'j', 'n', 'b', 'y'],
        'j': ['h', 'k', 'm', 'n', 'u'],
        'k': ['j', 'l', 'm', 'i'],
        'l': ['k', 'ñ', 'm', 'o'],
        'ñ': ['l', 'p', 'm'],
        'z': ['a', 's', 'x'],
        'x': ['z', 's', 'd', 'c'],
        'c': ['x', 'd', 'f', 'v'],
        'v': ['c', 'f', 'g', 'b'],
        'b': ['v', 'g', 'h', 'n'],
        'n': ['b', 'h', 'j', 'm'],
        'm': ['n', 'j', 'k', 'l']
    }


class SpanishNoiseSynthesizer:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed
        self.stats = defaultdict(int)
        self.examples_by_rule = defaultdict(list)
        
    def load_lexicon(self, lexicon_path: str) -> List[str]:
        """Load Spanish lexicon from file."""
        words = []
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if len(word) >= 4 and word.isalpha():
                        # Normalize to NFC
                        word = unicodedata.normalize('NFC', word)
                        words.append(word)
        except FileNotFoundError:
            print(f"Warning: Lexicon file {lexicon_path} not found. Using sample words.")
            words = self._get_sample_words()
        return words
    
    def load_frequency_table(self, freq_path: str) -> Dict[str, float]:
        """Load frequency table from TSV file."""
        freq_dict = {}
        try:
            with open(freq_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                header_skipped = False
                for row_num, row in enumerate(reader):
                    if len(row) >= 2:
                        # Skip header row if it contains non-numeric frequency
                        if not header_skipped:
                            try:
                                float(row[1])
                            except ValueError:
                                print(f"Skipping header row: {row}")
                                header_skipped = True
                                continue
                        
                        try:
                            word, freq = row[0].strip().lower(), float(row[1])
                            freq_dict[word] = freq
                        except ValueError as e:
                            print(f"Warning: Could not parse row {row_num + 1}: {row}. Error: {e}")
                            continue
        except FileNotFoundError:
            print(f"Warning: Frequency file {freq_path} not found. Using uniform frequencies.")
        return freq_dict
    
    def _get_sample_words(self) -> List[str]:
        """Fallback sample Spanish words for testing."""
        return [
            "palabra", "canción", "español", "máquina", "corazón", "justicia",
            "universidad", "historia", "ejemplo", "tiempo", "gobierno", "problema",
            "sistema", "proceso", "servicio", "desarrollo", "información", "trabajo",
            "producto", "empresa", "mercado", "cliente", "proyecto", "grupo",
            "resultado", "actividad", "programa", "momento", "situación", "persona"
        ]
    
    def calculate_variants_per_word(self, word: str, frequency: float, 
                                  min_variants: int, max_variants: int,
                                  freq_scale: float) -> int:
        """Calculate number of variants based on word frequency."""
        if frequency == 0:
            return min_variants
        
        # Log-scale frequency weighting
        import math
        log_freq = math.log(frequency + 1)
        normalized_freq = min(1.0, log_freq / 10.0)  # Adjust scaling as needed
        
        variants = min_variants + int(freq_scale * normalized_freq * (max_variants - min_variants))
        return max(min_variants, min(max_variants, variants))
    
    # Noise generation functions
    def apply_diacritic_errors(self, word: str, prob: float) -> Optional[str]:
        """Apply diacritic-related errors."""
        if self.rng.random() > prob:
            return None
            
        result = word
        diacritic_chars = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'ü': 'u'
        }
        
        # Drop accents
        if self.rng.random() < 0.4:
            for accented, base in diacritic_chars.items():
                result = result.replace(accented, base)
        
        # Add wrong accents
        elif self.rng.random() < 0.3:
            vowels = 'aeiou'
            for i, char in enumerate(result):
                if char in vowels and self.rng.random() < 0.2:
                    accented_options = {'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú'}
                    if char in accented_options:
                        result = result[:i] + accented_options[char] + result[i+1:]
                        break
        
        # ñ confusions
        elif self.rng.random() < 0.3:
            if 'ñ' in result:
                if self.rng.random() < 0.5:
                    result = result.replace('ñ', 'ni')
                else:
                    result = result.replace('ñ', 'n')
        
        return result if result != word else None
    
    def apply_historical_variants(self, word: str, prob: float) -> Optional[str]:
        """Apply historical orthographic variants."""
        if self.rng.random() > prob:
            return None
            
        result = word
        changes_made = False
        
        for old_char, new_chars in Config.HISTORICAL_MAPPINGS.items():
            if old_char in result and self.rng.random() < 0.3:
                new_char = self.rng.choice(new_chars)
                # Apply to first occurrence to avoid over-transformation
                result = result.replace(old_char, new_char, 1)
                changes_made = True
                break
        
        return result if changes_made else None
    
    def apply_ocr_confusions(self, word: str, prob: float) -> Optional[str]:
        """Apply OCR character confusions."""
        if self.rng.random() > prob:
            return None
            
        result = list(word)
        changes_made = False
        
        # Single character confusions
        for i, char in enumerate(result):
            if char in Config.OCR_CONFUSIONS and self.rng.random() < 0.15:
                confusion = self.rng.choice(Config.OCR_CONFUSIONS[char])
                result[i] = confusion
                changes_made = True
        
        # Multi-character confusions (rn -> m)
        result_str = ''.join(result)
        for multi_char, confusions in Config.OCR_CONFUSIONS.items():
            if len(multi_char) > 1 and multi_char in result_str and self.rng.random() < 0.1:
                confusion = self.rng.choice(confusions)
                result_str = result_str.replace(multi_char, confusion, 1)
                changes_made = True
                break
        
        return result_str if changes_made else None
    
    def apply_keyboard_errors(self, word: str, prob: float) -> Optional[str]:
        """Apply keyboard typo errors."""
        if self.rng.random() > prob:
            return None
            
        result = list(word)
        error_type = self.rng.choice(['substitution', 'insertion', 'deletion', 'transposition', 'doubling'])
        
        if error_type == 'substitution' and len(result) > 0:
            pos = self.rng.randint(0, len(result) - 1)
            char = result[pos]
            if char in Config.KEYBOARD_ADJACENCY:
                result[pos] = self.rng.choice(Config.KEYBOARD_ADJACENCY[char])
        
        elif error_type == 'insertion' and len(result) > 0:
            pos = self.rng.randint(0, len(result))
            if pos > 0 and result[pos-1] in Config.KEYBOARD_ADJACENCY:
                char_to_insert = self.rng.choice(Config.KEYBOARD_ADJACENCY[result[pos-1]])
                result.insert(pos, char_to_insert)
        
        elif error_type == 'deletion' and len(result) > 1:
            pos = self.rng.randint(0, len(result) - 1)
            result.pop(pos)
        
        elif error_type == 'transposition' and len(result) > 1:
            pos = self.rng.randint(0, len(result) - 2)
            result[pos], result[pos + 1] = result[pos + 1], result[pos]
        
        elif error_type == 'doubling' and len(result) > 0:
            pos = self.rng.randint(0, len(result) - 1)
            result.insert(pos + 1, result[pos])
        
        return ''.join(result)
    
    def apply_phonetic_confusions(self, word: str, prob: float) -> Optional[str]:
        """Apply phonetic confusions."""
        if self.rng.random() > prob:
            return None
            
        result = word
        changes_made = False
        
        for sound, confusions in Config.PHONETIC_MAPPINGS.items():
            if sound in result and self.rng.random() < 0.2:
                confusion = self.rng.choice(confusions)
                result = result.replace(sound, confusion, 1)
                changes_made = True
                break
        
        return result if changes_made else None
    
    def apply_merge_split_errors(self, word: str, prob: float) -> Optional[str]:
        """Apply word merge/split errors."""
        if self.rng.random() > prob:
            return None
            
        error_type = self.rng.choice(['merge_prefix', 'split_word', 'truncate'])
        
        if error_type == 'merge_prefix' and len(word) > 6:
            prefixes = ['de', 'la', 'el', 'un', 'en', 'al', 'del']
            prefix = self.rng.choice(prefixes)
            return prefix + word
        
        elif error_type == 'split_word' and len(word) > 6:
            split_pos = self.rng.randint(2, len(word) - 2)
            return word[:split_pos] + ' ' + word[split_pos:]
        
        elif error_type == 'truncate' and len(word) > 5:
            truncate_chars = self.rng.randint(1, 3)
            return word[:-truncate_chars]
        
        return None
    
    def apply_punctuation_errors(self, word: str, prob: float) -> Optional[str]:
        """Apply punctuation insertion errors."""
        if self.rng.random() > prob:
            return None
            
        if len(word) < 4:
            return None
            
        punct_chars = [',', '.', ';', ':', '-', "'"]
        pos = self.rng.randint(1, len(word) - 1)
        punct = self.rng.choice(punct_chars)
        
        return word[:pos] + punct + word[pos:]
    
    def apply_unicode_corruption(self, word: str, prob: float) -> Optional[str]:
        """Apply Unicode normalization errors."""
        if self.rng.random() > prob:
            return None
            
        # Convert to NFD then back, or introduce combining character errors
        decomposed = unicodedata.normalize('NFD', word)
        
        # Sometimes strip combining marks
        if self.rng.random() < 0.5:
            result = ''.join(c for c in decomposed if not unicodedata.combining(c))
            return result
        
        return decomposed
    
    def generate_noisy_variant(self, word: str, enabled_rules: Dict[str, bool]) -> Tuple[str, List[str]]:
        """Generate a single noisy variant with applied rules tracking."""
        result = word
        applied_rules = []
        
        # Determine number of edits (1-2 for most words, up to 3 for long words)
        max_edits = 1 if len(word) <= 6 else (2 if len(word) <= 10 else 3)
        num_edits = self.rng.randint(1, max_edits)
        
        # Apply rules in random order
        available_rules = [
            ('diacritic', self.apply_diacritic_errors, Config.RULE_PROBABILITIES['diacritic']),
            ('historical', self.apply_historical_variants, Config.RULE_PROBABILITIES['historical']),
            ('ocr_confusion', self.apply_ocr_confusions, Config.RULE_PROBABILITIES['ocr_confusion']),
            ('typo', self.apply_keyboard_errors, Config.RULE_PROBABILITIES['typo']),
            ('phonetic', self.apply_phonetic_confusions, Config.RULE_PROBABILITIES['phonetic']),
            ('merge_split', self.apply_merge_split_errors, Config.RULE_PROBABILITIES['merge_split']),
            ('punctuation', self.apply_punctuation_errors, Config.RULE_PROBABILITIES['punctuation']),
            ('unicode_corruption', self.apply_unicode_corruption, Config.RULE_PROBABILITIES['unicode_corruption'])
        ]
        
        # Filter by enabled rules
        available_rules = [(name, func, prob) for name, func, prob in available_rules 
                          if enabled_rules.get(name, True)]
        
        self.rng.shuffle(available_rules)
        
        edits_applied = 0
        for rule_name, rule_func, rule_prob in available_rules:
            if edits_applied >= num_edits:
                break
                
            variant = rule_func(result, rule_prob)
            if variant is not None and variant != result:
                result = variant
                applied_rules.append(rule_name)
                edits_applied += 1
        
        # If no changes were made and we don't want identity, try one more time
        if result == word and applied_rules == [] and self.rng.random() > Config.RULE_PROBABILITIES['identity']:
            # Force apply the most likely rule
            rule_name, rule_func, _ = available_rules[0] if available_rules else ('diacritic', self.apply_diacritic_errors, 1.0)
            variant = rule_func(result, 0.8)  # High probability
            if variant is not None and variant != result:
                result = variant
                applied_rules.append(rule_name)
        
        return result, applied_rules
    
    def is_valid_output(self, text: str, allow_punctuation: bool = False) -> bool:
        """Check if generated text is valid (alphabetic + allowed chars)."""
        if not text:
            return False
            
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzáéíóúüñç')
        if allow_punctuation:
            allowed_chars.update(',.-;:\' ')
            
        return all(c.lower() in allowed_chars for c in text)
    
    def generate_pairs(self, lexicon: List[str], freq_dict: Dict[str, float],
                      min_variants: int, max_variants: int, freq_scale: float,
                      identity_prob: float, enabled_rules: Dict[str, bool],
                      allow_punctuation: bool = False) -> List[Tuple[str, str, str, float, int]]:
        """Generate all noisy pairs."""
        pairs = []
        
        for word in lexicon:
            frequency = freq_dict.get(word, 0.0)
            n_variants = self.calculate_variants_per_word(word, frequency, min_variants, max_variants, freq_scale)
            
            # Identity mapping
            if self.rng.random() < identity_prob:
                pairs.append((word, word, 'identity', frequency, 0))
                self.stats['identity'] += 1
                n_variants -= 1
            
            # Generate noisy variants
            for _ in range(n_variants):
                max_attempts = 10
                for attempt in range(max_attempts):
                    noisy, rules = self.generate_noisy_variant(word, enabled_rules)
                    
                    if self.is_valid_output(noisy, allow_punctuation) and noisy != word:
                        rule_str = '+'.join(rules) if rules else 'none'
                        edit_distance = self._levenshtein_distance(word, noisy)
                        pairs.append((noisy, word, rule_str, frequency, edit_distance))
                        
                        # Update stats
                        for rule in rules:
                            self.stats[rule] += 1
                            if len(self.examples_by_rule[rule]) < 10:
                                self.examples_by_rule[rule].append(f"{word} -> {noisy}")
                        break
        
        return pairs
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def save_pairs(self, pairs: List[Tuple[str, str, str, float, int]], output_path: str):
        """Save pairs to TSV file."""
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['input', 'target', 'source_rule', 'freq', 'edit_count', 'seed'])
            
            for noisy, target, rule, freq, edit_dist in pairs:
                writer.writerow([noisy, target, rule, freq, edit_dist, self.seed])
    
    def save_stats(self, pairs: List[Tuple[str, str, str, float, int]], output_dir: str):
        """Save statistics and examples."""
        # Calculate statistics
        total_pairs = len(pairs)
        edit_distances = [edit_dist for _, _, _, _, edit_dist in pairs]
        avg_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0
        
        stats_data = {
            'total_pairs': total_pairs,
            'avg_edit_distance': avg_edit_distance,
            'rule_counts': dict(self.stats),
            'edit_distance_distribution': dict(Counter(edit_distances)),
            'seed': self.seed
        }
        
        # Save JSON stats
        with open(f"{output_dir}/stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        # Save examples by rule
        with open(f"{output_dir}/examples_by_rule.txt", 'w', encoding='utf-8') as f:
            for rule, examples in self.examples_by_rule.items():
                f.write(f"=== {rule.upper()} EXAMPLES ===\n")
                for example in examples[:10]:
                    f.write(f"{example}\n")
                f.write("\n")
    
    def run_tests(self, lexicon: List[str]) -> bool:
        """Run basic sanity tests."""
        print("Running sanity tests...")
        
        test_words = lexicon[:10] if len(lexicon) >= 10 else lexicon
        enabled_rules = {rule: True for rule in ['diacritic', 'historical', 'ocr_confusion']}
        
        all_variants = set()
        total_tests = 0
        passed_tests = 0
        
        for word in test_words:
            for _ in range(5):  # Test 5 variants per word
                noisy, rules = self.generate_noisy_variant(word, enabled_rules)
                total_tests += 1
                
                # Test: variant should be different (unless identity)
                if 'identity' not in rules and noisy == word:
                    print(f"FAIL: No change applied to '{word}' -> '{noisy}'")
                    continue
                    
                # Test: variant should be valid
                if not self.is_valid_output(noisy):
                    print(f"FAIL: Invalid output '{noisy}' for input '{word}'")
                    continue
                    
                # Test: no exact duplicates (unless intended)
                variant_key = (word, noisy)
                if variant_key in all_variants:
                    print(f"WARNING: Duplicate variant '{word}' -> '{noisy}'")
                all_variants.add(variant_key)
                
                passed_tests += 1
                print(f"PASS: '{word}' -> '{noisy}' (rules: {rules})")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        print(f"\nTest Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        return success_rate >= 0.8


def main():
    parser = argparse.ArgumentParser(description='Spanish Word-Noise Synthesizer')
    parser.add_argument('--lexicon', default='spanish_lexicon.txt', help='Input lexicon file')
    parser.add_argument('--freq', default='freq_table.tsv', help='Frequency table file')
    parser.add_argument('--out', default='noisy_pairs.tsv', help='Output TSV file')
    parser.add_argument('--min_variants', type=int, default=2, help='Minimum variants per word')
    parser.add_argument('--max_variants', type=int, default=4, help='Maximum variants per word')
    parser.add_argument('--freq_scale', type=float, default=1.0, help='Frequency scaling factor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--identity_prob', type=float, default=0.1, help='Identity mapping probability')
    parser.add_argument('--enable_punctuation', action='store_true', help='Enable punctuation errors')
    parser.add_argument('--enable_unicode', action='store_true', help='Enable Unicode corruption')
    parser.add_argument('--disable_historical', action='store_true', help='Disable historical variants')
    parser.add_argument('--disable_ocr', action='store_true', help='Disable OCR confusions')
    parser.add_argument('--disable_diacritic', action='store_true', help='Disable diacritic errors')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    
    args = parser.parse_args()
    
    # Initialize synthesizer
    synthesizer = SpanishNoiseSynthesizer(seed=args.seed)
    
    # Load data
    print(f"Loading lexicon from {args.lexicon}...")
    lexicon = synthesizer.load_lexicon(args.lexicon)
    print(f"Loaded {len(lexicon)} words")
    
    if args.test:
        success = synthesizer.run_tests(lexicon)
        sys.exit(0 if success else 1)
    
    print(f"Loading frequencies from {args.freq}...")
    freq_dict = synthesizer.load_frequency_table(args.freq)
    print(f"Loaded frequencies for {len(freq_dict)} words")
    
    # Configure enabled rules
    enabled_rules = {
        'diacritic': not args.disable_diacritic,
        'historical': not args.disable_historical,
        'ocr_confusion': not args.disable_ocr,
        'typo': True,
        'phonetic': True,
        'merge_split': True,
        'punctuation': args.enable_punctuation,
        'unicode_corruption': args.enable_unicode
    }
    
    print("Enabled rules:", [rule for rule, enabled in enabled_rules.items() if enabled])
    
    # Generate pairs
    print("Generating noisy pairs...")
    pairs = synthesizer.generate_pairs(
        lexicon, freq_dict, args.min_variants, args.max_variants, 
        args.freq_scale, args.identity_prob, enabled_rules, args.enable_punctuation
    )
    
    print(f"Generated {len(pairs)} noisy pairs")
    
    # Save outputs
    output_dir = Path(args.out).parent
    synthesizer.save_pairs(pairs, args.out)
    synthesizer.save_stats(pairs, output_dir)
    
    print(f"Saved pairs to {args.out}")
    print(f"Saved stats to {output_dir}/stats.json")
    print(f"Saved examples to {output_dir}/examples_by_rule.txt")


if __name__ == '__main__':
    main()