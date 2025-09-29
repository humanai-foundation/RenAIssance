# Synthetic Noise Pair Generation Report

## Overview

This report documents the comprehensive pipeline used to generate synthetic noisy text pairs for Spanish OCR error correction and historical text normalization. The system targets 16th-17th century orthographic variants and modern OCR errors, producing training data suitable for sequence-to-sequence models like ByT5 and T5.

## Pipeline Architecture

The synthetic noise generation pipeline consists of several key components working in sequence:

### 1. Input Processing
- **Lexicon Input**: Clean Spanish words (≥4 characters, alphabetic only, UTF-8 encoded)
- **Frequency Table**: Word frequency data in TSV format for weighted sampling
- **Text Preprocessing**: Unicode normalization, hyphen merging, sentence splitting

### 2. Core Synthesizer Engine
The `SpanishNoiseSynthesizer` class implements the main noise generation logic with:
- **Configurable Rule Probabilities**: 9 distinct noise categories with weighted application
- **Reproducible Randomness**: Seed-based generation for consistent results
- **Multi-rule Application**: Sequential application of multiple noise rules per word/sentence
- **Validation Framework**: Output validation and quality control

### 3. Noise Generation Rules

The system implements **9 comprehensive noise categories** with 50+ specific error types:

#### 3.1 Diacritic Errors (28% probability)
- Accent dropping: `canción → cancion`
- Wrong accent placement: `cancion → cancíon`
- Ñ confusions: `año → anio, ano`
- Umlaut errors: `pingüino → pinguino`

#### 3.2 Historical Orthography (20% probability)
- Character substitutions: `x ↔ j`, `v ↔ u`, `i ↔ j`
- Archaic spellings: `ç ↔ c/z/s`, `y ↔ ll/i`
- Period-specific variants: `gu ↔ qu`, `f ↔ h`
- Long s normalization: `ſ → s`

#### 3.3 OCR Confusions (25% probability)
- Character confusions: `i/l/1`, `o/0`, `rn → m`
- Multi-character errors: `cl → d`
- Shape-based confusions: `c ↔ e`, `t ↔ f`, `s ↔ f`
- Digit-letter confusion: `s → 5`

#### 3.4 Keyboard Errors (10% probability)
- Adjacent key substitutions based on Spanish QWERTY layout
- Character insertion/deletion
- Transposition errors
- Double-press artifacts

#### 3.5 Phonetic Confusions (14% probability)
- Sound-based substitutions: `b ↔ v`, `g ↔ j`
- Pronunciation variants: `c ↔ s/z`, `ll ↔ y`
- Silent letter deletion: `h → ''`

#### 3.6 Merge/Split Errors (3% probability)
- Word boundary errors: `para el → parael`
- Truncation: `hablando → habland`
- Prefix/suffix errors: `deshacer → eshacer`

#### 3.7 Punctuation Errors (5% probability)
- Mid-word punctuation insertion
- Apostrophe confusion
- Hyphen placement errors

#### 3.8 Unicode Corruption (2% probability)
- NFC/NFD normalization mixing
- Combining mark errors
- Encoding artifact simulation

#### 3.9 Identity Mappings (10% probability)
- No-change examples: `palabra → palabra`
- Essential for teaching models when to preserve text

## Detailed Pipeline Workflow

### Word-Level Generation (`generating_synthetic_noise.py`)

1. **Input Loading**
   ```
   Lexicon File → SpanishNoiseSynthesizer → Frequency Dictionary
   ```

2. **Variant Calculation**
   - Frequency-based variant count: 2-8 variants per word
   - Edit distance control: 1-3 edits based on word length
   - Weighted sampling based on word frequency

3. **Noise Application**
   ```
   For each word:
   ├── Identity mapping (10% chance)
   └── Generate N variants:
       ├── Apply enabled noise rules sequentially
       ├── Validate output (alphabetic, length constraints)
       ├── Calculate edit distance
       └── Record rule statistics
   ```

4. **Output Generation**
   - TSV format: `input → target | rule | frequency | edit_distance`
   - Statistics file: rule counts, edit distance distribution
   - Examples file: sample transformations by rule type

### Sentence-Level Generation (`Sentence_Level_Noise_Synthesis.py`)

1. **Text Processing**
   ```
   Raw Text → Merge Hyphens → Split Sentences → Clean Sentences
   ```

2. **Multi-Edit Application**
   - Random edit count: 5-10 edits per sentence
   - Sequential rule application
   - Cumulative transformation

3. **Pair Generation**
   ```
   For each sentence:
   ├── Apply random number of edits (5-10)
   ├── Sequential rule application
   ├── Validate output
   └── Create (noisy, clean) pairs
   ```

### Large-Scale Processing (`noise_synthesis_billion_words.ipynb`)

1. **Batch Processing**
   - Directory-based file processing
   - Parallel file handling
   - Progress tracking and error handling

2. **Scalable Architecture**
   - Memory-efficient processing
   - Configurable batch sizes
   - Debug mode for testing

## Configuration Parameters

### Rule Probabilities
```python
RULE_PROBABILITIES = {
    'identity': 0.10,           # Preserve unchanged text
    'diacritic': 0.28,          # Most common errors
    'historical': 0.20,         # Period-appropriate variants
    'ocr_confusion': 0.25,      # OCR artifacts
    'typo': 0.10,              # Modern keyboard errors
    'phonetic': 0.14,          # Sound-based confusions
    'merge_split': 0.03,       # Word boundary errors
    'punctuation': 0.05,       # Punctuation artifacts
    'unicode_corruption': 0.02  # Encoding issues
}
```

### Generation Controls
- **Variant Count**: 2-8 variants per word (frequency-weighted)
- **Edit Distance**: 1-3 edits per transformation
- **Identity Probability**: 10% unchanged examples
- **Validation**: Alphabetic-only output with length constraints

## Quality Assurance

### Built-in Validation
- **Output Validation**: Ensures valid Spanish text output
- **Rule Consistency**: Verifies rule application accuracy
- **Edit Distance Tracking**: Monitors transformation complexity
- **Statistics Generation**: Comprehensive rule usage analytics

### Statistical Controls
- **Reproducible Results**: Seed-based random generation
- **Balanced Distribution**: Frequency-weighted sampling
- **Quality Metrics**: Edit distance distribution, rule coverage
- **Example Tracking**: Sample transformations for manual review

## Output Formats

### Primary Output (TSV)
```
input          target         source_rule           freq     edit_count
cancion        canción        diacritic            0.0032   1
dixo           dijo           historical           0.0021   1
mil1on         millón         ocr_confusion+diacritic  0.0087   2
```

### Statistics (JSON)
- Total pair counts
- Rule usage statistics
- Edit distance distribution
- Generation metadata

### Examples by Rule
- Sample transformations for each noise category
- Manual validation examples
- Rule-specific pattern demonstrations

## Applications and Use Cases

### OCR Error Correction
- Training data for modern OCR post-processing
- Character confusion pattern learning
- Historical document digitization support

### Historical Text Normalization
- 16th-17th century Spanish orthography
- Period-appropriate variant mapping
- Archaic spelling modernization

### Model Training
- ByT5 sequence-to-sequence training
- T5 text correction fine-tuning
- Custom transformer model development

## Technical Implementation

### Core Classes
- **`SpanishNoiseSynthesizer`**: Main generation engine
- **`Config`**: Rule definitions and probabilities
- **Rule Functions**: Individual noise application methods

### Key Methods
- `generate_noisy_variant()`: Core noise application
- `generate_pairs()`: Batch pair generation
- `apply_*_errors()`: Category-specific transformations
- `calculate_variants_per_word()`: Frequency-based sampling

### Error Handling
- Graceful failure recovery
- Maximum attempt limits
- Validation fallbacks
- Comprehensive logging

## Performance Characteristics

- **Scalability**: Handles large corpora efficiently
- **Memory Usage**: Streaming processing for large files
- **Speed**: Optimized for batch processing
- **Reproducibility**: Deterministic output with seed control

## Conclusion

The synthetic noise generation pipeline provides a comprehensive, historically-informed approach to creating training data for Spanish text correction systems. By combining multiple noise categories with realistic probability distributions and quality controls, the system generates high-quality synthetic pairs suitable for modern NLP model training while maintaining linguistic accuracy and historical authenticity.

The modular architecture allows for easy customization of noise types and probabilities, making it adaptable to different use cases and linguistic requirements. The built-in validation and statistical tracking ensure consistent, high-quality output suitable for production model training.
