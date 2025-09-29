# Spanish Word-Noise Synthesizer

A comprehensive tool for generating synthetic noisy text pairs targeting OCR error correction and 16th-17th century Spanish orthographic normalization. Produces training data suitable for ByT5, T5, and other sequence-to-sequence models.

## Features

- **9 noise categories** with 50+ specific error types
- **Historically accurate** 16th-17th century Spanish variants
- **OCR-realistic** character confusions and artifacts  
- **Frequency-weighted sampling** for balanced datasets
- **Configurable parameters** for all noise types
- **Reproducible results** with seed control
- **Built-in validation** and statistics generation

## Quick Start

```bash
# Basic usage with default settings
python generate_spanish_noise.py \
  --lexicon spanish_lexicon.txt \
  --freq freq_table.tsv \
  --out noisy_pairs.tsv

# Advanced usage with custom parameters
python generate_spanish_noise.py \
  --lexicon spanish_lexicon.txt \
  --freq freq_table.tsv \
  --out noisy_pairs.tsv \
  --min_variants 3 \
  --max_variants 6 \
  --seed 1234 \
  --identity_prob 0.15 \
  --enable_punctuation \
  --enable_unicode

# Run built-in tests
python generate_spanish_noise.py --test
```

## Input File Formats

### Lexicon File (`spanish_lexicon.txt`)
```
palabra
canción
español
máquina
corazón
```
- One word per line
- Lowercase, NFC-normalized UTF-8
- Words ≥ 4 characters
- Only alphabetic characters

### Frequency Table (`freq_table.tsv`)
```
palabra	0.0045
canción	0.0032
español	0.0156
máquina	0.0023
```
- Tab-separated values
- Format: `word<TAB>frequency`
- Frequency as decimal (e.g., 0.0045 = 0.45%)

## Output Files

### Main Output (`noisy_pairs.tsv`)
```
input	target	source_rule	freq	edit_count	seed
cancion	canción	diacritic	0.0032	1	42
dixo	dijo	historical	0.0021	1	42
mil1on	millón	ocr_confusion+diacritic	0.0087	2	42
```

### Statistics (`stats.json`)
```json
{
  "total_pairs": 15432,
  "avg_edit_distance": 1.34,
  "rule_counts": {
    "diacritic": 4321,
    "historical": 3102,
    "ocr_confusion": 3856
  },
  "edit_distance_distribution": {
    "1": 8934,
    "2": 4521,
    "3": 1977
  },
  "seed": 42
}
```

### Examples (`examples_by_rule.txt`)
```
=== DIACRITIC EXAMPLES ===
canción -> cancion
máquina -> maquina
corazón -> corazon

=== HISTORICAL EXAMPLES ===
dijo -> dixo
justicia -> iusticia
universidad -> vniuersidad
```

## Noise Categories

### 1. Diacritic Errors
- **Accent dropping**: `canción → cancion`
- **Wrong accents**: `cancion → cancíon`  
- **Accent position**: `máquina → maquiná`
- **Ñ confusions**: `año → anio, ano`
- **Umlaut errors**: `pingüino → pinguino`

### 2. Historical Orthography
- **x ↔ j**: `dixo ↔ dijo, exército ↔ ejército`
- **v ↔ u**: `vniuersidad ↔ universidad`
- **i ↔ j**: `iusticia ↔ justicia`
- **ç variants**: `corazón ↔ coraçon`
- **Long s**: `conſejo → consejo`
- **f ↔ h**: `facienda → hacienda`
- **gu/qu**: `guerra ↔ querra`

### 3. OCR Confusions
- **i/l/1 confusion**: `millón → mil1on, milIon`
- **o/0 confusion**: `corona → c0rona`
- **rn → m**: `barrio → bamio`  
- **cl → d**: `clamar → damar`
- **c ↔ e**: `casa → easa`
- **t ↔ f**: `tanto → fanfo`
- **s ↔ f**: `casa → cafa`

### 4. Keyboard Errors
- **Adjacent key substitution**: `casa → caaa` (s→a)
- **Character insertion**: `tiempo → tiuempo`
- **Character deletion**: `palabra → plabra`
- **Transposition**: `forma → fomra`
- **Double-press**: `casa → caasa`

### 5. Phonetic Confusions
- **b ↔ v**: `vaso → baso`
- **g ↔ j**: `gente → jente`
- **c ↔ s/z**: `cinco → sinco`
- **ll ↔ y**: `llorar → yorar`
- **h deletion**: `hacer → acer`

### 6. Merge/Split Errors
- **Word merging**: `para el → parael`
- **Word splitting**: `algunos → al gunos`
- **Truncation**: `hablando → habland`
- **Prefix errors**: `deshacer → eshacer`

### 7. Punctuation Errors
- **Mid-word punctuation**: `corazón → cora,zón`
- **Apostrophe insertion**: `bien → bi'en`
- **Hyphen confusion**: `bienestar → bien-estar`

### 8. Unicode Corruption 
- **NFC/NFD mixing**: `é → é` (composed vs decomposed)
- **Combining mark errors**: Missing or doubled accents
- **Encoding artifacts**: Byte-level corruption simulation

### 9. Identity Mappings
- **No change**: `palabra → palabra`
- **Teaches model** when to leave words unchanged
- **Essential for** production deployment

## Configuration

### Rule Probabilities
```python
RULE_PROBABILITIES = {
    'identity': 0.10,      # Keep 10% unchanged
    'diacritic': 0.28,     # Most common errors
    'historical': 0.20,    # Period-appropriate variants
    'ocr_confusion': 0.25, # OCR artifacts
    'typo': 0.10,          # Modern keyboard errors
    'phonetic': 0.14,      # Sound-based confusions
    'merge_split': 0.03,   # Word boundary errors
    'punctuation': 0.05,   # Punctuation artifacts
    'unicode_corruption': 0.02  # Encoding issues
}
```

### Frequency-Based Sampling
The tool generates more variants for high-frequency words:
- **Low frequency**: 2-3 variants
- **Medium frequency**: 3-5 variants  
- **High frequency**: 4-8 variants
- **Configurable** via `--freq_scale` parameter

### Edit Distance Control
- **Short words (4-6 chars)**: Mostly 1 edit
- **Medium words (7-10 chars)**: 1-2 edits
- **Long words (11+ chars)**: Up to 3 edits
- **Average distance**: ~1.3 edits per variant

## Command Line Options

```bash
# Input/Output
--lexicon FILE          Input lexicon file (default: spanish_lexicon.txt)
--freq FILE             Frequency table (default: freq_table.tsv) 
--out FILE              Output TSV file (default: noisy_pairs.tsv)

# Variant Generation
--min_variants N        Minimum variants per word (default: 2)
--max_variants N        Maximum variants per word (default: 4)
--freq_scale F          Frequency scaling factor (default: 1.0)
--identity_prob F       Probability of identity mapping (default: 0.1)

# Rule Control
--enable_punctuation    Enable punctuation errors
--enable_unicode        Enable Unicode corruption
--disable_historical    Disable historical variants
--disable_ocr          Disable OCR confusions  
--disable_diacritic    Disable diacritic errors

# Utility
--seed N               Random seed for reproducibility (default: 42)
--test                 Run validation tests only
```

## Model Training Recommendations

### For ByT5 Models
```python
# Recommended hyperparameters
max_source_length = 128      # Handle long corrupted inputs
max_target_length = 64       # Clean outputs are shorter
batch_size = 32              # Adjust based on GPU memory
learning_rate = 1e-4         # Conservative for fine-tuning
num_epochs = 3-5             # Avoid overfitting
```

### For T5 Models
```python
# Preprocessing
tokenizer.add_prefix("correct Spanish: ")  # Task specification
max_source_length = 100
max_target_length = 50
```

### Data Splitting
```bash
# Generate separate datasets
python generate_spanish_noise.py --seed 42 --out train_pairs.tsv
python generate_spanish_noise.py --seed 123 --out dev_pairs.tsv  
python generate_spanish_noise.py --seed 456 --out test_pairs.tsv
```

### Training Strategy
1. **Start with identity mappings** for stability
2. **Gradually increase noise** complexity
3. **Use curriculum learning**: easy → hard errors
4. **Monitor edit distance** distribution in outputs
5. **Validate on historical texts** when available

## Validation and Quality Control

### Built-in Tests
```bash
python generate_spanish_noise.py --test
```
- Checks output validity (alphabetic characters)
- Verifies rule application consistency
- Validates edit distance distribution
- Reports success/failure rates

### Manual Inspection
```bash
# Generate small sample for manual review
python generate_spanish_noise.py \
  --min_variants 1 --max_variants 2 \
  --out sample_pairs.tsv

# Review examples by rule
cat examples_by_rule.txt
```

### Statistical Validation  
- **Edit distance**: Should average 1.0-1.5 edits
- **Rule distribution**: Check `stats.json` for balance
- **Length preservation**: Input/output length should be similar
- **Character validity**: All outputs should be valid Spanish text

## Examples

### Historical Text Normalization
```
Input:  "Eſta es vna hiſtoria muy antigua del reyno de Eſpaña"
Target: "Esta es una historia muy antigua del reino de España"
Rules:  historical+diacritic
```

### OCR Error Correction  
```
Input:  "E1 go6ierno ha anuncíado nuevas medídas"
Target: "El gobierno ha anunciado nuevas medidas" 
Rules:  ocr_confusion+diacritic
```

### Mixed Corruption
```
Input:  "Ia vniuersidad ofrecera proqramas innovadores"
Target: "La universidad ofrecerá programas innovadores"
Rules:  historical+ocr_confusion+diacritic+typo
```

## Troubleshooting

### Empty Output
- Check input file encoding (must be UTF-8)
- Verify word length ≥ 4 characters
- Ensure alphabetic-only input words

### Low Variant Count
- Increase `--max_variants` or `--freq_scale`
- Check frequency table format
- Verify rule probabilities > 0

### Invalid Characters in Output
- Disable `--enable_punctuation` for strict alphabetic output
- Check Unicode normalization settings
- Review OCR confusion mappings

### Poor Model Performance
- Increase identity mapping probability (--identity_prob)
- Reduce edit distance by lowering variant counts
- Balance rule probabilities based on your use case
- Add more training data with `--freq_scale`

## License and Citation

This tool generates synthetic data for research and commercial use. When using in academic work, please cite the noise categories and methodology described in this documentation.

## Contributing

To add new noise rules:
1. Add mapping tables to `Config` class
2. Implement noise function following existing patterns  
3. Add rule to `generate_noisy_variant()` method
4. Update probability configuration
5. Add tests and examples

For bug reports and feature requests, please include:
- Input files (sample)
- Command line arguments used
- Expected vs actual output
- Error messages or unexpected behavior