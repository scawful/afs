# Training Data Quality Analysis Tools

Comprehensive automated quality analysis, bias detection, and improvement tools for training datasets used in fine-tuning large language models.

## Features

### 1. Dataset Statistics
- Sample count and size
- Instruction length distribution (mean, std dev)
- Output length distribution
- Vocabulary analysis
- Category distribution

### 2. Quality Metrics
- **Instruction Clarity Score** (0.0-1.0)
  - Specificity of instructions
  - Clarity and readability
  - Presence of examples and requirements

- **Output Correctness Score** (0.0-1.0)
  - Syntax validity
  - Code structure quality
  - Completeness and documentation

- **Overall Quality Score** - Weighted average (40% clarity + 60% correctness)

### 3. Duplicate Detection
- Exact duplicate identification
- Semantic duplicate detection (configurable similarity threshold)
- Deduplication status tracking

### 4. Anomaly Detection
- Length outliers (configurable z-score threshold)
- Whitespace/empty anomalies
- Special character ratio anomalies
- Outlier scoring

### 5. Bias Detection
- **Gender Bias** - Pronoun distribution, gendered language
- **Cultural Bias** - Name diversity, regional representation
- **Technical Bias** - Framework diversity, paradigm diversity
- Actionable recommendations for balancing

### 6. Improvement Recommendations
- Specific actions for data quality improvement
- Data augmentation opportunities
- Bias mitigation strategies
- Removal recommendations for problematic samples

## Installation

The quality tools are part of the AFS package:

```bash
cd /Users/scawful/src/lab/afs
pip install -e .
```

## Quick Start

### Generate Quality Report

```bash
# Basic report
python scripts/quality_report.py training_data/dataset.jsonl

# With custom output and domain
python scripts/quality_report.py data.jsonl \
  --output results/report.json \
  --domain assembly

# Multiple files
python scripts/quality_report.py training_data/*.jsonl \
  --output combined_report.json
```

### Generate Interactive HTML Report

```bash
python scripts/quality_report_html.py dataset.jsonl \
  --output report.html \
  --domain code
```

Open `report.html` in a browser for interactive visualization.

### Improve Your Dataset

```bash
# Remove duplicates and anomalies
python scripts/improve_dataset.py data.jsonl \
  --remove-duplicates \
  --remove-anomalies \
  --output cleaned_data.jsonl

# Keep only high-quality samples
python scripts/improve_dataset.py data.jsonl \
  --min-quality 0.7 \
  --output filtered_data.jsonl

# Generate improvement report
python scripts/improve_dataset.py data.jsonl \
  --remove-duplicates \
  --min-quality 0.6 \
  --report improvements.json
```

## Usage Examples

### Python API

```python
from afs.quality import DatasetAnalyzer

# Load your data
samples = [
    {"instruction": "...", "output": "..."},
    # ... more samples
]

# Analyze
analyzer = DatasetAnalyzer(domain="code")
report = analyzer.analyze(samples, dataset_name="my_dataset")

# Access results
print(f"Average quality: {report.average_quality_score:.1%}")
print(f"Duplicates found: {sum(1 for s in report.sample_qualities if s.is_duplicate)}")

# Save reports
report.save_json("report.json")
report.save_samples_jsonl("samples.jsonl")
```

### Bias Analysis

```python
from afs.quality.bias import detect_biases

# Detect biases in dataset
bias_report = detect_biases(samples)

print(f"Gender bias score: {bias_report.gender_bias.bias_score:.2f}")
print(f"Cultural bias score: {bias_report.cultural_bias.bias_score:.2f}")
print(f"Technical bias score: {bias_report.technical_bias.bias_score:.2f}")

# Get recommendations
for rec in bias_report.recommendations:
    print(f"- {rec}")
```

### Find Duplicates and Anomalies

```python
from afs.quality.metrics import DuplicateDetector, AnomalyDetector

# Find duplicates
dup_detector = DuplicateDetector()
duplicates = dup_detector.find_duplicates(samples)

for idx, info in duplicates.items():
    print(f"Sample {idx}: {info.deduplication_status}")

# Find anomalies
anom_detector = AnomalyDetector()
anomalies = anom_detector.find_anomalies(samples)

for idx, info in anomalies.items():
    print(f"Sample {idx}: {info.anomaly_reasons}")
```

## Report Formats

### JSON Report

Contains:
- Dataset metadata (name, path, timestamp)
- Statistics (counts, lengths, vocabulary)
- Quality summary (average score, distribution)
- Bias analysis (all three bias types + recommendations)
- Sample count breakdown (high/medium/low quality)

```bash
python scripts/quality_report.py data.jsonl --output report.json
```

### Sample-Level JSONL

Per-sample detailed metrics:

```bash
# Generated automatically with JSON report
# Format: <report_stem>_samples.jsonl
```

### HTML Report

Interactive report with:
- Quality distribution charts
- Bias breakdown
- Issue summary
- Responsive design
- Downloadable metrics

```bash
python scripts/quality_report_html.py data.jsonl --output report.html
```

## Domain Support

Customize analysis for your domain:

```python
# For assembly code
analyzer = DatasetAnalyzer(domain="assembly")

# For Python/general code
analyzer = DatasetAnalyzer(domain="code")

# For general text
analyzer = DatasetAnalyzer(domain="general")
```

Different domains use domain-specific syntax validation.

## Configuration

### Quality Score Weights

Default weights (40% instruction clarity, 60% output correctness):

```python
# Customize if needed
analyzer = DatasetAnalyzer()
# Modify scoring in _analyze_sample method
```

### Duplicate Detection Threshold

```python
detector = DuplicateDetector(
    exact_match=True,
    semantic_threshold=0.95  # Default: 95% similarity
)
```

### Anomaly Detection Threshold

```python
detector = AnomalyDetector(
    length_threshold=2.0,  # Default: 2 std devs
    enable_content_checks=True
)
```

## Improvement Workflow

### 1. Analyze Current Dataset

```bash
python scripts/quality_report.py original.jsonl --output analysis.json
```

Review metrics:
- Average quality score
- Duplicates and anomalies
- Bias scores
- Improvement recommendations

### 2. Apply Improvements

```bash
python scripts/improve_dataset.py original.jsonl \
  --remove-duplicates \
  --remove-anomalies \
  --min-quality 0.6 \
  --output improved.jsonl
```

### 3. Verify Results

```bash
python scripts/quality_report.py improved.jsonl --output improved_analysis.json
```

Compare before/after metrics to confirm improvements.

### 4. Use Improved Data for Training

```python
# Load cleaned data for fine-tuning
with open("improved.jsonl") as f:
    training_data = [json.loads(line) for line in f]

# Train with confidence in data quality
from afs.training import train_model
train_model(training_data, ...)
```

## Output Interpretation

### Quality Scores

- **0.8-1.0** (High Quality)
  - Ready for training
  - Good candidates for augmentation
  - Minimal issues

- **0.5-0.8** (Medium Quality)
  - Acceptable for training
  - May benefit from improvement
  - Review for issues

- **0.0-0.5** (Low Quality)
  - Consider filtering out
  - Fix if possible
  - Review for patterns

### Bias Scores

- **0.0-0.2** (Well Balanced)
  - Good representation
  - No major bias detected

- **0.2-0.5** (Some Bias)
  - Room for improvement
  - Follow recommendations

- **0.5-1.0** (High Bias)
  - Significant imbalance
  - Action recommended

## Performance Considerations

### Large Datasets

For datasets > 100k samples, analyze in chunks:

```python
from pathlib import Path
import json

analyzer = DatasetAnalyzer()
chunk_size = 10000
reports = []

for i in range(0, total_samples, chunk_size):
    chunk = load_chunk(i, chunk_size)
    report = analyzer.analyze(chunk)
    reports.append(report)
```

### Memory Usage

Quality analysis requires keeping all samples in memory. Typical memory usage:
- 1000 samples: ~5-10 MB
- 10000 samples: ~50-100 MB
- 100000 samples: ~500 MB - 1 GB

## Examples

See `/examples/quality_analysis_example.py` for comprehensive examples:

```bash
PYTHONPATH=src python3 examples/quality_analysis_example.py
```

Includes:
1. Basic dataset analysis
2. Bias detection
3. Duplicate/anomaly detection
4. Complete improvement workflow

## Testing

Run tests to verify functionality:

```bash
python -m pytest tests/test_quality.py -v
```

## Modules

### `afs.quality.analyzer`
Main dataset analyzer and quality reporting.

### `afs.quality.metrics`
Quality metrics, duplicate detection, anomaly detection.

### `afs.quality.bias`
Gender, cultural, and technical bias detection.

## Scripts

### `quality_report.py`
Generate JSON quality report for datasets.

```bash
python scripts/quality_report.py <dataset> [options]
```

Options:
- `-o`, `--output`: Output report path
- `-d`, `--domain`: Domain (general|assembly|code)
- `--no-samples`: Skip per-sample details
- `-v`, `--verbose`: Verbose logging

### `quality_report_html.py`
Generate interactive HTML report.

```bash
python scripts/quality_report_html.py <dataset> -o report.html
```

### `improve_dataset.py`
Apply improvements to datasets.

```bash
python scripts/improve_dataset.py <dataset> [options]
```

Options:
- `-o`, `--output`: Output dataset path
- `--remove-duplicates`: Remove duplicate samples
- `--remove-anomalies`: Remove anomalies
- `--min-quality`: Minimum quality threshold
- `-r`, `--report`: Save improvement report
- `--augment`: Augment high-quality samples

## Contributing

To extend quality analysis:

1. Add custom metrics to `QualityMetrics` class
2. Implement domain-specific checks
3. Add tests in `tests/test_quality.py`
4. Update documentation

## API Reference

See docstrings and `/docs/QUALITY_ANALYSIS.md` for complete API documentation.

## Troubleshooting

### No anomalies detected when expected

- Ensure dataset has sufficient variety for statistical detection
- Lower `length_threshold` parameter if needed
- Check anomaly detector configuration

### Memory issues with large datasets

- Analyze in chunks using `chunk_size`
- Use lower-resolution sampling
- Increase available RAM

### Quality scores seem incorrect

- Verify domain setting matches data type
- Check sample format (must have 'instruction'/'output' fields)
- Review quality computation weights

### Bias scores unexpected

- Ensure sufficient samples for statistical analysis
- Customize bias detector for your use case
- Review detection logic in `afs.quality.bias`

## License

Part of AFS (Agentic File System) framework.

## Support

For issues or questions:
1. Check documentation in `/docs/QUALITY_ANALYSIS.md`
2. Review examples in `/examples/`
3. Run tests to verify installation
4. Check module docstrings
