# Training Data Quality Analysis

Comprehensive tools for analyzing, detecting issues in, and improving training datasets.

## Overview

The quality analysis module provides:

1. **Dataset Statistics** - Size, diversity, length distributions, vocabulary analysis
2. **Quality Metrics** - Per-sample clarity, correctness, and overall scores
3. **Bias Detection** - Gender, cultural, and technical bias analysis
4. **Duplicate Detection** - Exact and semantic duplicate identification
5. **Anomaly Detection** - Outlier and malformed sample detection
6. **Improvement Recommendations** - Actionable suggestions for dataset enhancement

## Quick Start

### Generate a Quality Report

```bash
# Basic report
python scripts/quality_report.py training_data/dataset.jsonl

# With custom output
python scripts/quality_report.py data.jsonl --output results/quality_report.json

# For assembly domain
python scripts/quality_report.py data.jsonl --domain assembly

# Multiple datasets
python scripts/quality_report.py training_data/*.jsonl --output combined_report.json
```

### Generate Interactive HTML Report

```bash
python scripts/quality_report_html.py dataset.jsonl --output report.html --domain assembly
```

### Improve Your Dataset

```bash
# Remove duplicates and anomalies
python scripts/improve_dataset.py data.jsonl --remove-duplicates --remove-anomalies

# Keep only high-quality samples
python scripts/improve_dataset.py data.jsonl --min-quality 0.7

# Combine multiple improvements
python scripts/improve_dataset.py data.jsonl \
  --remove-duplicates \
  --remove-anomalies \
  --min-quality 0.6 \
  --output cleaned_data.jsonl \
  --report improvements.json

# Augment high-quality samples
python scripts/improve_dataset.py data.jsonl --augment --output augmented_data.jsonl
```

## Modules

### `afs.quality.analyzer`

Main dataset analyzer providing comprehensive analysis.

```python
from afs.quality import DatasetAnalyzer, analyze_dataset

# Analyze a dataset
analyzer = DatasetAnalyzer(domain="assembly")
report = analyzer.analyze(samples, dataset_name="my_dataset")

# Or use convenience function
report = analyze_dataset(samples, dataset_name="my_data", domain="code")

# Access results
print(f"Average quality: {report.average_quality_score:.1%}")
print(f"High quality samples: {sum(1 for s in report.sample_qualities if s.overall_quality_score >= 0.8)}")

# Save reports
report.save_json("report.json")
report.save_samples_jsonl("sample_details.jsonl")
```

**Key Classes:**

- `DatasetAnalyzer` - Main analyzer
- `DatasetStatistics` - Overall dataset metrics
- `QualityReport` - Complete analysis results
- `SampleQuality` - Per-sample metrics

### `afs.quality.metrics`

Quality scoring and issue detection.

```python
from afs.quality.metrics import QualityMetrics, DuplicateDetector, AnomalyDetector

metrics = QualityMetrics(domain="code")

# Instruction clarity
clarity = metrics.compute_instruction_clarity("Write a function to calculate factorial")
print(f"Clarity score: {clarity.overall_score():.2f}")

# Output correctness
correctness = metrics.compute_output_correctness("def factorial(n):\n    return n")
print(f"Correctness score: {correctness.overall_score():.2f}")

# Find duplicates
detector = DuplicateDetector()
duplicates = detector.find_duplicates(samples)

# Find anomalies
anomaly_detector = AnomalyDetector()
anomalies = anomaly_detector.find_anomalies(samples)
```

**Key Classes:**

- `QualityMetrics` - Compute clarity and correctness scores
- `InstructionClarity` - Clarity metrics for instructions
- `OutputCorrectness` - Correctness metrics for outputs
- `DuplicateDetector` - Find duplicate samples
- `AnomalyDetector` - Find anomalous samples

### `afs.quality.bias`

Bias detection and analysis.

```python
from afs.quality.bias import BiasAnalyzer, detect_biases

# Analyze bias
bias_report = detect_biases(samples)

print(f"Gender bias score: {bias_report.gender_bias.bias_score:.2f}")
print(f"Cultural bias score: {bias_report.cultural_bias.bias_score:.2f}")
print(f"Technical bias score: {bias_report.technical_bias.bias_score:.2f}")

# Get recommendations
for rec in bias_report.recommendations:
    print(f"- {rec}")
```

**Key Classes:**

- `BiasAnalyzer` - Main bias analyzer
- `BiasReport` - Bias analysis results
- `GenderBiasDetector` - Detect gender bias
- `CulturalBiasDetector` - Detect cultural bias
- `TechnicalBiasDetector` - Detect technical bias

## Quality Metrics

### Instruction Clarity Score (0.0-1.0)

Measures how clear and specific the instruction is:

- **Specificity** - Uses specific verbs vs generic ones
- **Clarity** - Has imperative form, clear conditions
- **Structure** - Reasonable sentence length
- **Metadata** - Contains examples, requirements, context

**Factors:**
- Generic words: "what", "how", "explain"
- Specific words: "calculate", "convert", "assemble"
- Presence of examples and requirements

### Output Correctness Score (0.0-1.0)

Measures quality and correctness of the output:

- **Syntax Validity** - Passes domain-specific syntax check
- **Structure** - Well-organized with consistent indentation
- **Completeness** - Full vs partial implementation
- **Documentation** - Contains helpful comments

**Factors:**
- Syntax validation
- Comment ratio
- Structural consistency
- Length/completeness

### Overall Quality Score

Weighted average:
```
overall_quality = 0.4 * instruction_clarity + 0.6 * output_correctness
```

Higher weight on output because correctness is critical for training.

## Bias Detection

### Gender Bias

Analyzes pronoun usage and gendered language:

- Pronoun distribution (he/she/neutral)
- Gendered occupational examples
- Imbalanced representation

**Metrics:**
- `gender_ratio` - Deviation from 50/50 balance
- `bias_score` - Overall gender bias (0.0-1.0)

### Cultural Bias

Detects Western/English-centric examples:

- Cultural references (American, English, Western)
- Name diversity
- Regional representation

**Metrics:**
- `language_diversity` - Ratio of diverse names
- `regional_bias` - Western-centric content ratio
- `bias_score` - Overall cultural bias (0.0-1.0)

### Technical Bias

Analyzes code diversity:

- Framework diversity
- Programming paradigm diversity
- Code style consistency
- Accessibility (comments, docstrings)

**Metrics:**
- `framework_diversity` - Number of different frameworks
- `paradigm_diversity` - Variety of approaches
- `accessibility_score` - Documentation ratio
- `bias_score` - Overall technical bias (0.0-1.0)

## Duplicate Detection

Two levels of duplicate detection:

### Exact Duplicates
Complete text matches. Always indicative of redundancy.

### Semantic Duplicates
High similarity (default: 95%+ similar) using longest common substring.

**Usage:**
```python
detector = DuplicateDetector(exact_match=True, semantic_threshold=0.95)
duplicates = detector.find_duplicates(samples)

for idx, info in duplicates.items():
    if info.exact_duplicates:
        print(f"Sample {idx} has exact duplicates at indices {info.exact_duplicates}")
    for dup_idx, sim in info.semantic_duplicates:
        print(f"Sample {idx} is {sim:.1%} similar to sample {dup_idx}")
```

## Anomaly Detection

Detects outliers and malformed samples:

### Length Outliers
Samples significantly longer/shorter than average (>2 std devs).

### Content Anomalies
- Mostly whitespace
- Excessive special characters
- Empty or null samples

**Usage:**
```python
detector = AnomalyDetector(length_threshold=2.0)
anomalies = detector.find_anomalies(samples)

for idx, info in anomalies.items():
    print(f"Sample {idx}: {info.anomaly_reasons}")
```

## Report Outputs

### JSON Report Format

```json
{
  "dataset": {
    "name": "my_dataset",
    "path": "training_data/dataset.jsonl",
    "timestamp": "2024-01-14T10:30:00"
  },
  "statistics": {
    "total_samples": 1000,
    "unique_samples": 950,
    "instruction": {
      "count": 1000,
      "avg_length": 25.5,
      "std_length": 12.3,
      "vocab_size": 2500
    },
    "output": {
      "count": 1000,
      "avg_length": 150.2,
      "std_length": 85.4,
      "vocab_size": 8500
    }
  },
  "summary": {
    "average_quality_score": 0.76,
    "quality_distribution": {
      "0.0-0.2": 5,
      "0.2-0.4": 45,
      "0.4-0.6": 150,
      "0.6-0.8": 500,
      "0.8-1.0": 300
    },
    "improvement_opportunities": [...]
  },
  "bias": {
    "gender_bias": {...},
    "cultural_bias": {...},
    "technical_bias": {...},
    "overall_bias_score": 0.42,
    "recommendations": [...]
  }
}
```

### Sample-Level Details

```jsonl
{"index": 0, "instruction_clarity_score": 0.85, "output_correctness_score": 0.92, ...}
{"index": 1, "instruction_clarity_score": 0.62, "output_correctness_score": 0.58, ...}
...
```

### HTML Report Features

- Interactive quality distribution chart
- Summary metrics cards
- Bias analysis breakdown
- Issue detection summary
- Improvement recommendations
- Dataset statistics table
- Responsive design

## Improvement Workflow

### 1. Generate Quality Report

```bash
python scripts/quality_report.py data.jsonl --output report.json
```

### 2. Review Report

- Check average quality score
- Identify duplicates and anomalies
- Review bias metrics
- Read improvement recommendations

### 3. Apply Improvements

```bash
# Remove identified issues
python scripts/improve_dataset.py data.jsonl \
  --remove-duplicates \
  --remove-anomalies \
  --min-quality 0.5 \
  --output cleaned_data.jsonl

# Verify improvements
python scripts/quality_report.py cleaned_data.jsonl --output cleaned_report.json
```

### 4. Compare Results

Look at before/after metrics to verify improvements.

## Best Practices

### 1. Regular Quality Checks

- Analyze new datasets before training
- Periodically check existing datasets
- Track quality metrics over time

### 2. Domain-Specific Analysis

```python
# For assembly/code
analyzer = DatasetAnalyzer(domain="assembly")

# For general text
analyzer = DatasetAnalyzer(domain="general")
```

### 3. Handling Issues

**Duplicates:**
- Remove exactly duplicates immediately
- Review semantic duplicates (might be valid variations)
- Consider consolidating similar samples

**Anomalies:**
- Investigate before removing
- Some outliers might be valuable edge cases
- Manual review recommended

**Low Quality:**
- Review a sample of low-quality items
- Consider fixing (improving clarity/correctness)
- Remove if not fixable

**Bias:**
- Aim for balanced representation
- Use recommendations as starting point
- Consider domain requirements

### 4. Quality Thresholds

- **High Quality**: > 0.8 (good for augmentation)
- **Medium Quality**: 0.5-0.8 (acceptable for training)
- **Low Quality**: < 0.5 (consider removing)

Default removal threshold: 0.5 quality score

### 5. Validation

Always validate improvements:
```bash
# Before
python scripts/quality_report.py original.jsonl

# After
python scripts/quality_report.py improved.jsonl
```

## Integration with Training

After quality analysis and improvement:

1. **Use improved datasets for fine-tuning**
   ```python
   from afs.training import train_model

   # Load cleaned data
   with open("cleaned_data.jsonl") as f:
       training_data = [json.loads(line) for line in f]

   # Train with confidence in data quality
   train_model(training_data, ...)
   ```

2. **Monitor quality over time**
   - Periodically re-analyze training data
   - Compare quality metrics across versions
   - Track improvements from data cleaning

3. **Use high-quality samples for augmentation**
   - Identify top 20% by quality score
   - Use as templates for data augmentation
   - Maintain diversity while improving quality

## Customization

### Custom Domain Analysis

```python
class MyDomainMetrics(QualityMetrics):
    def __init__(self):
        super().__init__(domain="mydomain")

    def _validate_syntax(self, text: str) -> bool:
        # Custom validation logic
        return custom_validate(text)
```

### Custom Bias Detectors

```python
from afs.quality.bias import BiasAnalyzer

class CustomBiasAnalyzer(BiasAnalyzer):
    def __init__(self):
        super().__init__()
        # Add custom detectors
```

## Examples

### Complete Analysis Pipeline

```python
from pathlib import Path
from afs.quality import DatasetAnalyzer

# Load data
with open("data.jsonl") as f:
    samples = [json.loads(line) for line in f]

# Analyze
analyzer = DatasetAnalyzer(domain="assembly")
report = analyzer.analyze(samples, dataset_name="my_data")

# Generate reports
report.save_json("report.json")
report.save_samples_jsonl("samples.jsonl")

# Get insights
print(f"Quality: {report.average_quality_score:.1%}")
print(f"Duplicates: {sum(1 for s in report.sample_qualities if s.is_duplicate)}")
print(f"Anomalies: {sum(1 for s in report.sample_qualities if s.is_anomaly)}")
```

### Filtering Samples

```python
from afs.quality import DatasetAnalyzer

analyzer = DatasetAnalyzer()
report = analyzer.analyze(samples)

# Keep only high-quality samples
high_quality = [
    samples[s.index]
    for s in report.sample_qualities
    if s.overall_quality_score >= 0.8
]

# Keep non-duplicate, non-anomalous samples
clean = [
    samples[s.index]
    for s in report.sample_qualities
    if not s.is_duplicate and not s.is_anomaly
]
```

### Bias-Aware Sampling

```python
report = analyzer.analyze(samples)

# Get recommendations
for rec in report.bias_report.recommendations:
    print(f"Action: {rec}")

# Identify high-risk samples
for idx, reason in report.bias_report.high_risk_samples[:10]:
    print(f"Sample {idx}: {reason}")
    # May want to review or modify these samples
```

## Troubleshooting

### Memory Issues with Large Datasets

For very large datasets, analyze in chunks:

```python
from pathlib import Path
import json

analyzer = DatasetAnalyzer()
chunk_size = 10000

for i in range(0, total_samples, chunk_size):
    chunk = load_chunk(i, chunk_size)
    chunk_report = analyzer.analyze(chunk)
    save_partial_report(chunk_report, f"report_chunk_{i}.json")
```

### Slow Analysis

- Use specific domain for faster syntax validation
- Reduce semantic duplicate threshold if not needed
- Analyze a sample subset first for quick check

### Unexpected Quality Scores

Review:
- Domain setting matches your data
- Sample format (must have 'instruction'/'output' fields)
- Weights in quality computation (currently 0.4/0.6)

## API Reference

See docstrings in:
- `afs.quality.analyzer.DatasetAnalyzer`
- `afs.quality.metrics.QualityMetrics`
- `afs.quality.bias.BiasAnalyzer`

## Contributing

To add custom quality checks:

1. Extend `QualityMetrics` class
2. Override or add methods for custom scoring
3. Update bias detectors as needed
4. Add tests in `tests/test_quality.py`
