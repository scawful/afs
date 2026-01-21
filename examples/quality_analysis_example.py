#!/usr/bin/env python3
"""Example of using the quality analysis module.

This example demonstrates how to:
1. Analyze a dataset for quality issues
2. Generate reports
3. Apply improvements
4. Compare before/after results
"""

import json
from pathlib import Path

from afs.quality import DatasetAnalyzer, analyze_dataset
from afs.quality.bias import detect_biases
from afs.quality.metrics import DuplicateDetector, AnomalyDetector


def example_basic_analysis():
    """Example 1: Basic dataset quality analysis."""
    print("=" * 80)
    print("Example 1: Basic Dataset Quality Analysis")
    print("=" * 80)

    # Sample training data
    samples = [
        {
            "instruction": "Write a Python function to calculate factorial",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        },
        {
            "instruction": "Create a class for managing user accounts",
            "output": "class UserAccount:\n    def __init__(self, username, email):\n        self.username = username\n        self.email = email",
        },
        {
            "instruction": "Implement binary search algorithm",
            "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        },
        # Duplicate
        {
            "instruction": "Write a Python function to calculate factorial",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        },
        # Low quality
        {"instruction": "x", "output": "y"},
    ]

    # Analyze
    analyzer = DatasetAnalyzer(domain="code")
    report = analyzer.analyze(samples, dataset_name="example_code_dataset")

    # Print results
    print(f"\nDataset: {report.dataset_name}")
    print(f"Total samples: {report.statistics.total_samples}")
    print(f"Unique samples: {report.statistics.unique_samples}")
    print(f"Average quality score: {report.average_quality_score:.1%}\n")

    print("Quality Breakdown:")
    dist = report.quality_distribution
    for range_label, count in dist.items():
        print(f"  {range_label}: {count} samples")

    print("\nIssues Detected:")
    duplicates = sum(1 for s in report.sample_qualities if s.is_duplicate)
    anomalies = sum(1 for s in report.sample_qualities if s.is_anomaly)
    print(f"  Duplicates: {duplicates}")
    print(f"  Anomalies: {anomalies}")

    print("\nImprovement Opportunities:")
    for i, rec in enumerate(report.improvement_opportunities[:3], 1):
        print(f"  {i}. {rec}")

    return report


def example_bias_analysis():
    """Example 2: Bias detection analysis."""
    print("\n" + "=" * 80)
    print("Example 2: Bias Detection Analysis")
    print("=" * 80)

    # Sample data with potential bias
    samples = [
        {"instruction": "He wrote elegant code", "output": "result"},
        {"instruction": "She is a great engineer", "output": "result"},
        {"instruction": "The American standard approach", "output": "result"},
        {"instruction": "Using React framework", "output": "result"},
        {"instruction": "Using Vue framework", "output": "result"},
    ]

    # Analyze bias
    bias_report = detect_biases(samples)

    print(f"\nBias Scores (0.0 = balanced, 1.0 = highly biased):")
    print(f"  Gender Bias: {bias_report.gender_bias.bias_score:.2f}")
    print(f"    - Male pronouns: {bias_report.gender_bias.pronoun_counts.get('male', 0)}")
    print(f"    - Female pronouns: {bias_report.gender_bias.pronoun_counts.get('female', 0)}")

    print(f"\n  Cultural Bias: {bias_report.cultural_bias.bias_score:.2f}")
    print(f"    - Language diversity: {bias_report.cultural_bias.language_diversity:.1%}")
    print(f"    - Regional bias: {bias_report.cultural_bias.regional_bias:.1%}")

    print(f"\n  Technical Bias: {bias_report.technical_bias.bias_score:.2f}")
    print(f"    - Framework diversity: {bias_report.technical_bias.framework_diversity:.1%}")
    print(f"    - Paradigm diversity: {bias_report.technical_bias.paradigm_diversity:.1%}")

    print(f"\nOverall Bias Score: {bias_report.overall_bias_score:.2f}/1.0")

    print("\nRecommendations:")
    for i, rec in enumerate(bias_report.recommendations, 1):
        print(f"  {i}. {rec}")


def example_issue_detection():
    """Example 3: Finding duplicates and anomalies."""
    print("\n" + "=" * 80)
    print("Example 3: Issue Detection (Duplicates & Anomalies)")
    print("=" * 80)

    samples = [
        {"instruction": "Write a function", "output": "def func(): pass"},
        {"instruction": "Write a function", "output": "def func(): pass"},  # Exact duplicate
        {"instruction": "Another function", "output": "def another(): return 42"},
        {"instruction": "", "output": "empty instruction"},  # Anomaly
        {"instruction": "x" * 5000, "output": "y" * 5000},  # Length outlier
    ]

    # Find duplicates
    dup_detector = DuplicateDetector()
    duplicates = dup_detector.find_duplicates(samples)

    print(f"\nDuplicates Found: {len(duplicates)}")
    for idx, info in duplicates.items():
        if info.exact_duplicates:
            print(f"  Sample {idx}: Exact duplicate of {info.exact_duplicates}")

    # Find anomalies
    anom_detector = AnomalyDetector()
    anomalies = anom_detector.find_anomalies(samples)

    print(f"\nAnomalies Found: {len(anomalies)}")
    for idx, info in anomalies.items():
        print(f"  Sample {idx}:")
        for reason in info.anomaly_reasons:
            print(f"    - {reason}")


def example_comprehensive_workflow():
    """Example 4: Complete analysis and improvement workflow."""
    print("\n" + "=" * 80)
    print("Example 4: Comprehensive Workflow")
    print("=" * 80)

    # Create sample dataset
    samples = [
        {
            "instruction": "Explain what this code does",
            "output": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b",
            "category": "explanation",
        },
        {
            "instruction": "Explain what this code does",
            "output": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b",
            "category": "explanation",
        },  # Duplicate
        {
            "instruction": "Write a function to sum a list",
            "output": "def sum_list(items):\n    total = 0\n    for item in items:\n        total += item\n    return total",
            "category": "coding",
        },
        {
            "instruction": "Bad input",
            "output": "",
            "category": "coding",
        },  # Low quality
    ]

    # Step 1: Initial analysis
    print("\nStep 1: Analyzing dataset...")
    analyzer = DatasetAnalyzer(domain="code")
    initial_report = analyzer.analyze(samples, dataset_name="original")

    print(f"  Total samples: {initial_report.statistics.total_samples}")
    print(f"  Average quality: {initial_report.average_quality_score:.1%}")
    print(f"  Duplicates: {sum(1 for s in initial_report.sample_qualities if s.is_duplicate)}")

    # Step 2: Identify high-quality samples
    print("\nStep 2: Identifying high-quality samples...")
    high_quality_indices = [
        s.index for s in initial_report.sample_qualities if s.overall_quality_score >= 0.6
    ]
    print(f"  Found {len(high_quality_indices)} high-quality samples")

    # Step 3: Create improved dataset
    print("\nStep 3: Creating improved dataset...")
    improved_samples = [samples[i] for i in high_quality_indices]

    # Step 4: Verify improvements
    print("\nStep 4: Verifying improvements...")
    improved_report = analyzer.analyze(improved_samples, dataset_name="improved")

    print(f"  New sample count: {improved_report.statistics.total_samples}")
    print(f"  New average quality: {improved_report.average_quality_score:.1%}")
    print(f"  Retention rate: {len(improved_samples)/len(samples):.1%}")

    print("\nComparison:")
    print(f"  Quality improvement: {improved_report.average_quality_score - initial_report.average_quality_score:+.1%}")


if __name__ == "__main__":
    # Run all examples
    example_basic_analysis()
    example_bias_analysis()
    example_issue_detection()
    example_comprehensive_workflow()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
