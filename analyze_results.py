#!/usr/bin/env python3
"""
Results Analysis Script for PRASE-Python Experiments

This script analyzes experiment results from multiple runs and computes
mean and standard deviation for key metrics.

Usage:
    python analyze_results.py <experiment_results_directory>

Example:
    python analyze_results.py experiment_results_20240117_153045/
"""

import os
import re
import sys
import json
from collections import defaultdict
from pathlib import Path
import statistics


def parse_log_file(log_path):
    """
    Parse a log file and extract performance metrics.

    Returns:
        dict: Metrics extracted from the log file
    """
    metrics = {
        "hits_at_1": None,
        "hits_at_10": None,
        "mrr": None,
        "precision": None,
        "recall": None,
    }

    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Look for common metric patterns
        # Adjust these regex patterns based on actual output format

        # Example patterns (modify based on actual output):
        hits1_match = re.search(r"Hits@1[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if hits1_match:
            metrics["hits_at_1"] = float(hits1_match.group(1))

        hits10_match = re.search(r"Hits@10[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if hits10_match:
            metrics["hits_at_10"] = float(hits10_match.group(1))

        mrr_match = re.search(r"MRR[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if mrr_match:
            metrics["mrr"] = float(mrr_match.group(1))

        precision_match = re.search(r"Precision[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if precision_match:
            metrics["precision"] = float(precision_match.group(1))

        recall_match = re.search(r"Recall[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if recall_match:
            metrics["recall"] = float(recall_match.group(1))

    return metrics


def analyze_results(results_dir):
    """
    Analyze all experiment results in the given directory.

    Args:
        results_dir: Path to experiment results directory

    Returns:
        dict: Organized results by dataset
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return None

    # Find all log files
    log_files = list(results_path.glob("*.log"))

    if not log_files:
        print(f"Warning: No log files found in {results_dir}")
        return None

    # Organize results by dataset
    dataset_results = defaultdict(list)

    for log_file in log_files:
        # Extract dataset name and run number from filename
        # Format: {dataset_name}_run{N}.log
        match = re.match(r"(.+)_run(\d+)\.log", log_file.name)
        if match:
            dataset_name = match.group(1)
            run_number = int(match.group(2))

            metrics = parse_log_file(log_file)
            dataset_results[dataset_name].append(
                {"run": run_number, "metrics": metrics, "log_file": str(log_file)}
            )

    return dict(dataset_results)


def compute_statistics(values):
    """
    Compute mean and standard deviation for a list of values.

    Args:
        values: List of numeric values

    Returns:
        tuple: (mean, std_dev) or (None, None) if insufficient data
    """
    # Filter out None values
    valid_values = [v for v in values if v is not None]

    if len(valid_values) == 0:
        return None, None
    elif len(valid_values) == 1:
        return valid_values[0], 0.0
    else:
        mean = statistics.mean(valid_values)
        std_dev = statistics.stdev(valid_values)
        return mean, std_dev


def print_results(dataset_results):
    """
    Print formatted results with statistics.

    Args:
        dataset_results: Dictionary of results organized by dataset
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80 + "\n")

    for dataset_name, runs in sorted(dataset_results.items()):
        print(f"\nDataset: {dataset_name}")
        print("-" * 80)

        # Collect metrics across runs
        metrics_by_type = defaultdict(list)

        for run_data in runs:
            print(f"  Run {run_data['run']}: {run_data['log_file']}")
            for metric_name, value in run_data["metrics"].items():
                metrics_by_type[metric_name].append(value)

        print("\n  Statistics across runs:")
        print("  " + "-" * 76)

        # Compute and print statistics for each metric
        for metric_name, values in sorted(metrics_by_type.items()):
            mean, std = compute_statistics(values)

            if mean is not None:
                print(f"  {metric_name:20s}: {mean:.4f} ± {std:.4f}")
            else:
                print(f"  {metric_name:20s}: No data available")

        print()

    print("=" * 80)


def export_to_json(dataset_results, output_path):
    """
    Export results to JSON format.

    Args:
        dataset_results: Dictionary of results
        output_path: Path to save JSON file
    """
    # Compute statistics for export
    export_data = {}

    for dataset_name, runs in dataset_results.items():
        metrics_by_type = defaultdict(list)

        for run_data in runs:
            for metric_name, value in run_data["metrics"].items():
                metrics_by_type[metric_name].append(value)

        stats = {}
        for metric_name, values in metrics_by_type.items():
            mean, std = compute_statistics(values)
            stats[metric_name] = {
                "mean": mean,
                "std": std,
                "values": values,
                "count": len([v for v in values if v is not None]),
            }

        export_data[dataset_name] = {"runs": runs, "statistics": stats}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults exported to: {output_path}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <experiment_results_directory>")
        print("\nExample:")
        print("  python analyze_results.py experiment_results_20240117_153045/")
        sys.exit(1)

    results_dir = sys.argv[1]

    print(f"Analyzing results in: {results_dir}")

    # Analyze results
    dataset_results = analyze_results(results_dir)

    if dataset_results is None:
        sys.exit(1)

    # Print summary
    print_results(dataset_results)

    # Export to JSON
    json_output = os.path.join(results_dir, "analysis_summary.json")
    export_to_json(dataset_results, json_output)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
