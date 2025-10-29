"""Module for analyzing bias in code datasets."""

import numpy as np
from scipy import stats
from typing import Dict, List, Any

def analyze_language_distribution(dataset: Dict[str, int]) -> Dict[str, float]:
    """Analyze the distribution of programming languages in the dataset."""
    total = sum(dataset.values())
    return {lang: count/total for lang, count in dataset.items()}

def analyze_complexity_distribution(complexities: Dict[str, float]) -> Dict[str, Any]:
    """Analyze the distribution of code complexity scores."""
    values = list(complexities.values())
    return {
        "mean": np.mean(values),
        "std_dev": np.std(values),
        "percentiles": np.percentile(values, [25, 50, 75]).tolist()
    }

def analyze_license_distribution(licenses: Dict[str, int]) -> Dict[str, float]:
    """Analyze the distribution of license types."""
    total = sum(licenses.values())
    return {license_type: count/total for license_type, count in licenses.items()}

def calculate_fairness_metrics(dataset_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Calculate fairness metrics for the dataset."""
    metrics = {
        "demographic_parity": _calculate_demographic_parity(dataset_metrics),
        "equal_opportunity": _calculate_equal_opportunity(dataset_metrics)
    }
    return metrics

def _calculate_demographic_parity(metrics: Dict[str, Any]) -> float:
    """Calculate demographic parity across language distributions."""
    lang_counts = metrics["language_counts"]
    total = sum(lang_counts.values())
    expected_share = 1.0 / len(lang_counts)
    actual_shares = [count/total for count in lang_counts.values()]
    return 1.0 - np.mean([abs(share - expected_share) for share in actual_shares])

def _calculate_equal_opportunity(metrics: Dict[str, Any]) -> float:
    """Calculate equal opportunity metric based on complexity scores."""
    complexity_scores = metrics["complexity_scores"]
    means = [scores["mean"] for scores in complexity_scores.values()]
    return 1.0 - np.std(means) / np.mean(means)