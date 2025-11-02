import pytest
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.bias_analysis import (
    LanguageRepresentationAnalyzer,
    PopularityBiasAnalyzer,
    CodeQualityBiasAnalyzer,
    ComprehensiveBiasAnalyzer
)

class TestBiasAnalysis:
    """Test suite for bias analysis in the dataset."""
    
    def test_language_distribution(self, temp_workspace):
        """Test language distribution analysis."""
        # Create test dataset with known distribution
        dataset = {
            "python": 50,
            "java": 30,
            "cpp": 15,
            "javascript": 5
        }
        
        distribution = analyze_language_distribution(dataset)
        
        # Check for expected properties
        assert 0.4 <= distribution["python"] <= 0.6  # Python should be 40-60%
        assert 0.2 <= distribution["java"] <= 0.4    # Java should be 20-40%
        assert sum(distribution.values()) == 1.0     # Should sum to 100%

    def test_complexity_bias(self, temp_workspace):
        """Test code complexity distribution analysis."""
        # Generate synthetic complexity scores
        complexities = {
            "file1.py": 5,   # Low complexity
            "file2.py": 10,  # Medium complexity
            "file3.py": 15,  # High complexity
            "file4.py": 8    # Medium complexity
        }
        
        distribution = analyze_complexity_distribution(complexities)
        
        assert "mean" in distribution
        assert "std_dev" in distribution
        assert "percentiles" in distribution
        assert len(distribution["percentiles"]) == 3  # 25th, 50th, 75th

    def test_license_fairness(self, temp_workspace):
        """Test license distribution fairness."""
        licenses = {
            "MIT": 60,
            "Apache-2.0": 20,
            "GPL-3.0": 15,
            "BSD-3-Clause": 5
        }
        
        distribution = analyze_license_distribution(licenses)
        
        # Check for license diversity
        assert len(distribution) >= 3  # At least 3 different licenses
        assert distribution["MIT"] < 0.7  # MIT should not dominate completely

    def test_fairness_metrics(self, temp_workspace):
        """Test fairness metrics calculation."""
        dataset_metrics = {
            "language_counts": {
                "python": 100,
                "java": 80,
                "cpp": 60,
                "javascript": 40
            },
            "complexity_scores": {
                "python": {"mean": 8, "std": 2},
                "java": {"mean": 10, "std": 3},
                "cpp": {"mean": 12, "std": 4},
                "javascript": {"mean": 7, "std": 2}
            }
        }
        
        fairness_scores = calculate_fairness_metrics(dataset_metrics)
        
        # Check demographic parity
        assert "demographic_parity" in fairness_scores
        assert 0 <= fairness_scores["demographic_parity"] <= 1
        
        # Check equal opportunity
        assert "equal_opportunity" in fairness_scores
        assert 0 <= fairness_scores["equal_opportunity"] <= 1

    def test_complexity_thresholds(self, test_config):
        """Test complexity threshold fairness across languages."""
        complexities = {
            "python_file.py": 12,
            "java_file.java": 14,
            "cpp_file.cpp": 13,
            "js_file.js": 11
        }
        
        distribution = analyze_complexity_distribution(complexities)
        
        # Check if any language is unfairly penalized
        assert distribution["mean"] <= test_config["thresholds"]["complexity_threshold"]
        assert distribution["std_dev"] < test_config["thresholds"]["complexity_threshold"] / 2

    def test_statistical_significance(self):
        """Test statistical significance of bias metrics."""
        # Generate two distributions
        dist1 = np.random.normal(10, 2, 1000)  # Distribution for one language
        dist2 = np.random.normal(11, 2, 1000)  # Distribution for another language
        
        from scipy import stats
        
        # Perform statistical test
        t_stat, p_value = stats.ttest_ind(dist1, dist2)
        
        # Check if the difference is statistically significant
        assert p_value < 0.05  # Assuming 5% significance level

    def test_bias_mitigation_impact(self, temp_workspace):
        """Test the impact of bias mitigation strategies."""
        # Original biased dataset
        original_metrics = {
            "language_distribution": {
                "python": 0.7,  # Heavily biased towards Python
                "java": 0.2,
                "other": 0.1
            }
        }
        
        # Simulated balanced dataset after mitigation
        balanced_metrics = {
            "language_distribution": {
                "python": 0.4,
                "java": 0.3,
                "other": 0.3
            }
        }
        
        # Calculate improvement
        improvement = {
            lang: balanced_metrics["language_distribution"][lang] - 
                  original_metrics["language_distribution"][lang]
            for lang in original_metrics["language_distribution"]
        }
        
        # Verify improvement in balance
        assert improvement["python"] < 0  # Python representation should decrease
        assert improvement["other"] > 0   # Other languages should increase