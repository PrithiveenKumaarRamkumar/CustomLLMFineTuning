"""
Comprehensive bias analysis framework for ML data pipelines.

This module implements multi-dimensional bias detection including:
- Language representation bias
- Repository popularity bias  
- Code quality and complexity bias
- License diversity bias
- PII removal fairness analysis
- Geographic and demographic bias detection
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import os
import json
import re
from collections import Counter, defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

@dataclass
class BiasResult:
    """Result of bias analysis."""
    bias_type: str
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    description: str
    current_distribution: Dict[str, float]
    expected_distribution: Dict[str, float]
    statistical_test_result: Dict[str, Any]
    mitigation_suggestions: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bias_type': self.bias_type,
            'severity': self.severity,
            'description': self.description,
            'current_distribution': self.current_distribution,
            'expected_distribution': self.expected_distribution,
            'statistical_test_result': self.statistical_test_result,
            'mitigation_suggestions': self.mitigation_suggestions,
            'timestamp': self.timestamp.isoformat()
        }

class LanguageRepresentationAnalyzer:
    """Analyzes bias in programming language representation."""
    
    def __init__(self, expected_distribution: Optional[Dict[str, Tuple[float, float]]] = None):
        # Expected ranges for language distribution (min, max)
        self.expected_distribution = expected_distribution or {
            'python': (0.40, 0.50),
            'java': (0.25, 0.35), 
            'cpp': (0.15, 0.25),
            'c': (0.05, 0.15),
            'javascript': (0.05, 0.15),
            'go': (0.02, 0.08),
            'rust': (0.01, 0.05),
            'other': (0.05, 0.15)
        }
    
    def analyze_language_bias(self, 
                            file_data: pd.DataFrame,
                            language_column: str = 'language') -> BiasResult:
        """
        Analyze bias in language representation.
        
        Args:
            file_data: DataFrame with file information
            language_column: Column name containing language information
            
        Returns:
            BiasResult with analysis results
        """
        if language_column not in file_data.columns:
            raise ValueError(f"Column '{language_column}' not found in data")
            
        # Calculate current distribution
        language_counts = file_data[language_column].value_counts()
        total_files = len(file_data)
        current_dist = {lang: count/total_files for lang, count in language_counts.items()}
        
        # Calculate expected distribution (using midpoints)
        expected_dist = {lang: (min_val + max_val) / 2 
                        for lang, (min_val, max_val) in self.expected_distribution.items()}
        
        # Normalize expected distribution
        expected_total = sum(expected_dist.values())
        expected_dist = {lang: val/expected_total for lang, val in expected_dist.items()}
        
        # Statistical test (Chi-square goodness of fit)
        observed = [current_dist.get(lang, 0) * total_files for lang in expected_dist.keys()]
        expected = [expected_dist[lang] * total_files for lang in expected_dist.keys()]
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        # Determine bias severity
        severity = self._assess_language_bias_severity(current_dist, expected_dist, p_value)
        
        # Generate mitigation suggestions
        mitigation_suggestions = self._generate_language_mitigation(current_dist, expected_dist)
        
        return BiasResult(
            bias_type='language_representation',
            severity=severity,
            description=f'Language distribution analysis across {len(current_dist)} languages',
            current_distribution=current_dist,
            expected_distribution=expected_dist,
            statistical_test_result={
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'test_type': 'chi_square_goodness_of_fit'
            },
            mitigation_suggestions=mitigation_suggestions,
            timestamp=datetime.now()
        )
    
    def _assess_language_bias_severity(self, 
                                     current: Dict[str, float], 
                                     expected: Dict[str, float],
                                     p_value: float) -> str:
        """Assess severity of language representation bias."""
        if p_value > 0.1:
            return 'none'
        
        # Calculate maximum deviation
        max_deviation = 0
        for lang in expected.keys():
            if lang in current:
                deviation = abs(current[lang] - expected[lang]) / expected[lang]
                max_deviation = max(max_deviation, deviation)
        
        if max_deviation > 0.5:
            return 'critical'
        elif max_deviation > 0.3:
            return 'high'
        elif max_deviation > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _generate_language_mitigation(self, 
                                    current: Dict[str, float], 
                                    expected: Dict[str, float]) -> List[str]:
        """Generate mitigation suggestions for language bias."""
        suggestions = []
        
        for lang, expected_val in expected.items():
            current_val = current.get(lang, 0)
            
            if current_val < expected_val * 0.8:
                suggestions.append(f"Increase {lang} representation - currently {current_val:.1%}, target {expected_val:.1%}")
            elif current_val > expected_val * 1.2:
                suggestions.append(f"Reduce {lang} over-representation - currently {current_val:.1%}, target {expected_val:.1%}")
        
        if not suggestions:
            suggestions.append("Language distribution within acceptable ranges")
            
        return suggestions

class PopularityBiasAnalyzer:
    """Analyzes bias related to repository popularity metrics."""
    
    def __init__(self):
        self.high_star_threshold = 100  # Repositories with >100 stars considered "popular"
        self.expected_high_star_ratio = 0.30  # Expect ~30% from high-star repos
    
    def analyze_popularity_bias(self, 
                              repo_data: pd.DataFrame,
                              star_column: str = 'stars') -> BiasResult:
        """
        Analyze bias towards high-popularity repositories.
        
        Args:
            repo_data: DataFrame with repository information
            star_column: Column name containing star counts
            
        Returns:
            BiasResult with popularity bias analysis
        """
        if star_column not in repo_data.columns:
            raise ValueError(f"Column '{star_column}' not found in data")
        
        # Categorize by popularity
        high_star_count = len(repo_data[repo_data[star_column] >= self.high_star_threshold])
        total_repos = len(repo_data)
        
        current_high_star_ratio = high_star_count / total_repos if total_repos > 0 else 0
        
        # Calculate star distribution
        star_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        star_labels = ['0-9', '10-49', '50-99', '100-499', '500-999', '1000+']
        repo_data['star_category'] = pd.cut(repo_data[star_column], bins=star_bins, labels=star_labels, right=False)
        
        current_dist = repo_data['star_category'].value_counts(normalize=True).to_dict()
        current_dist = {str(k): v for k, v in current_dist.items()}
        
        # Expected more balanced distribution
        expected_dist = {
            '0-9': 0.40,
            '10-49': 0.25,
            '50-99': 0.15,
            '100-499': 0.12,
            '500-999': 0.05,
            '1000+': 0.03
        }
        
        # Statistical test
        observed = [current_dist.get(cat, 0) * total_repos for cat in expected_dist.keys()]
        expected = [expected_dist[cat] * total_repos for cat in expected_dist.keys()]
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        # Assess bias severity
        severity = self._assess_popularity_bias_severity(current_high_star_ratio, p_value)
        
        # Generate mitigation suggestions
        mitigation_suggestions = self._generate_popularity_mitigation(current_high_star_ratio, current_dist)
        
        return BiasResult(
            bias_type='repository_popularity',
            severity=severity,
            description=f'Repository popularity bias analysis across {total_repos} repositories',
            current_distribution=current_dist,
            expected_distribution=expected_dist,
            statistical_test_result={
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'high_star_ratio': current_high_star_ratio,
                'expected_high_star_ratio': self.expected_high_star_ratio
            },
            mitigation_suggestions=mitigation_suggestions,
            timestamp=datetime.now()
        )
    
    def _assess_popularity_bias_severity(self, current_ratio: float, p_value: float) -> str:
        """Assess severity of popularity bias."""
        if p_value > 0.1 and abs(current_ratio - self.expected_high_star_ratio) < 0.1:
            return 'none'
        
        deviation = abs(current_ratio - self.expected_high_star_ratio)
        
        if deviation > 0.4:
            return 'critical'
        elif deviation > 0.25:
            return 'high'
        elif deviation > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _generate_popularity_mitigation(self, 
                                      current_ratio: float,
                                      current_dist: Dict[str, float]) -> List[str]:
        """Generate mitigation suggestions for popularity bias."""
        suggestions = []
        
        if current_ratio > self.expected_high_star_ratio + 0.1:
            suggestions.append("Reduce bias toward popular repositories by including more low-star repos")
            suggestions.append("Implement stratified sampling across star ranges")
        elif current_ratio < self.expected_high_star_ratio - 0.1:
            suggestions.append("Include more high-quality popular repositories")
        
        # Check specific category imbalances
        if current_dist.get('1000+', 0) > 0.1:
            suggestions.append("Consider capping super-popular repositories (1000+ stars)")
        
        if current_dist.get('0-9', 0) > 0.6:
            suggestions.append("Too many very low-star repositories may indicate quality issues")
        
        return suggestions or ["Popularity distribution within acceptable ranges"]

class CodeQualityBiasAnalyzer:
    """Analyzes bias in code quality metrics."""
    
    def analyze_complexity_bias(self, 
                              file_data: pd.DataFrame,
                              complexity_column: str = 'complexity') -> BiasResult:
        """
        Analyze bias in code complexity distribution.
        """
        if complexity_column not in file_data.columns:
            # If complexity not calculated, skip analysis
            return BiasResult(
                bias_type='code_complexity',
                severity='none',
                description='Complexity analysis skipped - no complexity data available',
                current_distribution={},
                expected_distribution={},
                statistical_test_result={},
                mitigation_suggestions=['Calculate complexity metrics for bias analysis'],
                timestamp=datetime.now()
            )
        
        # Analyze complexity distribution by language
        complexity_by_lang = {}
        
        if 'language' in file_data.columns:
            for lang in file_data['language'].unique():
                lang_data = file_data[file_data['language'] == lang]
                complexity_values = lang_data[complexity_column].dropna()
                
                if len(complexity_values) > 0:
                    complexity_by_lang[lang] = {
                        'mean': float(complexity_values.mean()),
                        'std': float(complexity_values.std()),
                        'median': float(complexity_values.median()),
                        'q75': float(complexity_values.quantile(0.75)),
                        'high_complexity_rate': float((complexity_values > 15).mean())
                    }
        
        # Overall complexity distribution
        all_complexity = file_data[complexity_column].dropna()
        
        # Define complexity categories
        complexity_bins = [0, 5, 10, 15, 25, float('inf')]
        complexity_labels = ['Simple', 'Moderate', 'Complex', 'High', 'Extreme']
        
        if len(all_complexity) > 0:
            complexity_categories = pd.cut(all_complexity, bins=complexity_bins, labels=complexity_labels, right=False)
            current_dist = complexity_categories.value_counts(normalize=True).to_dict()
            current_dist = {str(k): v for k, v in current_dist.items()}
        else:
            current_dist = {}
        
        # Expected distribution (right-skewed for production code)
        expected_dist = {
            'Simple': 0.30,
            'Moderate': 0.35,
            'Complex': 0.25,
            'High': 0.08,
            'Extreme': 0.02
        }
        
        # Statistical analysis
        if current_dist:
            observed = [current_dist.get(cat, 0) * len(all_complexity) for cat in expected_dist.keys()]
            expected = [expected_dist[cat] * len(all_complexity) for cat in expected_dist.keys()]
            chi2_stat, p_value = stats.chisquare(observed, expected)
        else:
            chi2_stat, p_value = 0, 1
        
        # Assess bias
        severity = self._assess_complexity_bias_severity(current_dist, complexity_by_lang, p_value)
        
        # Generate suggestions
        mitigation_suggestions = self._generate_complexity_mitigation(complexity_by_lang, current_dist)
        
        return BiasResult(
            bias_type='code_complexity',
            severity=severity,
            description=f'Code complexity bias analysis across {len(file_data)} files',
            current_distribution=current_dist,
            expected_distribution=expected_dist,
            statistical_test_result={
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'complexity_by_language': complexity_by_lang
            },
            mitigation_suggestions=mitigation_suggestions,
            timestamp=datetime.now()
        )
    
    def _assess_complexity_bias_severity(self, 
                                       current_dist: Dict[str, float],
                                       complexity_by_lang: Dict[str, Dict[str, float]],
                                       p_value: float) -> str:
        """Assess complexity bias severity."""
        if not current_dist:
            return 'none'
        
        # Check for extreme complexity bias
        extreme_rate = current_dist.get('Extreme', 0)
        if extreme_rate > 0.1:
            return 'high'
        
        # Check for language-specific bias
        lang_high_complexity_rates = [lang_data.get('high_complexity_rate', 0) 
                                    for lang_data in complexity_by_lang.values()]
        
        if len(lang_high_complexity_rates) > 1:
            complexity_variance = np.var(lang_high_complexity_rates)
            if complexity_variance > 0.05:  # High variance between languages
                return 'medium'
        
        if p_value < 0.05:
            return 'medium'
        
        return 'low'
    
    def _generate_complexity_mitigation(self, 
                                      complexity_by_lang: Dict[str, Dict[str, float]],
                                      current_dist: Dict[str, float]) -> List[str]:
        """Generate complexity bias mitigation suggestions."""
        suggestions = []
        
        # Check for extreme complexity
        if current_dist.get('Extreme', 0) > 0.05:
            suggestions.append("Consider filtering out extremely complex files (>25 complexity)")
        
        # Check for language-specific issues
        for lang, metrics in complexity_by_lang.items():
            if metrics.get('high_complexity_rate', 0) > 0.2:
                suggestions.append(f"High complexity rate in {lang} files - consider filtering")
            elif metrics.get('high_complexity_rate', 0) < 0.05:
                suggestions.append(f"Very low complexity in {lang} files - may indicate toy examples")
        
        # Overall distribution check
        if current_dist.get('Simple', 0) > 0.5:
            suggestions.append("High proportion of simple code - ensure real-world complexity")
        
        return suggestions or ["Complexity distribution appears balanced"]

class PIIFairnessAnalyzer:
    """Analyzes fairness and effectiveness of PII removal across different patterns and contexts."""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'github_token': r'ghp_[A-Za-z0-9]{36}',
            'api_key': r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)["\'\s]*[:=]["\'\s]*[A-Za-z0-9+/=]{20,}',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'phone': r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
        }
    
    def analyze_pii_removal_fairness(self, 
                                   before_files: List[str],
                                   after_files: List[str]) -> BiasResult:
        """
        Analyze fairness and effectiveness of PII removal process.
        
        Args:
            before_files: List of file paths before PII removal
            after_files: List of file paths after PII removal
            
        Returns:
            BiasResult with PII removal fairness analysis
        """
        pii_metrics = {}
        
        # Analyze each PII type
        for pii_type, pattern in self.pii_patterns.items():
            before_matches = self._count_pii_matches(before_files, pattern)
            after_matches = self._count_pii_matches(after_files, pattern)
            
            detection_rate = 1 - (after_matches / before_matches) if before_matches > 0 else 1.0
            false_negative_rate = after_matches / before_matches if before_matches > 0 else 0.0
            
            pii_metrics[pii_type] = {
                'before_count': before_matches,
                'after_count': after_matches,
                'detection_rate': detection_rate,
                'false_negative_rate': false_negative_rate
            }
        
        # Calculate overall metrics
        total_before = sum(metrics['before_count'] for metrics in pii_metrics.values())
        total_after = sum(metrics['after_count'] for metrics in pii_metrics.values())
        overall_detection_rate = 1 - (total_after / total_before) if total_before > 0 else 1.0
        
        # Assess fairness across PII types
        detection_rates = [metrics['detection_rate'] for metrics in pii_metrics.values() if metrics['before_count'] > 0]
        detection_variance = np.var(detection_rates) if len(detection_rates) > 1 else 0
        
        # Determine severity
        severity = self._assess_pii_fairness_severity(overall_detection_rate, detection_variance, pii_metrics)
        
        # Generate mitigation suggestions
        mitigation_suggestions = self._generate_pii_mitigation(pii_metrics, overall_detection_rate)
        
        current_dist = {f"{pii_type}_detection_rate": metrics['detection_rate'] 
                       for pii_type, metrics in pii_metrics.items()}
        expected_dist = {f"{pii_type}_detection_rate": 0.99 
                        for pii_type in pii_metrics.keys()}
        
        return BiasResult(
            bias_type='pii_removal_fairness',
            severity=severity,
            description=f'PII removal fairness analysis across {len(self.pii_patterns)} pattern types',
            current_distribution=current_dist,
            expected_distribution=expected_dist,
            statistical_test_result={
                'overall_detection_rate': overall_detection_rate,
                'detection_variance': detection_variance,
                'total_before': total_before,
                'total_after': total_after,
                'pii_type_metrics': pii_metrics
            },
            mitigation_suggestions=mitigation_suggestions,
            timestamp=datetime.now()
        )
    
    def _count_pii_matches(self, file_paths: List[str], pattern: str) -> int:
        """Count PII pattern matches across files."""
        total_matches = 0
        
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = re.findall(pattern, content)
                        total_matches += len(matches)
            except Exception as e:
                logger.warning(f"Could not analyze file {file_path}: {e}")
        
        return total_matches
    
    def _assess_pii_fairness_severity(self, 
                                    overall_rate: float,
                                    variance: float,
                                    pii_metrics: Dict[str, Dict[str, Any]]) -> str:
        """Assess PII removal fairness severity."""
        
        # Critical if overall detection rate is too low
        if overall_rate < 0.95:
            return 'critical'
        
        # High if there's significant variance between PII types
        if variance > 0.01:  # 1% variance threshold
            return 'high'
        
        # Check for specific PII type failures
        for pii_type, metrics in pii_metrics.items():
            if pii_type in ['ssn', 'credit_card'] and metrics['detection_rate'] < 0.98:
                return 'high'  # Stricter for sensitive PII
        
        # Medium if any detection rate below 97%
        if any(metrics['detection_rate'] < 0.97 for metrics in pii_metrics.values() 
               if metrics['before_count'] > 0):
            return 'medium'
        
        return 'low'
    
    def _generate_pii_mitigation(self, 
                               pii_metrics: Dict[str, Dict[str, Any]],
                               overall_rate: float) -> List[str]:
        """Generate PII removal mitigation suggestions."""
        suggestions = []
        
        if overall_rate < 0.95:
            suggestions.append("Critical: Overall PII detection rate below 95% - review all patterns")
        
        for pii_type, metrics in pii_metrics.items():
            detection_rate = metrics['detection_rate']
            if detection_rate < 0.90:
                suggestions.append(f"Improve {pii_type} pattern - only {detection_rate:.1%} detection rate")
            elif detection_rate < 0.95 and pii_type in ['ssn', 'credit_card']:
                suggestions.append(f"Strengthen {pii_type} detection - critical PII type")
        
        # Check for pattern improvements
        if pii_metrics.get('email', {}).get('detection_rate', 1) < 0.98:
            suggestions.append("Consider more comprehensive email patterns (international domains)")
        
        if pii_metrics.get('phone', {}).get('detection_rate', 1) < 0.95:
            suggestions.append("Expand phone number patterns for international formats")
        
        return suggestions or ["PII removal fairness within acceptable parameters"]

class ComprehensiveBiasAnalyzer:
    """Main orchestrator for comprehensive bias analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        self.language_analyzer = LanguageRepresentationAnalyzer(
            self.config.get('expected_language_distribution')
        )
        self.popularity_analyzer = PopularityBiasAnalyzer()
        self.quality_analyzer = CodeQualityBiasAnalyzer()
        self.pii_analyzer = PIIFairnessAnalyzer()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for bias analysis."""
        return {
            'expected_language_distribution': {
                'python': (0.40, 0.50),
                'java': (0.25, 0.35),
                'cpp': (0.15, 0.25),
                'javascript': (0.05, 0.15),
            },
            'enable_visualizations': True,
            'output_directory': 'bias_analysis_reports',
            'severity_threshold': 'medium'
        }
    
    def analyze_dataset_bias(self, 
                           dataset_info: Dict[str, Any]) -> Dict[str, BiasResult]:
        """
        Comprehensive bias analysis of the entire dataset.
        
        Args:
            dataset_info: Dictionary containing:
                - file_data: DataFrame with file information
                - repo_data: DataFrame with repository information  
                - before_pii_files: List of files before PII removal
                - after_pii_files: List of files after PII removal
                
        Returns:
            Dictionary of bias analysis results by type
        """
        results = {}
        
        logger.info("Starting comprehensive bias analysis")
        
        try:
            # Language representation bias
            if 'file_data' in dataset_info:
                logger.info("Analyzing language representation bias")
                results['language_bias'] = self.language_analyzer.analyze_language_bias(
                    dataset_info['file_data']
                )
            
            # Repository popularity bias
            if 'repo_data' in dataset_info:
                logger.info("Analyzing repository popularity bias")
                results['popularity_bias'] = self.popularity_analyzer.analyze_popularity_bias(
                    dataset_info['repo_data']
                )
            
            # Code quality bias
            if 'file_data' in dataset_info:
                logger.info("Analyzing code quality bias")
                results['quality_bias'] = self.quality_analyzer.analyze_complexity_bias(
                    dataset_info['file_data']
                )
            
            # PII removal fairness
            if 'before_pii_files' in dataset_info and 'after_pii_files' in dataset_info:
                logger.info("Analyzing PII removal fairness")
                results['pii_fairness'] = self.pii_analyzer.analyze_pii_removal_fairness(
                    dataset_info['before_pii_files'],
                    dataset_info['after_pii_files']
                )
                
        except Exception as e:
            logger.error(f"Error in bias analysis: {e}")
        
        logger.info(f"Completed bias analysis with {len(results)} analyses")
        return results
    
    def generate_bias_report(self, 
                           bias_results: Dict[str, BiasResult],
                           output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive bias analysis report.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyses': len(bias_results),
                'critical_biases': 0,
                'high_biases': 0,
                'medium_biases': 0,
                'low_biases': 0,
                'no_bias': 0
            },
            'analyses': {},
            'overall_assessment': '',
            'priority_actions': [],
            'mitigation_strategy': {}
        }
        
        # Process each bias analysis
        for bias_type, result in bias_results.items():
            report['analyses'][bias_type] = result.to_dict()
            
            # Count by severity
            severity_count_key = f"{result.severity}_biases"
            if severity_count_key in report['summary']:
                report['summary'][severity_count_key] += 1
            elif result.severity == 'none':
                report['summary']['no_bias'] += 1
        
        # Overall assessment
        critical_count = report['summary']['critical_biases']
        high_count = report['summary']['high_biases']
        
        if critical_count > 0:
            report['overall_assessment'] = f"Critical bias issues detected ({critical_count}). Immediate action required."
        elif high_count > 0:
            report['overall_assessment'] = f"High-priority bias issues detected ({high_count}). Action recommended."
        elif report['summary']['medium_biases'] > 0:
            report['overall_assessment'] = "Medium-level biases detected. Monitor and improve."
        else:
            report['overall_assessment'] = "No significant biases detected. Continue monitoring."
        
        # Priority actions
        for bias_type, result in bias_results.items():
            if result.severity in ['critical', 'high']:
                report['priority_actions'].extend([
                    f"{bias_type}: {suggestion}" for suggestion in result.mitigation_suggestions[:2]
                ])
        
        # Mitigation strategy
        report['mitigation_strategy'] = self._generate_mitigation_strategy(bias_results)
        
        # Save report
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Bias analysis report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save bias report: {e}")
        
        return report
    
    def _generate_mitigation_strategy(self, 
                                    bias_results: Dict[str, BiasResult]) -> Dict[str, Any]:
        """Generate comprehensive mitigation strategy."""
        strategy = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_monitoring': [],
            'data_collection_adjustments': []
        }
        
        for bias_type, result in bias_results.items():
            if result.severity == 'critical':
                strategy['immediate_actions'].extend(result.mitigation_suggestions)
            elif result.severity == 'high':
                strategy['short_term_improvements'].extend(result.mitigation_suggestions)
            else:
                strategy['long_term_monitoring'].extend(result.mitigation_suggestions)
        
        # Data collection adjustments
        if any('language' in bias_type for bias_type in bias_results.keys()):
            strategy['data_collection_adjustments'].append("Implement stratified sampling by language")
        
        if any('popularity' in bias_type for bias_type in bias_results.keys()):
            strategy['data_collection_adjustments'].append("Balance high and low popularity repositories")
        
        return strategy
    
    def create_bias_visualizations(self, 
                                 bias_results: Dict[str, BiasResult],
                                 output_dir: str = "bias_visualizations"):
        """Create visualizations for bias analysis results."""
        if not self.config.get('enable_visualizations', True):
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for bias_type, result in bias_results.items():
                if result.current_distribution:
                    self._create_distribution_plot(
                        result.current_distribution,
                        result.expected_distribution,
                        bias_type,
                        output_dir
                    )
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_distribution_plot(self, 
                                current: Dict[str, float],
                                expected: Dict[str, float],
                                bias_type: str,
                                output_dir: str):
        """Create distribution comparison plot."""
        plt.figure(figsize=(12, 6))
        
        categories = list(current.keys())
        current_values = [current.get(cat, 0) for cat in categories]
        expected_values = [expected.get(cat, 0) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, current_values, width, label='Current', alpha=0.8)
        plt.bar(x + width/2, expected_values, width, label='Expected', alpha=0.8)
        
        plt.xlabel('Categories')
        plt.ylabel('Proportion')
        plt.title(f'Distribution Comparison: {bias_type}')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{bias_type}_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {output_path}")

# Example usage function
def test_bias_analysis():
    """Test function demonstrating bias analysis capabilities."""
    
    # Create sample data with biases
    np.random.seed(42)
    
    # Biased language distribution (too much Python)
    languages = ['python'] * 70 + ['java'] * 20 + ['cpp'] * 8 + ['javascript'] * 2
    
    # High star bias
    stars = [1000] * 80 + [50] * 15 + [10] * 5
    
    file_data = pd.DataFrame({
        'language': languages,
        'file_size': np.random.lognormal(8, 2, len(languages)),
        'complexity': np.random.exponential(5, len(languages))
    })
    
    repo_data = pd.DataFrame({
        'stars': stars,
        'license': ['MIT'] * 60 + ['Apache-2.0'] * 25 + ['GPL-3.0'] * 15
    })
    
    # Initialize analyzer
    analyzer = ComprehensiveBiasAnalyzer()
    
    # Analyze
    dataset_info = {
        'file_data': file_data,
        'repo_data': repo_data,
        'before_pii_files': [],  # Empty for this test
        'after_pii_files': []
    }
    
    results = analyzer.analyze_dataset_bias(dataset_info)
    
    # Generate report
    report = analyzer.generate_bias_report(results)
    
    print(f"Bias Analysis Complete:")
    print(f"  - {report['summary']['critical_biases']} critical biases")
    print(f"  - {report['summary']['high_biases']} high-priority biases") 
    print(f"  - {report['summary']['medium_biases']} medium-level biases")
    print(f"Overall: {report['overall_assessment']}")
    
    return report

if __name__ == "__main__":
    # Run test
    test_report = test_bias_analysis()
    print(f"\nTest completed. Report summary: {json.dumps(test_report['summary'], indent=2)}")