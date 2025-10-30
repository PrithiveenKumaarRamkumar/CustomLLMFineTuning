"""Module for detecting anomalies in code datasets."""

import numpy as np
from typing import Dict, List, Any, Optional
import chardet
import re

class AnomalyDetector:
    """Detector for various types of anomalies in code datasets."""

    def __init__(self, thresholds: Dict[str, Any]):
        """Initialize the anomaly detector with thresholds."""
        self.thresholds = thresholds
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'api_key': r'(?i)(api[_-]?key|access[_-]?token)["\']?\s*[=:]\s*["\']?([a-zA-Z0-9]{32,})["\']?',
            'password': r'(?i)(password|passwd|pwd)["\']?\s*[=:]\s*["\']?([^"\'\s]+)["\']?'
        }
        
    def analyze_single_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single file for anomalies.
        
        Args:
            file_data: Dictionary containing processed file data
            
        Returns:
            Dictionary containing anomaly detection results
        """
        if not file_data:
            return {
                "file_size": [],
                "complexity": {"score": 0, "metrics": {}},
                "pii_detected": []
            }
            
        # Process anomalies for a single file
        return {
            "file_size": self._detect_size_anomalies([file_data]),
            "complexity": self._analyze_complexity([file_data]),
            "pii_detected": self._detect_pii([file_data])
        }
        
        content = file_data["content"]
        metadata = file_data["metadata"]
        
        anomalies = {
            "has_anomalies": False,
            "size_anomaly": len(content) > 10_000_000,
            "pii_found": False,
            "stats": {
                "size": len(content),
                "cleaned_ratio": metadata["cleaned_size"] / metadata["original_size"]
            }
        }
        
        # Check for PII
        pii_results = self.detect_pii_patterns(content)
        if any(pii_results.get("detected_patterns", [])):
            anomalies["has_anomalies"] = True
            anomalies["pii_found"] = True
            
        return anomalies

    def detect_file_size_anomalies(self, sizes: List[int]) -> List[int]:
        """Detect anomalies in file sizes."""
        threshold = 10_000_000  # 10MB threshold
        anomalies = []
        for i, size in enumerate(sizes):
            if size > threshold:
                anomalies.append(i)
        return anomalies
        
    def detect_complexity_anomalies(self, complexities: Dict[str, int]) -> List[str]:
        """Detect anomalies in code complexity."""
        threshold = 15  # Complexity threshold
        return [filename for filename, complexity in complexities.items() if complexity > threshold]
        
    def detect_duplicates(self, files: Dict[str, str]) -> List[tuple]:
        """Detect duplicate code content."""
        duplicates = []
        seen = {}
        for filename, content in files.items():
            if content in seen:
                duplicates.append((seen[content], filename))
            else:
                seen[content] = filename
        return duplicates
        
    def detect_statistical_anomalies(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """Detect statistical anomalies using z-score."""
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        # Get unique indices where z-score exceeds threshold
        anomaly_indices = list(set(np.where(z_scores > threshold)[0]))
        return sorted(anomaly_indices)  # Sort for consistent order
        
    def detect_pii_patterns(self, content: str) -> Dict[str, Any]:
        """Detect potential PII in code content."""
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'api_key': r'(?i)(api[-_]?key|sk[-_]?(?:test|live)_[a-zA-Z0-9]+)',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        results = {
            "detected_patterns": [],
            "details": {},
            "risk_level": "low"
        }
        
        pattern_matches = 0
        for pattern_name, pattern in pii_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                results["detected_patterns"].append(pattern_name)
                results["details"][pattern_name] = matches
                pattern_matches += 1
        
        # Set risk level based on number of patterns found
        if pattern_matches >= 3:
            results["risk_level"] = "high"
        elif pattern_matches == 2:
            results["risk_level"] = "medium"
        
        return results
        
        # Phone number detection is added but not matched yet
        if re.search(pii_patterns['phone'], content):
            results["detected_patterns"].append('phone')
        if re.search(pii_patterns['api_key'], content):
            results["detected_patterns"].append('api_key')
        
        # Deduplicate patterns
        results["detected_patterns"] = list(set(results["detected_patterns"]))
        
        return results
        
    def analyze_language_bias(self, language_dist: Dict[str, int]) -> Dict[str, Any]:
        """Analyze bias in programming language distribution."""
        total = sum(language_dist.values())
        proportions = {lang: count/total for lang, count in language_dist.items()}
        mean_prop = 1.0 / len(language_dist)
        max_bias = max(abs(prop - mean_prop) for prop in proportions.values())
        bias_threshold = 0.5  # 50% deviation threshold for significant bias
        
        # Calculate recommendations for balancing
        recommendations = []
        ideal_count = total / len(language_dist)
        for lang, count in language_dist.items():
            diff = abs(count - ideal_count)
            if count < ideal_count * 0.5:  # Only recommend for significant imbalances
                recommendations.append(f"Increase {lang} samples by {int(diff)}")
            elif count > ideal_count * 1.5:
                recommendations.append(f"Decrease {lang} samples by {int(diff)}")
        
        return {
            'proportions': proportions,
            'bias_scores': {lang: abs(prop - mean_prop) for lang, prop in proportions.items()},
            'mean_proportion': mean_prop,
            'has_significant_bias': max_bias > bias_threshold,
            'recommendations': recommendations if max_bias > bias_threshold else []
        }
        
    def analyze_dataset(self, files: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze dataset for various types of anomalies."""
        return {
            "file_size_anomalies": self._detect_size_anomalies(files),
            "complexity_anomalies": self._detect_complexity_anomalies(files),
            "pii_detections": self._detect_pii(files),
            "encoding_issues": self._detect_encoding_issues(files)
        }

    def _detect_size_anomalies(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect files with anomalous sizes."""
        anomalies = []
        for file in files:
            # Handle case where file might be a tuple from PII removal
            if isinstance(file, tuple):
                content = file[0]
                file_info = {'file_path': 'unknown', 'content': content}
            else:
                file_info = file

            # Get size from metadata or content length
            size = 0
            if isinstance(file_info, dict):
                if 'metadata' in file_info and 'size' in file_info['metadata']:
                    size = file_info['metadata']['size']
                elif 'content' in file_info:
                    size = len(file_info['content'])
                elif isinstance(file_info.get('content'), str):
                    size = len(file_info['content'])
            else:
                size = len(str(file_info))
                
            if size > self.thresholds['file_size_max']:
                anomalies.append({
                    'file': file_info.get('file_path', 'unknown'),
                    'size': size,
                    'type': 'oversized'
                })
            elif size < 100:  # Minimum size threshold
                anomalies.append({
                    'file': file_info.get('file_path', 'unknown'),
                    'size': size,
                    'type': 'undersized'
                })
        return anomalies

    def _detect_complexity_anomalies(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect files with high code complexity."""
        anomalies = []
        for file in files:
            complexity = 0  # Placeholder for actual complexity calculation
            max_complexity = self.thresholds.get('complexity_max', 15)
            if complexity > max_complexity:
                anomalies.append({
                    'file': file['file_path'],
                    'complexity': complexity,
                    'type': 'high_complexity'
                })
        return anomalies

    def _detect_pii(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential PII in files."""
        detections = []
        for file in files:
            # Handle tuple case from PII removal
            if isinstance(file, tuple):
                content = file[0]
                file_info = {'path': 'unknown', 'content': content}
            else:
                file_info = file
                content = file_info.get('content', '')
            
            if not isinstance(content, str):
                continue
            
            pii_found = {}
            for pii_type, pattern in self.pii_patterns.items():
                try:
                    matches = re.findall(pattern, content)
                    if matches:
                        pii_found[pii_type] = len(matches)
                except TypeError:
                    continue
            
            if pii_found:
                detections.append({
                    'file': file['path'],
                    'detections': pii_found
                })
        return detections

    def _detect_encoding_issues(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect files with encoding issues."""
        issues = []
        for file in files:
            # Handle case where file might be a tuple from PII removal
            if isinstance(file, tuple):
                content = file[0]  # Get content from tuple
                file_info = {'path': 'unknown', 'content': content}
            else:
                file_info = file
                content = file_info.get('content', '')

            # Ensure content is in bytes
            if isinstance(content, str):
                content = content.encode('utf-8')
            elif not isinstance(content, (bytes, bytearray)):
                content = str(content).encode('utf-8')
            
            detected = chardet.detect(content)
            if detected['encoding'] != 'utf-8' or detected['confidence'] < 0.9:
                issues.append({
                    'file': file_info.get('path', 'unknown'),
                    'detected_encoding': detected['encoding'],
                    'confidence': detected['confidence']
                })
        return issues

    def _analyze_complexity(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        if not files:
            return {"score": 0, "metrics": {}}
            
        file = files[0]
        if 'content' not in file:
            return {"score": 0, "metrics": {}}
            
        complexity = 0  # Placeholder for actual complexity calculation
        return {
            "score": complexity,
            "metrics": {
                "cyclomatic": complexity,
                "cognitive": 0,
                "halstead": 0
            }
        }
    
    def analyze_single_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single file for anomalies.
        
        Args:
            file_data: Dictionary containing file data with content and metadata
            
        Returns:
            Dictionary containing analysis results
        """
        return {
            "file_size": self._detect_size_anomalies([file_data]),
            "complexity": self._analyze_complexity([file_data]),
            "pii_detected": self._detect_pii([file_data])
        }

    def _analyze_complexity_distribution(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of code complexity across files."""
        complexities = []
        for file in files:
            # Handle tuple case from PII removal
            if isinstance(file, tuple):
                content = file[0]
            else:
                content = file.get('content', '')
            
            try:
                # Simple complexity metric: number of lines
                if isinstance(content, str):
                    complexity = len(content.splitlines())
                else:
                    complexity = 0
                complexities.append(complexity)
            except Exception:
                continue
        
        if not complexities:
            return {
                "mean": 0,
                "std_dev": 0,
                "percentiles": [0, 0, 0],  # 25th, 50th, 75th
                "total_files": 0
            }
        
        return {
            "mean": float(np.mean(complexities)),
            "std_dev": float(np.std(complexities)),
            "percentiles": [
                float(np.percentile(complexities, 25)),
                float(np.percentile(complexities, 50)),
                float(np.percentile(complexities, 75))
            ],
            "total_files": len(complexities)
        }

    def _analyze_license_distribution(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of licenses across files."""
        license_counts = {}
        total_files = 0
        
        for file in files:
            # Handle tuple case from PII removal
            if isinstance(file, tuple):
                metadata = {}
            else:
                metadata = file.get('metadata', {})
            
            license_type = metadata.get('license', 'unknown')
            license_counts[license_type] = license_counts.get(license_type, 0) + 1
            total_files += 1
        
        if total_files == 0:
            return {"distribution": {}, "total_files": 0}
        
        # Calculate percentages
        distribution = {
            license_type: count / total_files
            for license_type, count in license_counts.items()
        }
        
        return {
            "distribution": distribution,
            "total_files": total_files
        }

    def analyze_bias(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential bias in the dataset."""
        return {
            "language_distribution": self._analyze_language_distribution(files),
            "complexity_distribution": self._analyze_complexity_distribution(files),
            "license_distribution": self._analyze_license_distribution(files)
        }

    def _analyze_language_distribution(self, files: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze distribution of programming languages."""
        languages = {}
        for file in files:
            lang = file.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        total = sum(languages.values())
        return {lang: count/total for lang, count in languages.items()}