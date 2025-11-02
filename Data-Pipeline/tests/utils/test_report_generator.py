import pytest
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

class QAReportGenerator:
    """Generate comprehensive test reports with visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "details": {}
        }

    def add_test_results(self, results: Dict[str, Any]):
        """Add test results to the report."""
        self.results["details"].update(results)
        
        # Update summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("status") == "passed")
        failed_tests = total_tests - passed_tests
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }

    def add_coverage_data(self, coverage_data: Dict[str, Any]):
        """Add code coverage information to the report."""
        self.results["coverage"] = coverage_data

    def add_bias_metrics(self, bias_metrics: Dict[str, Any]):
        """Add bias analysis metrics to the report."""
        self.results["bias_analysis"] = bias_metrics

    def generate_visualizations(self):
        """Generate visualizations for the report."""
        # Test Results Pie Chart
        plt.figure(figsize=(10, 6))
        plt.pie(
            [self.results["summary"]["passed_tests"], 
             self.results["summary"]["failed_tests"]],
            labels=["Passed", "Failed"],
            colors=["green", "red"],
            autopct="%1.1f%%"
        )
        plt.title("Test Results Distribution")
        plt.savefig(self.output_dir / "test_results_pie.png")
        plt.close()

        # Coverage Heatmap
        if "coverage" in self.results:
            coverage_data = self.results["coverage"]
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                coverage_data["matrix"],
                annot=True,
                fmt=".1f",
                cmap="YlOrRd"
            )
            plt.title("Code Coverage Heatmap")
            plt.savefig(self.output_dir / "coverage_heatmap.png")
            plt.close()

        # Bias Analysis Visualization
        if "bias_analysis" in self.results:
            bias_data = self.results["bias_analysis"]
            plt.figure(figsize=(10, 6))
            
            # Language Distribution
            if "language_distribution" in bias_data:
                plt.bar(
                    bias_data["language_distribution"].keys(),
                    bias_data["language_distribution"].values()
                )
                plt.title("Language Distribution in Dataset")
                plt.xticks(rotation=45)
                plt.ylabel("Percentage")
                plt.savefig(self.output_dir / "language_distribution.png")
                plt.close()

    def generate_report(self) -> Path:
        """Generate the final HTML report."""
        report_path = self.output_dir / "test_report.html"
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Execution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; margin-bottom: 20px; }}
                .details {{ margin-top: 20px; }}
                .visualization {{ margin: 20px 0; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Test Execution Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {self.results["summary"]["total_tests"]}</p>
                <p class="passed">Passed: {self.results["summary"]["passed_tests"]}</p>
                <p class="failed">Failed: {self.results["summary"]["failed_tests"]}</p>
                <p>Pass Rate: {self.results["summary"]["pass_rate"]:.2f}%</p>
            </div>
            
            <div class="visualization">
                <h2>Test Results Distribution</h2>
                <img src="test_results_pie.png" alt="Test Results Pie Chart">
            </div>
            
            {self._generate_coverage_section() if "coverage" in self.results else ""}
            {self._generate_bias_section() if "bias_analysis" in self.results else ""}
            
            <div class="details">
                <h2>Test Details</h2>
                {self._generate_test_details()}
            </div>
        </body>
        </html>
        """
        
        report_path.write_text(html_content)
        return report_path

    def _generate_coverage_section(self) -> str:
        """Generate the coverage section of the report."""
        return f"""
        <div class="coverage">
            <h2>Code Coverage</h2>
            <p>Overall Coverage: {self.results["coverage"]["total_coverage"]:.2f}%</p>
            <img src="coverage_heatmap.png" alt="Coverage Heatmap">
        </div>
        """

    def _generate_bias_section(self) -> str:
        """Generate the bias analysis section of the report."""
        return f"""
        <div class="bias-analysis">
            <h2>Bias Analysis</h2>
            <img src="language_distribution.png" alt="Language Distribution">
            <h3>Fairness Metrics</h3>
            <ul>
                {"".join(f"<li>{k}: {v:.2f}</li>" for k, v in 
                        self.results["bias_analysis"].get("fairness_metrics", {}).items())}
            </ul>
        </div>
        """

    def _generate_test_details(self) -> str:
        """Generate the test details section."""
        details = []
        for test_name, result in self.results["details"].items():
            status_class = "passed" if result["status"] == "passed" else "failed"
            details.append(f"""
            <div class="test-case">
                <h3 class="{status_class}">{test_name}</h3>
                <p>Status: {result["status"]}</p>
                {f'<p>Error: {result["error"]}</p>' if "error" in result else ""}
                {f'<p>Duration: {result["duration"]:.2f}s</p>' if "duration" in result else ""}
            </div>
            """)
        return "\n".join(details)