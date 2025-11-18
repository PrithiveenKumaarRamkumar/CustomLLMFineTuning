#!/usr/bin/env python3
"""
Test script for CustomLLM Inference API

This script tests all API endpoints to ensure proper functionality.
Run this after starting the API server to validate the implementation.
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any
import httpx
import argparse


class APITester:
    """Test suite for the CustomLLM Inference API."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.results = {
            "passed": 0,
            "failed": 0,
            "tests": []
        }
    
    async def run_all_tests(self):
        """Run all API tests."""
        print(f"🧪 Testing CustomLLM Inference API at {self.base_url}")
        print("=" * 60)
        
        # Test basic connectivity
        await self.test_root_endpoint()
        await self.test_status_endpoint()
        await self.test_health_endpoint()
        await self.test_metrics_endpoint()
        
        # Test model info (if model is loaded)
        await self.test_model_info_endpoint()
        
        # Test inference endpoints (requires API key and loaded model)
        if self.api_key:
            await self.test_predict_endpoint()
            await self.test_batch_predict_endpoint()
            await self.test_authentication()
            await self.test_validation_errors()
        else:
            print("⚠️ Skipping authenticated tests (no API key provided)")
        
        # Print summary
        self.print_summary()
    
    async def test_root_endpoint(self):
        """Test the root endpoint."""
        await self._test_endpoint(
            "GET",
            "/",
            "Root endpoint",
            expected_status=200,
            validate_response=lambda r: "message" in r and "version" in r
        )
    
    async def test_status_endpoint(self):
        """Test the status endpoint."""
        await self._test_endpoint(
            "GET",
            "/status",
            "Status endpoint",
            expected_status=200,
            validate_response=lambda r: "service" in r and "status" in r
        )
    
    async def test_health_endpoint(self):
        """Test the health endpoint."""
        await self._test_endpoint(
            "GET",
            "/health",
            "Health endpoint",
            expected_status=[200, 503],  # May be unhealthy if model not loaded
            validate_response=lambda r: "status" in r and "api_status" in r
        )
    
    async def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/metrics")
                
                if response.status_code == 200:
                    content = response.text
                    required_metrics = [
                        "api_requests_total",
                        "api_request_duration_seconds", 
                        "gpu_memory_usage_bytes",
                        "tokens_per_second"
                    ]
                    
                    missing_metrics = [m for m in required_metrics if m not in content]
                    
                    if not missing_metrics:
                        self._record_test("Metrics endpoint", True, "All required metrics present")
                    else:
                        self._record_test(
                            "Metrics endpoint", 
                            False, 
                            f"Missing metrics: {missing_metrics}"
                        )
                else:
                    self._record_test(
                        "Metrics endpoint", 
                        False, 
                        f"Unexpected status: {response.status_code}"
                    )
                    
            except Exception as e:
                self._record_test("Metrics endpoint", False, str(e))
    
    async def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        await self._test_endpoint(
            "GET",
            "/info",
            "Model info endpoint",
            expected_status=[200, 503],  # May fail if model not loaded
            validate_response=lambda r: True  # Just check if it responds
        )
    
    async def test_predict_endpoint(self):
        """Test the main prediction endpoint."""
        test_payload = {
            "prompt": "Hello, world!",
            "max_length": 10,
            "temperature": 0.7
        }
        
        await self._test_endpoint(
            "POST",
            "/predict",
            "Prediction endpoint",
            headers=self.headers,
            json_payload=test_payload,
            expected_status=[200, 503],  # May fail if model not loaded
            validate_response=lambda r: "generated_text" in r if isinstance(r, dict) else True
        )
    
    async def test_batch_predict_endpoint(self):
        """Test the batch prediction endpoint."""
        test_payload = {
            "prompts": ["Hello", "World"],
            "max_length": 10,
            "temperature": 0.7
        }
        
        await self._test_endpoint(
            "POST",
            "/predict/batch",
            "Batch prediction endpoint",
            headers=self.headers,
            json_payload=test_payload,
            expected_status=[200, 503],  # May fail if model not loaded
            validate_response=lambda r: "results" in r if isinstance(r, dict) else True
        )
    
    async def test_authentication(self):
        """Test authentication requirements."""
        test_payload = {
            "prompt": "Test prompt",
            "max_length": 10
        }
        
        # Test without authentication
        await self._test_endpoint(
            "POST",
            "/predict",
            "Authentication (no token)",
            json_payload=test_payload,
            expected_status=401,
            validate_response=lambda r: "error" in r if isinstance(r, dict) else True
        )
        
        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        await self._test_endpoint(
            "POST",
            "/predict",
            "Authentication (invalid token)",
            headers=invalid_headers,
            json_payload=test_payload,
            expected_status=401,
            validate_response=lambda r: "error" in r if isinstance(r, dict) else True
        )
    
    async def test_validation_errors(self):
        """Test request validation."""
        # Test invalid payload
        invalid_payload = {
            "prompt": "",  # Empty prompt should fail
            "max_length": -1,  # Invalid max_length
            "temperature": 5.0  # Invalid temperature
        }
        
        await self._test_endpoint(
            "POST",
            "/predict",
            "Validation errors",
            headers=self.headers,
            json_payload=invalid_payload,
            expected_status=422,
            validate_response=lambda r: "error" in r if isinstance(r, dict) else True
        )
    
    async def _test_endpoint(
        self,
        method: str,
        endpoint: str,
        test_name: str,
        headers: Dict[str, str] = None,
        json_payload: Dict[str, Any] = None,
        expected_status: int = 200,
        validate_response: callable = None
    ):
        """Generic endpoint testing method."""
        if isinstance(expected_status, int):
            expected_status = [expected_status]
        
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}{endpoint}"
                
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=json_payload)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                # Check status code
                if response.status_code not in expected_status:
                    self._record_test(
                        test_name,
                        False,
                        f"Expected status {expected_status}, got {response.status_code}"
                    )
                    return
                
                # Parse response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                # Validate response content
                if validate_response and not validate_response(response_data):
                    self._record_test(
                        test_name,
                        False,
                        "Response validation failed"
                    )
                    return
                
                self._record_test(test_name, True, f"Status: {response.status_code}")
                
            except Exception as e:
                self._record_test(test_name, False, str(e))
    
    def _record_test(self, name: str, passed: bool, message: str = ""):
        """Record test result."""
        if passed:
            self.results["passed"] += 1
            print(f"✅ {name}: PASSED {message}")
        else:
            self.results["failed"] += 1
            print(f"❌ {name}: FAILED - {message}")
        
        self.results["tests"].append({
            "name": name,
            "passed": passed,
            "message": message
        })
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📈 Success Rate: {self.results['passed'] / (self.results['passed'] + self.results['failed']) * 100:.1f}%")
        
        if self.results["failed"] > 0:
            print("\n💡 Failed Tests:")
            for test in self.results["tests"]:
                if not test["passed"]:
                    print(f"   • {test['name']}: {test['message']}")


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test CustomLLM Inference API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000", 
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key", 
        default=None, 
        help="API key for authenticated endpoints (default: from API_KEY env var)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only basic connectivity tests"
    )
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv("API_KEY")
    
    # Create tester
    tester = APITester(args.url, api_key)
    
    # Run tests
    if args.quick:
        await tester.test_root_endpoint()
        await tester.test_status_endpoint()
        await tester.test_health_endpoint()
        tester.print_summary()
    else:
        await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if tester.results["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())