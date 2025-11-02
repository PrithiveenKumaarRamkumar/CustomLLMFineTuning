"""
Performance Testing Suite for Data Pipeline
Load testing, scalability testing, and resource monitoring.
"""

import pytest
import os
import time
import psutil
import tempfile
import threading
import multiprocessing
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
import statistics
import json
import matplotlib.pyplot as plt
import pandas as pd

# Import pipeline modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from anomaly_detection import PipelineAnomalyDetector
    from bias_analysis import ComprehensiveBiasAnalyzer
except ImportError:
    class PipelineAnomalyDetector:
        def __init__(self, config=None): pass
        def analyze_pipeline_stage(self, name, metrics): return []
    class ComprehensiveBiasAnalyzer:
        def __init__(self, config=None): pass
        def analyze_dataset_bias(self, info): return {}


@pytest.mark.performance
class TestPipelinePerformance:
    """Performance testing suite - 15 test cases"""
    
    def test_small_dataset_performance(self, temp_workspace):
        """Test performance with small dataset (100 files)"""
        
        dataset_size = 100
        results = self._run_performance_test(temp_workspace, dataset_size, "small")
        
        # Performance expectations for small dataset
        assert results['processing_time'] < 30  # Should complete within 30 seconds
        assert results['throughput'] > 3  # At least 3 files per second
        assert results['memory_peak_mb'] < 100  # Should use less than 100MB
        assert results['success_rate'] >= 0.95  # 95% success rate minimum
    
    def test_medium_dataset_performance(self, temp_workspace):
        """Test performance with medium dataset (1000 files)"""
        
        dataset_size = 1000
        results = self._run_performance_test(temp_workspace, dataset_size, "medium")
        
        # Performance expectations for medium dataset
        assert results['processing_time'] < 300  # Should complete within 5 minutes
        assert results['throughput'] > 3  # Maintain throughput
        assert results['memory_peak_mb'] < 500  # Should use less than 500MB
        assert results['success_rate'] >= 0.90  # 90% success rate minimum
    
    def test_large_dataset_performance(self, temp_workspace):
        """Test performance with large dataset (5000 files)"""
        
        dataset_size = 5000
        results = self._run_performance_test(temp_workspace, dataset_size, "large")
        
        # Performance expectations for large dataset
        assert results['processing_time'] < 1800  # Should complete within 30 minutes
        assert results['throughput'] > 2  # At least 2 files per second
        assert results['memory_peak_mb'] < 1000  # Should use less than 1GB
        assert results['success_rate'] >= 0.85  # 85% success rate minimum
    
    def test_memory_usage_scaling(self, temp_workspace):
        """Test memory usage scaling across different dataset sizes"""
        
        dataset_sizes = [100, 500, 1000, 2000]
        memory_usage_results = []
        
        for size in dataset_sizes:
            results = self._run_performance_test(temp_workspace, size, f"scaling_{size}")
            memory_usage_results.append({
                'size': size,
                'memory_peak_mb': results['memory_peak_mb'],
                'memory_avg_mb': results['memory_avg_mb']
            })
        
        # Verify linear or sub-linear memory scaling
        sizes = [r['size'] for r in memory_usage_results]
        memory_peaks = [r['memory_peak_mb'] for r in memory_usage_results]
        
        # Memory should not grow exponentially
        memory_growth_rate = memory_peaks[-1] / memory_peaks[0]
        size_growth_rate = sizes[-1] / sizes[0]
        
        assert memory_growth_rate <= size_growth_rate * 2  # At most 2x linear growth
    
    def test_concurrent_processing_performance(self, temp_workspace):
        """Test performance with concurrent processing"""
        
        num_workers = min(4, multiprocessing.cpu_count())
        dataset_size_per_worker = 200
        
        def worker_task(worker_id):
            worker_dir = os.path.join(temp_workspace, f'worker_{worker_id}')
            os.makedirs(worker_dir, exist_ok=True)
            
            return self._run_performance_test(
                worker_dir, 
                dataset_size_per_worker, 
                f"concurrent_worker_{worker_id}"
            )
        
        # Run concurrent workers
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze concurrent performance
        total_files = num_workers * dataset_size_per_worker
        overall_throughput = total_files / total_time
        
        # Concurrent processing should be faster than sequential
        sequential_estimate = sum(r['processing_time'] for r in results)
        speedup = sequential_estimate / total_time
        
        assert speedup > 1.5  # At least 1.5x speedup from concurrency
        assert overall_throughput > 5  # Good concurrent throughput
        assert all(r['success_rate'] >= 0.90 for r in results)  # All workers successful
    
    def test_processing_time_distribution(self, temp_workspace):
        """Test distribution of processing times across files"""
        
        # Create files of varying complexity
        file_complexities = {
            'simple': 'print("hello")',
            'medium': '''
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n-1)

for i in range(10):
    print(f"Factorial of {i} is {calculate_factorial(i)}")
''',
            'complex': '''
import itertools
import functools

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.cache = {}
    
    @functools.lru_cache(maxsize=128)
    def process_item(self, item):
        # Simulate complex processing
        result = 0
        for i in range(len(str(item))):
            result += i ** 2
        return result
    
    def batch_process(self):
        return [self.process_item(item) for item in self.data]

# Generate test data and process
data = list(range(100))
processor = DataProcessor(data)
results = processor.batch_process()
'''
        }
        
        processing_times = []
        
        # Test each complexity level
        for complexity, content in file_complexities.items():
            complexity_times = []
            
            # Create multiple files of same complexity
            for i in range(20):
                filepath = os.path.join(temp_workspace, f'{complexity}_{i}.py')
                with open(filepath, 'w') as f:
                    f.write(content)
                
                # Measure processing time for individual file
                start_time = time.time()
                self._process_single_file(filepath)
                file_time = time.time() - start_time
                
                complexity_times.append(file_time)
            
            processing_times.extend(complexity_times)
        
        # Analyze processing time distribution
        mean_time = statistics.mean(processing_times)
        std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        
        # Times should be reasonable and not too variable
        assert mean_time < 1.0  # Average less than 1 second per file
        assert std_dev < mean_time  # Standard deviation less than mean (reasonable variability)
    
    def test_memory_leak_detection(self, temp_workspace):
        """Test for memory leaks during extended processing"""
        
        initial_memory = self._get_memory_usage()
        memory_samples = [initial_memory]
        
        # Process files in batches and monitor memory
        batch_size = 50
        num_batches = 10
        
        for batch_num in range(num_batches):
            batch_dir = os.path.join(temp_workspace, f'batch_{batch_num}')
            os.makedirs(batch_dir, exist_ok=True)
            
            # Create batch files
            for i in range(batch_size):
                filepath = os.path.join(batch_dir, f'file_{i}.py')
                with open(filepath, 'w') as f:
                    f.write(f'print("Batch {batch_num}, File {i}")')
            
            # Process batch
            batch_files = list(Path(batch_dir).glob('*.py'))
            self._process_file_batch([str(f) for f in batch_files])
            
            # Sample memory usage
            current_memory = self._get_memory_usage()
            memory_samples.append(current_memory)
            
            # Clean up batch files to isolate memory usage
            for filepath in batch_files:
                os.remove(filepath)
        
        # Analyze memory trend
        memory_growth = memory_samples[-1] - memory_samples[0]
        
        # Should not have significant memory growth (< 50MB growth allowed)
        assert memory_growth < 50, f"Potential memory leak detected: {memory_growth}MB growth"
    
    def test_cpu_utilization_efficiency(self, temp_workspace):
        """Test CPU utilization efficiency during processing"""
        
        # Monitor CPU usage during processing
        cpu_samples = []
        
        def cpu_monitor():
            while not stop_monitoring.is_set():
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
        
        # Create test dataset
        dataset_size = 500
        self._create_test_dataset(temp_workspace, dataset_size, "cpu_test")
        
        # Start CPU monitoring
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=cpu_monitor)
        monitor_thread.start()
        
        # Run processing
        start_time = time.time()
        results = self._run_performance_test(temp_workspace, dataset_size, "cpu_efficiency")
        processing_time = time.time() - start_time
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # CPU utilization should be reasonable
            assert avg_cpu > 10, "CPU utilization too low - inefficient processing"
            assert avg_cpu < 95, "CPU utilization too high - system overload"
            assert max_cpu < 100, "CPU should not be maxed out consistently"
    
    def test_disk_io_performance(self, temp_workspace):
        """Test disk I/O performance and efficiency"""
        
        # Create files of different sizes
        file_sizes = {
            'small': 1024,      # 1KB
            'medium': 10240,    # 10KB  
            'large': 102400,    # 100KB
            'xlarge': 1024000   # 1MB
        }
        
        io_results = {}
        
        for size_name, size_bytes in file_sizes.items():
            # Create test file
            filepath = os.path.join(temp_workspace, f'io_test_{size_name}.py')
            content = f'# {size_name} file\n' + 'x = "a"\n' * (size_bytes // 10)
            
            # Measure write performance
            write_start = time.time()
            with open(filepath, 'w') as f:
                f.write(content)
            write_time = time.time() - write_start
            
            # Measure read performance
            read_start = time.time()
            with open(filepath, 'r') as f:
                _ = f.read()
            read_time = time.time() - read_start
            
            # Measure processing performance
            process_start = time.time()
            self._process_single_file(filepath)
            process_time = time.time() - process_start
            
            io_results[size_name] = {
                'size_bytes': size_bytes,
                'write_time': write_time,
                'read_time': read_time,
                'process_time': process_time,
                'write_throughput_mb_s': (size_bytes / (1024 * 1024)) / write_time,
                'read_throughput_mb_s': (size_bytes / (1024 * 1024)) / read_time
            }
        
        # Verify I/O performance is reasonable
        for size_name, metrics in io_results.items():
            # Read/write should be fast
            assert metrics['write_time'] < 1.0, f"Write too slow for {size_name} file"
            assert metrics['read_time'] < 0.5, f"Read too slow for {size_name} file"
            
            # Throughput should be reasonable
            if metrics['size_bytes'] > 10000:  # Only check for larger files
                assert metrics['write_throughput_mb_s'] > 1, f"Write throughput too low for {size_name}"
                assert metrics['read_throughput_mb_s'] > 5, f"Read throughput too low for {size_name}"
    
    def test_error_handling_performance_impact(self, temp_workspace):
        """Test performance impact of error handling"""
        
        # Create mix of valid and problematic files
        total_files = 200
        error_rate = 0.1  # 10% error rate
        num_error_files = int(total_files * error_rate)
        
        # Create normal files
        normal_files = []
        for i in range(total_files - num_error_files):
            filepath = os.path.join(temp_workspace, f'normal_{i}.py')
            with open(filepath, 'w') as f:
                f.write(f'print("Normal file {i}")')
            normal_files.append(filepath)
        
        # Create problematic files
        error_files = []
        for i in range(num_error_files):
            filepath = os.path.join(temp_workspace, f'error_{i}.py')
            
            if i % 3 == 0:
                # Binary file
                with open(filepath, 'wb') as f:
                    f.write(b'\x00\x01\x02\xff\xfe')
            elif i % 3 == 1:
                # Syntax error
                with open(filepath, 'w') as f:
                    f.write('def broken_function(\n    print("syntax error"')
            else:
                # Permission error (will be simulated)
                with open(filepath, 'w') as f:
                    f.write('print("test")')
                os.chmod(filepath, 0o000)
            
            error_files.append(filepath)
        
        all_files = normal_files + error_files
        
        # Test processing with error handling
        start_time = time.time()
        results = self._process_file_batch_with_error_handling(all_files)
        processing_time = time.time() - start_time
        
        # Clean up permissions
        for filepath in error_files:
            try:
                os.chmod(filepath, 0o644)
            except:
                pass
        
        # Error handling should not significantly slow down processing
        expected_time_per_file = 0.05  # 50ms per file
        expected_total_time = total_files * expected_time_per_file
        
        # Allow 50% overhead for error handling
        assert processing_time < expected_total_time * 1.5
        
        # Should successfully process most normal files
        assert results['files_processed'] >= len(normal_files) * 0.9
        assert results['error_count'] >= num_error_files * 0.5
    
    def test_throughput_consistency(self, temp_workspace):
        """Test consistency of processing throughput over time"""
        
        # Process files in multiple batches and measure throughput
        batch_size = 100
        num_batches = 5
        throughput_measurements = []
        
        for batch_num in range(num_batches):
            batch_dir = os.path.join(temp_workspace, f'throughput_batch_{batch_num}')
            os.makedirs(batch_dir, exist_ok=True)
            
            # Create batch
            for i in range(batch_size):
                filepath = os.path.join(batch_dir, f'file_{i}.py')
                with open(filepath, 'w') as f:
                    f.write(f'print("Throughput test batch {batch_num}, file {i}")')
            
            # Measure throughput for this batch
            batch_files = [os.path.join(batch_dir, f'file_{i}.py') for i in range(batch_size)]
            
            start_time = time.time()
            self._process_file_batch(batch_files)
            batch_time = time.time() - start_time
            
            batch_throughput = batch_size / batch_time
            throughput_measurements.append(batch_throughput)
        
        # Analyze throughput consistency
        mean_throughput = statistics.mean(throughput_measurements)
        throughput_std = statistics.stdev(throughput_measurements)
        
        # Throughput should be consistent (coefficient of variation < 30%)
        coefficient_of_variation = throughput_std / mean_throughput
        assert coefficient_of_variation < 0.3, f"Throughput too inconsistent: CV = {coefficient_of_variation:.2f}"
        
        # All measurements should be within reasonable range of mean
        for throughput in throughput_measurements:
            deviation = abs(throughput - mean_throughput) / mean_throughput
            assert deviation < 0.5, f"Throughput measurement too far from mean: {deviation:.2f}"
    
    def test_resource_cleanup_efficiency(self, temp_workspace):
        """Test efficiency of resource cleanup after processing"""
        
        initial_memory = self._get_memory_usage()
        initial_open_files = len(psutil.Process().open_files())
        
        # Process large batch and monitor resources
        batch_size = 1000
        self._create_test_dataset(temp_workspace, batch_size, "cleanup_test")
        
        # Process with explicit resource monitoring
        mid_processing_memory = None
        mid_processing_files = None
        
        def monitor_during_processing():
            nonlocal mid_processing_memory, mid_processing_files
            time.sleep(5)  # Wait for processing to start
            mid_processing_memory = self._get_memory_usage()
            mid_processing_files = len(psutil.Process().open_files())
        
        monitor_thread = threading.Thread(target=monitor_during_processing)
        monitor_thread.start()
        
        # Run processing
        results = self._run_performance_test(temp_workspace, batch_size, "cleanup")
        
        monitor_thread.join()
        
        # Check resources after processing
        final_memory = self._get_memory_usage()
        final_open_files = len(psutil.Process().open_files())
        
        # Memory should return close to initial level
        memory_recovery = abs(final_memory - initial_memory)
        assert memory_recovery < 20, f"Poor memory cleanup: {memory_recovery}MB not recovered"
        
        # File handles should be cleaned up
        file_handle_recovery = abs(final_open_files - initial_open_files)
        assert file_handle_recovery < 10, f"File handles not cleaned up: {file_handle_recovery} handles leaked"
    
    def test_scalability_limits(self, temp_workspace):
        """Test system behavior at scalability limits"""
        
        # Gradually increase load until system limits
        dataset_sizes = [100, 500, 1000, 2000, 4000]
        scalability_results = []
        
        for size in dataset_sizes:
            try:
                results = self._run_performance_test(temp_workspace, size, f"scalability_{size}")
                
                scalability_results.append({
                    'size': size,
                    'success': True,
                    'processing_time': results['processing_time'],
                    'throughput': results['throughput'],
                    'memory_peak': results['memory_peak_mb'],
                    'success_rate': results['success_rate']
                })
                
                # Stop if performance degrades significantly
                if len(scalability_results) > 1:
                    current_throughput = results['throughput']
                    previous_throughput = scalability_results[-2]['throughput']
                    
                    if current_throughput < previous_throughput * 0.5:
                        break  # Significant performance degradation
                        
            except Exception as e:
                scalability_results.append({
                    'size': size,
                    'success': False,
                    'error': str(e)
                })
                break  # System limit reached
        
        # Analyze scalability
        successful_results = [r for r in scalability_results if r['success']]
        
        # Should handle at least moderate scale
        assert len(successful_results) >= 3, "System doesn't scale to moderate sizes"
        
        # Throughput shouldn't degrade too much with scale
        if len(successful_results) >= 2:
            first_throughput = successful_results[0]['throughput']
            last_throughput = successful_results[-1]['throughput']
            throughput_retention = last_throughput / first_throughput
            
            assert throughput_retention > 0.3, f"Throughput degrades too much with scale: {throughput_retention:.2f}"
    
    # Helper methods for performance testing
    
    def _run_performance_test(self, workspace_dir, dataset_size, test_name):
        """Run performance test with specified parameters"""
        
        # Create test dataset
        self._create_test_dataset(workspace_dir, dataset_size, test_name)
        
        # Monitor resources
        initial_memory = self._get_memory_usage()
        memory_samples = []
        
        def memory_monitor():
            while not stop_monitoring.is_set():
                memory_samples.append(self._get_memory_usage())
                time.sleep(0.5)
        
        # Start monitoring
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        # Run processing
        start_time = time.time()
        
        try:
            file_paths = list(Path(workspace_dir).glob(f'{test_name}_*.py'))
            processed_count = self._process_file_batch([str(f) for f in file_paths])
            success = True
            
        except Exception as e:
            processed_count = 0
            success = False
            
        processing_time = time.time() - start_time
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Calculate metrics
        throughput = processed_count / processing_time if processing_time > 0 else 0
        success_rate = processed_count / dataset_size if dataset_size > 0 else 0
        
        memory_peak = max(memory_samples) if memory_samples else initial_memory
        memory_avg = statistics.mean(memory_samples) if memory_samples else initial_memory
        
        return {
            'test_name': test_name,
            'dataset_size': dataset_size,
            'processing_time': processing_time,
            'throughput': throughput,
            'success_rate': success_rate,
            'memory_peak_mb': memory_peak,
            'memory_avg_mb': memory_avg,
            'files_processed': processed_count,
            'success': success
        }
    
    def _create_test_dataset(self, workspace_dir, size, name_prefix):
        """Create test dataset of specified size"""
        
        file_templates = [
            'def function_{i}():\n    return {i}',
            'class Class{i}:\n    def __init__(self): self.value = {i}',
            '# File {i}\nprint("Processing {i}")',
            'import os\nvar{i} = {i}\nprint(f"Value: {{var{i}}}")'
        ]
        
        for i in range(size):
            template = file_templates[i % len(file_templates)]
            content = template.format(i=i)
            
            filepath = os.path.join(workspace_dir, f'{name_prefix}_{i:04d}.py')
            with open(filepath, 'w') as f:
                f.write(content)
    
    def _process_file_batch(self, file_paths):
        """Simulate processing a batch of files"""
        processed_count = 0
        
        for filepath in file_paths:
            try:
                self._process_single_file(filepath)
                processed_count += 1
            except:
                continue  # Skip failed files
        
        return processed_count
    
    def _process_file_batch_with_error_handling(self, file_paths):
        """Process files with explicit error handling"""
        processed_count = 0
        error_count = 0
        
        for filepath in file_paths:
            try:
                self._process_single_file(filepath)
                processed_count += 1
            except Exception:
                error_count += 1
        
        return {
            'files_processed': processed_count,
            'error_count': error_count
        }
    
    def _process_single_file(self, filepath):
        """Simulate processing a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simulate some processing work
            lines = content.split('\n')
            word_count = sum(len(line.split()) for line in lines)
            
            # Simulate variable processing time based on content
            processing_delay = min(0.001 * word_count, 0.05)  # Max 50ms
            time.sleep(processing_delay)
            
        except Exception as e:
            # Re-raise to allow error handling testing
            raise e
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)


if __name__ == '__main__':
    # Run performance tests
    pytest.main([__file__, '-v', '-m', 'performance'])