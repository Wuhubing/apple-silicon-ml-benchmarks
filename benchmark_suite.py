import tensorflow as tf
import coremltools as ct
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime

class BenchmarkSuite:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device_info': self._get_device_info(),
            'compute_unit_tests': {},
            'batch_size_tests': {},
            'memory_tests': {}
        }
        
        self.compute_units = [
            ct.ComputeUnit.ALL,
            ct.ComputeUnit.CPU_AND_GPU,
            ct.ComputeUnit.CPU_ONLY,
            ct.ComputeUnit.CPU_AND_NE
        ]
    
    def _get_device_info(self):
        """Get device information"""
        import platform
        return {
            'machine': platform.machine(),
            'processor': platform.processor(),
            'system': platform.system(),
            'version': platform.version()
        }
    
    def run_compute_unit_tests(self, model):
        """Test different compute unit configurations"""
        print("\nStarting compute unit tests...")
        test_image = Image.new('RGB', (224, 224), color='white')
        
        for compute_unit in self.compute_units:
            print(f"\nTesting compute unit: {compute_unit}")
            try:
                mlmodel = ct.convert(
                    model,
                    inputs=[ct.ImageType(
                        name="input_1",
                        shape=(1, 224, 224, 3),
                        scale=1/255.0
                    )],
                    compute_units=compute_unit
                )
                
                # Warm-up
                for _ in range(10):
                    _ = mlmodel.predict({"input_1": test_image})
                
                # Test
                latencies = []
                for i in range(100):
                    start = time.perf_counter()
                    _ = mlmodel.predict({"input_1": test_image})
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)  # Convert to milliseconds
                    
                    if (i + 1) % 20 == 0:
                        print(f"Completed {i + 1}/100 iterations")
                
                self.results['compute_unit_tests'][str(compute_unit)] = {
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'p90': np.percentile(latencies, 90),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'raw_latencies': latencies
                }
                
            except Exception as e:
                print(f"Error testing {compute_unit}: {e}")
    
    def save_results(self):
        """Save test results"""
        # Convert numpy arrays to lists
        results_copy = self.results.copy()
        for cu_test in results_copy['compute_unit_tests'].values():
            if 'raw_latencies' in cu_test:
                cu_test['raw_latencies'] = [float(x) for x in cu_test['raw_latencies']]
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(results_copy, f, indent=4)
        print("\nResults saved to benchmark_results.json")

def main():
    suite = BenchmarkSuite()
    
    try:
        print("Loading MobileNetV2 model...")
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )
        
        suite.run_compute_unit_tests(model)
        suite.save_results()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()