import tensorflow as tf
import coremltools as ct
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime

def run_comprehensive_benchmarks():
    print("Starting comprehensive performance tests...")
    
    # 1. Load model
    print("\n1. Loading MobileNetV2 model...")
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )
    
    # 2. Test different compute unit configurations
    compute_units = [
        ct.ComputeUnit.ALL,
        ct.ComputeUnit.CPU_AND_GPU,
        ct.ComputeUnit.CPU_ONLY,
        ct.ComputeUnit.CPU_AND_NE
    ]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'compute_unit_tests': {},
        'batch_size_tests': {},
        'memory_tests': {}
    }
    
    # Create test input
    test_image = Image.new('RGB', (224, 224), color='white')
    
    for compute_unit in compute_units:
        print(f"\nTesting compute unit: {compute_unit}")
        try:
            # Convert model
            mlmodel = ct.convert(
                model,
                inputs=[ct.ImageType(
                    name="input_1",
                    shape=(1, 224, 224, 3),
                    scale=1/255.0
                )],
                compute_units=compute_unit
            )
            
            # Run inference tests
            latencies = []
            for _ in range(100):  # 100 iterations
                start = time.perf_counter()
                _ = mlmodel.predict({"input_1": test_image})
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to milliseconds
            
            results['compute_unit_tests'][str(compute_unit)] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'p90': np.percentile(latencies, 90),
                'min': np.min(latencies),
                'max': np.max(latencies)
            }
            
        except Exception as e:
            print(f"Error testing {compute_unit}: {e}")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nBenchmark complete, results saved to benchmark_results.json")
    return results

if __name__ == "__main__":
    run_comprehensive_benchmarks()