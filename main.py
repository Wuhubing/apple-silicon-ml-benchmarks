import tensorflow as tf
import coremltools as ct
import numpy as np
import time
from PIL import Image

def create_and_convert_model():
    # 1. Load pretrained model
    print("Loading MobileNetV2 model...")
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )
    
    # 2. Convert to Core ML model
    print("Converting to Core ML model...")
    mlmodel = ct.convert(
        model,
        inputs=[ct.ImageType(
            name="input_1",
            shape=(1, 224, 224, 3),
            scale=1/255.0
        )],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",  # Explicitly specify mlprogram format
        minimum_deployment_target=ct.target.iOS15
    )
    
    # 3. Save model with .mlpackage extension
    mlmodel.save("MobileNetV2.mlpackage")  # Changed from .mlmodel to .mlpackage
    print("Model saved as MobileNetV2.mlpackage")
    
    return mlmodel

def benchmark_model(model, iterations=100):
    # Create test image using PIL
    test_image = Image.new('RGB', (224, 224), color='white')
    
    # Warmup
    print("Warming up model...")
    for _ in range(10):
        _ = model.predict({"input_1": test_image})
    
    # Performance testing
    print(f"Starting performance test ({iterations} iterations)...")
    latencies = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        _ = model.predict({"input_1": test_image})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{iterations} iterations")
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p90_latency = np.percentile(latencies, 90)
    
    print("\nPerformance Statistics:")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Standard Deviation: {std_latency:.2f} ms")
    print(f"P90 Latency: {p90_latency:.2f} ms")
    
    return {
        'mean': avg_latency,
        'std': std_latency,
        'p90': p90_latency,
        'raw_latencies': latencies
    }

def main():
    try:
        # 1. Create and convert model
        model = create_and_convert_model()
        
        # 2. Run performance tests
        results = benchmark_model(model)
        
        # 3. Save results
        np.save('benchmark_results.npy', results)
        print("\nBenchmark results saved to benchmark_results.npy")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main() 