import ssl
import tensorflow as tf
import numpy as np
from model_optimizer import ModelOptimizer
from benchmark import PerformanceBenchmark
from PIL import Image

# Resolve SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

class TestPipeline:
    def __init__(self):
        self.model_optimizer = ModelOptimizer()
        self.benchmark = PerformanceBenchmark()
        
    def run_pipeline(self):
        try:
            # 1. Create and optimize model
            print("Step 1: Creating and optimizing model")
            original_model = self.create_model()
            optimized_model = self.optimize_model(original_model)
            
            # 2. Run benchmarks
            print("\nStep 2: Running benchmarks")
            test_results = self.run_benchmarks(optimized_model)
            
            # 3. Save results
            print("\nStep 3: Saving results")
            self.save_results(test_results)
            
        except Exception as e:
            print(f"Error: {e}")
    
    def create_model(self):
        """Create base model"""
        print("Loading MobileNetV2 model...")
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )
        return model
    
    def optimize_model(self, model):
        """Optimize model"""
        print("Converting to Core ML format...")
        coreml_model = self.model_optimizer.convert_to_coreml(model)
        
        # Save in .mlpackage format
        print("Saving model...")
        coreml_model.save("OptimizedModel.mlpackage")
        print("Optimized model saved as OptimizedModel.mlpackage")
        return coreml_model
    
    def create_test_input(self):
        """Create test input data"""
        # Create a 224x224 RGB image
        img = Image.new('RGB', (224, 224), color='white')
        # Ensure returning PIL.Image object
        return img
    
    def run_benchmarks(self, model):
        """Run performance tests"""
        results = {}
        
        # Prepare test data
        test_input = self.create_test_input()
        
        # Test latency
        print("Testing latency...")
        latency_results = self.benchmark.measure_latency(
            model, 
            test_input, 
            iterations=100
        )
        results['latency'] = latency_results
        
        # Test throughput
        print("Testing throughput...")
        throughput_results = self.benchmark.measure_throughput(
            model,
            test_input,
            batch_sizes=[1, 4, 8]  # Reduced batch sizes to avoid memory issues
        )
        results['throughput'] = throughput_results
        
        return results
    
    def save_results(self, results):
        """Save test results"""
        np.save('benchmark_results.npy', results)
        
        # Print results summary
        print("\nTest Results Summary:")
        print(f"Mean Latency: {results['latency']['mean']:.2f} ms")
        print(f"Latency Std Dev: {results['latency']['std']:.2f} ms")
        print(f"P90 Latency: {results['latency']['p90']:.2f} ms")
        print("\nThroughput by Batch Size:")
        for batch_size, throughput in results['throughput'].items():
            print(f"Batch Size {batch_size}: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    pipeline = TestPipeline()
    pipeline.run_pipeline()