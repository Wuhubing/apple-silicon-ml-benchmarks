import time
import numpy as np
from PIL import Image

class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            'latency': [],
            'memory': [],
            'power': []
        }
    
    def measure_latency(self, model, input_image, iterations=100):
        """Measure inference latency"""
        latencies = []
        
        # Ensure input is PIL.Image
        if not isinstance(input_image, Image.Image):
            raise TypeError("Input must be a PIL.Image object")
        
        # Warm-up
        for _ in range(10):
            _ = model.predict({"input_1": input_image})
            
        # Test
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model.predict({"input_1": input_image})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds
            
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'p90': np.percentile(latencies, 90),
            'raw_latencies': latencies  # Add raw data
        }
    
    def measure_throughput(self, model, input_image, batch_sizes=[1,4,8]):
        """Measure throughput"""
        results = {}
        
        # Ensure input is PIL.Image
        if not isinstance(input_image, Image.Image):
            raise TypeError("Input must be a PIL.Image object")
            
        for batch in batch_sizes:
            start = time.perf_counter()
            # Process batch times consecutively
            for _ in range(batch):
                _ = model.predict({"input_1": input_image})
            end = time.perf_counter()
            
            results[batch] = batch / (end - start)
        return results