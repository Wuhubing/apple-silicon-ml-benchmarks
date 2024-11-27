import psutil
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    def __init__(self):
        self.results = {
            'memory': [],
            'cpu_usage': [],
            'power': [],
            'temperature': []
        }
        
    def monitor_resources(self, duration=10, interval=0.1):
        """Monitor system resource usage"""
        start_time = time.time()
        while time.time() - start_time < duration:
            # Memory usage
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # CPU usage
            cpu = psutil.cpu_percent(interval=0.1)
            
            # Record data
            self.results['memory'].append(memory)
            self.results['cpu_usage'].append(cpu)
            
            time.sleep(interval)
            
    def analyze_model_efficiency(self, model, test_image, batch_sizes=[1,2,4,8,16]):
        """Analyze model efficiency with different batch sizes"""
        efficiency_metrics = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            start_time = time.time()
            
            # Record initial resource usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run inference
            for _ in range(batch_size):
                _ = model.predict({"input_1": test_image})
            
            # Calculate metrics
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            efficiency_metrics[batch_size] = {
                'time_per_sample': (end_time - start_time) / batch_size,
                'memory_per_sample': (final_memory - initial_memory) / batch_size,
                'throughput': batch_size / (end_time - start_time)
            }
            
        return efficiency_metrics
    
    def visualize_efficiency(self, metrics):
        """Visualize efficiency metrics"""
        # Set overall style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Common plot style
        plot_style = {
            'linewidth': 2.5,
            'markersize': 10,
            'marker': 'o',
            'markeredgecolor': 'white',
            'markeredgewidth': 2
        }
        
        # 1. Throughput analysis
        ax1 = plt.subplot(2, 2, 1)
        batch_sizes = list(metrics.keys())
        throughputs = [m['throughput'] for m in metrics.values()]
        
        sns.lineplot(x=batch_sizes, y=throughputs, **plot_style)
        ax1.fill_between(batch_sizes, throughputs, alpha=0.2)
        
        ax1.set_title('Throughput Analysis', pad=20, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Throughput (samples/sec)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add data labels
        for x, y in zip(batch_sizes, throughputs):
            ax1.annotate(f'{y:.2f}', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=10)
        
        # [Rest of the visualization code remains the same, just remove Chinese comments]
        
        # Save figure
        plt.savefig('efficiency_analysis.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print("\nEfficiency analysis saved to efficiency_analysis.png")

def main():
    # Create test image
    test_image = Image.new('RGB', (224, 224), color='white')
    
    # Load saved model
    import coremltools as ct
    model = ct.models.MLModel('MobileNetV2.mlpackage')
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Perform efficiency analysis
    print("Starting efficiency analysis...")
    metrics = analyzer.analyze_model_efficiency(model, test_image)
    
    # Visualize results
    analyzer.visualize_efficiency(metrics)
    
    # Monitor resources
    print("\nStarting resource monitoring...")
    analyzer.monitor_resources(duration=30)  # Monitor for 30 seconds
    
    # Save results
    np.save('performance_metrics.npy', {
        'efficiency': metrics,
        'resources': analyzer.results
    })
    print("\nAnalysis results saved to performance_metrics.npy")

if __name__ == "__main__":
    main()