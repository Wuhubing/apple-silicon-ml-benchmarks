import tensorflow as tf
import coremltools as ct
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ComputeUnitOptimizer:
    def __init__(self):
        self.compute_units = [
            ct.ComputeUnit.ALL,               # All available compute units
            ct.ComputeUnit.CPU_AND_GPU,       # CPU and GPU only
            ct.ComputeUnit.CPU_ONLY,          # CPU only
            ct.ComputeUnit.CPU_AND_NE         # CPU and Neural Engine
        ]
        self.results = {}
        
        # Set plotting style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Configure plot parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def convert_model(self, model, compute_unit):
        """Convert model with specified compute unit"""
        print(f"\nUsing compute unit: {compute_unit}")
        mlmodel = ct.convert(
            model,
            inputs=[ct.ImageType(
                name="input_1",
                shape=(1, 224, 224, 3),
                scale=1/255.0
            )],
            compute_units=compute_unit,
            minimum_deployment_target=ct.target.iOS15
        )
        return mlmodel
    
    def benchmark_model(self, model, iterations=100):
        """Benchmark model performance"""
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Warmup
        for _ in range(10):
            _ = model.predict({"input_1": test_image})
        
        # Test latency
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model.predict({"input_1": test_image})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'p90': np.percentile(latencies, 90),
            'raw_latencies': latencies
        }
    
    def run_optimization(self):
        """Run optimization tests"""
        print("Loading MobileNetV2 model...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )
        
        for compute_unit in self.compute_units:
            try:
                coreml_model = self.convert_model(base_model, compute_unit)
                print("Running benchmark...")
                results = self.benchmark_model(coreml_model)
                
                self.results[compute_unit] = results
                print(f"Mean Latency: {results['mean']:.2f} ms")
                print(f"Std Dev: {results['std']:.2f} ms")
                print(f"P90 Latency: {results['p90']:.2f} ms")
                
            except Exception as e:
                print(f"Error testing {compute_unit}: {e}")
    
    def visualize_results(self):
        """Create professional visualizations"""
        fig = plt.figure(figsize=(15, 12), dpi=300)
        
        # 1. Latency Comparison Bar Plot
        plt.subplot(2, 1, 1)
        compute_units = list(self.results.keys())
        means = [self.results[cu]['mean'] for cu in compute_units]
        stds = [self.results[cu]['std'] for cu in compute_units]
        
        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(compute_units)))
        bars = plt.bar(range(len(compute_units)), means, yerr=stds, 
                      capsize=5, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms',
                    ha='center', va='bottom')
        
        plt.xticks(range(len(compute_units)), 
                  [str(cu).split('.')[-1] for cu in compute_units],
                  rotation=45)
        plt.title('Latency Comparison Across Compute Units', 
                 pad=20, weight='bold')
        plt.xlabel('Compute Unit Configuration')
        plt.ylabel('Latency (ms)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Latency Distribution Plot
        plt.subplot(2, 1, 2)
        
        # Create DataFrame for seaborn
        data = []
        for cu in compute_units:
            df = pd.DataFrame({
                'Latency': self.results[cu]['raw_latencies'],
                'Compute Unit': str(cu).split('.')[-1]
            })
            data.append(df)
        df_combined = pd.concat(data)
        
        # Plot violin plot with individual points
        sns.violinplot(data=df_combined, x='Compute Unit', y='Latency',
                      palette='viridis', inner='box')
        sns.swarmplot(data=df_combined, x='Compute Unit', y='Latency',
                     color='white', alpha=0.5, size=4)
        
        plt.title('Latency Distribution by Compute Unit', 
                 pad=20, weight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add statistical annotations
        stats_text = "Statistical Summary:\n"
        for cu in compute_units:
            stats = self.results[cu]
            cu_name = str(cu).split('.')[-1]
            stats_text += f"\n{cu_name}:\n"
            stats_text += f"Mean: {stats['mean']:.2f}ms\n"
            stats_text += f"P90: {stats['p90']:.2f}ms"
        
        plt.text(1.15, 0.5, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('compute_unit_comparison.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        print("\nResults saved to compute_unit_comparison.png")

if __name__ == "__main__":
    optimizer = ComputeUnitOptimizer()
    optimizer.run_optimization()
    optimizer.visualize_results()