import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_results():
    # Load test results
    with open('benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    # Set plot style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Compute unit performance comparison
    plt.subplot(2, 1, 1)
    compute_units = list(results['compute_unit_tests'].keys())
    mean_latencies = [results['compute_unit_tests'][cu]['mean'] 
                     for cu in compute_units]
    
    sns.barplot(x=compute_units, y=mean_latencies)
    plt.title('Different Compute Unit Configurations Performance')
    plt.xlabel('Compute Unit')
    plt.ylabel('Latency (ms)')
    plt.xticks(rotation=45)
    
    # 2. Performance distribution plot
    plt.subplot(2, 1, 2)
    data = []
    for cu in compute_units:
        stats = results['compute_unit_tests'][cu]
        latencies = stats['raw_latencies']
        sns.kdeplot(data=latencies, label=cu)
    
    plt.title('Latency Distribution by Compute Unit')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPerformance analysis plots saved to performance_analysis.png")
    
    # Generate analysis report
    generate_report(results)

def generate_report(results):
    """Generate analysis report"""
    report = []
    report.append("# Performance Analysis Report\n")
    
    # 1. Device information
    report.append("## 1. Device Information")
    for key, value in results['device_info'].items():
        report.append(f"- {key}: {value}")
    report.append("")
    
    # 2. Performance analysis
    report.append("## 2. Performance Analysis\n")
    for cu, stats in results['compute_unit_tests'].items():
        report.append(f"### {cu}")
        report.append(f"- Mean Latency: {stats['mean']:.2f} ms")
        report.append(f"- P90 Latency: {stats['p90']:.2f} ms")
        report.append(f"- Min Latency: {stats['min']:.2f} ms")
        report.append(f"- Max Latency: {stats['max']:.2f} ms")
        report.append("")
    
    # 3. Optimization recommendations
    report.append("## 3. Optimization Recommendations\n")
    best_cu = min(results['compute_unit_tests'].items(), 
                 key=lambda x: x[1]['mean'])
    report.append(f"1. Best performing configuration: {best_cu[0]}")
    report.append("2. Optimization suggestions:")
    report.append("   - Consider using batch processing for higher throughput")
    report.append("   - Optimize memory access patterns")
    report.append("   - Implement model quantization")
    
    with open('analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    print("\nAnalysis report saved to analysis_report.md")

if __name__ == "__main__":
    analyze_results()