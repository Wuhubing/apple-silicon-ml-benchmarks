# Performance Analysis Report

## 1. Overview
Test conducted on: 2024-11-27T00:40:29.998668

## 2. Compute Unit Performance Analysis

### ComputeUnit.ALL
- Mean Latency: 0.87 ms
- P90 Latency: 0.68 ms
- Min Latency: 0.54 ms
- Max Latency: 25.95 ms

### ComputeUnit.CPU_AND_GPU
- Mean Latency: 3.13 ms
- P90 Latency: 1.94 ms
- Min Latency: 1.64 ms
- Max Latency: 138.10 ms

### ComputeUnit.CPU_ONLY
- Mean Latency: 2.54 ms
- P90 Latency: 2.66 ms
- Min Latency: 2.43 ms
- Max Latency: 3.90 ms

### ComputeUnit.CPU_AND_NE
- Mean Latency: 0.64 ms
- P90 Latency: 0.67 ms
- Min Latency: 0.55 ms
- Max Latency: 1.71 ms

## 3. Optimization Recommendations

1. Best performing configuration: ComputeUnit.CPU_AND_NE
2. Optimization suggestions:
   - Consider using batch processing for higher throughput
   - Optimize memory access patterns
   - Implement model quantization