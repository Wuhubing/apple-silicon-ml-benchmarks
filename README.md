# apple-silicon-ml-benchmarks

A comprehensive benchmarking and optimization framework for deep learning models on Apple Silicon, with a specific focus on MobileNetV2 deployment across different compute unit configurations.

## Project Overview

This project provides a systematic approach to evaluate and optimize deep learning model performance on Apple Silicon platforms. Key features:

- Comprehensive performance analysis across different compute units (CPU, GPU, Neural Engine)
- Detailed latency and throughput measurements
- Automated benchmarking pipeline
- Visualization tools for performance analysis
- Optimization recommendations for different deployment scenarios

## Requirements

- macOS with Apple Silicon (M1/M2)
- Python 3.8+
- TensorFlow 2.12+
- CoreML Tools 6.1+
- NumPy
- Matplotlib
- Seaborn
- PIL

## Installation

```bash
git clone https://github.com/yourusername/apple-silicon-ml-benchmarks.git
cd apple-silicon-ml-benchmarks
pip install -r requirements.txt
```

## Usage

1. Run comprehensive benchmarks:
```bash
python benchmark_suite.py
```

2. Analyze results:
```bash
python analyze_results.py
```

3. Run specific performance tests:
```bash
python run_benchmarks.py
```



## Key Features

- **Multiple Compute Unit Testing**: Evaluate performance across ALL, CPU_AND_GPU, CPU_ONLY, and CPU_AND_NE configurations
- **Comprehensive Metrics**: Measure latency, throughput, and resource utilization
- **Automated Analysis**: Generate detailed performance reports and visualizations
- **Optimization Guidelines**: Practical recommendations for deployment optimization

## Results

The framework provides detailed analysis including:
- Performance comparison across compute units
- Latency distribution analysis
- Resource utilization patterns
- Optimization recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{apple_silicon_ml_benchmarks,
  title={Technical Report: Performance Analysis and Optimization of
MobileNetV2 on Apple M2: A Detailed Study of Neural Engine
and Compute Unit Selection Strategies},
  author={Weibing Wang},
  year={2024},
  url={https://github.com/Wuhubing/apple-silicon-ml-benchmarks}
}
```

## Contact

- Weibing Wang
- UW-Madison
- Email:wwang652@wisc.edu or weibingwangwe@outlook.com
```
