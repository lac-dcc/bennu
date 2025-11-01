# Benchmark Suite in ONNX Neural Network Models

This repository contains a benchmark collection of ONNX-format neural network models used to evaluate autotuning strategies for graph-level optimizations. These benchmarks were used in our research published in:

- **CC 2025** — Fusion of Operators of Computational Graphs via Greedy Clustering: The XNNC Experience - DOI: [10.1145/3708493.3712689](10.1145/3708493.3712689) 
- **TACO 2024** — The Droplet Search Algorithm for Kernel Scheduling - DOI: [10.1145/3650109](10.1145/3650109)

### What the benchmarks are
The suite consists of real ONNX neural network models (e.g., CNNs, transformers, MLPs) that serve as representative workloads for hardware-oriented autotuning and fusion research. Each model is provided in ONNX format together with metadata such as graph size, number of operators, and target synthesis metrics when applicable.

All benchmarks were obtained from the [ONNX Model Zoo](https://github.com/onnx/models) script on GitHub.

### Why they are useful
These models were used to evaluate autotuning heuristics that explore optimization choices such as graph-level fusion, and kernel scheduling.
They enable **reproducible experiments**, fair comparison across tuning strategies, and extension of research for neural networks.

### What you can do with them
- Use the models as input to your **compiler**, or **autotuner**.  
- Benchmark fusion or scheduling heuristics using consistent ONNX graphs.  
- Extend the dataset with new models or new ground-truth metrics.  
- Use them in tutorials or coursework on deep-learning compilation, accelerator design, or autotuning.

### How to cite
If you use this benchmark suite, please cite the following works:

```
@article{canesche2024droplet,
  title={The droplet search algorithm for kernel scheduling},
  author={Canesche, Michael and Ros{\'a}rio, Vanderson and Borin, Edson and Quint{\~a}o Pereira, Fernando},
  journal={ACM Transactions on Architecture and Code Optimization},
  volume={21},
  number={2},
  pages={1--28},
  year={2024},
  publisher={ACM New York, NY}
}

@inproceedings{canesche2025fusion,
  title={Fusion of Operators of Computational Graphs via Greedy Clustering: The XNNC Experience},
  author={Canesche, Michael and do Rosario, Vanderson Martins and Borin, Edson and Quint{\~a}o Pereira, Fernando Magno},
  booktitle={Proceedings of the 34th ACM SIGPLAN International Conference on Compiler Construction},
  pages={117--127},
  year={2025}
}
```

