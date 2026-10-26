# Toy Dimensionality Reduction Benchmarks

This directory contains code for benchmarking SDR on toy dimensionality reduction tasks.

# Downstream Prediction Evaluation for KRR and KNN models (Appendix G)

To evaluate SDR on downstream predictions against other supervised DR methods with Kernel Ridge Regressor (KRR) and the K-Nearest Neighbors (k-NN) models on toy datasets (Tables 6 and 7 in Appendix G), run the following command:
```bash
poetry run python benchmarks_dr/pure_dr_toy_benchmarks.py
```

To create visualizations of the embeddings for Swiss-Roll and S-curve datasets in 2D (Figure 9 in Appnedix G) run the following command:
```bash
poetry run python benchmarks/plot_synthetic_embedding_comparison.py
```