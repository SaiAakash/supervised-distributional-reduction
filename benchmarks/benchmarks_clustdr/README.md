# Distributional Reduction Benchmarks

This folder contains the code to run the main experiments and ablation studies under the distributional reduction setting with plotting utilities to reproduce the plots in the paper (Section 5.1, Appendix Section C.2 and Appendix F).

## Experiments in Section 5.1
To run the distributional reduction experiments in Section 5.1, run the following commands:
```bash
poetry run python benchmarks_clustdr/classification_joint_benchmarks.py --datasets coil20 snareseq fmnist --coil20-path data/coil20_50d.pt --snareseq-path data/SNAREseq_atac.pt --fmnist-path data/fmnist_processed.pt
```

The results by default will be saved into a folder called `Joint_Clust_DR_results` inside the `benchmark_outputs` folder. To generate the plots (Figure 2 and 3 in the paper), run:
```bash
poetry run python benchmarks_clustdr/plot_summary_metrics.py
```
This will store the evaluation metrics and runtime plots for all the three datasets into the `combined_plots` folder inside the `Joint_Clust_DR_results` folder.

## Experiments in Appendix C.2
To run the experiment comparing the FGW-only and the SDR variants (producing Table 3 in the paper), run the following commands for the COIL-20 and SNAREseq datasets respectively:
```bash
poetry run python benchmarks_clustdr/sdr_objective_comparison.py --datasets snareseq --coil20-path data/coil20_50d.pt --results-dir benchmark_outputs/fgw_vs_sdr/coil20

poetry run python benchmarks_clustdr/sdr_objective_comparison.py --datasets snareseq --snareseq-path data/SNAREseq_atac.pt --results-dir benchmark_outputs/fgw_vs_sdr/snareseq
```

To generate example visualizations of the embeddings for the two variants (like Figure 6 in the paper), run the following command:
```bash
poetry run python benchmarks_clustdr/plot_objective_embedding_grid --layout 2x2 --output-dir benchmark_outputs/fgw_vs_sdr
```

## Experiments in Appendix F ($\eta$ ablation)
To run the $\eta$ ablation experiments (producing Figures 7 and 8 in the paper), run the following commands for the COIL-20 and SNAREseq datasets respectively:
```bash
poetry run python benchmarks_clustdr/sdr_eta_ablation.py --datasets coil20 --coil20-path data/coil20_50d.pt --results-dir benchmark_outputs/eta_ablation/coil20
poetry run python benchmarks_clustdr/plot_eta_ablation_metrics.py --summary-path benchmark_outputs/eta_ablation/coil20 --baseline-dir benchmark_outputs/Joint_Clust_DR_results/coil20
```
```bash
poetry run python benchmarks_clustdr/sdr_eta_ablation.py --datasets snareseq --snareseq-path data/SNAREseq_atac.pt --results-dir benchmark_outputs/eta_ablation/snareseq
poetry run python benchmarks_clustdr/plot_eta_ablation_metrics.py --summary-path benchmark_outputs/eta_ablation/snareseq --baseline-dir benchmark_outputs/Joint_Clust_DR_results/snareseq
```