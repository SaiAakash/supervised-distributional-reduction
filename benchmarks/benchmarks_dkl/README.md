# Kernel Learning Benchmarks

## Experiments in Section 5.2

To obtain results on the regression benchmarks (metrics in Table 1), run the following command:
```bash
poetry run python benchmarks_dkl/gp_benchmarks.py --datasets boston energy concrete --boston-path data/boston.csv.xls --energy-path data/ENB2012_data.xlsx --concrete-path data/Concrete_Data.xls 
```

To obtain results on the image classification benchmarks (metrics in Table 2), run the following command:
```bash
poetry run python benchmarks_dkl/image_classification_benchmarks.py --datasets coil20 mnist --coil20-path data/coil20.pt --mnist-path data/mnist_test_stratified_2k.pt
```

## Uncertainty Calibration Experiments
To create the MLL vs MACE tradeoff plot (Figure 4 in Section 5.2) and the calibration curves for the regression datasets in Appendix J run the following command:
```bash
poetry run python benchmarks_dkl/plot_gp_calibration_curves.py --boston-path data/boston.csv.xls --concrete-path data/Concrete_Data.xls --energy-path data/ENB2012_data.xlsx
```

## Experiments in Appendix I ($\beta$ Ablation)
To run the ablation study for the $\beta$ parameter (Figure 10 in Appendix I of the paper), run the following commands:
```bash
poetry run python benchmarks_dkl/sdr_beta_ablation.py --datasets boston concrete energy --boston-path data/boston.csv.xls --concrete-path data/Concrete_Data.xls --energy-path data/ENB2012_data.xlsx
poetry run python benchmarks_dkll/plot_beta_ablation.py
```