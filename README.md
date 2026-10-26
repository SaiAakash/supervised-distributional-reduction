# Supervised Distributional Reduction (SDR)


Code implementation of the paper **Supervised Distributional Reduction via Optimal Transport and Dependence Maximization** accepted at NeurIPS 2026.

Paper: [https://arxiv.org/pdf/2605.27619](https://arxiv.org/pdf/2605.27619)

*Supervised Distributional Reduction* is an algorithm based on the Fused Gromov-Wasserstein(FGW) framework to jointly address clustering and dimensionality reduction with explicit dependence maximization to produce target-aware embeddings.

## Setup

We use `poetry` ([Poetry Docs](https://python-poetry.org/docs/#installation)) for managing dependencies and the preferred version of Python to be used for this codebase is 3.12. Run the following commands to install the required dependencies:

```bash
poetry env use python3.12
poetry lock && poetry install
```

## Repo Structure
- `src/`: Contains the implementation of the SDR algorithm, along with the necessary components for computing affinities and other distributional reduction algorithms like DistR, Clust-then-DR and DR-then-Clust.
- `benchmarks/benchmarks_clust_dr/`: Contains all the required scripts to run the main experiments under the distributional reduction setting with plotting utilities to reproduce the plots in the paper (Section 5.1, Appendix Section C.2 and Appendix F).
- `benchmarks/benchmarks_dkl/`: Contains all the required scripts to run the main experiments for the kernel learning benchmarks with plotting utilities to reproduce the plots in the paper (Section 5.2, Appendix I and Appendix J).
- `benchmarks/benchmarks_dr/`: Contains the required script to run the pure DR experiments on toy examples with a corresponding plotting utility script (Appendix G).
- `benchmarks/ablation_studies/`: Contains additional ablation studies apart from the ones existing in the previous three folders including experiments like: (i) comparison of joint with sequential baselines, (ii) ablation on various terms of the SDR objective (pure FGW, DistR + CKA, SDR), (iii) Comparison with an accept-reject step for the $T$-block to guarantee monotonic descent, etc.

The instructions to run each of these experiments is provided inside the corresponding sub-folders except for the ablation studies because that folder just involves running the scripts in a standalone fashion and should be fairly straightforward.

## Citing SDR
If you find the SDR work and its implementation useful, please consider citing:
```bibtex
@misc{ramesh2026superviseddistributionalreductionoptimal,
      title={Supervised Distributional Reduction via Optimal Transport and Dependence Maximization}, 
      author={Sai-Aakash Ramesh and Archit Sood and Andrew Corbett and Tim Dodwell},
      year={2026},
      eprint={2605.27619},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.27619}, 
}
```

## Acknowledgement

We thank the authors of the [Distributional Reduction](https://arxiv.org/pdf/2402.02239) paper for their implementation on top of which the SDR implementation has been built.


