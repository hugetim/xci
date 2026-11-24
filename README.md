# XCI v0.1

XCI v0.1 is an extension of Epoch AI's Bayesian [Epoch Capabilities Index](https://epoch.ai/benchmarks/eci) (ECI) methodology. This repo holds the code that fits it to Epoch's benchmark data. It is used in the post [Decomposing the Epoch Capabilities Index](https://hugetim.github.io/posts/eci-models/).

## The model

XCI v0.1 keeps Epoch's model ([epoch-research/eci-public](https://github.com/epoch-research/eci-public)) and changes its likelihood:

1. **Extended-support beta** (the XB of [Kosmidis and Zeileis 2026](https://doi.org/10.1093/jrsssc/qlaf039)): latent scores live on [−δ, 1 + δ] and are clamped to [0, 1], so exact 0% and 100% scores are used as they are, not clipped.
2. **Median-centered:** the benchmark curve is the latent score a model beats half the time.
3. **Capability noise** on each score, sd τ_b on the capability scale, added to the beta's variance by variance matching.
4. **Benchmark-specific τ_b**, hierarchical like Epoch's σ_b.

The clip on the curve is 10⁻¹² (Epoch: 10⁻⁶). Items 1–3 together are the XB+ of the post [Handling 100% scores in the Bayesian ECI](https://hugetim.github.io/posts/eci-clip/#xb-median-centered-beta-with-capability-scale-noise-also). The docstrings of [`xci.py`](xci.py) give the formulas and priors.

## Running it

You need Linux (or WSL) and conda.

```bash
conda env create -f environment.yml
conda activate xci
bash setup.sh                 # clones Epoch's eci-public at the pinned commit and installs it
curl -sSfLO https://epoch.ai/data/benchmark_data.zip && unzip -q benchmark_data.zip -d data
python xci.py data/epoch_capabilities_index out
```

A fit takes about 20 minutes on 4 cores. To refit new data, `--init-from` starts each chain at its own posterior draw of an earlier fit.

Outputs in `out/`: `eci.csv` (ECI by model, with 90% intervals), `edi.csv` (difficulty, slope, σ_b and τ_b by benchmark), `hyper.json`, `diagnostics.json` (convergence against Epoch's checks and the SHA-256 of the fit input) and `draws.npz` (posterior draws).

## Credits and license

Data: Epoch AI, "Capabilities & benchmarking", <https://epoch.ai/benchmarks>, used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Epoch's model code (`eci-public`) is MIT-licensed and is fetched by `setup.sh`, not included here.

The code in this repo is released under the MIT license (see `LICENSE`).
