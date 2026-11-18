"""Fit XCI v0.1, an extension of Epoch AI's Bayesian ECI methodology, to Epoch's benchmark data.

Epoch's model (eci.bayesian in https://github.com/epoch-research/eci-public): each model m has a capability C_m and
each benchmark b a difficulty D_b, slope alpha_b and noise level sigma_b; a score is beta-distributed around the
curve mu = sigmoid(alpha_b (C_m - D_b)), with concentration kappa_b = 1 / (4 sigma_b^2) - 1, after clipping scores to
[0.001, 0.999] and mu to [1e-6, 1 - 1e-6].

XCI v0.1 keeps Epoch's curve, priors, constraints and anchoring, and changes the likelihood in four ways:

1. Extended-support beta (the XB of Kosmidis and Zeileis 2026, JRSS C 75: 139-157). The latent score
   y* = -delta + (1 + 2 delta) x lives on [-delta, 1 + delta] and the observed score is y* clamped to [0, 1]. Scores
   are not clipped: an exact 0 contributes P(x <= x0) and an exact 1 P(x >= 1 - x0), x0 = delta / (1 + 2 delta).
   Prior: delta ~ HalfNormal(0.05), truncated at 0.1.
2. Median-centred: the latent beta's median, not its mean, is mu (the curve is the latent score a model beats half
   the time). a solves I_mu(a, kappa - a) = 1/2, read from a precomputed table (median_table.py).
3. Capability noise by variance matching: a model's standing on a benchmark also varies on the capability scale
   with sd tau_b. Its effect on the score's variance, (alpha_b tau_b)^2 (mu (1 - mu))^2, is added to Epoch's:
       kappa + 1 = 1 / (4 sigma_b^2 + (alpha_b tau_b)^2 mu (1 - mu))   (capped smoothly so that kappa >= 0.43)
4. Benchmark-specific capability noise, hierarchical like Epoch's sigma_b: tau_b = tau exp(tau_spread z_b),
   z_b ~ Normal(0, 1), tau ~ HalfNormal(1) (raw capability scale), tau_spread ~ Epoch's prior for tau_sigma.

The clip on the curve is mu in [1e-12, 1 - 1e-12] (Epoch: 1e-6), so that the likelihood of a censored 0 or 1 does not
flatten out before the data stop pulling.

Sampling: nutpie (NUTS with a low-rank-plus-diagonal mass matrix), 4 chains of 1,500 tuning and 2,000 kept draws by
default. Outputs follow eci.bayesian.fit_bayesian_eci: each draw is mapped to the ECI scale (Claude 3.5 Sonnet = 130,
GPT-5 = 150), summarised by the posterior mean and 5th-95th percentiles, and checked against Epoch's convergence
standards (no divergences, bulk ESS >= 400, R-hat <= 1.01).

Usage:
    python xci.py DATA_DIR OUT_DIR [--init-from PREVIOUS_OUT_DIR] [--chains 4 --cores 4 --tune 1500 --draws 2000]
DATA_DIR is the epoch_capabilities_index/ folder of Epoch's benchmark data (https://epoch.ai/data/benchmark_data.zip);
its processed_data_for_eci.csv is the fit input, used as is.
Outputs in OUT_DIR: eci.csv (ECI by model), edi.csv (difficulty, slope, sigma_b and tau_b by benchmark, ECI scale
except sigma_b), hyper.json, diagnostics.json, draws.npz.
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
import arviz as az  # noqa: E402
from eci.bayesian import (  # noqa: E402
    DEFAULT_ANCHOR_ECI_HIGH, DEFAULT_ANCHOR_ECI_LOW, DEFAULT_ANCHOR_MODEL_HIGH, DEFAULT_ANCHOR_MODEL_LOW)

from median_table import herm_a  # noqa: E402

VERSION = "0.1"
ECI_PUBLIC_COMMIT = "e04929a"  # the eci-public commit this was checked against (setup.sh pins it)
MU_FLOOR = 1e-12               # clip on the curve: mu in [MU_FLOOR, 1 - MU_FLOOR]
LAMBDA = 1.0                   # capability noise scales with the slope: logit-scale sd alpha_b^LAMBDA tau_b
S2_MAX, SHARP = 0.7, 50.0      # smooth cap on the variance factor: kappa >= 1 / 0.7 - 1 = 0.43 (table edge 0.3)
MIN_ESS, MAX_RHAT = 400, 1.01  # Epoch's convergence standards (eci.bayesian _MIN_ESS, _MAX_RHAT)
HYPER = ["tau_alpha", "sigma_median", "tau_sigma", "delta", "tau_cap", "tau_spread"]
DIAG_VARS = ["capability", "difficulty", "slope", "sigma", "tau_b"] + HYPER


def beta_logcdf_small(a, b, x, n_terms=200):
    """log I_x(a, b), the beta CDF at a small x, as a PyTensor expression with stable gradients.

    PyTensor's own beta logcdf returns NaN gradients when a is small and b large, which is exactly where a weak
    model's exact 0 sits. Instead, DLMF 8.17.8:
        I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * sum_n (a + b)_n / (a + 1)_n x^n,
    whose terms are all positive, summed in logs. Successive terms have ratio (a + b + n - 1) x / (a + n), which tends
    to x, so the series converges for any x < 1. Against scipy.special.betainc, in logs, for mu = a / (a + b) up to
    0.1 and kappa = a + b up to 1,000 (Epoch's fit has kappa up to ~770): better than 1e-11 with 200 terms at x up to
    0.084, the largest x the cap on delta allows. Gradients agree to 1e-9.
    """
    import pytensor.tensor as pt
    n = np.arange(1, n_terms + 1, dtype=float)
    x = pt.as_tensor_variable(x)
    lx = pt.log(x)
    logt = pt.cumsum(pt.log(a[:, None] + b[:, None] + n[None, :] - 1) - pt.log(a[:, None] + n[None, :]) + lx, axis=1)
    return (a * lx + b * pt.log1p(-x) - pt.log(a) - (pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b))
            + pt.log1p(pt.exp(logt).sum(axis=1)))


def kappa_formula(sigma, slope_raw, tau, mu, xp=np):
    """kappa + 1 = 1 / s2, s2 = 4 sigma^2 + (slope^LAMBDA tau)^2 mu (1 - mu), with s2 capped smoothly at S2_MAX by a
    soft minimum, -log(exp(-SHARP s2) + exp(-SHARP S2_MAX)) / SHARP, which equals s2 to ~1e-10 for s2 <= 0.25.
    slope_raw is on the raw capability scale (geometric mean 1), as is tau."""
    s2 = 4 * sigma ** 2 + (slope_raw ** LAMBDA * tau) ** 2 * mu * (1 - mu)
    s2 = -xp.logaddexp(-SHARP * s2, -SHARP * S2_MAX) / SHARP
    return 1 / s2 - 1


def xci_model(df):
    """The XCI v0.1 PyMC model for Epoch's processed fit input (columns Model, benchmark, performance in [0, 1]).
    Every prior and constraint not described in the module docstring is Epoch's (eci.bayesian.build_bayesian_model):
    Uniform(-50, 50) capabilities, flat sum-to-zero difficulties within the same box, ZeroSumNormal log-slopes with
    marginal sd tau_alpha, truncated LogNormal sigma_b, and Epoch's hyperpriors."""
    import pymc as pm
    import pytensor.tensor as pt
    from pymc.distributions.transforms import ZeroSumTransform
    from eci.bayesian import BayesianPriors, _index_data, _lognormal, _CD_BOUND, _MAX_SIGMA
    priors = BayesianPriors()
    _, model_names, _, bench_names, model_idx, bench_idx = _index_data(df)
    y = df.performance.to_numpy(float)
    assert ((y >= 0) & (y <= 1)).all()
    cat = np.where(y <= 0, 0, np.where(y >= 1, 2, 1))
    inner, lo, hi = cat == 1, cat == 0, cat == 2
    with pm.Model(coords={"model": model_names, "benchmark": bench_names}) as model:
        tau_alpha = _lognormal("tau_alpha", priors.tau_alpha)
        sigma_median = _lognormal("sigma_median", priors.sigma_median)
        tau_sigma = _lognormal("tau_sigma", priors.tau_sigma)
        sigma = pm.Truncated("sigma", pm.LogNormal.dist(mu=pm.math.log(sigma_median), sigma=tau_sigma),
                             lower=0.0, upper=_MAX_SIGMA, dims="benchmark")
        capability = pm.Uniform("capability", -_CD_BOUND, _CD_BOUND, dims="model")
        difficulty = pm.Flat("difficulty", dims="benchmark", transform=ZeroSumTransform([-1]))
        pm.Potential("difficulty_bound", pm.math.switch(pm.math.all(pm.math.abs(difficulty) <= _CD_BOUND), 0.0, -np.inf))
        log_slope = pm.ZeroSumNormal("log_slope", sigma=tau_alpha * np.sqrt(len(bench_names) / (len(bench_names) - 1)),
                                     dims="benchmark")
        slope = pm.Deterministic("slope", pm.math.exp(log_slope), dims="benchmark")
        delta = pm.Truncated("delta", pm.HalfNormal.dist(0.05), lower=0.0, upper=0.1)
        mu = pt.clip(pm.math.sigmoid(slope[bench_idx] * (capability[model_idx] - difficulty[bench_idx])), MU_FLOOR, 1 - MU_FLOOR)
        tau_cap = pm.HalfNormal("tau_cap", 1.0)  # the median of tau_b
        tau_spread = _lognormal("tau_spread", priors.tau_sigma)
        tau_b = pm.Deterministic("tau_b", tau_cap * pt.exp(tau_spread * pm.Normal("tau_z", 0.0, 1.0, dims="benchmark")),
                                 dims="benchmark")
        kappa = kappa_formula(sigma[bench_idx], slope[bench_idx], tau_b[bench_idx], mu, xp=pt)
        a = herm_a(mu, kappa, xp=pt); b = kappa - a
        x0 = delta / (1 + 2 * delta)
        lp_in = pm.logp(pm.Beta.dist(a[inner], b[inner]), (y[inner] + delta) / (1 + 2 * delta)) - pt.log1p(2 * delta)
        lp_lo = beta_logcdf_small(a[lo], b[lo], x0)
        lp_hi = beta_logcdf_small(b[hi], a[hi], x0)   # P(x >= 1 - x0) for Beta(a, b) = P(x' <= x0) for Beta(b, a)
        pm.Potential("xci", lp_in.sum() + lp_lo.sum() + lp_hi.sum())
    print(f"XCI v{VERSION}: {int(lo.sum())} scores at 0, {int(hi.sum())} at 1, {int(inner.sum())} interior", flush=True)
    return model


def init_from_fit(fit_dir, model, eci_guess=None):
    """Posterior means of an earlier fit (its draws.npz), on this model's raw scale, as starting points. draws.npz holds
    ECI-scale draws plus each draw's map ECI = a + b * raw, so raw capability = (eci - a) / b, difficulty likewise,
    raw slope = ECI-scale slope * b. The earlier fit may be of older data: a model it lacks starts at eci_guess[model]
    (Epoch's published ECI) mapped to raw, else at the median capability; a benchmark it lacks starts at the median
    difficulty and sigma, the mean log slope and tau_z = 0."""
    dr = np.load(Path(fit_dir) / "draws.npz")
    a, b = dr["a"][:, None].astype(float), dr["b"][:, None].astype(float)
    src_m, src_b = list(dr["models"]), list(dr["benchmarks"])
    mods, bens = list(model.coords["model"]), list(model.coords["benchmark"])
    eci_guess = eci_guess or {}

    def pick(values, src, names, fill):
        pos = {n: i for i, n in enumerate(src)}
        return np.array([values[pos[n]] if n in pos else fill(n) for n in names], dtype=float)

    cap_src = ((dr["eci"].astype(float) - a) / b).mean(0)
    a_m, b_m = float(a.mean()), float(b.mean())
    cap = pick(cap_src, src_m, mods,
               lambda n: (eci_guess[n] - a_m) / b_m if n in eci_guess else float(np.median(cap_src)))
    dif_src = ((dr["edi"].astype(float) - a) / b).mean(0)
    lsl_src = np.log(dr["slope"].astype(float) * b).mean(0)
    sig_src = dr["sigma"].astype(float).mean(0)
    dif = pick(dif_src, src_b, bens, lambda n: float(np.median(dif_src)))
    lsl = pick(lsl_src, src_b, bens, lambda n: float(lsl_src.mean()))
    init = {"capability": cap, "difficulty": dif - dif.mean(), "log_slope": lsl - lsl.mean(),
            "sigma": pick(sig_src, src_b, bens, lambda n: float(np.median(sig_src)))}
    new_m, new_b = [m for m in mods if m not in set(src_m)], [x for x in bens if x not in set(src_b)]
    if new_m or new_b:
        print("init-from: %d new models (%s), %d new benchmarks (%s)" % (len(new_m), ", ".join(new_m[:5]), len(new_b),
                                                                         ", ".join(new_b[:5])), flush=True)
    for k in HYPER:
        if "hyper_" + k in dr.files:
            init[k] = float(np.median(dr["hyper_" + k]))
    if {"tau_b", "hyper_tau_cap", "hyper_tau_spread"} <= set(dr.files):
        z = (np.log(dr["tau_b"].astype(float) / dr["hyper_tau_cap"].astype(float)[:, None])
             / dr["hyper_tau_spread"].astype(float)[:, None]).mean(0)
        init["tau_z"] = pick(z, src_b, bens, lambda n: 0.0)
    return init


def progress_logger(every, t_start):
    """nutpie progress_callback that prints one plain line at most every `every` seconds. Returns nutpie.sample keyword
    arguments; {} when every <= 0."""
    if every <= 0:
        return {}
    import threading
    lock, last = threading.Lock(), [0.0]

    def fmt(sec):
        sec = int(sec); return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"

    def cb(prog):
        now = time.time()
        with lock:
            done = all(p.finished_draws >= p.total_draws for p in prog)
            if now - last[0] < every and not done:
                return
            last[0] = now
        fin = sum(p.finished_draws for p in prog); tot = sum(p.total_draws for p in prog)
        el = now - t_start
        eta = fmt(el * (tot - fin) / fin) if fin else "?"
        chains = " ".join(f"{p.finished_draws}/{p.total_draws}{'w' if p.tuning else ''}" for p in prog)
        print(f"[progress {fmt(el)}] draws {chains} (w = warmup) | {100 * fin / max(tot, 1):.0f}% | "
              f"divergences {sum(p.divergences for p in prog)} | left ~{eta}", flush=True)

    return dict(progress_callback=cb, progress_rate=1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir", help="Epoch's epoch_capabilities_index/ folder (holds processed_data_for_eci.csv)")
    ap.add_argument("out_dir")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--target-accept", type=float, default=0.95, help="NUTS target acceptance rate (Epoch: 0.95)")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--progress-every", type=float, default=60.0, metavar="SECONDS",
                    help="print a progress line this often; 0 turns it off")
    ap.add_argument("--init-from", metavar="FIT_DIR",
                    help="start chains at an earlier XCI fit's posterior means (its draws.npz)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import nutpie
    src = Path(args.data_dir) / "processed_data_for_eci.csv"
    df = pd.read_csv(src)
    model = xci_model(df)
    models, benches = list(model.coords["model"]), list(model.coords["benchmark"])
    print(f"{len(df)} scores, {len(models)} models, {len(benches)} benchmarks", flush=True)

    # Capabilities and difficulties start at 0 on the raw scale; only the slopes, noise and hyperparameters are
    # jittered. nutpie's default jitter (+-1 on the unconstrained scale) puts Uniform(-50, 50) starts up to +-23 away,
    # where the curve saturates and the clip on mu flattens the gradient.
    free = {rv.name for rv in model.free_RVs}
    jitter = {model[v] for v in ("log_slope", "sigma", *HYPER) if v in free}
    init = None
    if args.init_from:
        pub = Path(args.data_dir) / "eci_scores.csv"  # Epoch's published ECI: starting points for models new since then
        guess = pd.read_csv(pub).set_index("Model").eci.to_dict() if pub.exists() else None
        init = {k: v for k, v in init_from_fit(args.init_from, model, guess).items() if k in free}
        jitter = {model[v] for v in HYPER if v not in init and v in free}
    t0 = time.time()
    compiled = nutpie.compile_pymc_model(model, jitter_rvs=jitter, initial_points=init)
    t1 = time.time()
    trace = nutpie.sample(compiled, chains=args.chains, cores=args.cores, tune=args.tune, draws=args.draws,
                          seed=args.seed, adaptation="low_rank", target_accept=args.target_accept,
                          progress_bar=False, **progress_logger(args.progress_every, t1))
    t2 = time.time()
    post = trace.posterior.to_dataset() if hasattr(trace.posterior, "to_dataset") else trace.posterior
    stats = trace.sample_stats.to_dataset() if hasattr(trace.sample_stats, "to_dataset") else trace.sample_stats

    summ = az.summary(post, var_names=DIAG_VARS, kind="diagnostics")
    div = int(stats["diverging"].sum())
    diag = dict(model=f"XCI v{VERSION}", divergences=div, min_ess_bulk=float(summ["ess_bulk"].min()),
                min_ess_tail=float(summ["ess_tail"].min()), max_rhat=float(summ["r_hat"].max()),
                worst_rhat=str(summ["r_hat"].idxmax()), worst_ess=str(summ["ess_bulk"].idxmin()),
                chains=args.chains, tune=args.tune, draws=args.draws, adaptation="low_rank",
                target_accept=args.target_accept, seed=args.seed, init_from=args.init_from,
                compile_s=round(t1 - t0, 1), sample_s=round(t2 - t1, 1),
                n_scores=len(df), n_models=len(models), n_benchmarks=len(benches),
                processed_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                sbeta=True, median_centre=True, var_match=LAMBDA, tau_bench=True, mu_floor=MU_FLOOR,
                delta_prior="halfnormal0.05", eci_public_commit=ECI_PUBLIC_COMMIT,
                fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    diag["passes_epoch_checks"] = bool(div == 0 and diag["min_ess_bulk"] >= MIN_ESS and diag["max_rhat"] <= MAX_RHAT)
    print(json.dumps(diag, indent=1), flush=True)
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1))

    def stack(name, dim):
        return post[name].stack(s=("chain", "draw")).transpose("s", dim).values

    C, D = stack("capability", "model"), stack("difficulty", "benchmark")
    lo, hi = models.index(DEFAULT_ANCHOR_MODEL_LOW), models.index(DEFAULT_ANCHOR_MODEL_HIGH)
    b = (DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW) / (C[:, hi] - C[:, lo])
    a = DEFAULT_ANCHOR_ECI_LOW - b * C[:, lo]
    s = dict(a=a, b=b, eci=a[:, None] + b[:, None] * C, edi=a[:, None] + b[:, None] * D,
             slope=stack("slope", "benchmark") / b[:, None], sigma=stack("sigma", "benchmark"),
             tau_b=stack("tau_b", "benchmark"))  # tau_b on the raw capability scale (x b for ECI points)

    def summarise(x):
        return x.mean(0), np.quantile(x, 0.05, axis=0), np.quantile(x, 0.95, axis=0)

    m, l, h = summarise(s["eci"])
    pd.DataFrame({"Model": models, "eci": m, "eci_q05": l, "eci_q95": h}).to_csv(out / "eci.csv", index=False)
    rows = {"benchmark": benches}
    for k, x in (("edi", s["edi"]), ("slope", s["slope"]), ("sigma", s["sigma"]), ("tau_b", s["tau_b"] * b[:, None])):
        rows[k], rows[k + "_q05"], rows[k + "_q95"] = summarise(x)
    pd.DataFrame(rows).to_csv(out / "edi.csv", index=False)
    hyp = {k: dict(zip(("mean", "q05", "q95"), (float(v[0]) for v in summarise(post[k].values.reshape(-1, 1)))))
           for k in HYPER}
    (out / "hyper.json").write_text(json.dumps(hyp, indent=1))
    np.savez_compressed(out / "draws.npz", models=np.array(models), benchmarks=np.array(benches),
                        **{k: v.astype(np.float32) for k, v in s.items()},
                        **{"hyper_" + k: post[k].values.reshape(-1).astype(np.float32) for k in HYPER})


if __name__ == "__main__":
    main()
