#!/usr/bin/env python3
"""
tinyhippo_discreteness_v2.py
==============================

Upgraded version of tinyhippo_ca3_discreteness.py. Same goal (get a
same-metric discreteness readout for tinyHippo's simulated CA3/CA1 to
compare against DANU's real-CA1 alpha bimodality = 0.243), but fixes the
two problems the last two runs exposed:

  1. Sarle's bimodality coefficient (BC) alone can be inflated by a
     skewed, heavy-tailed *unimodal* distribution (e.g. a population
     mostly at baseline with occasional synchronous burst outliers) --
     which is exactly what the v2-retuned CA3/CA1 traces turned out to
     be, not real two-state switching. BC > 0.555 is no longer treated
     as sufficient evidence of bimodality by itself.

  2. Onset transients (all neurons sharing the same initial condition at
     t=0) can dominate a short simulation window and masquerade as a
     "state" -- this produced the original CA1_PYR BC=0.715 artifact.

Fixes:
  - Burn-in trim: drop the first `burn_in_ms` of simulated time before
    any metric is computed.
  - Real multimodality check: Hartigan's dip test (via the `diptest`
    package if installed -- directly tests for a dip between two modes,
    not fooled by skew) AND a 1- vs 2-component GaussianMixture BIC
    comparison (if 2 components isn't a meaningfully better fit, it's
    not bimodal regardless of what BC says). Both are reported alongside
    BC so a skew artifact can't slip through unnoticed again.
  - Multi-seed sweep: runs the network across several `seed_connect`
    values and reports BC / dip-p / delta-BIC as distributions (mean +-
    std, plus all raw values), not single-run numbers.

Install the optional dip-test package with:
    pip install diptest --break-system-packages
If unavailable, the script still runs -- it just relies on the GMM-BIC
check alone and prints a note that dip-test was skipped.

Dependencies: numpy, scipy, scikit-learn, matplotlib, nest (PyNEST),
              diptest (optional)
Requires tinyhippo.py (or tinyhippo_v2_bistable_ca3.py) importable for
build_ca1_ca3_izh.
"""

import os
os.makedirs("outputs", exist_ok=True)

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import nest

# --- Choose which network builder to test. This is set to the ORIGINAL,
# unretuned tinyhippo.py (p_ca3_EE=0.04, original drive rates) so CA1's
# numbers reflect CA1 as anatomically specified -- downstream of baseline
# CA3, not downstream of the v2 CA3 retuned for bistability. CA1 receives
# Schaffer collateral input from CA3, so CA1's dynamics are NOT invariant
# to CA3's tuning even though CA1's own recurrent parameters don't change
# between tinyhippo.py and tinyhippo_v2_bistable_ca3.py -- only this
# (original) network gives a fair "as-designed CA1" to compare against
# DANU's real CA1. Swap back to tinyhippo_v2_bistable_ca3 only if you
# specifically want CA1-downstream-of-retuned-CA3 as a separate condition,
# not as a stand-in for baseline CA1.
from tiny import build_ca1_ca3_izh

try:
    import diptest as _diptest_pkg
    _HAVE_DIPTEST = True
except ImportError:
    _HAVE_DIPTEST = False
    print("[warn] 'diptest' package not installed -- Hartigan's dip test will be "
          "skipped, relying on GMM-BIC only. Install with: "
          "pip install diptest --break-system-packages")


# --------------------------------------------------------------------------
# Binning: spikes -> population-vector matrix [n_bins x n_units]
# --------------------------------------------------------------------------

def population_vector_from_recorder(spk_recorder, unit_ids, t_stop_ms, bin_ms,
                                      burn_in_ms=0.0):
    """Bin a NEST spike_recorder's events into a [n_bins x n_units] count matrix.

    Bins entirely before `burn_in_ms` are dropped, so onset transients
    (e.g. a synchronous first-bin burst from shared initial conditions)
    can't dominate the discreteness metrics.
    """
    ev = nest.GetStatus(spk_recorder, "events")[0]
    times = np.asarray(ev["times"], dtype=float)
    senders = np.asarray(ev["senders"], dtype=int)

    unit_ids = np.asarray(unit_ids, dtype=int)
    id_to_col = {uid: i for i, uid in enumerate(unit_ids)}

    edges = np.arange(0.0, t_stop_ms + bin_ms, bin_ms)
    n_bins_all = len(edges) - 1
    mat_all = np.zeros((n_bins_all, len(unit_ids)), dtype=np.float32)

    bin_idx = np.digitize(times, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins_all)
    for t_i, s in zip(bin_idx[valid], senders[valid]):
        col = id_to_col.get(int(s))
        if col is not None:
            mat_all[t_i, col] += 1.0

    centers_all = edges[:-1] + bin_ms / 2.0

    keep = centers_all >= burn_in_ms
    return centers_all[keep], mat_all[keep]


# --------------------------------------------------------------------------
# Discreteness metrics
# --------------------------------------------------------------------------

def bimodality_coefficient(x):
    """Sarle's bimodality coefficient. BC > 5/9 (~0.555) is the
    conventional (but skew-sensitive, see module docstring) cutoff."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 3:
        return np.nan
    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, fisher=True, bias=False)
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skew ** 2 + 1.0) / (kurt + correction)


def dip_test(x):
    """Hartigan's dip test via the `diptest` package, if installed.
    Returns (dip_statistic, p_value) or (nan, nan) if unavailable.
    Low p-value (<0.05) = reject unimodality = evidence FOR bimodality
    (or more generally multimodality). This is not fooled by skew the
    way BC is, since it directly tests for a dip in the CDF rather than
    inferring shape from skew/kurtosis moments.
    """
    if not _HAVE_DIPTEST:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan, np.nan
    dip, pval = _diptest_pkg.diptest(x)
    return dip, pval


def gmm_bic_comparison(x, seed=0):
    """Fit 1- and 2-component GaussianMixture to x and compare via BIC.
    Returns (bic_1, bic_2, delta_bic) where delta_bic = bic_1 - bic_2.
    delta_bic > 0 means the 2-component model fits better (lower BIC);
    the more positive, the stronger the evidence for a real second mode.
    As a rule of thumb, delta_bic > ~10 is considered strong evidence.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    x = x[np.isfinite(x).ravel()]
    if x.shape[0] < 10:
        return np.nan, np.nan, np.nan

    gmm1 = GaussianMixture(n_components=1, random_state=seed, n_init=3).fit(x)
    gmm2 = GaussianMixture(n_components=2, random_state=seed, n_init=5).fit(x)

    bic1 = gmm1.bic(x)
    bic2 = gmm2.bic(x)
    return bic1, bic2, (bic1 - bic2)


def project_and_cluster(mat, n_states=2, seed=0):
    """Reduce a [n_bins x n_units] count matrix to a 1D trace (first PC),
    then hard-assign each bin to one of n_states via a GaussianMixture on
    that trace. Returns (projection, hard_states, soft_probs).
    """
    if mat.shape[0] < 5 or mat.shape[1] < 2:
        raise ValueError("Not enough bins/units to project/cluster.")

    pca = PCA(n_components=1, random_state=seed)
    proj = pca.fit_transform(mat).ravel()

    gmm = GaussianMixture(n_components=n_states, random_state=seed, n_init=5)
    gmm.fit(proj.reshape(-1, 1))
    soft = gmm.predict_proba(proj.reshape(-1, 1))
    hard = np.argmax(soft, axis=1)

    means = gmm.means_.ravel()
    order = np.argsort(means)
    remap = {old: new for new, old in enumerate(order)}
    hard = np.array([remap[h] for h in hard])
    soft = soft[:, order]

    return proj, hard, soft


def dwell_times(state_seq, bin_ms):
    if len(state_seq) == 0:
        return np.array([])
    changes = np.where(np.diff(state_seq) != 0)[0] + 1
    run_starts = np.concatenate(([0], changes))
    run_ends = np.concatenate((changes, [len(state_seq)]))
    run_lengths_bins = run_ends - run_starts
    return run_lengths_bins * bin_ms


def switch_rate(state_seq):
    if len(state_seq) < 2:
        return np.nan
    return np.mean(np.diff(state_seq) != 0)


# --------------------------------------------------------------------------
# Per-run analysis (single seed)
# --------------------------------------------------------------------------

def analyze_population(label, spk_recorder, unit_ids, sim_ms, bin_ms=20.0,
                        burn_in_ms=300.0, seed=0, verbose=True):
    t_centers, mat = population_vector_from_recorder(
        spk_recorder, unit_ids, sim_ms, bin_ms, burn_in_ms=burn_in_ms
    )

    total_spikes = int(mat.sum())
    if total_spikes < 20 or mat.shape[0] < 10:
        if verbose:
            print(f"[{label}] too few spikes/bins after burn-in trim -- skipping "
                  f"(total_spikes={total_spikes}, bins={mat.shape[0]})")
        return None

    proj, hard, soft = project_and_cluster(mat, seed=seed)
    bc = bimodality_coefficient(proj)
    dip_stat, dip_p = dip_test(proj)
    bic1, bic2, delta_bic = gmm_bic_comparison(proj, seed=seed)
    dwell_ms = dwell_times(hard, bin_ms)
    sw_rate = switch_rate(hard)

    # verdict: only call it "bimodal" if BC clears the cutoff AND at
    # least one of the two independent checks agrees (dip-test p<0.05,
    # or delta_bic > 10 favoring 2 components)
    bc_flag = (not np.isnan(bc)) and bc > 5.0 / 9.0
    dip_flag = (not np.isnan(dip_p)) and dip_p < 0.05
    bic_flag = (not np.isnan(delta_bic)) and delta_bic > 10.0
    corroborated = bc_flag and (dip_flag or bic_flag)

    if verbose:
        print(f"\n[{label}] total spikes={total_spikes}, bins={mat.shape[0]} "
              f"(after {burn_in_ms:.0f} ms burn-in)")
        print(f"[{label}] BC={bc:.3f} (cutoff 0.555){' [FLAG]' if bc_flag else ''}")
        if _HAVE_DIPTEST:
            print(f"[{label}] dip stat={dip_stat:.4f}, p={dip_p:.4f}"
                  f"{' [FLAG: reject unimodal]' if dip_flag else ' [unimodal not rejected]'}")
        else:
            print(f"[{label}] dip test skipped (package not installed)")
        print(f"[{label}] GMM 1-comp BIC={bic1:.1f}, 2-comp BIC={bic2:.1f}, "
              f"delta={delta_bic:.1f}{' [FLAG: 2-comp favored]' if bic_flag else ''}")
        print(f"[{label}] VERDICT: "
              f"{'genuinely bimodal (corroborated)' if corroborated else 'NOT corroborated -- likely BC artifact (e.g. skewed bursts)' if bc_flag else 'not bimodal'}")
        print(f"[{label}] switch_rate={sw_rate:.3f}, mean_dwell={dwell_ms.mean():.1f} ms, "
              f"n_runs={len(dwell_ms)}")

    return dict(label=label, t=t_centers, mat=mat, proj=proj, hard=hard, soft=soft,
                bc=bc, dip_stat=dip_stat, dip_p=dip_p, bic1=bic1, bic2=bic2,
                delta_bic=delta_bic, corroborated=corroborated,
                dwell_ms=dwell_ms, switch_rate=sw_rate, bin_ms=bin_ms,
                burn_in_ms=burn_in_ms)


# --------------------------------------------------------------------------
# Multi-seed sweep
# --------------------------------------------------------------------------

def run_single_seed(seed_connect, sim_ms, bin_ms, burn_in_ms,
                     theta_on=False, swr_on=False, verbose=False):
    net = build_ca1_ca3_izh(theta_on=theta_on, swr_on=swr_on, seed_connect=seed_connect)
    nest.Simulate(float(sim_ms))

    pops = {
        "CA3_PYR": (net["spk_ca3_pyr"], net["CA3_PYR"]),
        "CA3_INT": (net["spk_ca3_int"], net["CA3_INT"]),
        "CA1_PYR": (net["spk_pyr"], net["PYR"]),
        "CA1_BASKET": (net["spk_ba"], net["BASKET"]),
    }

    results = {}
    for label, (spk, ids) in pops.items():
        r = analyze_population(label, spk, ids, sim_ms, bin_ms=bin_ms,
                                burn_in_ms=burn_in_ms, seed=seed_connect, verbose=verbose)
        results[label] = r
    return results


def sweep(seeds=(1, 2, 3, 4, 5), sim_ms=8000.0, bin_ms=20.0, burn_in_ms=300.0,
          theta_on=False, swr_on=False):
    print(f"\n===== Multi-seed sweep: {len(seeds)} seeds, sim_ms={sim_ms}, "
          f"burn_in={burn_in_ms} ms =====")

    all_results = {label: [] for label in ("CA3_PYR", "CA3_INT", "CA1_PYR", "CA1_BASKET")}

    for i, seed in enumerate(seeds):
        print(f"\n--- seed {seed} ({i+1}/{len(seeds)}) ---")
        res = run_single_seed(seed, sim_ms, bin_ms, burn_in_ms,
                               theta_on=theta_on, swr_on=swr_on, verbose=True)
        for label, r in res.items():
            if r is not None:
                all_results[label].append(r)

    # Summary table
    print("\n===== Sweep summary (mean +- std across seeds) =====")
    print(f"{'population':12s} {'n':>3s} {'BC':>16s} {'dip_p':>16s} "
          f"{'delta_BIC':>16s} {'% corroborated':>16s}")
    summary = {}
    for label, runs in all_results.items():
        if not runs:
            print(f"{label:12s}   no usable runs")
            continue
        bcs = np.array([r["bc"] for r in runs])
        dips = np.array([r["dip_p"] for r in runs])
        bics = np.array([r["delta_bic"] for r in runs])
        corrob = np.array([r["corroborated"] for r in runs])
        summary[label] = dict(bc=bcs, dip_p=dips, delta_bic=bics, corroborated=corrob)

        if np.all(np.isnan(dips)):
            dip_str = f"{'n/a':>16s}"
        else:
            dip_str = f"{np.nanmean(dips):6.3f} +- {np.nanstd(dips):5.3f}"

        print(f"{label:12s} {len(runs):3d} "
              f"{bcs.mean():6.3f} +- {bcs.std():5.3f}  "
              f"{dip_str}  "
              f"{bics.mean():6.1f} +- {bics.std():5.1f}  "
              f"{100*corrob.mean():5.1f}%")

    print("\nCompare against DANU real-CA1 alpha BC = 0.243 (non-bimodal).")
    print("A population here is only trustworthy as 'discrete/bimodal' if its "
          "'% corroborated' column is high, not just its mean BC.")

    plot_sweep(summary)
    return summary


def plot_sweep(summary, out_path="outputs/tinyhippo_discreteness_sweep.png"):
    labels = [l for l in summary if summary[l]]
    if not labels:
        print("Nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bc_data = [summary[l]["bc"] for l in labels]
    try:
        # matplotlib >= 3.9
        axes[0].boxplot(bc_data, tick_labels=labels)
    except TypeError:
        # matplotlib < 3.9
        axes[0].boxplot(bc_data, labels=labels)
    axes[0].axhline(5.0 / 9.0, color="r", linestyle="--", lw=1, label="BC cutoff (0.555)")
    axes[0].axhline(0.243, color="g", linestyle="--", lw=1, label="DANU real-CA1 alpha")
    axes[0].set_ylabel("Bimodality coefficient")
    axes[0].set_title("BC across seeds")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    corrob_pct = [100 * summary[l]["corroborated"].mean() for l in labels]
    axes[1].bar(labels, corrob_pct, color="0.5")
    axes[1].set_ylabel("% seeds with corroborated bimodality")
    axes[1].set_title("Dip-test / GMM-BIC corroboration rate")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved sweep figure to {out_path}")


if __name__ == "__main__":
    sweep(seeds=(1, 2, 3, 4, 5), sim_ms=8000.0, bin_ms=20.0, burn_in_ms=300.0,
          theta_on=False, swr_on=False)