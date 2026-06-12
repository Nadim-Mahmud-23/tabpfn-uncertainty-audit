"""Monte Carlo validation of the Mondrian noise floor (paper Sec. 4).

Simulates the *exact* finite-sample law of realized per-group coverage under
split conformal prediction -- no experimental data needed.

Facts used (see Vovk 2012; Angelopoulos & Bates 2023, Sec. 3):
  * With n calibration points and k = ceil((n+1)(1-alpha)), the conditional
    coverage given the calibration draw is C ~ Beta(k, n+1-k).
  * Empirical coverage measured on a test stratum of size m is then
    Binomial(m, C) / m  (test-side measurement noise).

Two regimes are simulated for a taxonomy with per-group calibration sizes
{n_g} and test sizes {m_g}:
  MONDRIAN   : each group g gets its own threshold -> independent
               C_g ~ Beta(k_g, n_g+1-k_g), then Binomial(m_g, C_g)/m_g.
  ZERO-BIAS  : all groups share the global threshold (n = sum n_g) and all
  MARGINAL     structural biases b_g are zero -> common C ~ Beta(k, n+1-k),
               per-group Binomial(m_g, C)/m_g. This is the gap one observes
               under marginal calibration *even when there is nothing to fix*.

Outputs: printed table of E[gap] (+/- sd) for the paper's axes, and
figures/mc_noise_floor.pdf (simulated E[gap] vs min_g n_g, with the
closed-form heuristic floor overlaid).

Run:  python scripts/mc_noise_floor.py            (~10 s, CPU only)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
REPS = 200_000
TEST_RATIO = 2400 / 2000  # test split / calibration split in the paper

# Per-group calibration sizes, seed 0 (paper Table 1).
AXES = {
    "Sex (Employment)": [1046, 954],
    "Sex (Income)": [961, 1039],
    "Sex (PublicCov)": [1112, 888],
    "Race (Employment)": [1271, 312, 333, 84],
    "Race (Income)": [1194, 342, 378, 86],
    "Race (PublicCov)": [1122, 308, 439, 131],
    "Age (Employment)": [577, 393, 406, 272, 352],
    "Age (Income)": [231, 664, 625, 340, 140],
    "Age (PublicCov)": [651, 557, 450, 342],
}


def k_of(n, alpha):
    return int(np.ceil((n + 1) * (1 - alpha)))


def simulate_gap(n_groups, alpha, reps=REPS, mondrian=True, rng=RNG):
    """Return per-rep max-min empirical per-group coverage."""
    n_groups = np.asarray(n_groups)
    m_groups = np.maximum(1, np.round(n_groups * TEST_RATIO).astype(int))
    G = len(n_groups)
    cov = np.empty((reps, G))
    if mondrian:
        for j, (n_g, m_g) in enumerate(zip(n_groups, m_groups)):
            k = k_of(n_g, alpha)
            C = rng.beta(k, n_g + 1 - k, size=reps)
            cov[:, j] = rng.binomial(m_g, C) / m_g
    else:  # zero-bias marginal: one shared threshold from the pooled set
        n = int(n_groups.sum())
        k = k_of(n, alpha)
        C = rng.beta(k, n + 1 - k, size=reps)
        for j, m_g in enumerate(m_groups):
            cov[:, j] = rng.binomial(m_g, C) / m_g
    return cov.max(axis=1) - cov.min(axis=1)


def heuristic_floor(n_groups, alpha):
    n_min = min(n_groups)
    G = len(n_groups)
    return np.sqrt(alpha * (1 - alpha) / n_min) * np.sqrt(2 * np.log(G))


def main():
    print(f"{'axis':24s} {'alpha':>5s} {'floor':>7s} {'E[gap] Mond':>12s} "
          f"{'E[gap] zero-bias marg':>22s}")
    rows = {}
    for name, ngs in AXES.items():
        for alpha in (0.20, 0.10, 0.05):
            g_m = simulate_gap(ngs, alpha, mondrian=True)
            g_0 = simulate_gap(ngs, alpha, mondrian=False)
            fl = heuristic_floor(ngs, alpha)
            rows[(name, alpha)] = (fl, g_m.mean(), g_m.std(), g_0.mean(), g_0.std())
            print(f"{name:24s} {alpha:5.2f} {fl:7.3f} "
                  f"{g_m.mean():6.3f} +/- {g_m.std():5.3f} "
                  f"{g_0.mean():10.3f} +/- {g_0.std():5.3f}")

    # ---- figure: E[gap] vs min n_g at alpha = 0.1 -----------------------
    alpha = 0.10
    nmins = np.unique(np.round(np.geomspace(25, 2000, 24)).astype(int))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    colors = {2: "#4477AA", 4: "#EE6677", 5: "#228833", 8: "#AA3377"}
    for G in (2, 4, 5, 8):
        sim = []
        for nm in nmins:
            # smallest stratum nm; remaining strata 6x larger (qualitatively
            # matching the paper's imbalanced axes)
            ngs = [nm] + [6 * nm] * (G - 1)
            sim.append(simulate_gap(ngs, alpha, reps=40_000).mean())
        ax.plot(nmins, sim, "-", color=colors[G], lw=1.6,
                label=rf"simulated, $|\mathcal{{G}}|={G}$")
        ax.plot(nmins, [heuristic_floor([nm] + [6 * nm] * (G - 1), alpha)
                        for nm in nmins],
                "--", color=colors[G], lw=1.1, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"smallest calibration stratum $\min_g n_g$")
    ax.set_ylabel(r"expected coverage gap $\mathbb{E}[\max_g \hat c_g - \min_g \hat c_g]$")
    ax.legend(frameon=False, fontsize=8,
              title=r"solid: Monte Carlo $\;$ dashed: Eq. floor", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/mc_noise_floor.pdf")
    print("\nwrote figures/mc_noise_floor.pdf")


if __name__ == "__main__":
    main()
