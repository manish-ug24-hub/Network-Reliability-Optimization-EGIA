"""
egiza_nrop.py
=============
Enhanced Gazelle-Inspired Algorithm (EGIZA) for the Network Reliability
Optimisation Problem (NROP), implemented from the paper
"Network Reliability Optimisation Using the Enhanced Gazelle-Inspired
Algorithm (EGIZA)".

Contents
--------
1. NROPInstance      - network (complete candidate graph), costs, reliability
                       estimator, Monte-Carlo validator, repair operator
2. make_fitness      - Mode 1 (min cost s.t. R >= R0) and
                       Mode 2 (max reliability s.t. cost <= c0), penalty based
3. egiza             - Algorithm 1 of the paper
                         variant="paper"    : literal pseudocode (binary state)
                         variant="improved" : continuous latent state + greedy
                                              selection (fixes the evasion no-op)
4. ga_baseline       - simple GA stand-in for the paper's GPC baseline
5. egiza_continuous  - same dynamics for CEC-style continuous benchmarks
6. main()            - demo experiment (python egiza_nrop.py --help)

Assumptions (the paper does not specify them, so they are parameters):
  * edge costs  c_ij ~ U{1..10}      * R0 = 0.95 in Mode 1
  * c0 = 0.4 * (cost of the complete graph) in Mode 2
  * reliability estimator = node-cut (degree) product  prod_i (1 - q^{d_i}).
    Replace NROPInstance.reliability_estimate with your own Upper Bound
    Method if you have the exact formula from the GPC paper.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

LEVY_CLIP = 10.0  # guards against |v| ~ 0 blow-ups in the Levy step


# --------------------------------------------------------------------------
# 1. Problem instance
# --------------------------------------------------------------------------
class NROPInstance:
    """Complete candidate graph on n nodes; x is a binary vector over edges."""

    def __init__(self, n: int, p: float, seed: int = 0,
                 cost_low: int = 1, cost_high: int = 10):
        rng = np.random.default_rng(seed)
        self.n, self.p, self.q = n, p, 1.0 - p
        self.rows, self.cols = np.triu_indices(n, 1)
        self.D = self.rows.size                       # number of candidate edges
        self.cost = rng.integers(cost_low, cost_high + 1, self.D).astype(float)
        self.order = np.argsort(self.cost, kind="stable")   # cheapest first
        self.eid = np.full((n, n), -1, dtype=int)     # (i, j) -> edge index
        self.eid[self.rows, self.cols] = np.arange(self.D)
        self.eid[self.cols, self.rows] = np.arange(self.D)

    # ---- basic quantities -------------------------------------------------
    def total_cost(self, x) -> float:
        return float(np.dot(x, self.cost))

    def degrees(self, x) -> np.ndarray:
        m = np.asarray(x, dtype=bool)
        return (np.bincount(self.rows[m], minlength=self.n)
                + np.bincount(self.cols[m], minlength=self.n))

    def _components(self, x):
        m = np.asarray(x, dtype=bool)
        g = csr_matrix((np.ones(int(m.sum())), (self.rows[m], self.cols[m])),
                       shape=(self.n, self.n))
        return connected_components(g, directed=False)

    # ---- reliability ------------------------------------------------------
    def reliability_estimate(self, x) -> float:
        """O(n) node-cut estimator: every node must keep >= 1 working link."""
        d = self.degrees(x)
        return float(np.prod(1.0 - self.q ** d))

    def reliability_mc(self, x, samples: int = 5000, seed: int = 123) -> float:
        """Monte-Carlo all-terminal reliability (use to validate the estimator)."""
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(x)
        r, c = self.rows[idx], self.cols[idx]
        ok = 0
        for _ in range(samples):
            keep = rng.random(idx.size) < self.p
            g = csr_matrix((np.ones(int(keep.sum())), (r[keep], c[keep])),
                           shape=(self.n, self.n))
            ok += connected_components(g, directed=False)[0] == 1
        return ok / samples

    # ---- repair operator --------------------------------------------------
    def repair(self, x) -> np.ndarray:
        """Make x a connected graph with min degree >= 2 using cheapest edges.
        Cost: O(|E| log |E|) for the (pre-computed) sort + linear scans."""
        x = np.asarray(x, dtype=np.int8).copy()
        # (a) connectivity: greedily add the cheapest edge joining two components
        ncomp, labels = self._components(x)
        if ncomp > 1:
            comp = labels.copy()
            for e in self.order:
                if x[e]:
                    continue
                u, v = self.rows[e], self.cols[e]
                if comp[u] != comp[v]:
                    x[e] = 1
                    comp[comp == comp[v]] = comp[u]
                    ncomp -= 1
                    if ncomp == 1:
                        break
        # (b) every node needs degree >= 2 (no single link isolates a node)
        deg = self.degrees(x)
        for i in np.flatnonzero(deg < 2):
            while deg[i] < 2:
                cand = [j for j in range(self.n) if j != i and not x[self.eid[i, j]]]
                j = min(cand, key=lambda k: self.cost[self.eid[i, k]])
                x[self.eid[i, j]] = 1
                deg[i] += 1
                deg[j] += 1
        return x


# --------------------------------------------------------------------------
# 2. Fitness (lower is better), penalty formulation
# --------------------------------------------------------------------------
def make_fitness(inst: NROPInstance, mode: int, R0: float = 0.95,
                 c0: float | None = None, lam: float = 1000.0, mu: float = 1000.0):
    if mode == 1:      # min cost  s.t. R >= R0
        def f(x):
            return inst.total_cost(x) + lam * max(0.0, R0 - inst.reliability_estimate(x))
    elif mode == 2:    # max R  s.t. cost <= c0   (minimise -R)
        if c0 is None:
            c0 = 0.4 * inst.cost.sum()
        def f(x):
            return (-inst.reliability_estimate(x)
                    + mu * max(0.0, inst.total_cost(x) - c0) / c0)
    else:
        raise ValueError("mode must be 1 or 2")
    return f


# --------------------------------------------------------------------------
# 3. EGIZA  (Algorithm 1)
# --------------------------------------------------------------------------
def levy_sigma(s: float) -> float:
    return (math.gamma(1 + s) * math.sin(math.pi * s / 2)
            / (math.gamma((1 + s) / 2) * s * 2 ** ((s - 1) / 2))) ** (1 / s)


def levy_step(rng, D: int, s: float, sigma: float) -> np.ndarray:
    u = rng.normal(0.0, sigma, D)
    v = rng.normal(0.0, 1.0, D)
    return np.clip(u / np.abs(v) ** (1 / s), -LEVY_CLIP, LEVY_CLIP)


def _sync(x_cont, b):
    """Make the continuous latent vector consistent with the repaired bits."""
    return np.where(b == 1, np.maximum(x_cont, 0.5), np.minimum(x_cont, 0.4999))


def egiza(fitness, repair, D, N=30, max_fe=10_000, alpha_max=1.0,
          beta_min=0.1, beta_max=1.5, s=1.5, variant="improved", seed=None):
    """
    variant="paper"    : literal Algorithm 1. The state is binary after every
                         step, so the evasion move alpha*r1*(x_i - x_pred) is
                         rounded/clipped straight back to x_i (a no-op).
    variant="improved" : keeps a continuous latent position in [0,1]; bits are
                         obtained by thresholding at 0.5, then repaired. Uses
                         greedy replacement (keep the better of old/new).
    Returns (best_bits, best_fitness, history[(fe, best_so_far), ...]).
    """
    literal = variant == "paper"
    rng = np.random.default_rng(seed)
    sigma = levy_sigma(s)                       # constant -> computed once
    T_max = max(1, (max_fe - N) // N)           # same FE budget for every run

    X = rng.random((N, D))
    B = (X >= 0.5).astype(np.int8)
    fit = np.empty(N)
    fe = 0
    for i in range(N):                          # steps 1-3
        B[i] = repair(B[i])
        X[i] = B[i].astype(float) if literal else _sync(X[i], B[i])
        fit[i] = fitness(B[i])
        fe += 1
    k = int(np.argmin(fit))
    Xs, Bs, fs = X[k].copy(), B[k].copy(), fit[k]
    history = [(fe, fs)]

    for t in range(1, T_max + 1):               # step 4
        frac = t / T_max
        Pe = 0.9 - 0.8 * frac                   # exploration probability
        alpha = alpha_max * (1 - frac)          # evasion step
        beta = beta_min + (beta_max - beta_min) * frac   # grazing step
        Xp = X[int(np.argmax(fit))].copy()      # predator = worst individual

        for i in range(N):
            if rng.random() < Pe:               # predator-evasion (explore)
                cand = X[i] + alpha * rng.random() * (X[i] - Xp)
            else:                               # grazing (exploit around x*)
                L = levy_step(rng, D, s, sigma)
                cand = Xs + beta * L * (X[i] - Xs)

            if literal:                         # step 18: binary projection
                cand = np.clip(np.floor(cand + 0.5), 0, 1)
                b = cand.astype(np.int8)
            else:
                cand = np.clip(cand, 0.0, 1.0)
                b = (cand >= 0.5).astype(np.int8)

            b = repair(b)                       # step 19
            cand = b.astype(float) if literal else _sync(cand, b)
            f = fitness(b)                      # step 20
            fe += 1

            if literal or f <= fit[i]:          # paper: unconditional update
                X[i], B[i], fit[i] = cand, b, f
            if f < fs:                          # steps 21-23
                Xs, Bs, fs = cand.copy(), b.copy(), f
        history.append((fe, fs))
    return Bs, fs, history


# --------------------------------------------------------------------------
# 4. GA baseline (stand-in for GPC - NOT the authors' GPC implementation)
# --------------------------------------------------------------------------
def ga_baseline(fitness, repair, D, N=30, max_fe=10_000, pc=0.9, pm=None,
                seed=None, tour_k=3):
    rng = np.random.default_rng(seed)
    pm = pm if pm is not None else 2.0 / D
    pop = np.array([repair(rng.integers(0, 2, D)) for _ in range(N)], dtype=np.int8)
    fit = np.array([fitness(x) for x in pop])
    fe = N
    k = int(np.argmin(fit))
    best, fb = pop[k].copy(), fit[k]
    history = [(fe, fb)]
    while fe < max_fe:
        children = [best.copy()]                         # elitism
        while len(children) < N:
            a = min(rng.integers(0, N, tour_k), key=lambda j: fit[j])
            b = min(rng.integers(0, N, tour_k), key=lambda j: fit[j])
            c = pop[a].copy()
            if rng.random() < pc:
                mask = rng.random(D) < 0.5
                c = np.where(mask, pop[a], pop[b])
            flip = rng.random(D) < pm
            c = np.where(flip, 1 - c, c).astype(np.int8)
            children.append(repair(c))
        pop = np.array(children, dtype=np.int8)
        fit = np.array([fitness(x) for x in pop])
        fe += N
        k = int(np.argmin(fit))
        if fit[k] < fb:
            best, fb = pop[k].copy(), fit[k]
        history.append((fe, fb))
    return best, fb, history


# --------------------------------------------------------------------------
# 5. Continuous EGIZA (for CEC-style benchmark functions)
# --------------------------------------------------------------------------
def egiza_continuous(func, D, lb=-100.0, ub=100.0, N=30, max_fe=10_000,
                     alpha_max=1.0, beta_min=0.1, beta_max=1.5, s=1.5, seed=None):
    rng = np.random.default_rng(seed)
    sigma = levy_sigma(s)
    T_max = max(1, (max_fe - N) // N)
    X = lb + rng.random((N, D)) * (ub - lb)
    fit = np.array([func(x) for x in X])
    k = int(np.argmin(fit))
    Xs, fs = X[k].copy(), fit[k]
    history = [(N, fs)]
    fe = N
    for t in range(1, T_max + 1):
        frac = t / T_max
        Pe = 0.9 - 0.8 * frac
        alpha = alpha_max * (1 - frac)
        beta = beta_min + (beta_max - beta_min) * frac
        Xp = X[int(np.argmax(fit))].copy()
        for i in range(N):
            if rng.random() < Pe:
                cand = X[i] + alpha * rng.random() * (X[i] - Xp)
            else:
                cand = Xs + beta * levy_step(rng, D, s, sigma) * (X[i] - Xs)
            cand = np.clip(cand, lb, ub)
            f = func(cand)
            fe += 1
            if f <= fit[i]:
                X[i], fit[i] = cand, f
            if f < fs:
                Xs, fs = cand.copy(), f
        history.append((fe, fs))
    return Xs, fs, history


# --------------------------------------------------------------------------
# 6. Demo experiment
# --------------------------------------------------------------------------
def run_mode(inst, mode, runs, max_fe, R0, c0_frac, plot_path=None):
    c0 = c0_frac * inst.cost.sum()
    fit_fn = make_fitness(inst, mode, R0=R0, c0=c0)
    algos = {
        "EGIZA (paper-literal)": lambda s: egiza(fit_fn, inst.repair, inst.D,
                                                  max_fe=max_fe, variant="paper", seed=s),
        "EGIZA (improved)":      lambda s: egiza(fit_fn, inst.repair, inst.D,
                                                  max_fe=max_fe, variant="improved", seed=s),
        "GA stand-in for GPC":   lambda s: ga_baseline(fit_fn, inst.repair, inst.D,
                                                        max_fe=max_fe, seed=s),
    }
    label = "Mode 1: min cost s.t. R>=%.2f" % R0 if mode == 1 else \
            "Mode 2: max R s.t. cost<=%.1f" % c0
    print(f"\n=== n={inst.n}, p={inst.p}, {label}, {runs} runs, {max_fe} FE ===")
    print(f"{'algorithm':24s}{'mean fit':>12s}{'std':>10s}{'mean cost':>11s}"
          f"{'mean R_est':>12s}{'R_MC(best)':>12s}{'time/run':>10s}")
    curves = {}
    for name, runner in algos.items():
        fits, costs, rels, hists = [], [], [], []
        t0 = time.time()
        best_overall = None
        for r in range(runs):
            x, f, h = runner(1000 + r)
            fits.append(f); costs.append(inst.total_cost(x))
            rels.append(inst.reliability_estimate(x)); hists.append([v for _, v in h])
            if best_overall is None or f < best_overall[1]:
                best_overall = (x, f)
        dt = (time.time() - t0) / runs
        mc = inst.reliability_mc(best_overall[0])
        print(f"{name:24s}{np.mean(fits):12.4f}{np.std(fits):10.4f}{np.mean(costs):11.2f}"
              f"{np.mean(rels):12.5f}{mc:12.5f}{dt:9.2f}s")
        L = min(map(len, hists))
        curves[name] = np.mean([h[:L] for h in hists], axis=0)
    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            for name, c in curves.items():
                plt.plot(np.linspace(0, max_fe, len(c)), c, label=name)
            plt.xlabel("Function evaluations"); plt.ylabel("Mean best fitness")
            plt.title(label); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig(plot_path, dpi=150); plt.close()
            print("saved", plot_path)
        except Exception as e:  # matplotlib optional
            print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser(description="EGIZA for network reliability optimisation")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p", type=float, default=0.8)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--fe", type=int, default=10_000)
    ap.add_argument("--R0", type=float, default=0.95)
    ap.add_argument("--c0_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    inst = NROPInstance(a.n, a.p, seed=a.seed)
    for mode in (1, 2):
        run_mode(inst, mode, a.runs, a.fe, a.R0, a.c0_frac,
                 plot_path=f"convergence_mode{mode}.png" if a.plot else None)

    # sanity check of the continuous variant on a CEC-style function
    rastrigin = lambda x: 10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    _, fbest, _ = egiza_continuous(rastrigin, D=10, lb=-5.12, ub=5.12, seed=1)
    print(f"\nContinuous EGIZA on Rastrigin (D=10, 10k FE): best = {fbest:.4f}")


if __name__ == "__main__":
    main()
