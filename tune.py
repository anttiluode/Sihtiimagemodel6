"""Validation-only tuning. Writes params.json. Never opens the test photographs.

Every arm gets its own grid on the SAME validation observations, selected by
validation MSE (the only metric used for selection).
"""
import argparse
import json
import time

import numpy as np

from bench import Bench, ROOT
from data import natural
from metrics import score

GRID = dict(csc_lam=(5e-4, 1e-3, 2e-3), csc_gamma=(0.3, 1.0, 3.0),
            ind_lam=(1e-4, 3e-4, 1e-3, 2e-3), epll_beta_scale=(0.5, 1.0, 2.0))
DEPTH, SIGMA, ITERS = 8, 0.01, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "params.json"))
    args = ap.parse_args()
    t0 = time.time()
    train, val, _ = natural()                      # the test crops are generated but never used here
    bench = Bench(train, DEPTH, SIGMA, cache=ROOT / "runs" / "fit_natural.pkl")
    Y = bench.observe_set(val, seed=11)
    log = []

    def mse_of(recons):
        return score(recons, val)["mse"]

    best = {}
    res = {}
    for lam in GRID["csc_lam"]:
        for g in GRID["csc_gamma"]:
            m = mse_of([bench.csc.solve(y, lam, g, iters=ITERS) for y in Y])
            res[(lam, g)] = m
            log.append(f"csc lam={lam:g} gamma={g:g}: val mse {m:.6f}")
    (best["csc_lam"], best["csc_gamma"]) = min(res, key=res.get)
    res = {}
    for lam in GRID["ind_lam"]:
        res[lam] = mse_of([bench.ind.solve(y, lam) for y in Y])
        log.append(f"independent lam={lam:g}: val mse {res[lam]:.6f}")
    best["ind_lam"] = min(res, key=res.get)
    res = {}
    W = [bench.gauss.wiener(y, bench.A, SIGMA) for y in Y]
    for bs in GRID["epll_beta_scale"]:
        b = tuple(bs * v for v in (1, 4, 8, 16, 32, 64))
        res[bs] = mse_of([bench.ep.solve(y, SIGMA, w, betas=b) for y, w in zip(Y, W)])
        log.append(f"epll beta_scale={bs:g}: val mse {res[bs]:.6f}")
    best["epll_beta_scale"] = min(res, key=res.get)
    log.append(f"wiener (no knobs): val mse {mse_of(W):.6f}")
    # convergence check for the chosen CSC setting
    m2 = mse_of([bench.csc.solve(y, best["csc_lam"], best["csc_gamma"], iters=2 * ITERS) for y in Y])
    log.append(f"csc chosen setting at {2*ITERS} iterations: val mse {m2:.6f} (convergence check)")
    for k in GRID:
        if best[k] in (min(GRID[k]), max(GRID[k])):
            log.append(f"NOTE: {k}={best[k]:g} sits on the edge of its grid")
    params = dict(depth=DEPTH, sigma=SIGMA, csc_iters=ITERS, **{k: float(v) for k, v in best.items()},
                  tune_log=log, tune_seconds=round(time.time() - t0))
    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    print("\n".join(log))
    print(json.dumps({k: v for k, v in params.items() if k != "tune_log"}, indent=2))


if __name__ == "__main__":
    main()
