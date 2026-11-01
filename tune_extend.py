"""Stage 2 of validation tuning (run after tune.py), written because stage 1 put three
choices on grid edges and showed CSC is not converged at 400 iterations (800 was worse).
Extends those grids on the SAME validation observations and makes CSC's iteration count an
explicit, recorded choice. Appends to params.json's tune_log. Never touches test data."""
import json
import time

import numpy as np

from bench import Bench, ROOT
from data import natural
from metrics import score

P = ROOT / "params.json"


def main():
    p = json.loads(P.read_text())
    log = p["tune_log"] + ["--- stage 2 (tune_extend.py): grids extended past stage-1 edges ---"]
    t0 = time.time()
    train, val, _ = natural()
    b = Bench(train, p["depth"], p["sigma"], cache=ROOT / "runs" / "fit_natural.pkl")
    Y = b.observe_set(val, seed=11)
    mse = lambda r: score(r, val)["mse"]

    res = {}
    for lam in (7e-4, 1e-3, 1.5e-3):
        for g in (0.03, 0.1, 0.3):
            res[(lam, g)] = mse([b.csc.solve(y, lam, g, iters=400) for y in Y])
            log.append(f"csc lam={lam:g} gamma={g:g} iters=400: val mse {res[(lam, g)]:.6f}")
    lam, g = min(res, key=res.get)
    its = {400: res[(lam, g)]}
    for it in (1600,):
        sols = [b.csc.solve(y, lam, g, iters=it, return_parts=True) for y in Y]
        its[it] = mse([s[0] for s in sols])
        log.append(f"csc lam={lam:g} gamma={g:g} iters={it}: val mse {its[it]:.6f}")
    # how far from the optimum of its own objective is the 400-iteration answer?
    f = lambda it, y: b.csc.objective(y, *b.csc.solve(y, lam, g, iters=it, return_parts=True)[1:], lam, g)
    rel = np.mean([(f(400, y) - f(1600, y)) / f(1600, y) for y in Y[:8]])
    log.append(f"csc objective excess at 400 vs 1600 iterations (8 val images): {100*rel:.2f}%")
    p.update(csc_lam=lam, csc_gamma=g, csc_iters=min(its, key=its.get))

    r = {}
    for lam_i in (2e-3, 4e-3, 8e-3):
        r[lam_i] = mse([b.ind.solve(y, lam_i) for y in Y])
        log.append(f"independent lam={lam_i:g}: val mse {r[lam_i]:.6f}")
    p["ind_lam"] = min(r, key=r.get)

    W = [b.gauss.wiener(y, b.A, p["sigma"]) for y in Y]
    r = {}
    for bs in (0.125, 0.25, 0.5):
        betas = tuple(bs * v for v in (1, 4, 8, 16, 32, 64))
        r[bs] = mse([b.ep.solve(y, p["sigma"], w, betas=betas) for y, w in zip(Y, W)])
        log.append(f"epll beta_scale={bs:g}: val mse {r[bs]:.6f}")
    p["epll_beta_scale"] = min(r, key=r.get)
    for k, grid in (("csc_gamma", (0.03, 0.1, 0.3)), ("ind_lam", (2e-3, 4e-3, 8e-3)),
                    ("epll_beta_scale", (0.125, 0.25, 0.5)), ("csc_lam", (7e-4, 1e-3, 1.5e-3))):
        if p[k] in (min(grid), max(grid)):
            log.append(f"NOTE after stage 2: {k}={p[k]:g} still on the edge of its extended grid")
    log.append(f"stage 2 chose csc_lam={p['csc_lam']:g} csc_gamma={p['csc_gamma']:g} "
               f"csc_iters={p['csc_iters']} ind_lam={p['ind_lam']:g} epll_beta_scale={p['epll_beta_scale']:g}")
    p["tune_log"] = log
    p["tune_seconds"] = p.get("tune_seconds", 0) + round(time.time() - t0)
    P.write_text(json.dumps(p, indent=2))
    print("\n".join(log))


if __name__ == "__main__":
    main()
