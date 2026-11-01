"""Stage 3 of validation tuning — the amendment of docs/PREREG.md (before any test run).

1. no_inhibition (Astra's tighter coordination control) on the Sihti fog: its own lam and scale.
2. The heat fog (true low-pass, matched diffusion time): every arm re-tuned on validation.
Validation observations only; the test photographs are never opened.
"""
import json
import time

from bench import Bench, ROOT
from data import natural
from fog import FOGS
from metrics import score

P = ROOT / "params.json"
NOINH = dict(lam=(3e-4, 1e-3, 3e-3), scale=(0.05, 0.15, 0.5))


def tune_noinh(b, Y, val, gamma, log, tag):
    r = {}
    for lam in NOINH["lam"]:
        for s in NOINH["scale"]:
            r[(lam, s)] = score([b.csc.solve_no_inhibition(y, lam, gamma, s) for y in Y], val)["mse"]
            log.append(f"[{tag}] no_inhibition lam={lam:g} scale={s:g}: val mse {r[(lam, s)]:.6f}")
    best = min(r, key=r.get)
    for k, v, g in (("noinh_lam", best[0], NOINH["lam"]), ("noinh_scale", best[1], NOINH["scale"])):
        if v in (min(g), max(g)):
            log.append(f"[{tag}] NOTE: {k}={v:g} on the edge of its grid")
    return dict(noinh_lam=best[0], noinh_scale=best[1])


def main():
    p = json.loads(P.read_text())
    log = p["tune_log"] + ["--- stage 3 (tune_amend.py): no_inhibition control + heat fog ---"]
    t0 = time.time()
    train, val, _ = natural()
    cache = ROOT / "runs" / "fit_natural.pkl"
    b = Bench(train, p["depth"], p["sigma"], cache=cache)
    Y = b.observe_set(val, seed=11)
    p.update(tune_noinh(b, Y, val, p["csc_gamma"], log, "sihti"))
    # epll's beta scale was still on the low edge after stage 2: extend once more
    W0 = [b.gauss.wiener(y, b.A, p["sigma"]) for y in Y]
    r = {}
    for bs in (0.03125, 0.0625, 0.125):
        betas = tuple(bs * v for v in (1, 4, 8, 16, 32, 64))
        r[bs] = score([b.ep.solve(y, p["sigma"], w, betas=betas) for y, w in zip(Y, W0)], val)["mse"]
        log.append(f"[sihti] epll beta_scale={bs:g}: val mse {r[bs]:.6f}")
    p["epll_beta_scale"] = min(r, key=r.get)
    if p["epll_beta_scale"] == 0.03125:
        log.append("[sihti] NOTE: epll_beta_scale=0.03125 on the edge of its grid")

    hf = FOGS["heat"]
    bh = Bench(train, hf["depth"], p["sigma"], cache=cache, dt=hf["dt"])
    Yh = bh.observe_set(val, seed=11)
    mse = lambda r: score(r, val)["mse"]
    h = dict(depth=hf["depth"], dt=hf["dt"])
    r = {}
    for lam in (5e-4, 1e-3, 2e-3):
        for g in (0.03, 0.1, 0.3):
            r[(lam, g)] = mse([bh.csc.solve(y, lam, g, iters=p["csc_iters"]) for y in Yh])
            log.append(f"[heat] csc lam={lam:g} gamma={g:g}: val mse {r[(lam, g)]:.6f}")
    h["csc_lam"], h["csc_gamma"] = min(r, key=r.get)
    r = {}
    for lam in (1e-3, 2e-3, 4e-3, 8e-3):
        r[lam] = mse([bh.ind.solve(y, lam) for y in Yh])
        log.append(f"[heat] independent lam={lam:g}: val mse {r[lam]:.6f}")
    h["ind_lam"] = min(r, key=r.get)
    W = [bh.gauss.wiener(y, bh.A, p["sigma"]) for y in Yh]
    log.append(f"[heat] wiener: val mse {mse(W):.6f}")
    r = {}
    for bs in (0.03125, 0.0625, 0.125, 0.25, 0.5):
        betas = tuple(bs * v for v in (1, 4, 8, 16, 32, 64))
        r[bs] = mse([bh.ep.solve(y, p["sigma"], w, betas=betas) for y, w in zip(Yh, W)])
        log.append(f"[heat] epll beta_scale={bs:g}: val mse {r[bs]:.6f}")
    h["epll_beta_scale"] = min(r, key=r.get)
    h.update(tune_noinh(bh, Yh, val, h["csc_gamma"], log, "heat"))
    log.append(f"[heat] chose {json.dumps(h)}")
    p["heat"] = h
    p["tune_log"] = log
    p["tune_seconds"] = p.get("tune_seconds", 0) + round(time.time() - t0)
    P.write_text(json.dumps(p, indent=2))
    print("\n".join(log[-40:]))


if __name__ == "__main__":
    main()
