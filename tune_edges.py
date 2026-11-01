"""Stage 4 of validation tuning: extend every choice that stage 3 left on a grid edge.
Validation observations only. Appends to params.json's tune_log."""
import json
import time

from bench import Bench, ROOT
from data import natural
from metrics import score

P = ROOT / "params.json"
NI_LAM, NI_SCALE = (0.003, 0.01, 0.03, 0.1), (0.005, 0.015, 0.05)


def noinh(b, Y, val, gamma, log, tag):
    r = {}
    for lam in NI_LAM:
        for s in NI_SCALE:
            r[(lam, s)] = score([b.csc.solve_no_inhibition(y, lam, gamma, s) for y in Y], val)["mse"]
            log.append(f"[{tag}] no_inhibition lam={lam:g} scale={s:g}: val mse {r[(lam, s)]:.6f}")
    lam, s = min(r, key=r.get)
    if lam in (min(NI_LAM), max(NI_LAM)) or s in (min(NI_SCALE), max(NI_SCALE)):
        log.append(f"[{tag}] NOTE: no_inhibition choice lam={lam:g} scale={s:g} on the edge of its extended grid")
    return dict(noinh_lam=lam, noinh_scale=s)


def main():
    p = json.loads(P.read_text())
    log = p["tune_log"] + ["--- stage 4 (tune_edges.py): extend stage-3 edge choices ---"]
    t0 = time.time()
    train, val, _ = natural()
    cache = ROOT / "runs" / "fit_natural.pkl"
    b = Bench(train, p["depth"], p["sigma"], cache=cache)
    Y = b.observe_set(val, seed=11)
    p.update(noinh(b, Y, val, p["csc_gamma"], log, "sihti"))

    h = p["heat"]
    bh = Bench(train, h["depth"], p["sigma"], cache=cache, dt=h["dt"])
    Yh = bh.observe_set(val, seed=11)
    mse = lambda r: score(r, val)["mse"]
    r = {}
    for g in (0.003, 0.01, 0.03):
        r[g] = mse([bh.csc.solve(y, h["csc_lam"], g, iters=p["csc_iters"]) for y in Yh])
        log.append(f"[heat] csc lam={h['csc_lam']:g} gamma={g:g}: val mse {r[g]:.6f}")
    h["csc_gamma"] = min(r, key=r.get)
    r = {}
    for lam in (0.00025, 0.0005, 0.001):
        r[lam] = mse([bh.ind.solve(y, lam) for y in Yh])
        log.append(f"[heat] independent lam={lam:g}: val mse {r[lam]:.6f}")
    h["ind_lam"] = min(r, key=r.get)
    h.update(noinh(bh, Yh, val, h["csc_gamma"], log, "heat"))
    for k, v, g in (("heat csc_gamma", h["csc_gamma"], (0.003, 0.01, 0.03)),
                    ("heat ind_lam", h["ind_lam"], (0.00025, 0.0005, 0.001))):
        if v == min(g):
            log.append(f"NOTE: {k}={v:g} on the low edge of its extended grid")
    log.append(f"stage 4 final: sihti noinh_lam={p['noinh_lam']:g} noinh_scale={p['noinh_scale']:g}; heat {json.dumps(h)}")
    p["heat"] = h
    p["tune_log"] = log
    p["tune_seconds"] = p.get("tune_seconds", 0) + round(time.time() - t0)
    P.write_text(json.dumps(p, indent=2))
    print("\n".join(l for l in log if "stage 4" in l or "NOTE" in l or "chose" in l or "final" in l))


if __name__ == "__main__":
    main()
