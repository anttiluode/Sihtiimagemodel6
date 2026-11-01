"""POST-HOC diagnostics (written after the test results were seen; not gates).

Why did CSC lose the edge-F clauses of G2b while winning MSE by 27%? Splits edge-F into
precision (are the result's edges real?) and recall (are the truth's edges found?), and
measures spurious edges in flat regions (truth gradient energy in its lowest third).
Same observations, seeds and parameters as run6.py.
"""
import json

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

from bench import Bench, ROOT, load_params
from data import natural
from metrics import edges, edge_counts

ARMS = ("wiener", "independent", "no_inhibition", "epll", "csc")


def split(recons, truths):
    c = np.array([edge_counts(x, t) for x, t in zip(recons, truths)], float).sum(0)
    flat_fp, flat_px = 0, 0
    for x, t in zip(recons, truths):
        g = gaussian_gradient_magnitude(t, 1.0)
        flat = g <= np.quantile(g, 1 / 3)
        flat_fp += (edges(x) & flat).sum()
        flat_px += flat.sum()
    return dict(precision=c[0] / max(c[1], 1), recall=c[2] / max(c[3], 1),
                edges_per_image=c[1] / len(truths), truth_edges_per_image=c[3] / len(truths),
                flat_spurious_rate=flat_fp / flat_px)


def main():
    train, _, test = natural()
    cache = ROOT / "runs" / "fit_natural.pkl"
    res = {}
    for fog in ("sihti", "heat"):
        p = load_params(fog=fog)
        dt = p.get("dt", 0.25)
        b = Bench(train, p["depth"], p["sigma"], cache=cache, dt=dt)
        out = b.run(b.observe_set(test, seed=23), p, arms=ARMS, seed=23)
        res[fog] = {a: split(out[a], test) for a in ARMS}
        res[fog]["truth"] = split(test, test)
    (ROOT / "results" / "natural" / "posthoc_edges.json").write_text(json.dumps(res, indent=1))
    for fog, r in res.items():
        print(f"\n{fog} fog | precision | recall | edges/img (truth {r['truth']['truth_edges_per_image']:.0f}) | spurious edge rate in flat third")
        for a in ARMS:
            v = r[a]
            print(f"{a:14s} {v['precision']:.3f} {v['recall']:.3f} {v['edges_per_image']:6.1f} {v['flat_spurious_rate']:.4f}")


if __name__ == "__main__":
    main()
