"""The preregistered test (docs/PREREG.md), or a run on your own folders.

    python run6.py                                         # natural test -> results/natural/
    python run6.py --train-folder train --eval-folder unseen --eval-limit 32 --name things
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench import Bench, ARMS, ROOT, load_params, score_all, pans, flicker
from data import natural, load_images, check_disjoint, TEST_PHOTOS
from fog import FOGS
from figures import comparison_figure, temporal_figure, atoms_figure

TEMPORAL_ARMS = ("wiener", "wiener_sample", "independent", "no_inhibition", "epll", "csc")


def boot_ratio(a, b, n=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def image_gates(sc, suffix=""):
    g = {}
    m = lambda k: sc[k]["mse_per_image"]
    r, lo, hi = boot_ratio(m("csc"), m("wiener"))
    g["G1 focus beyond linear" + suffix] = dict(
        ratio=r, ci=[lo, hi], edge_f=[sc["csc"]["edge_f"], sc["wiener"]["edge_f"]],
        passed=bool(r <= .95 and hi < 1 and sc["csc"]["edge_f"] > sc["wiener"]["edge_f"]))
    r, lo, hi = boot_ratio(m("csc"), m("independent"))
    g["G2 beats independent patches" + suffix] = dict(
        ratio=r, ci=[lo, hi], orient_deg=[sc["csc"]["orient_deg"], sc["independent"]["orient_deg"]],
        passed=bool(r <= .95 and hi < 1 and sc["csc"]["orient_deg"] < sc["independent"]["orient_deg"]))
    r, lo, hi = boot_ratio(m("csc"), m("no_inhibition"))
    g["G2b competition itself pays" + suffix] = dict(
        ratio=r, ci=[lo, hi], edge_f=[sc["csc"]["edge_f"], sc["no_inhibition"]["edge_f"]],
        passed=bool(r <= .95 and hi < 1 and sc["csc"]["edge_f"] > sc["no_inhibition"]["edge_f"]))
    r, lo, hi = boot_ratio(m("csc"), m("epll"))
    g["G3 beats Model 4 engine" + suffix] = dict(ratio=r, ci=[lo, hi], passed=bool(r <= .97 and hi < 1))
    return g


def gates(sc, fl):
    g = image_gates(sc)
    g["G4 temporal"] = dict(
        flicker=dict(csc=fl["csc"], wiener=fl["wiener"], independent=fl["independent"]),
        passed=bool(fl["csc"] <= 1.25 * fl["wiener"] and fl["csc"] <= fl["independent"]))
    ws = sc["wiener_sample"]
    g["G5 instrument: right energy, wrong places"] = dict(
        band_ratio=ws["band_ratio"], mse=[ws["mse"], sc["wiener"]["mse"]],
        edge_f=[ws["edge_f"], sc["csc"]["edge_f"]],
        passed=bool(ws["band_ratio"] >= .9 and ws["mse"] > sc["wiener"]["mse"]
                    and ws["edge_f"] < sc["csc"]["edge_f"]))
    return g


def table(sc, fl=None):
    lines = ["| arm | MSE | edge F | orient. err (deg) | fine-band ratio |" + (" flicker |" if fl else ""),
             "|---|---:|---:|---:|---:|" + ("---:|" if fl else "")]
    for arm, s in sc.items():
        f = f" {fl[arm]:.5f} |" if fl and arm in fl else (" — |" if fl else "")
        lines.append(f"| {arm} | {s['mse']:.5f} | {s['edge_f']:.3f} | {s['orient_deg']:.1f} | "
                     f"{s['band_ratio']:.2f} |" + f)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-folder")
    ap.add_argument("--eval-folder")
    ap.add_argument("--train-limit", type=int, default=3000)
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    p = load_params()
    t0 = time.time()

    if args.train_folder:
        name = args.name or "folders"
        train = load_images(args.train_folder, args.train_limit)
        test = load_images(args.eval_folder, args.eval_limit)
        check_disjoint(train, test)
        cache = ROOT / "runs" / f"fit_{name}.pkl"
        out_dir = ROOT / "runs" / name
    else:
        name = "natural"
        train, _, test = natural()
        cache = ROOT / "runs" / "fit_natural.pkl"
        out_dir = ROOT / "results" / "natural"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = Bench(train, p["depth"], p["sigma"], cache=cache)
    Y = bench.observe_set(test, seed=23)
    out = bench.run(Y, p, arms=ARMS + (("csc_converged",) if not args.train_folder else ()), seed=23)
    sc = score_all(out, test)
    print(table(sc))

    result = dict(params={k: v for k, v in p.items() if k != "tune_log"}, n_test=len(test), scores=sc)
    comparison_figure(test, out, out_dir / "comparison.png")
    atoms_figure(bench.D, out_dir / "atoms.png")

    if name == "natural":
        seqs = pans(TEST_PHOTOS, 8, 6, seed=31)
        fl = {a: [] for a in TEMPORAL_ARMS}
        rng_seed = 31
        for i, (frames, offs) in enumerate(seqs):
            Yf = bench.observe_set(frames, seed=rng_seed * 1000 + i)
            o = bench.run(Yf, p, arms=TEMPORAL_ARMS, seed=rng_seed * 1000 + i)
            for a in TEMPORAL_ARMS:
                fl[a].append(flicker(o[a], offs))
            if i == 0:
                temporal_figure(frames, Yf, o, offs, out_dir / "temporal.png")
        flm = {a: float(np.nanmean(v)) for a, v in fl.items()}
        result["flicker"] = flm
        result["flicker_per_sequence"] = fl
        result["gates"] = gates(sc, flm)
        print(table(sc, flm))
        for k, v in result["gates"].items():
            print(f"{k}: {'PASS' if v['passed'] else 'FAIL'}  {json.dumps({x: y for x, y in v.items() if x != 'passed'})}")

        # gated replication: true heat fog, matched diffusion time, its own validation tuning
        ph = load_params(fog="heat")
        bh = Bench(train, ph["depth"], p["sigma"], cache=cache, dt=ph["dt"])
        oh = bh.run(bh.observe_set(test, seed=23), ph, seed=23)
        sch = score_all(oh, test)
        comparison_figure(test, oh, out_dir / "comparison_heat.png")
        result["heat"] = dict(params={k: v for k, v in ph.items() if k != "tune_log"}, scores=sch,
                              gates=image_gates(sch, " (heat fog)"))
        print("\nheat fog (gated replication)\n" + table(sch))
        for k, v in result["heat"]["gates"].items():
            print(f"{k}: {'PASS' if v['passed'] else 'FAIL'}  {json.dumps({x: y for x, y in v.items() if x != 'passed'})}")
        # report-only: heavier Sihti fog, Sihti parameters
        f16 = FOGS["sihti16"]
        b16 = Bench(train, f16["depth"], p["sigma"], cache=cache, dt=f16["dt"])
        sc16 = score_all(b16.run(b16.observe_set(test, seed=23), p, seed=23), test)
        result["depth16_report_only"] = sc16
        print("\nSihti fog depth 16 (report only)\n" + table(sc16))

    result["seconds"] = round(time.time() - t0)
    extra = [result["depth16_report_only"], result["heat"]["scores"]] if "heat" in result else []
    for s in [result["scores"]] + extra:
        for v in s.values():
            v["mse_per_image"] = [round(x, 7) for x in v["mse_per_image"]]
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {out_dir} in {result['seconds']} s")


if __name__ == "__main__":
    main()
