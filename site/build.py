"""Build site/index.html from template.html and the result files, so every number on the page comes from results/.

Usage: python site/build.py   (from the repo root, after python -m oulad.run and python -m oulad.threshold)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def curves(pred, col="p_boosted"):
    """Share of at-risk students missed and share of all students flagged, for thresholds 0.00..1.00,
    overall and per age band and gender, plus a 50-bin histogram of P(pass) by outcome."""
    y, p = pred["passed"].to_numpy(), pred[col].to_numpy()
    ts = np.round(np.arange(0, 1.0001, 0.01), 2)
    groups = {"all": np.ones(len(y), bool),
              "age_0_35": (pred["age_band"] == "0-35").to_numpy(),
              "age_35_plus": (pred["age_band"] != "0-35").to_numpy(),
              "female": (pred["gender"] == "F").to_numpy(),
              "male": (pred["gender"] == "M").to_numpy()}
    out = {"t": ts.tolist(), "missed": {}, "flagged": {}, "passing_flagged": {}}
    for g, m in groups.items():
        risk, ok = (y == 0) & m, (y == 1) & m
        out["missed"][g] = [round(float((p[risk] >= t).mean()), 4) for t in ts]
        out["flagged"][g] = [round(float((p[m] < t).mean()), 4) for t in ts]
        out["passing_flagged"][g] = [round(float((p[ok] < t).mean()), 4) for t in ts]
    edges = np.linspace(0, 1, 51)
    out["hist"] = {"edges": edges.round(2).tolist(),
                   "at_risk": np.histogram(p[y == 0], edges)[0].tolist(),
                   "passed": np.histogram(p[y == 1], edges)[0].tolist()}
    return out


def main():
    summary = json.loads((RES / "summary.json").read_text())
    threshold = json.loads((RES / "threshold.json").read_text())
    feats = pd.read_pickle(ROOT / "data" / "processed" / "features_day14.pkl")
    pred = pd.read_csv(RES / "later_run_predictions_day14.csv")
    keys = ["code_module", "code_presentation", "id_student"]
    pred = pred.merge(feats[keys + ["age_band", "gender"]], on=keys, how="left")
    total = len(pd.read_csv(ROOT / "data" / "raw" / "studentInfo.csv", usecols=["id_student"]))
    counts = {}
    for c in (14, 28):
        kept = len(pd.read_pickle(ROOT / "data" / "processed" / f"features_day{c}.pkl"))
        counts[f"day{c}"] = {"enrolments": total, "kept": kept, "excluded": total - kept}
    data = {"summary": summary, "threshold": threshold, "curves_day14": curves(pred), "counts": counts}
    html = (ROOT / "site" / "template.html").read_text(encoding="utf8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (ROOT / "site" / "index.html").write_text(html, encoding="utf8")
    print(f"wrote site/index.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
