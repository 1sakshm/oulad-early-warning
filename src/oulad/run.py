"""Fit every model on both splits and both cutoffs, then compute the metrics in ANALYSIS_PLAN.md.

Usage: python -m oulad.run   (from src/, after python -m oulad.features)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import models as M
from .splits import later_run, random_splits

ROOT = Path(__file__).resolve().parents[2]
PROC, OUT = ROOT / "data" / "processed", ROOT / "results"
N_BOOT, MIN_GROUP = 1000, 200
MODELS = ["majority", "logistic", "boosted", "gru", "boosted_protected"]
PRIMARY = "boosted"


# ---------------------------------------------------------------- fitting

def fit_predict(df, tr, te, cutoff, seeds_gru):
    """P(pass) on the test rows for every model, fitted on the training rows only."""
    train, test = df.loc[tr], df.loc[te]
    y = train["passed"].to_numpy()
    types = M.top_types(train)
    Xtr, Xte = M.align(M.design(train, types), M.design(test, types))
    preds = {"majority": np.full(len(te), y.mean())}
    preds["logistic"] = M.logistic().fit(Xtr, y).predict_proba(Xte)[:, 1]
    preds["boosted"] = M.boosted().fit(Xtr, y).predict_proba(Xte)[:, 1]
    Ptr, Pte = M.align(M.design(train, types, True), M.design(test, types, True))
    preds["boosted_protected"] = M.boosted().fit(Ptr, y).predict_proba(Pte)[:, 1]

    seq = M.daily_clicks(df.loc[np.concatenate([tr, te])], cutoff, types)
    s_tr, s_te = seq[:len(tr)], seq[len(tr):]
    z_tr, z_te = M.static_for_gru(Xtr, Xte)
    runs = [M.fit_gru(s_tr, z_tr, y, train["id_student"].to_numpy(), s_te, z_te, seed) for seed in seeds_gru]
    preds["gru"] = np.mean(runs, axis=0)
    preds["gru_seed_aucs"] = [roc_auc_score(test["passed"], r) for r in runs]
    return preds


# ---------------------------------------------------------------- metrics

def student_resamples(test, n=N_BOOT, seed=0):
    """Bootstrap resamples of students; each returns the positions of all that student's test rows."""
    rng = np.random.default_rng(seed)
    pos = pd.Series(np.arange(len(test))).groupby(test["id_student"].to_numpy()).apply(np.array)
    groups = pos.to_numpy()
    return [np.concatenate(groups[rng.integers(0, len(groups), len(groups))]) for _ in range(n)]


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def group_levels(test):
    imd = test["imd_band"].map(lambda b: "unknown" if pd.isna(b) else
                               f"{(int(b.split('-')[0]) // 20) * 20}-{(int(b.split('-')[0]) // 20) * 20 + 20}%")
    return {
        "gender": test["gender"],
        "age_band": test["age_band"].map(lambda a: "0-35" if a == "0-35" else "35+"),
        "imd_quintile": imd,
        "disability": test["disability"],
    }


def rates(y, p):
    flagged = p < 0.5
    at_risk, passed = y == 0, y == 1
    fnr = (~flagged & at_risk).sum() / max(at_risk.sum(), 1)
    fpr = (flagged & passed).sum() / max(passed.sum(), 1)
    return fnr, fpr


def fairness(test, p, boots):
    y = test["passed"].to_numpy()
    out = {}
    for attr, level in group_levels(test).items():
        level = level.to_numpy()
        rows, used = {}, []
        for lv in sorted(set(level)):
            m = level == lv
            fnr, fpr = rates(y[m], p[m])
            rows[lv] = {"n": int(m.sum()), "auc": round(auc(y[m], p[m]), 4),
                        "fnr": round(fnr, 4), "fpr": round(fpr, 4)}
            if m.sum() >= MIN_GROUP:
                used.append(lv)

        def gaps(idx):
            a, f = [], []
            for lv in used:
                m = level[idx] == lv
                a.append(auc(y[idx][m], p[idx][m]))
                f.append(rates(y[idx][m], p[idx][m])[0])
            return np.nanmax(a) - np.nanmin(a), max(f) - min(f)

        point = gaps(np.arange(len(y)))
        bs = np.array([gaps(b) for b in boots])
        out[attr] = {"levels": rows, "levels_in_gap": used,
                     "auc_gap": round(point[0], 4), "auc_gap_ci": ci(bs[:, 0]),
                     "fnr_gap": round(point[1], 4), "fnr_gap_ci": ci(bs[:, 1])}
    return out


# ---------------------------------------------------------------- main

def run_cutoff(cutoff):
    df = pd.read_pickle(PROC / f"features_day{cutoff}.pkl").reset_index(drop=True)
    tr, te = later_run(df)
    test = df.loc[te]
    y = test["passed"].to_numpy()
    print(f"day {cutoff}: later run {len(tr)} train / {len(te)} test")

    later = fit_predict(df, tr, te, cutoff, seeds_gru=range(5))
    boots = student_resamples(test)
    res = {"n_train": len(tr), "n_test": len(te), "pass_rate_test": round(float(y.mean()), 4),
           "later_run": {}, "random_split": {}, "transfer_drop": {}}
    later_boot = {}
    for m in MODELS:
        p = later[m]
        later_boot[m] = np.array([auc(y[b], p[b]) for b in boots])
        res["later_run"][m] = {"auc": round(auc(y, p), 4), "auc_ci": ci(later_boot[m])}
    res["later_run"]["gru"]["auc_per_seed"] = [round(a, 4) for a in later["gru_seed_aucs"]]

    rand_boot = {m: [] for m in MODELS}
    rand_auc = {m: [] for m in MODELS}
    for seed, rtr, rte in random_splits(df, tr, te):
        print(f"  random split seed {seed}")
        pr = fit_predict(df, rtr, rte, cutoff, seeds_gru=[seed])
        rtest = df.loc[rte]
        ry = rtest["passed"].to_numpy()
        rb = student_resamples(rtest, seed=seed + 1)
        for m in MODELS:
            rand_auc[m].append(auc(ry, pr[m]))
            rand_boot[m].append([auc(ry[b], pr[m][b]) for b in rb])
    for m in MODELS:
        res["random_split"][m] = {"auc_mean": round(float(np.mean(rand_auc[m])), 4),
                                  "auc_range": [round(min(rand_auc[m]), 4), round(max(rand_auc[m]), 4)]}
        if m == "majority":
            continue
        drop = np.mean(rand_boot[m], axis=0) - later_boot[m]
        res["transfer_drop"][m] = {
            "drop": round(float(np.mean(rand_auc[m]) - auc(y, later[m])), 4),
            "ci": ci(drop),
            "p_one_sided": round(float(np.mean(drop <= 0)), 4)}

    res["fairness"] = {m: fairness(test, later[m], boots) for m in ["logistic", "boosted", "gru"]}
    keys = test[["code_module", "code_presentation", "id_student", "passed"]].assign(
        **{f"p_{m}": later[m] for m in MODELS})
    return res, keys


def main():
    OUT.mkdir(exist_ok=True)
    summary, preds = {}, {}
    for c in (14, 28):
        summary[f"day{c}"], preds[c] = run_cutoff(c)
        preds[c].to_csv(OUT / f"later_run_predictions_day{c}.csv", index=False)

    # H2: day 28 vs day 14 on the enrolments in both later-run test sets (paired by student).
    k = ["code_module", "code_presentation", "id_student"]
    both = preds[28].merge(preds[14][k + [f"p_{PRIMARY}"]], on=k, suffixes=("_28", "_14"))
    y = both["passed"].to_numpy()
    boots = student_resamples(both)
    a28, a14 = both[f"p_{PRIMARY}_28"].to_numpy(), both[f"p_{PRIMARY}_14"].to_numpy()
    diff = np.array([auc(y[b], a28[b]) - auc(y[b], a14[b]) for b in boots])
    summary["h2_day28_minus_day14"] = {"n": len(both), "diff": round(auc(y, a28) - auc(y, a14), 4),
                                       "ci": ci(diff), "p_one_sided": round(float(np.mean(diff <= 0)), 4)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("h2")}, indent=2))


if __name__ == "__main__":
    main()
