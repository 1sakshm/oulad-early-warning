"""Descriptive, post hoc numbers for the website (logged in the deviation log). Nothing here changes the planned results.

Usage: python -m oulad.describe   (from src/, after python -m oulad.run)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import models as M
from .features import KEY, load_clicks, read
from .run import OUT, PROC
from .splits import later_run


def unit_sample(test, p, n=2000, seed=0):
    """A random sample of later-run test students for the dot chart: P(pass), outcome, age band, gender."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test), size=min(n, len(test)), replace=False)
    t = test.iloc[idx]
    return {"p": np.round(p[idx], 3).tolist(),
            "passed": t["passed"].astype(int).tolist(),
            "older": (t["age_band"] != "0-35").astype(int).tolist(),
            "female": (t["gender"] == "F").astype(int).tolist(),
            "result": t["final_result"].map({"Pass": 0, "Distinction": 1, "Fail": 2, "Withdrawn": 3}).tolist(),
            "n_total": int(len(test))}


def click_curves(df, first=-14, last=27):
    """Mean clicks per student per day, by outcome and age band, for enrolments still registered on day 28."""
    clicks = load_clicks(last + 1)
    for k in ("code_module", "code_presentation"):
        clicks[k] = clicks[k].astype(str)
    clicks = clicks[clicks["date"] >= first]
    grp = df[KEY].assign(group=np.where(df["passed"] == 1, "pass", "risk") + "_" +
                         np.where(df["age_band"] == "0-35", "young", "older"))
    sizes = grp["group"].value_counts()
    c = clicks.merge(grp, on=KEY, how="inner").groupby(["group", "date"])["sum_click"].sum()
    days = list(range(first, last + 1))
    out = {"days": days, "n": {g: int(v) for g, v in sizes.items()}}
    for g in sizes.index:
        out[g] = [round(float(c.get((g, d), 0)) / sizes[g], 2) for d in days]
    return out


def grouped_importance(model, X, y, groups, repeats=5, seed=0):
    """Drop in AUROC when a whole group of columns is shuffled together."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[name] = {"drop": round(float(np.mean(drops)), 4), "sd": round(float(np.std(drops)), 4)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["drop"]))


def main():
    df = pd.read_pickle(PROC / "features_day14.pkl").reset_index(drop=True)
    tr, te = later_run(df)
    train, test = df.loc[tr], df.loc[te]
    types = M.top_types(train)
    Xtr, Xte = M.align(M.design(train, types), M.design(test, types))
    model = M.boosted().fit(Xtr, train["passed"].to_numpy())
    p = model.predict_proba(Xte)[:, 1]
    y = test["passed"].to_numpy()

    groups = {
        "Days since last active": ["days_since_active"],
        "Total clicks and clicks per week": ["clicks_total", "clicks_per_week"],
        "Number of active days": ["active_days"],
        "Clicks before the course started": ["clicks_before_start"],
        "Clicks by activity type": [c for c in Xte.columns if c.startswith("type_")],
        "Assessments submitted and lateness": ["n_submitted", "n_due", "share_submitted", "mean_days_late", "mean_cma_score"],
        "When they registered": ["registered_days_before"],
        "Credits studied": ["studied_credits"],
        "Previous attempts": ["num_of_prev_attempts"],
        "Which module": [c for c in Xte.columns if c.startswith("module_")],
    }
    per_module = {}
    for m in sorted(test["code_module"].unique()):
        mask = (test["code_module"] == m).to_numpy()
        per_module[m] = {"n": int(mask.sum()), "auc": round(float(roc_auc_score(y[mask], p[mask])), 4),
                         "pass_rate": round(float(y[mask].mean()), 4)}

    info = read("studentInfo")
    mix = info.groupby("code_module")["final_result"].value_counts().unstack(fill_value=0)
    reg = read("studentRegistration")
    w = info.merge(reg, on=KEY)
    w = w[(w["final_result"] == "Withdrawn") & w["date_unregistration"].notna()]
    edges = list(range(-120, 281, 20))
    hist = np.histogram(w["date_unregistration"].clip(-120, 279), edges)[0]

    d28 = pd.read_pickle(PROC / "features_day28.pkl")
    d28 = d28[d28["code_module"].isin(test["code_module"].unique())]
    out = {
        "auc_check": round(float(roc_auc_score(y, p)), 4),
        "units": unit_sample(test, p),
        "clicks": click_curves(d28),
        "importance": grouped_importance(model, Xte, y, groups),
        "per_module": per_module,
        "outcomes": {"overall": info["final_result"].value_counts().to_dict(),
                     "by_module": {m: mix.loc[m].to_dict() for m in mix.index}},
        "withdrawal_days": {"edges": edges, "counts": hist.tolist(), "n": int(len(w))},
    }
    (OUT / "describe.json").write_text(json.dumps(out, indent=1))
    print("AUROC check", out["auc_check"], "| importance:", list(out["importance"].items())[:4])


if __name__ == "__main__":
    main()
