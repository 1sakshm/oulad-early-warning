"""Post hoc, secondary: a flagging threshold chosen from the training data instead of the fixed 0.5.

The threshold is set on out-of-fold training predictions (5 folds grouped by student) so that 80 % of at-risk
training students are flagged. It is then applied unchanged to the later-run test predictions.
Usage: python -m oulad.threshold (from src/, after python -m oulad.run).
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import models as M
from .run import OUT, PROC, ci, group_levels, rates, student_resamples
from .splits import later_run

TARGET_RECALL = 0.8


def out_of_fold(train, y):
    """P(pass) for every training row from a model that never saw that row's student."""
    p = np.zeros(len(train))
    for fit_idx, val_idx in GroupKFold(n_splits=5).split(train, groups=train["id_student"]):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        types = M.top_types(fit)
        Xf, Xv = M.align(M.design(fit, types), M.design(val, types))
        p[val_idx] = M.boosted().fit(Xf, y[fit_idx]).predict_proba(Xv)[:, 1]
    return p


def run(cutoff):
    df = pd.read_pickle(PROC / f"features_day{cutoff}.pkl").reset_index(drop=True)
    tr, te = later_run(df)
    train, test = df.loc[tr], df.loc[te]
    y_tr = train["passed"].to_numpy()

    # Flag when P(pass) < t; pick t so that 80 % of at-risk training students fall below it.
    oof = out_of_fold(train, y_tr)
    t = float(np.quantile(oof[y_tr == 0], TARGET_RECALL))

    keys = ["code_module", "code_presentation", "id_student"]
    pred = pd.read_csv(OUT / f"later_run_predictions_day{cutoff}.csv")
    test = test.merge(pred[keys + ["p_boosted"]], on=keys, how="left")
    y, p = test["passed"].to_numpy(), test["p_boosted"].to_numpy()
    flagged = p < t
    fnr, fpr = rates(y, p - t + 0.5)  # rates() flags below 0.5; shift so it flags below t
    res = {"threshold": round(t, 4), "flag_rate": round(float(flagged.mean()), 4),
           "missed_at_risk": round(fnr, 4), "passing_flagged": round(fpr, 4), "groups": {}}

    boots = student_resamples(test)
    for attr, level in group_levels(test).items():
        level = level.to_numpy()
        rows = {}
        for lv in sorted(set(level)):
            m = level == lv
            f, g = rates(y[m], p[m] - t + 0.5)
            rows[lv] = {"n": int(m.sum()), "missed_at_risk": round(f, 4), "passing_flagged": round(g, 4)}
        res["groups"][attr] = rows
        if len(rows) == 2:  # two-level attributes: gap with a bootstrap CI
            a, b = sorted(rows)
            def gap(idx):
                ma, mb = level[idx] == a, level[idx] == b
                return rates(y[idx][mb], p[idx][mb] - t + 0.5)[0] - rates(y[idx][ma], p[idx][ma] - t + 0.5)[0]
            res["groups"][attr]["missed_gap"] = {"which": f"{b} minus {a}",
                                                 "gap": round(gap(np.arange(len(y))), 4),
                                                 "ci": ci([gap(i) for i in boots])}
    return res


if __name__ == "__main__":
    out = {f"day{c}": run(c) for c in (14, 28)}
    (OUT / "threshold.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
