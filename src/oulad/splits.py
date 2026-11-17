"""The two ways of splitting enrolments into train and test, as fixed in ANALYSIS_PLAN.md."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .features import read


def modules_with_both_years():
    """Modules with at least one 2013 and one 2014 presentation, chosen from `courses` by rule."""
    courses = read("courses")
    years = courses.assign(year=courses["code_presentation"].str[:4]).groupby("code_module")["year"].agg(set)
    return sorted(m for m, ys in years.items() if {"2013", "2014"} <= ys)


def later_run(df):
    """Train on 2013 presentations, test on 2014 ones; test students never appear in training."""
    df = df[df["code_module"].isin(modules_with_both_years())]
    year = df["code_presentation"].str[:4]
    train = df[year == "2013"]
    test = df[(year == "2014") & ~df["id_student"].isin(train["id_student"])]
    return train.index.to_numpy(), test.index.to_numpy()


def random_splits(df, train_idx, test_idx, seeds=range(5)):
    """The same enrolments, split by student with the later-run test fraction."""
    pool = df.loc[np.concatenate([train_idx, test_idx])]
    frac = len(test_idx) / len(pool)
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        tr, te = next(gss.split(pool, groups=pool["id_student"]))
        yield seed, pool.index[tr].to_numpy(), pool.index[te].to_numpy()
