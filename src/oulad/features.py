"""Build one row per enrolment from the OULAD tables, using only what is known before the cutoff day."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
KEY = ["code_module", "code_presentation", "id_student"]
PROTECTED = ["gender", "age_band", "imd_band", "disability", "region", "highest_education"]


def read(name, **kw):
    return pd.read_csv(RAW / f"{name}.csv", na_values=["?"], **kw)


def load_clicks(max_day):
    """Clicks dated before max_day, joined to their activity type."""
    vle = read("vle", usecols=["id_site", "activity_type"])
    clicks = read(
        "studentVle",
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"},
    )
    clicks = clicks[clicks["date"] < max_day]
    return clicks.merge(vle, on="id_site", how="left")


def enrolments(cutoff):
    """studentInfo plus registration dates, minus students who left on or before the cutoff."""
    info = read("studentInfo")
    reg = read("studentRegistration")
    df = info.merge(reg, on=KEY, how="left")
    left_early = df["date_unregistration"].notna() & (df["date_unregistration"] <= cutoff)
    df = df[~left_early].copy()
    df["passed"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    df["imd_band"] = df["imd_band"].replace({"10-20": "10-20%"})
    return df, left_early.sum()


def activity_features(clicks, cutoff):
    c = clicks[clicks["date"] < cutoff]
    g = c.groupby(KEY, observed=True)
    out = pd.DataFrame({
        "clicks_total": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
        "last_active_day": g["date"].max(),
        "clicks_before_start": c[c["date"] < 0].groupby(KEY, observed=True)["sum_click"].sum(),
    })
    weeks = max(cutoff / 7, 1)
    out["clicks_per_week"] = c[c["date"] >= 0].groupby(KEY, observed=True)["sum_click"].sum() / weeks
    by_type = c.pivot_table(index=KEY, columns="activity_type", values="sum_click",
                            aggfunc="sum", fill_value=0, observed=True)
    by_type.columns = [f"type_{t}" for t in by_type.columns]
    out = out.join(by_type, how="outer").fillna({"clicks_before_start": 0, "clicks_per_week": 0})
    out["days_since_active"] = cutoff - out["last_active_day"]
    return out.drop(columns="last_active_day").reset_index()


def assessment_features(cutoff):
    """Assessments due before the cutoff. Scores only for computer-marked ones (see the deviation log)."""
    a = read("assessments")
    due = a[a["date"].notna() & (a["date"] < cutoff) & (a["assessment_type"] != "Exam")]
    sa = read("studentAssessment")
    sa = sa[(sa["is_banked"] == 0) & (sa["date_submitted"] < cutoff)]
    sub = due.merge(sa, on="id_assessment", how="inner")
    sub["days_late"] = (sub["date_submitted"] - sub["date"]).clip(lower=0)
    sub["cma_score"] = sub["score"].where(sub["assessment_type"] == "CMA")
    g = sub.groupby(KEY)
    out = pd.DataFrame({
        "n_submitted": g.size(),
        "mean_days_late": g["days_late"].mean(),
        "mean_cma_score": g["cma_score"].mean(),
    }).reset_index()
    n_due = due.groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()
    return out, n_due


def build(cutoff):
    """Feature table for one cutoff day. Returns (table, number excluded for leaving early)."""
    df, n_left = enrolments(cutoff)
    clicks = load_clicks(cutoff)
    act = activity_features(clicks, cutoff)
    asm, n_due = assessment_features(cutoff)
    for t in (act, asm):
        for k in ("code_module", "code_presentation"):
            t[k] = t[k].astype(str)
    df = df.merge(act, on=KEY, how="left").merge(asm, on=KEY, how="left")
    df = df.merge(n_due, on=["code_module", "code_presentation"], how="left")

    count_cols = [c for c in df.columns if c.startswith("type_")] + [
        "clicks_total", "active_days", "clicks_before_start", "clicks_per_week", "n_submitted", "n_due"]
    df[count_cols] = df[count_cols].fillna(0)
    df["share_submitted"] = np.where(df["n_due"] > 0, df["n_submitted"] / df["n_due"].clip(lower=1), np.nan)
    # Never active before the cutoff: count the whole window as inactive.
    df["days_since_active"] = df["days_since_active"].fillna(cutoff + 30)
    df["registered_days_before"] = -df["date_registration"]
    df["cutoff"] = cutoff
    return df, n_left


if __name__ == "__main__":
    for c in (14, 28):
        table, n_left = build(c)
        out = RAW.parent / "processed" / f"features_day{c}.pkl"
        out.parent.mkdir(exist_ok=True)
        table.to_pickle(out)
        print(f"day {c}: {len(table)} enrolments kept, {n_left} excluded for leaving on or before day {c}")
