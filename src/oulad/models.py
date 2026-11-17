"""The four models from ANALYSIS_PLAN.md, with settings fixed before any fitting."""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .features import PROTECTED, load_clicks

BASE = ["clicks_total", "active_days", "clicks_before_start", "clicks_per_week", "days_since_active",
        "n_submitted", "n_due", "share_submitted", "mean_days_late", "mean_cma_score",
        "registered_days_before", "studied_credits", "num_of_prev_attempts"]


def top_types(train, k=10):
    """The k activity types with the most clicks in the training data."""
    cols = [c for c in train.columns if c.startswith("type_")]
    return list(train[cols].sum().sort_values(ascending=False).index[:k])


def design(df, types, protected=False):
    """Feature matrix: base features, top activity types, the rest pooled, module one-hot."""
    all_types = [c for c in df.columns if c.startswith("type_")]
    X = df[BASE + types].copy()
    X["type_other"] = df[[c for c in all_types if c not in types]].sum(axis=1)
    X = X.join(pd.get_dummies(df["code_module"], prefix="module", dtype=float))
    if protected:
        X = X.join(pd.get_dummies(df[PROTECTED].fillna("unknown"), dtype=float))
    return X


def align(X_train, X_test):
    """Drop columns with no observed value in the training rows (e.g. assessment features when nothing was
    due before the cutoff), then give the test rows the same columns."""
    X_train = X_train.loc[:, X_train.notna().any()]
    return X_train, X_test.reindex(columns=X_train.columns, fill_value=0)


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def boosted():
    return HistGradientBoostingClassifier(random_state=0)


# ---------------------------------------------------------------- sequence model

def daily_clicks(df, cutoff, types):
    """Array (enrolments, cutoff + 1 days, channels): log(1 + clicks) per day and activity type.
    Step 0 holds all clicks before day 0; the last channel pools the other types."""
    clicks = load_clicks(cutoff)
    for k in ("code_module", "code_presentation"):
        clicks[k] = clicks[k].astype(str)
    names = [t.removeprefix("type_") for t in types]
    clicks["channel"] = clicks["activity_type"].map({n: i for i, n in enumerate(names)}).fillna(len(names))
    clicks["step"] = np.where(clicks["date"] < 0, 0, clicks["date"] + 1)
    row = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_frame(
        df[["code_module", "code_presentation", "id_student"]]))
    clicks = clicks.join(row.rename("row"), on=["code_module", "code_presentation", "id_student"], how="inner")
    agg = clicks.groupby(["row", "step", "channel"])["sum_click"].sum().reset_index()
    seq = np.zeros((len(df), cutoff + 1, len(names) + 1), dtype=np.float32)
    seq[agg["row"].to_numpy(), agg["step"].to_numpy(), agg["channel"].to_numpy().astype(int)] = agg["sum_click"]
    return np.log1p(seq)


class GRUNet(nn.Module):
    def __init__(self, n_channels, n_static, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden + n_static, 1)

    def forward(self, seq, static):
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)


def fit_gru(seq_tr, static_tr, y_tr, groups_tr, seq_te, static_te, seed, epochs=20, batch=256):
    """One-layer GRU, Adam 1e-3, 20 epochs; keeps the epoch with the lowest loss on a 10 % validation split
    grouped by student. Returns P(pass) for the test rows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(seq_tr, groups=groups_tr))
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    S, Z, Y = t(seq_tr), t(static_tr), t(y_tr)
    net = GRUNet(S.shape[2], Z.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state = np.inf, None
    for _ in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            opt.zero_grad()
            loss_fn(net(S[b], Z[b]), Y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(S[va], Z[va]), Y[va]).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        return torch.sigmoid(net(t(seq_te), t(static_te))).numpy()


def static_for_gru(X_train, X_test):
    """Impute and standardise the static features, fitted on the training rows only."""
    prep = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    return prep.fit_transform(X_train), prep.transform(X_test)
