import numpy as np
import pandas as pd
from utils.metrics import evaluate_predictions

def _percentile_rank_dict(values_dict):
    items = list(values_dict.items())
    if not items:
        return {}
    vals = np.array([v for _, v in items], dtype=float)
    finite = np.isfinite(vals)
    if finite.sum() == 0:
        return {k: 0.0 for k, _ in items}
    vmin = np.nanmin(vals[finite])
    vmax = np.nanmax(vals[finite])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return {k: 0.0 for k, _ in items}
    return {k: float((v - vmin) / (vmax - vmin)) if np.isfinite(v) else 0.0 for (k, _), v in zip(items, vals)}

def univariate_scores(X_train_df, y_train, num_cols, cat_cols):
    scores = {}
    y = pd.to_numeric(pd.Series(y_train), errors='coerce')

    for c in num_cols:
        x = pd.to_numeric(X_train_df[c], errors='coerce')
        m = (~x.isna()) & (~y.isna())
        if m.sum() < 8:
            scores[c] = 0.0
            continue
        xv = x[m].values
        yv = y[m].values
        if np.std(xv) == 0 or np.std(yv) == 0:
            scores[c] = 0.0
            continue
        r = np.corrcoef(xv, yv)[0, 1]
        scores[c] = float(r * r) if np.isfinite(r) else 0.0

    for c in cat_cols:
        g = X_train_df[c].astype(str)
        df = pd.DataFrame({'y': y, 'g': g}).dropna()
        if df['g'].nunique() < 2:
            scores[c] = 0.0
            continue
        grand = df['y'].mean()
        ssb = df.groupby('g')['y'].apply(lambda v: len(v) * (v.mean() - grand) ** 2).sum()
        sst = ((df['y'] - grand) ** 2).sum()
        scores[c] = float(ssb / sst) if sst > 0 else 0.0

    ranks = _percentile_rank_dict(scores)
    return scores, ranks

def permutation_rank(predict_fn, X_valid_trans, y_valid, original_to_cols, repeats=3):
    try:
        from scipy import sparse as _sparse
        if _sparse.issparse(X_valid_trans):
            X_valid_trans = X_valid_trans.toarray()
    except Exception:
        pass
    X_valid_trans = np.asarray(X_valid_trans)

    base_pred = predict_fn(X_valid_trans)
    base_metrics = evaluate_predictions(y_valid, base_pred)
    base_r2 = base_metrics['R2']
    base_mae = base_metrics['MAE']
    base_acc = base_metrics['Accuracy']

    dR2, dMAE, dACC = {}, {}, {}
    n = X_valid_trans.shape[0]
    for orig, idxs in original_to_cols.items():
        if len(idxs) == 0 or n == 0:
            dR2[orig] = 0.0; dMAE[orig] = 0.0; dACC[orig] = 0.0
            continue

        r2_drops, mae_incs, acc_drops = [], [], []
        for _ in range(int(repeats)):
            Xp = X_valid_trans.copy()
            perm = np.random.permutation(n)
            Xp[:, idxs] = Xp[perm][:, idxs]
            yp = predict_fn(Xp)
            m = evaluate_predictions(y_valid, yp)
            r2_drops.append(max(0.0, base_r2 - m['R2']))
            mae_incs.append(max(0.0, m['MAE'] - base_mae))
            acc_drops.append(max(0.0, base_acc - m['Accuracy']))

        dR2[orig] = float(np.mean(r2_drops)) if r2_drops else 0.0
        dMAE[orig] = float(np.mean(mae_incs)) if mae_incs else 0.0
        dACC[orig] = float(np.mean(acc_drops)) if acc_drops else 0.0

    dR2_rank = _percentile_rank_dict(dR2)
    return dR2, dMAE, dACC, dR2_rank
