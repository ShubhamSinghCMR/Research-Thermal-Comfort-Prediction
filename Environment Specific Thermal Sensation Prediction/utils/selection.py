import numpy as np

def stability_selection(picks_per_fold, scores_per_fold, tau=0.6, kmin=3, kmax=7):
    if not picks_per_fold:
        return [], {}, {}
    all_feats = set()
    for lst in picks_per_fold:
        all_feats.update(lst)
    freq_counts = {f:0 for f in all_feats}
    for lst in picks_per_fold:
        for f in set(lst):
            freq_counts[f] += 1
    nfolds = len(picks_per_fold)
    freq_map = {f: freq_counts[f]/nfolds for f in freq_counts}

    mean_scores = {}
    for sf in scores_per_fold:
        for f, s in sf.items():
            mean_scores.setdefault(f, []).append(s)
    mean_scores = {f: float(np.mean(v)) for f, v in mean_scores.items()}

    cands = [f for f in all_feats if freq_map.get(f,0.0) >= tau]
    if not cands:
        ordered = sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True)
        pool = [k for k,_ in ordered[:kmin]]
        return pool, freq_map, mean_scores

    ordered = sorted(cands, key=lambda f: mean_scores.get(f,0.0), reverse=True)
    k = max(kmin, min(kmax, len(ordered)))
    pool = ordered[:k]
    return pool, freq_map, mean_scores
