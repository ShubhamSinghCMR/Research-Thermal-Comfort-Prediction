
import os, sys, subprocess, argparse, logging
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def _call_python(file_path: Path, env=None):
    cmd = [sys.executable, str(file_path)]
    logging.info("Running: %s", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env or os.environ.copy())
    if cp.returncode != 0:
        raise SystemExit(f"Script failed: {file_path} (exit {cp.returncode})")

def _rank_results(csv_path: Path, out_path: Path):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    def _rank_env(g):
        order_cols, asc = [], []
        if "R2" in g.columns:   order_cols.append("R2");   asc.append(False)
        if "RMSE" in g.columns: order_cols.append("RMSE"); asc.append(True)
        if "MAE" in g.columns:  order_cols.append("MAE");  asc.append(True)
        gg = g.sort_values(order_cols, ascending=asc, kind="mergesort").copy()
        gg["Rank"] = range(1, len(gg) + 1)
        return gg

    # Pandas >= 2.2 adds include_groups=...
    try:
        ranked = df.groupby("Environment", group_keys=False).apply(_rank_env, include_groups=True)
    except TypeError:
        # Older pandas (no include_groups param)
        ranked = df.groupby("Environment", group_keys=False).apply(_rank_env)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    logging.info("Saved ranked results -> %s", out_path)

    winners = []
    for env, g in ranked.groupby("Environment"):
        top = g.sort_values(["Rank"]).iloc[0]
        winners.append((env, top["Model"], float(top.get("R2", float("nan")))))
    logging.info("Per-environment top by ranking: %s", winners)

def main():
    parser = argparse.ArgumentParser(description="Run ensemble, baselines, aggregate and rank results.")
    parser.add_argument("--stacker", default="", help='Override stacker method: ridge|nnls|avg|lgbm (default: config/AUTO)')
    parser.add_argument("--results_csv", default="output/comparison_all_models.csv", help="Where the unified results are written")
    parser.add_argument("--ranked_csv", default="output/comparison_all_models_ranked.csv", help="Path to save ranked results")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.stacker:
        env["FORCE_STACKER_METHOD"] = args.stacker.strip().lower()
        logging.info("FORCING STACKER METHOD: %s", env["FORCE_STACKER_METHOD"])

    # Ensure LGBM patch/warnings are active
    import utils.compat_lgbm  # noqa: F401

    # Run main ensemble
    try:
        import run_main_pipeline
        logging.info("=== Running main ensemble pipeline ===")
        run_main_pipeline.main()
    except Exception as e:
        logging.warning("Import run_main_pipeline failed (%s); fallback to subprocess.", e)
        _call_python(PROJECT_ROOT / "run_main_pipeline.py", env=env)

    # Run baselines
    try:
        from scripts import run_baselines
        logging.info("=== Running baselines ===")
        run_baselines.main()
    except Exception as e:
        logging.warning("Import scripts.run_baselines failed (%s); fallback to subprocess.", e)
        _call_python(PROJECT_ROOT / "scripts" / "run_baselines.py")

    # Aggregate & rank
    results_csv = PROJECT_ROOT / args.results_csv
    if not results_csv.exists():
        candidates = list((PROJECT_ROOT / "output").glob("**/*comparison*all*models*.csv"))
        if not candidates:
            raise SystemExit(f"Results CSV not found: {results_csv}")
        results_csv = candidates[0]
        logging.info("Auto-detected results file: %s", results_csv)

    ranked_csv = PROJECT_ROOT / args.ranked_csv
    _rank_results(results_csv, ranked_csv)

    logging.info("Done. Ranked file at: %s", ranked_csv)

if __name__ == "__main__":
    main()
