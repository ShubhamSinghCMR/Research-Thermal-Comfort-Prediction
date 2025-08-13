import os
import itertools
import importlib
import pandas as pd
from pathlib import Path

def set_cfg(selection_mode, selection_scope, use_stacking, calibration_on, kmin, kmax):
    from utils import config as C
    C.SELECTION_MODE = selection_mode
    C.SELECTION_SCOPE = selection_scope
    C.USE_STACKING = use_stacking
    C.CALIBRATION_ON = calibration_on
    C.KMIN = kmin
    C.KMAX = kmax
    return C

def run_once(outdir):
    os.environ['OUTPUT_DIR'] = str(outdir)
    import run_main_pipeline as rmp
    importlib.reload(rmp)
    rmp.main()

def main():
    base_out = Path("output/ablations")
    base_out.mkdir(parents=True, exist_ok=True)

    selection_modes = ['none', 'univariate_only', 'permutation_only', 'hybrid']
    selection_scopes = ['per_model', 'global']
    stacking_opts = [False, True]
    calib_opts = [False, True]
    k_values = [(3,3), (5,5), (7,7)]

    rows = []
    grid = list(itertools.product(selection_modes, selection_scopes, stacking_opts, calib_opts, k_values))
    for (smode, sscope, stack_on, calib_on, (kmin, kmax)) in grid:
        set_cfg(smode, sscope, stack_on, calib_on, kmin, kmax)
        variant = f"sm_{smode}__sc_{sscope}__stack_{int(stack_on)}__cal_{int(calib_on)}__k{kmin}"
        outdir = base_out / variant
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n==== Running {variant} ====")
        run_once(outdir)

        sum_path = outdir / "predicted_tsv_results.csv"
        if sum_path.exists():
            df = pd.read_csv(sum_path)
            df.insert(0, 'Variant', variant)
            rows.append(df)

    if rows:
        big = pd.concat(rows, axis=0, ignore_index=True)
        big.to_csv(base_out / "ablations_summary.csv", index=False)
        print(f"\nSaved ablations summary → {base_out / 'ablations_summary.csv'}")

if __name__ == "__main__":
    main()
