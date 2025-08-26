
import os, sys, logging
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STACKER_CANDIDATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main():
    # Ensure LGBM patch/warnings are active
    import utils.compat_lgbm  # noqa: F401
    import run_main_pipeline
    for m in STACKER_CANDIDATES:
        os.environ["FORCE_STACKER_METHOD"] = m
        logging.info(f"=== RUN with STACKING_METHOD={m} ===")
        run_main_pipeline.main()

if __name__ == "__main__":
    main()
