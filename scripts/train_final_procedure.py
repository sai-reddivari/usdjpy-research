"""Train the locked final procedure on development data and freeze it. Fits models: guarded.

    python scripts/train_final_procedure.py --allow-fitting --output DIR

This is the freeze stage of data_review/run_reserved_year.py: stages A-C
under the Run B settings with the elastic net seeded (random_state=0), the
whole Stage C shortlist passed to each booster, the sixteen-configuration
grids, fold-local early stopping, refit with inner fold 3's iteration count,
inner-only thresholds, and the own-price and constant baselines. The
procedure is specified in data_review/manifests/usdjpy_locked_procedure.md.

Safeguards:
- Nothing runs without --allow-fitting.
- --output must be a new folder outside the research tree, the private folder
  and the report folders. The frozen models of the study
  (data_review/data/results/reserved_year/frozen) can never be overwritten.
- Only development inputs are read: the 1,018 eligible development rows
  through 2025-10-20, outcomes through 2025-10-21. Reserved files are never
  opened. The licensed inputs are required (no 35-feature fallback).
- Installed library versions must equal those recorded at the freeze.

Re-running this does not reproduce the study's reserved-year evaluation; the
frozen models under data_review are the ones that were scored. A re-run on the
same machine and versions is expected, but not guaranteed, to give the same
models (tests/test_synthetic_differential.py exercises the wiring on synthetic
data only; no real-data refit was performed during the refactor).
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas  # noqa: E402

from usdjpy import config, experiments, models, selection, splits  # noqa: E402
from usdjpy.config import END, ET, LOCKED_VERSIONS, RESERVED_START  # noqa: E402

FINAL = experiments.RUNS["final"]


def installed_versions():
    import catboost
    import lightgbm
    import numpy
    import scipy
    import shap
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "pandas": pandas.__version__, "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "catboost": catboost.__version__,
            "shap": shap.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__}


def check_development_only(dataset):
    """The dataset must end with development outcomes; nothing on or after the reserved start."""
    last_decision = dataset.df["decision_date"].max()
    last_outcome = dataset.df["next_date"].max()
    assert last_decision < pandas.Timestamp(RESERVED_START), "a reserved decision is in memory"
    assert last_outcome <= pandas.Timestamp(END), "an outcome after the development end is in memory"


def freeze_to(output, dataset, fold, check_real_data=True):
    """Run the freeze into output/frozen and write frozen_manifest.json; returns the manifest."""
    started = datetime.now(ET).isoformat()
    t0 = time.perf_counter()
    check_development_only(dataset)
    if check_real_data:
        assert dataset.private and len(dataset.features) == 39 and len(dataset.df) == splits.EXPECTED_ELIGIBLE_ROWS
        assert str(dataset.usable_start.date()) == "2021-11-19"
    fold_record = splits.final_fold_record(dataset, fold)
    assert fold_record["purge_assertion"] and all(inner["purge_assertion"] for inner in fold_record["inner"])
    if check_real_data:
        assert fold_record["n_train"] == splits.EXPECTED_TRAIN_ROWS
    ledger = selection.Ledger()
    frozen_dir = output / "frozen"
    result = models.freeze_final_procedure(dataset, fold, frozen_dir, FINAL["selection_settings"],
                                           FINAL["elastic_net_random_state"], ledger)
    manifest = {
        "written": datetime.now(ET).isoformat(), "started": started, "clock": "America/New_York",
        "versions": installed_versions(), "required_versions": LOCKED_VERSIONS, "machine": platform.machine(),
        "specification": "manifests/usdjpy_locked_procedure.md (sections 3.1-3.10; addendum 2026-10-27)",
        "reserved_files_present_at_freeze": [], "no_reserved_row_in_memory": True,
        "dataset": {"n_eligible_rows": int(len(dataset.df)), "n_features": len(dataset.features),
                    "private_features_in_memory": dataset.private, "usable_start": str(dataset.usable_start.date())},
        "final_fold": fold_record, "settings": FINAL["selection_settings"].as_dict(),
        "elastic_net_random_state": FINAL["elastic_net_random_state"],
        "grids": {name: models.GRIDS16[name] for name in models.FROZEN_BOOSTERS},
        "extra_knobs": {name: {"name": knob, "values": values} for name, (knob, values) in models.EXTRA_KNOB.items()},
        "fixed_settings": {"lightgbm": models.LGB_FINAL_FIXED, "xgboost": models.XGB_FIXED, "catboost": models.CAT_FIXED,
                           "early_stopping": {"cap": models.FINAL_CAP, "patience": models.FINAL_PATIENCE,
                                              "metric": "auc", "fold_local": True},
                           "threads_per_model": models.THREADS,
                           "selector_threads": FINAL["selection_settings"].selector_threads},
        "shortlist": result["shortlist"], "shortlist_size": len(result["shortlist"]),
        "own_market_features": result["own_market_features"],
        "procedures": list(FINAL["procedures"]), "grid8_by_products": list(FINAL["grid8_by_products"]),
        "models": result["models"],
        "fits": {"by_stage": dict(ledger.fits), "total": int(sum(ledger.fits.values()))},
        "wall_time_seconds": {"by_stage": {k: round(v, 3) for k, v in ledger.seconds.items()},
                              "total": round(time.perf_counter() - t0, 2)},
        "warnings": ledger.warnings, "failures": ledger.failures,
    }
    if check_real_data:
        manifest["dataset"].update({
            "features_file_sha256": models.sha256_file(splits.PRIVATE_FEATURES),
            "labels_file_sha256": models.sha256_file(splits.LABELS),
            "exceptions_file_sha256": models.sha256_file(splits.EXCEPTIONS)})
    (frozen_dir / "frozen_manifest.json").write_text(json.dumps(models._jsonable(manifest), indent=1))
    return manifest


def prepare_output(path):
    """A new folder outside every protected location; an existing folder is refused."""
    path = Path(path).resolve()
    for protected in config.protected_roots():
        if config.is_inside(path, protected):
            raise SystemExit(f"Refusing: {path} lies inside a protected folder ({protected.name}).")
    if path.exists():
        raise SystemExit(f"Refusing: {path} already exists; the freeze needs a fresh output folder.")
    return config.output_dir(root=path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-fitting", action="store_true", help="required: this script fits models")
    parser.add_argument("--output", type=Path, required=True, help="a new folder for frozen/ and its manifest")
    args = parser.parse_args()
    if not args.allow_fitting:
        raise SystemExit("Refusing to fit: pass --allow-fitting to train the locked final procedure.")
    seen = installed_versions()
    differing = {k: (seen[k], v) for k, v in LOCKED_VERSIONS.items() if seen[k] != v}
    if differing:
        raise SystemExit(f"Installed versions differ from the freeze record (seen, required): {differing}")
    output = prepare_output(args.output)
    dataset = splits.load_dataset(private=True)
    fold = splits.make_final_fold(dataset)
    manifest = freeze_to(output, dataset, fold)
    print(f"Froze {len(manifest['models'])} procedures ({manifest['fits']['total']} fits) into {output / 'frozen'}.")


if __name__ == "__main__":
    main()
