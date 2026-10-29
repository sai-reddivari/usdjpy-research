"""The executed experiments as explicit settings: Run A, Run B, Run C and the locked final procedure.

Nothing here runs an experiment. Each entry names the selection settings,
grids, procedures and saved-results folder of a run that already exists, so a
reader can see what differed between runs and tests can check the settings
against what each run recorded.

- Run A: protocol v2.1 as approved. Early-stopped selection LightGBM, 400-fit
  wrapper budget, eight-configuration grids, stages A-D plus interaction
  comparisons. Driver: data_review/run_a.py.
- Run B: amended after Run A's results. Fixed 150-tree selector, 2,000-fit
  wrapper budget, sixteen-configuration grids (eight-grid choice kept as a
  by-product), matched own-price boosters. Driver: data_review/run_b.py;
  negative controls: data_review/run_b_controls.py.
- Run C: the report's name for the existing no-wrapper development replay.
  Run B settings with Stage D removed: each booster gets the whole Stage C
  shortlist. Saved under no_wrapper_development, models without a prefix there
  and with an nw_ prefix in the reserved year. Driver:
  data_review/run_no_wrapper.py. Not a new run.
- Final (locked): Run C's procedure fitted once on the 1,009-row final window,
  with the elastic net seeded (random_state=0). Frozen under
  reserved_year/frozen. Driver: data_review/run_reserved_year.py, and here
  scripts/train_final_procedure.py (guarded).

Runs A, B and C used an unseeded elastic net (random_state=None). Only the
final freeze is seeded. These settings record that; they must not be changed
to "improve" reproducibility of the historical runs.
"""

from . import models, selection
from .config import RESULTS

BOOSTERS = ("lightgbm", "xgboost", "catboost")
OWN_PRICE_BOOSTERS = tuple(f"{booster}_own_market" for booster in BOOSTERS)

# Run A's US-only rates ablation: the US-only value replaces each original column in its slot.
ABLATION_SLOTS = {"gap2y": "us2y", "gap2y_chg_1d": "us2y_chg_1d", "gap2y_chg_5d": "us2y_chg_5d",
                  "gapchg_x_vixchg": "us_only_gapchg_x_vixchg"}

# Run B negative controls: the cheap chain (stages A-D, no interaction comparisons, 150-tree scoring model).
NEGATIVE_CONTROL_SETTINGS = selection.SelectionSettings(
    fixed_iterations=selection.RUN_B.fixed_iterations, wrapper_budget=selection.RUN_B.wrapper_budget,
    tolerance=selection.RUN_B.tolerance, interactions=False, selector_threads=selection.RUN_B.selector_threads,
)
NEGATIVE_CONTROL_LABELLED_ROWS = 1277
NEGATIVE_CONTROL_ANCHOR_SHIFTS = (319, 426, 639)


def negative_control_shifts(n=NEGATIVE_CONTROL_LABELLED_ROWS):
    """The 20 circular label shifts of the Run B negative controls (data_review/run_b_controls.py).

    k_j = round(300 + 677 j / 19), j = 0..19, with the values nearest 319, 426
    and 639 replaced by exactly those values (ties to the lower index).
    """
    shifts = [int(round(300 + 677 * j / 19)) for j in range(20)]
    for anchor in NEGATIVE_CONTROL_ANCHOR_SHIFTS:
        free = [i for i in range(20) if shifts[i] not in NEGATIVE_CONTROL_ANCHOR_SHIFTS]
        position = min(free, key=lambda i: (abs(shifts[i] - anchor), i))
        shifts[position] = anchor
    assert len(set(shifts)) == 20 and all(min(k, n - k) >= 300 for k in shifts), shifts
    return shifts


RUNS = {
    "A": {
        "name": "Run A (protocol v2.1 as approved)",
        "selection_settings": selection.RUN_A,
        "elastic_net_random_state": None,
        "stage_d": True,
        "interaction_comparisons": True,
        "grids": models.GRIDS,
        "nested_grid": False,
        "boosters_on": "wrapper-selected subset",
        "baselines": ("majority", "logistic_own_market", "lgb_own_market"),
        "saved_models": ("majority", "logistic_own_market", "lgb_own_market", "lightgbm", "xgboost", "catboost"),
        "results_dir": RESULTS / "run_a",
        "variants": ("main", "ablation"),
    },
    "B": {
        "name": "Run B (fixed selector, larger budget, nested grids)",
        "selection_settings": selection.RUN_B,
        "elastic_net_random_state": None,
        "stage_d": True,
        "interaction_comparisons": True,
        "grids": models.GRIDS16,
        "nested_grid": True,
        "boosters_on": "wrapper-selected subset",
        "baselines": ("majority", "logistic_own_market") + OWN_PRICE_BOOSTERS,
        "saved_models": ("majority", "logistic_own_market")
        + tuple(name for booster in BOOSTERS
                for name in (booster, f"{booster}_grid8", f"{booster}_own_market", f"{booster}_own_market_grid8")),
        "results_dir": RESULTS / "run_b",
        "variants": ("main", "ablation"),
    },
    "C": {
        "name": "Run C (no-wrapper development replay)",
        "selection_settings": selection.RUN_B,
        "elastic_net_random_state": None,
        "stage_d": False,
        "interaction_comparisons": False,
        "grids": models.GRIDS16,
        "nested_grid": True,
        "boosters_on": "entire Stage C shortlist",
        "baselines": ("majority", "logistic_own_market") + OWN_PRICE_BOOSTERS,   # reused from Run B
        "saved_models": tuple(name for booster in BOOSTERS for name in (booster, f"{booster}_grid8")),
        "results_dir": RESULTS / "no_wrapper_development",
        "variants": ("main",),
    },
    "final": {
        "name": "Locked final procedure (frozen for the reserved year)",
        "selection_settings": selection.RUN_B,
        "elastic_net_random_state": 0,
        "stage_d": False,
        "interaction_comparisons": False,
        "grids": models.GRIDS16,
        "nested_grid": True,
        "boosters_on": "entire Stage C shortlist",
        "procedures": tuple(models.FROZEN_PROCEDURES),
        "grid8_by_products": tuple(models.FROZEN_GRID8),
        "results_dir": RESULTS / "reserved_year",
        "variants": ("main",),
    },
}

# Reserved-year paired comparisons (first minus second), in the saved order.
RESERVED_PAIRS = [
    ("nw_lightgbm", "lightgbm_own_market"), ("nw_xgboost", "xgboost_own_market"), ("nw_catboost", "catboost_own_market"),
    ("nw_lightgbm", "logistic_own_market"), ("nw_xgboost", "logistic_own_market"), ("nw_catboost", "logistic_own_market"),
    ("nw_catboost", "nw_lightgbm"), ("nw_catboost", "nw_xgboost"),
    ("lightgbm_own_market", "logistic_own_market"), ("xgboost_own_market", "logistic_own_market"),
    ("catboost_own_market", "logistic_own_market"),
    ("nw_lightgbm", "majority"), ("nw_xgboost", "majority"), ("nw_catboost", "majority"),
]

# Features derived from CFTC reports; rows without them are outside the 229-row primary scoring set.
CFTC_FEATURES = ["cftc_net", "cftc_net_chg_4reports", "cftc_release_age_days", "pos_z", "short_yen_stress"]
