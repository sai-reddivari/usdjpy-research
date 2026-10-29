"""Score the 13 frozen reserved-year models on the cached reserved rows. No fitting.

    python scripts/score_frozen_models.py [--frozen-dir DIR] [--output DIR]

Steps: check every frozen joblib file against its recorded sha256; re-predict
each recorded fingerprint (25 development training rows) exactly; load the
229 eligible reserved rows (all 39 features required, so the licensed
J-Quants inputs must be present); predict once per model; write per-row
predictions (decision date, quarter, label, probability, flag; no feature
values) and the quarterly AUC and paired-difference tables.

Outputs go to --output (default USDJPY_OUTPUT_DIR/score_frozen); a protected
research or report folder is refused. The saved research results are not
touched. Thresholds and iteration counts come from the frozen manifest; no
reserved outcome enters any choice.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas  # noqa: E402

from usdjpy import config, evaluation, experiments, models, splits  # noqa: E402


def score(frozen_dir, output):
    """Verify, predict and tabulate; returns the reserved AUC table."""
    manifest, frozen = models.load_frozen_models(frozen_dir)
    development = splits.load_dataset(private=True)
    models.assert_fingerprints(manifest, frozen, development, splits.make_final_fold(development))
    reserved, ineligible = splits.load_reserved_dataset()
    assert pandas.Timestamp(manifest["final_fold"]["max_outcome_time"]) < reserved.df["decision_time"].min(), "purge"
    predictions_dir = config.output_dir("predictions", root=output)
    labels = reserved.df["quarter"].map(splits.RESERVED_LABELS)
    names = list(experiments.RUNS["final"]["procedures"]) + list(experiments.RUNS["final"]["grid8_by_products"])
    tables = {}
    for name in names:
        probabilities = models.frozen_probabilities(manifest, frozen, name, reserved.df)
        frame = pandas.DataFrame({"decision_date": reserved.df["decision_date"].dt.date.astype(str), "quarter": labels,
                                  "y": reserved.y, "p": probabilities, "flagged": reserved.df["flagged"]})
        frame.to_csv(predictions_dir / f"{name}.csv", index=False)
        tables[name] = frame
    auc = evaluation.reserved_auc_table(tables)
    summary_dir = config.output_dir("summary", root=output)
    auc.to_csv(summary_dir / "auc_per_quarter.csv")
    evaluation.reserved_paired_table(auc, experiments.RESERVED_PAIRS).to_csv(summary_dir / "paired_differences.csv")
    ineligible.to_csv(summary_dir / "ineligible_rows.csv", index=False)
    return auc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-dir", type=Path, default=config.FROZEN_DIR)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_ROOT / "score_frozen")
    args = parser.parse_args()
    output = config.output_dir(root=args.output)
    auc = score(args.frozen_dir, output)
    print(auc[["R1", "R2", "R3", "R4", "mean_quarterly_auc", "pooled_auc"]].round(3).to_string())
    print(f"Wrote predictions and tables to {output}. No model was fitted.")


if __name__ == "__main__":
    main()
