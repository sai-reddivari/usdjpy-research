"""Rebuild the observation, aligned, label and feature tables from the cached raw data. Offline.

    python scripts/rebuild_aligned_tables.py                 # all tables, licensed ones to the private scratch
    python scripts/rebuild_aligned_tables.py --public-only   # the 35 public features only, no licensed inputs

Public outputs go to USDJPY_OUTPUT_DIR/rebuild (default ../submission_work/rebuild);
licensed outputs go to USDJPY_PRIVATE_SCRATCH/rebuild with mode 0600. The
research folder is read, never written. No network request is made: every
fetch runs with its acquisition switch off and raises if a cache is missing.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usdjpy import config, features, sources, timing  # noqa: E402


def rebuild(public_only):
    """Write every rebuilt table; return {name: path}."""
    out = config.output_dir("rebuild")
    written = {path.name: path for path in sources.write_observation_files(out)}
    private_observations = private_out = None
    if not public_only:
        private_out = config.private_scratch_dir("rebuild")
        private_observations = sources.write_private_jquants_observations(private_out)
        written[private_observations.name] = private_observations
    timing.write_aligned_tables(out, out, private_observations, private_out)
    # Features read the research observation files (byte-identical to the rebuilt ones, see
    # tests/test_data_equivalence.py) because the scheduled-event calendar is an input, not a rebuilt table.
    build = features.build_features(private=not public_only)
    if public_only:
        public = build.features[["decision_date"] + features.PUBLIC]
        public.to_csv(out / "usdjpy_features_public.csv", index=False)
    else:
        features.write_feature_tables(build, out, private_out)
    written.update({path.name: path for path in out.glob("*.csv")})
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-only", action="store_true", help="skip every licensed input and output")
    args = parser.parse_args()
    written = rebuild(args.public_only)
    print(f"Rebuilt {len(written)} tables from cache into {config.output_dir('rebuild')} (no request made).")


if __name__ == "__main__":
    main()
