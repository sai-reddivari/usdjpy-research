"""Folders, study windows and clocks shared by every module.

Locations come from environment variables, so the package works on another
machine without editing code. The defaults find the research caches next to
this submission folder; they are local conveniences, not requirements:

    USDJPY_RESEARCH_ROOT   the research folder with data/ and manifests/
                           (default: ../data_review next to submission/)
    USDJPY_SETUP_ROOT      the older setup folder holding the FMP request ledger
                           (default: ../setup_research)
    USDJPY_PRIVATE_DIR     licensed J-Quants caches and derived tables; never
                           copied into the project (default: ~/Library/
                           Application Support/CQF/private_data/jquants)
    USDJPY_PRIVATE_SCRATCH private scratch for rebuilt licensed tables
                           (default: a sibling of USDJPY_PRIVATE_DIR)
    USDJPY_OUTPUT_DIR      where scripts write (default: ../submission_work, outside the
                           package so rebuilt vendor-derived tables are never shipped)

The research folder, the setup folder and the private folder are reference
inputs. Nothing in this package writes into them: output_dir() refuses them.

Conventions: every clock is US Eastern (America/New_York, so EST in winter and
EDT in summer). Tokyo times are written once in JST and converted. Dates are
calendar dates on the source's own calendar.
"""

import os
from datetime import date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PACKAGE_ROOT = Path(__file__).resolve().parent
SUBMISSION_ROOT = PACKAGE_ROOT.parent


def path_from_environment(variable, default):
    """Path from an environment variable, else the default; '~' is expanded."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else Path(default)


RESEARCH_ROOT = path_from_environment("USDJPY_RESEARCH_ROOT", SUBMISSION_ROOT.parent / "data_review")
SETUP_ROOT = path_from_environment("USDJPY_SETUP_ROOT", SUBMISSION_ROOT.parent / "setup_research")
PRIVATE_DIR = path_from_environment(
    "USDJPY_PRIVATE_DIR",
    Path.home() / "Library" / "Application Support" / "CQF" / "private_data" / "jquants",
)
PRIVATE_SCRATCH = path_from_environment("USDJPY_PRIVATE_SCRATCH", PRIVATE_DIR.parent / "jquants_scratch")
OUTPUT_ROOT = path_from_environment("USDJPY_OUTPUT_DIR", SUBMISSION_ROOT.parent / "submission_work")

# Reference inputs inside the research folder.
RAW = RESEARCH_ROOT / "data" / "raw"                       # immutable cached provider responses
OBSERVATIONS = RESEARCH_ROOT / "data" / "observations"     # long observation files
ALIGNED = RESEARCH_ROOT / "data" / "aligned"               # decision-row tables and audits
RESULTS = RESEARCH_ROOT / "data" / "results"               # saved runs A, B, C and the reserved year
MANIFESTS = RESEARCH_ROOT / "manifests"

RESERVED_RAW = RAW / "reserved"
RESERVED_ALIGNED = ALIGNED / "reserved"
RESERVED_PRIVATE = PRIVATE_DIR / "reserved"
FROZEN_DIR = RESULTS / "reserved_year" / "frozen"

# Acquisition bookkeeping kept in the older setup folder.
LEGACY_CACHE = SETUP_ROOT / "cache"
FMP_LEDGER = SETUP_ROOT / "request_ledger.json"
FMP_LOCK = SETUP_ROOT / ".acquisition.lock"

# Study windows. Development = warm-up plus development; nothing after END is read for it.
START = date(2020, 10, 22)
END = date(2025, 10, 21)
RESERVED_START = date(2025, 10, 22)       # first reserved decision date
RESERVED_END = date(2026, 10, 21)         # outcomes must land on or before this date
RESERVED_FETCH_END = date(2026, 10, 28)   # a few days past the window, to confirm the last label window

ET = ZoneInfo("America/New_York")
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc

# Long observation schema shared by every observation file.
OBSERVATION_COLUMNS = [
    "series_id", "observation_date", "observation_time", "available_time",
    "value", "unit", "source_id", "availability_basis",
]

# Versions the frozen models were fitted and scored with (frozen_manifest.json).
LOCKED_VERSIONS = {
    "python": "3.14.8", "pandas": "3.0.6", "sklearn": "1.9.1", "lightgbm": "4.7.0",
    "xgboost": "3.4.1", "catboost": "1.2.10", "shap": "0.54.1",
}


def in_development(day):
    """True when the date lies inside the development window."""
    return START <= day <= END


def protected_roots():
    """Folders that hold reference inputs and must never be written by this package."""
    return [RESEARCH_ROOT.resolve(), SETUP_ROOT.resolve(), PRIVATE_DIR.resolve(),
            (SUBMISSION_ROOT / "latex").resolve(), (SUBMISSION_ROOT / "results").resolve(),
            (SUBMISSION_ROOT / "figures").resolve(), (SUBMISSION_ROOT / "provenance").resolve()]


def is_inside(path, folder):
    """True when path equals folder or lies below it."""
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def output_dir(*parts, root=None):
    """Create and return a writable output folder; refuse any protected location.

    The default root is OUTPUT_ROOT. A folder inside the research tree, the
    setup tree, the private folder or the report-facing submission folders
    raises ValueError, so a script can never overwrite a reference artifact.
    """
    folder = Path(root or OUTPUT_ROOT).joinpath(*parts).resolve()
    for protected in protected_roots():
        if is_inside(folder, protected):
            raise ValueError(f"Refusing to write into a protected reference folder: {protected.name}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def private_scratch_dir(*parts):
    """Create a 0700 private scratch folder for rebuilt licensed tables (outside every reference folder)."""
    folder = PRIVATE_SCRATCH.joinpath(*parts).resolve()
    if is_inside(folder, PRIVATE_DIR) or is_inside(folder, RESEARCH_ROOT) or is_inside(folder, SUBMISSION_ROOT):
        raise ValueError("The private scratch folder must lie outside the private cache and the project.")
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    return folder


def private_available():
    """True when the licensed J-Quants cache is present on this machine."""
    return (PRIVATE_DIR / "jquants_0000_raw_manifest.json").exists()


class PrivateDataUnavailable(FileNotFoundError):
    """Raised when a step needs the licensed J-Quants inputs and they are absent."""


def require_private_inputs(*paths):
    """Return the paths if every one exists; otherwise raise PrivateDataUnavailable.

    The locked procedure uses all 39 candidates, four of them licensed. A step
    that needs them must stop here rather than continue on the 35 public ones.
    """
    missing = [Path(path).name for path in paths if not Path(path).exists()]
    if missing:
        raise PrivateDataUnavailable(
            "Licensed J-Quants inputs are required and absent: " + ", ".join(missing)
            + ". Set USDJPY_PRIVATE_DIR to the private folder. The 35 public features are "
            "not a substitute for the locked 39-candidate procedure."
        )
    return paths
