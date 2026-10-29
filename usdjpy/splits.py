"""Quarters, outer folds, inner folds, purging, the final freeze fold and the reserved rows.

Relocated from data_review/exam3/splits.py and the fold/row parts of
data_review/exam3/reserved.py. Fold membership is checked row for row against
the historical code and the saved run records (tests/test_split_equivalence.py).
One deliberate change: when the licensed features are requested and absent,
load_dataset and load_reserved_dataset raise config.PrivateDataUnavailable
instead of silently using the 35 public columns.

Protocol v2.1 section 5; clarifications 1 and 4.

Dataset: the labelled decision rows from the usable start (first row with every
feature present) to the last labelled row, in date order, positional index 0..n-1.
Every split is a pair of positional index arrays into that frame.

Clocks: decision time 17:00 ET on decision_date; outcome time 17:00 ET on
next_date (the next retained session close). A validation quarter starting on
date D begins at 17:00 ET on D, the earliest decision time it can contain.

Purging: a training row is kept only if its outcome time is strictly before the
validation start time, i.e. next_date < D. The condition is asserted on every
boundary with the timestamps themselves (assert_purged).

Outer fold k (k = 1..8) validates on quarter Q(8+k) and trains on the purged
rows before it (expanding from the usable start). Its three inner folds are
numbered chronologically (1 = oldest): inner fold j validates on quarter
Q(8+k-4+j) (j = 1, 2, 3 -> the third-last, second-last and last quarter of the
outer training window) and trains on the rows before that quarter, purged at
that boundary. Inner validation rows are restricted to the purged outer
training set, so every inner row is an outer-training row and no outer
validation row (and no row purged at the outer boundary) enters any inner
computation.

Event-timing exceptions (clarification 4): decision dates flagged in
data/observations/event_timing_exceptions.csv are dropped from every training
set (outer and inner). They stay in the development OUTER validation sets
(the outer quarters, scored as they stand), where the `flagged` mask lets the
sensitivity run drop them from scoring. Inner validation sets are drawn from
the purged outer-training pool, from which flagged rows have already been
removed, so a flagged row that lies inside a training window (as the nine
Q16 rows do in the final, reserved-year fit) is excluded from inner
validation as well; it is scored in no inner fold and enters no fit. In the
eight development outer folds all nine flagged rows lie in Q16, so no
development training window contains one. (Comment clarified 2026-10-27;
no code changed.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pandas

from .config import (ALIGNED, END, ET, OBSERVATIONS, PRIVATE_DIR, RESERVED_ALIGNED, RESERVED_END,
                     RESERVED_PRIVATE, RESERVED_START, require_private_inputs)
from .features import FEATURES

FIRST_QUARTER_START = pandas.Timestamp('2021-10-22')
N_QUARTERS = 16                      # Q1..Q16 are the development quarters
FIRST_OUTER_QUARTER = 9              # outer folds validate on Q9..Q16
N_OUTER = 8
N_INNER = 3

PUBLIC_FEATURES = ALIGNED / 'usdjpy_features_public.csv'
PRIVATE_FEATURES = PRIVATE_DIR / 'usdjpy_features_full.csv'
LABELS = ALIGNED / 'usdjpy_5pm_et_label.csv'
EXCEPTIONS = OBSERVATIONS / 'event_timing_exceptions.csv'


def quarter_starts() -> list[pandas.Timestamp]:
    """Start dates of Q1..Q16 plus the start of the reserved period (Q17 position)."""
    return [FIRST_QUARTER_START + pandas.DateOffset(months=3 * i) for i in range(N_QUARTERS + 1)]


def quarter_bounds(q: int) -> tuple[pandas.Timestamp, pandas.Timestamp]:
    """[start, end] dates of quarter q (1-based); end is the last calendar day."""
    s = quarter_starts()
    return s[q - 1], s[q] - pandas.Timedelta(days=1)


def quarter_of(dates: pandas.Series) -> pandas.Series:
    """Quarter number (1..16) of each decision date; 0 before Q1, 17 after Q16."""
    s = quarter_starts()
    return pandas.Series(numpy.searchsorted(numpy.array(s, dtype='datetime64[ns]'), dates.values, side='right'),
                     index=dates.index)


def at_5pm_et(dates: pandas.Series | pandas.Timestamp):
    """17:00 America/New_York on the given calendar date(s)."""
    if isinstance(dates, pandas.Timestamp):
        return (dates.normalize() + pandas.Timedelta(hours=17)).tz_localize(ET)
    return (pandas.to_datetime(dates).dt.normalize() + pandas.Timedelta(hours=17)).dt.tz_localize(ET)


def flagged_dates(path=EXCEPTIONS) -> list[pandas.Timestamp]:
    """Decision dates listed in the development event-timing exception file."""
    if not path.exists():
        return []
    ex = pandas.read_csv(path, parse_dates=['decision_date'])
    return sorted(set(ex['decision_date']))


@dataclass
class Dataset:
    df: pandas.DataFrame                  # decision_date, next_date, decision_time, outcome_time, y, flagged, features
    features: list[str]               # feature columns present (39 private, 35 public)
    usable_start: pandas.Timestamp
    private: bool

    @property
    def y(self) -> numpy.ndarray:
        return self.df['y'].to_numpy()

    def X(self, cols: list[str] | None = None) -> pandas.DataFrame:
        return self.df[cols if cols is not None else self.features]


def load_dataset(private: bool = True) -> Dataset:
    """Labelled rows from the usable start with every feature present.

    private=True reads the 39-column file from the private folder (never copied
    anywhere) and raises PrivateDataUnavailable when it is absent: the locked
    procedure is defined on 39 candidates. private=False is the explicit
    35-column public dataset, for demonstrations only; its usable start and
    rows differ from the locked dataset.
    """
    use_private = private
    if use_private:
        require_private_inputs(PRIVATE_FEATURES)
    F = pandas.read_csv(PRIVATE_FEATURES if use_private else PUBLIC_FEATURES, parse_dates=['decision_date'])
    feats = [c for c in FEATURES if c in F.columns]
    L = pandas.read_csv(LABELS, parse_dates=['decision_date', 'next_date'])
    L = L[['decision_date', 'decision_time', 'next_date', 'positive_return']]
    df = F.merge(L, on='decision_date', how='inner').sort_values('decision_date')
    complete = df[feats].notna().all(axis=1)
    assert complete.any(), 'no row with every feature present'
    usable = df.loc[complete.idxmax(), 'decision_date']
    df = df[(df['decision_date'] >= usable) & df['positive_return'].notna()].reset_index(drop=True)
    assert df[feats].notna().all().all(), 'missing feature values after the usable start'
    assert (df['next_date'] > df['decision_date']).all()
    df['y'] = df['positive_return'].astype(int)
    df['outcome_time'] = at_5pm_et(df['next_date'])
    df['decision_time'] = at_5pm_et(df['decision_date'])
    df['quarter'] = quarter_of(df['decision_date'])
    df['flagged'] = df['decision_date'].isin(flagged_dates())
    return Dataset(df=df, features=feats, usable_start=usable, private=use_private)


@dataclass
class OuterFold:
    k: int                                   # 1..8
    quarter: int                             # 9..16
    valid_start: pandas.Timestamp                # calendar date of the quarter start
    train_idx: numpy.ndarray                    # purged, flagged rows dropped
    valid_idx: numpy.ndarray                    # the outer quarter (flagged rows kept; see Dataset.df['flagged'])
    inner: list[tuple[numpy.ndarray, numpy.ndarray]] = field(default_factory=list)   # (train_idx, valid_idx), fold 1 oldest
    inner_quarters: list[int] = field(default_factory=list)
    n_purged_outer: int = 0
    n_purged_inner: list[int] = field(default_factory=list)

    def sizes(self) -> dict:
        return {'train': int(len(self.train_idx)), 'valid': int(len(self.valid_idx)),
                'inner_train': [int(len(t)) for t, _ in self.inner],
                'inner_valid': [int(len(v)) for _, v in self.inner],
                'purged_outer': self.n_purged_outer, 'purged_inner': list(self.n_purged_inner)}


def assert_purged(ds: Dataset, train_idx: numpy.ndarray, valid_start_date: pandas.Timestamp, label: str = '') -> None:
    """max(outcome_time of training rows) < 17:00 ET on the validation start date."""
    if len(train_idx) == 0:
        return
    vstart = at_5pm_et(pandas.Timestamp(valid_start_date))
    mx = ds.df['outcome_time'].iloc[train_idx].max()
    assert mx < vstart, f'purge failed {label}: max outcome time {mx} >= validation start {vstart}'


def _purged_before(ds: Dataset, base: numpy.ndarray, start: pandas.Timestamp) -> tuple[numpy.ndarray, int]:
    """Rows of `base` whose outcome lands before `start`; also the count removed by purging alone."""
    d = ds.df.iloc[base]
    before = (d['decision_date'] < start).to_numpy()
    keep = before & (d['next_date'] < start).to_numpy() & (~d['flagged'].to_numpy())
    n_purged = int(before.sum() - (before & (d['next_date'] < start).to_numpy()).sum())
    return base[keep], n_purged


def make_outer_fold(ds: Dataset, k: int) -> OuterFold:
    assert 1 <= k <= N_OUTER
    q = FIRST_OUTER_QUARTER + k - 1
    starts = quarter_starts()
    vstart, vnext = starts[q - 1], starts[q]
    all_idx = numpy.arange(len(ds.df))
    train_idx, n_purged = _purged_before(ds, all_idx, vstart)
    dates = ds.df['decision_date']
    valid_idx = all_idx[((dates >= vstart) & (dates < vnext)).to_numpy()]
    assert_purged(ds, train_idx, vstart, f'outer fold {k}')
    fold = OuterFold(k=k, quarter=q, valid_start=vstart, train_idx=train_idx, valid_idx=valid_idx,
                     n_purged_outer=n_purged)
    for j in range(1, N_INNER + 1):
        qj = q - N_INNER + j - 1                 # j=1 -> q-3 (oldest), j=3 -> q-1 (latest)
        s, e = starts[qj - 1], starts[qj]
        itrain, n_ip = _purged_before(ds, train_idx, s)
        tdates = ds.df['decision_date'].iloc[train_idx]
        ivalid = train_idx[((tdates >= s) & (tdates < e)).to_numpy()]
        assert_purged(ds, itrain, s, f'outer fold {k} inner fold {j}')
        assert len(numpy.intersect1d(itrain, valid_idx)) == 0 and len(numpy.intersect1d(ivalid, valid_idx)) == 0
        assert numpy.isin(itrain, train_idx).all() and numpy.isin(ivalid, train_idx).all()
        assert len(itrain) > 0 and len(ivalid) > 0
        fold.inner.append((itrain, ivalid))
        fold.inner_quarters.append(qj)
        fold.n_purged_inner.append(n_ip)
    return fold


def all_outer_folds(ds: Dataset) -> list[OuterFold]:
    return [make_outer_fold(ds, k) for k in range(1, N_OUTER + 1)]


# ----------------------------------------------------------------------------
# Final (freeze) fold and reserved-year rows (relocated from data_review/exam3/reserved.py)
# ----------------------------------------------------------------------------
FINAL_QUARTER = 17                           # the reserved window starts at the Q17 position of the development grid
RESERVED_QUARTERS = [17, 18, 19, 20]         # R1..R4, three-month blocks from 2025-10-22
RESERVED_LABELS = {17: 'R1', 18: 'R2', 19: 'R3', 20: 'R4'}
RESERVED_EXCEPTIONS = RESERVED_ALIGNED / 'event_timing_exceptions_reserved.csv'
EXPECTED_TRAIN_ROWS = 1009
EXPECTED_ELIGIBLE_ROWS = 1018
EXPECTED_FLAGGED = 9


def reserved_quarter_starts() -> list[pandas.Timestamp]:
    """Start dates of R1..R4 plus the day after the reserved window (five timestamps)."""
    s = quarter_starts()
    base = s[FINAL_QUARTER - 1]
    assert base == pandas.Timestamp(RESERVED_START)
    out = [base + pandas.DateOffset(months=3 * i) for i in range(5)]
    assert out[-1] == pandas.Timestamp(RESERVED_END) + pandas.Timedelta(days=1)
    return out


def reserved_quarter_of(dates: pandas.Series) -> pandas.Series:
    """17..20 for decision dates inside the reserved window (quarter_of extended past Q16)."""
    s = reserved_quarter_starts()
    q = numpy.searchsorted(numpy.array(s, dtype='datetime64[ns]'), dates.values, side='right')
    return pandas.Series(q + FINAL_QUARTER - 1, index=dates.index)


def make_final_fold(ds: Dataset) -> OuterFold:
    """Final training window and inner folds Q14-Q16, purged and flagged rows dropped (locked procedure 3.6).

    The validation set is empty by construction: the reserved rows do not
    exist in memory when this fold is built. The cutoff assertion is against
    the first reserved decision time, 2025-10-22 17:00 ET.
    """
    assert len(ds.df) == EXPECTED_ELIGIBLE_ROWS and int(ds.df['flagged'].sum()) == EXPECTED_FLAGGED, \
        (len(ds.df), int(ds.df['flagged'].sum()))
    starts = quarter_starts()
    vstart = starts[FINAL_QUARTER - 1]
    assert vstart == pandas.Timestamp(RESERVED_START)
    all_idx = numpy.arange(len(ds.df))
    train_idx, n_purged = _purged_before(ds, all_idx, vstart)
    assert n_purged == 0, n_purged                       # the last labelled development outcome is 2025-10-21 17:00 ET
    assert len(train_idx) == EXPECTED_TRAIN_ROWS, len(train_idx)
    assert not ds.df['flagged'].iloc[train_idx].any()
    assert_purged(ds, train_idx, vstart, 'final fold (reserved cutoff)')
    cutoff = at_5pm_et(pandas.Timestamp(RESERVED_START))
    mx = ds.df['outcome_time'].iloc[train_idx].max()
    # The nine flagged rows (2025-10-02 and 2025-10-09 to 2025-10-20) are dropped, so the last training decision is
    # 2025-10-08 with outcome 2025-10-09 17:00 ET, earlier than the 2025-10-21 17:00 ET the locked procedure 3.6 quotes
    # (which is the last labelled development outcome, not the last training outcome). Both are before the cutoff.
    assert mx <= at_5pm_et(pandas.Timestamp(END)) and mx < cutoff, (mx, cutoff)
    fold = OuterFold(k=9, quarter=FINAL_QUARTER, valid_start=vstart, train_idx=train_idx,
                     valid_idx=numpy.array([], dtype=int), n_purged_outer=n_purged)
    for j in range(1, 4):
        qj = FINAL_QUARTER - 3 + j - 1                   # 14, 15, 16
        s, e = starts[qj - 1], starts[qj]
        itrain, n_ip = _purged_before(ds, train_idx, s)
        tdates = ds.df['decision_date'].iloc[train_idx]
        ivalid = train_idx[((tdates >= s) & (tdates < e)).to_numpy()]
        assert_purged(ds, itrain, s, f'final fold inner fold {j}')
        assert numpy.isin(itrain, train_idx).all() and numpy.isin(ivalid, train_idx).all()
        assert len(numpy.intersect1d(itrain, ivalid)) == 0 and len(itrain) > 0 and len(ivalid) > 0
        assert not ds.df['flagged'].iloc[itrain].any() and not ds.df['flagged'].iloc[ivalid].any()
        assert ds.df['decision_date'].iloc[itrain].max() < ds.df['decision_date'].iloc[ivalid].min()
        assert ds.df['quarter'].iloc[ivalid].eq(qj).all()
        fold.inner.append((itrain, ivalid))
        fold.inner_quarters.append(qj)
        fold.n_purged_inner.append(n_ip)
    assert fold.inner_quarters == [14, 15, 16]
    return fold


def final_fold_record(ds: Dataset, fold: OuterFold) -> dict:
    """Dates, counts and the purge facts of the final fold, for the frozen manifest."""
    tr = ds.df.iloc[fold.train_idx]
    rec = {'n_train': int(len(fold.train_idx)), 'n_eligible_development_rows': int(len(ds.df)),
           'n_flagged_dropped': int(ds.df['flagged'].sum()),
           'flagged_dates_dropped': [str(d.date()) for d in sorted(ds.df.loc[ds.df['flagged'], 'decision_date'])],
           'first_decision': str(tr['decision_date'].min().date()), 'last_decision': str(tr['decision_date'].max().date()),
           'max_outcome_time': str(tr['outcome_time'].max()), 'first_reserved_decision_time': str(at_5pm_et(pandas.Timestamp(RESERVED_START))),
           'purge_assertion': bool(tr['outcome_time'].max() < at_5pm_et(pandas.Timestamp(RESERVED_START))),
           'n_positive': int(ds.y[fold.train_idx].sum()), 'positive_rate': float(ds.y[fold.train_idx].mean()),
           'inner_quarters': list(fold.inner_quarters), 'inner': []}
    for j, (itr, iva) in enumerate(fold.inner, start=1):
        q = fold.inner_quarters[j - 1]
        s = quarter_starts()[q - 1]
        rec['inner'].append({'inner_fold': j, 'quarter': q, 'valid_start': str(s.date()),
                             'n_train': int(len(itr)), 'n_valid': int(len(iva)),
                             'train_last_decision': str(ds.df['decision_date'].iloc[itr].max().date()),
                             'train_max_outcome_time': str(ds.df['outcome_time'].iloc[itr].max()),
                             'valid_first_decision': str(ds.df['decision_date'].iloc[iva].min().date()),
                             'valid_last_decision': str(ds.df['decision_date'].iloc[iva].max().date()),
                             'n_valid_positive': int(ds.y[iva].sum()),
                             'purge_assertion': bool(ds.df['outcome_time'].iloc[itr].max() < at_5pm_et(s)),
                             'n_purged_at_boundary': int(fold.n_purged_inner[j - 1])})
    return rec




def load_reserved_dataset(features_path: Path | None = None, labels_path: Path | None = None,
                          exceptions_path: Path = RESERVED_EXCEPTIONS,
                          require_features: list[str] | None = None,
                          private: bool = True) -> tuple[Dataset, pandas.DataFrame]:
    """Reserved decision rows (2025-10-22 to 2026-10-21) with a complete label window and all 39 features.

    Same construction as splits.load_dataset, on the reserved files. Rows with
    a missing feature are not eligible (locked procedure section 5); they are
    returned separately (decision date, missing columns) so the driver records
    them, never dropped silently. Returns (dataset of eligible rows, ineligible rows).

    private=True requires the licensed reserved feature file (raises
    PrivateDataUnavailable without it); the 229-row primary scoring set is
    defined on all 39 features.
    """
    if features_path is not None:
        fp = features_path
    elif private:
        (fp,) = require_private_inputs(RESERVED_PRIVATE / 'usdjpy_features_full_reserved.csv')
    else:
        fp = RESERVED_ALIGNED / 'usdjpy_features_public_reserved.csv'
    F = pandas.read_csv(fp, parse_dates=['decision_date'])
    feats = [c for c in FEATURES if c in F.columns]
    L = pandas.read_csv(labels_path or (RESERVED_ALIGNED / 'usdjpy_5pm_et_label_reserved.csv'), parse_dates=['decision_date', 'next_date'])
    L = L[['decision_date', 'decision_time', 'next_date', 'positive_return']]
    df = F.merge(L, on='decision_date', how='inner').sort_values('decision_date')
    df = df[(df['decision_date'] >= pandas.Timestamp(RESERVED_START)) & (df['decision_date'] <= pandas.Timestamp(RESERVED_END))]
    df = df[df['positive_return'].notna()].reset_index(drop=True)
    assert (df['next_date'] <= pandas.Timestamp(RESERVED_END)).all(), 'a reserved decision has an outcome beyond the window'
    need = feats if require_features is None else [c for c in feats if c in require_features]
    complete = df[need].notna().all(axis=1)
    ineligible = pandas.DataFrame({'decision_date': df.loc[~complete, 'decision_date'].dt.date.astype(str),
                               'missing_features': [', '.join(c for c in feats if pandas.isna(r[c])) for _, r in df[~complete].iterrows()]})
    df = df[complete].reset_index(drop=True)
    assert (df['next_date'] > df['decision_date']).all()
    df['y'] = df['positive_return'].astype(int)
    df['outcome_time'] = at_5pm_et(df['next_date'])
    df['decision_time'] = at_5pm_et(df['decision_date'])
    df['quarter'] = reserved_quarter_of(df['decision_date'])
    assert df['quarter'].between(17, 20).all()
    exc = pandas.read_csv(exceptions_path, parse_dates=['decision_date']) if exceptions_path.exists() else pandas.DataFrame(columns=['decision_date', 'flagged'])
    flagged = set(exc.loc[exc['flagged'].astype(bool), 'decision_date']) if len(exc) else set()
    df['flagged'] = df['decision_date'].isin(flagged)
    return Dataset(df=df, features=feats, usable_start=df['decision_date'].min(), private=all(c in F.columns for c in FEATURES)), ineligible
