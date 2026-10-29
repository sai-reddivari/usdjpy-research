"""Decision rows, the sigma_20 volatility hurdle and the binary label.

- Decision time: 17:00 ET on every cleaned FMP session date d. The FMP daily
  close for d is provisionally treated as the price at that moment (an
  assumption the report states; it is not resolved by the data).
- Outcome: r(d) = close(d_next) / close(d) - 1, where d_next is the next
  cleaned session date. The outcome time is 17:00 ET on d_next.
- sigma_20(d): sample standard deviation (ddof 1) of the 20 most recent
  simple close-to-close returns, the last being the return into close(d).
- Label: positive_return(d) = 1 if r(d) > max(0.001, 0.5 * sigma_20(d)), else 0.
  The target is positive hurdle exceedance, not direction.
- The first 20 rows have no sigma_20 and so no label. The last date of the
  table is dropped because its outcome lies beyond the window.
- positive_ret_25bps is a fixed 0.25% comparison label, not used for models.
"""

import numpy
import pandas

from .config import END
from .timing import at_et, iso_text

RETURN_FLOOR = 0.001
HURDLE_SCALE = 0.5
VOL_WINDOW = 20
FIXED_HURDLE = 0.0025

# Column order of usdjpy_5pm_et_label.csv.
LABEL_COLUMNS = [
    "date", "decision_time", "weekday", "close", "prev_bd_return", "20d_vol", "scaled_20d_vol",
    "positive_move_threshold", "floor_binds", "next_date", "gap_days", "fri_to_mon", "next_close_px",
    "next_day_return", "positive_return", "positive_ret_25bps",
]


def build_label(closes, end=END):
    """Decision rows with the label and its diagnostics, from the cleaned ['date', 'close'] table.

    end is the last date of the table; that row is dropped because its outcome
    lies beyond the window.
    """
    label = closes.copy()
    label["prev_bd_return"] = label["close"] / label["close"].shift(1) - 1
    label["20d_vol"] = label["prev_bd_return"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1)
    label["next_date"] = label["date"].shift(-1)
    label["next_close_px"] = label["close"].shift(-1)
    label["next_day_return"] = label["next_close_px"] / label["close"] - 1
    label["scaled_20d_vol"] = HURDLE_SCALE * label["20d_vol"]
    label["positive_move_threshold"] = numpy.maximum(RETURN_FLOOR, label["scaled_20d_vol"])   # NaN stays NaN

    assert label["date"].iloc[-1] == end, "the close table must end on the window end"
    assert pandas.isna(label["next_day_return"].iloc[-1]), "the last row cannot have an outcome"
    label = label.iloc[:-1].copy()
    labelled = has_label(label)
    assert (~labelled).sum() == VOL_WINDOW, "exactly the warm-up rows lack a label"
    assert (~labelled.iloc[:VOL_WINDOW]).all() and labelled.iloc[VOL_WINDOW:].all(), "unlabelled row after warm-up"

    label["floor_binds"] = (label["scaled_20d_vol"] < RETURN_FLOOR).where(labelled)
    exceeds_hurdle = label["next_day_return"] > label["positive_move_threshold"]
    label["positive_return"] = exceeds_hurdle.astype("Int64").where(labelled)
    label["positive_ret_25bps"] = (label["next_day_return"] > FIXED_HURDLE).astype("Int64").where(labelled)
    dates = pandas.to_datetime(label["date"])
    label["weekday"] = dates.dt.day_name()
    label["gap_days"] = (pandas.to_datetime(label["next_date"]) - dates).dt.days
    label["fri_to_mon"] = (dates.dt.dayofweek == 4) & (label["gap_days"] == 3)
    label["decision_time"] = [at_et(day, 17) for day in label["date"]]
    return label


def has_label(label):
    """True on rows that carry a label (sigma_20 and next-day return both present)."""
    return label["positive_move_threshold"].notna() & label["next_day_return"].notna()


def label_for_csv(label):
    """The 16 output columns, with decision_date and decision_time as text."""
    out = label[LABEL_COLUMNS].rename(columns={"date": "decision_date"}).copy()
    out["decision_time"] = iso_text(out["decision_time"])
    return out
