"""FMP FX cleaning, source calendars, availability rules and the as-of join.

Cleaning (FMP daily FX). The vendor file carries weekend rows that are not
trading days, 25 December / 1 January stubs with zero or tiny volume, and one
bad low print on 2024-12-25. Weekend rows and stubs are dropped; only close is
kept, because open is the previous close, change is close minus open and vwap
is the OHLC average.

Availability. Every source keeps its own calendar. Each observation gets an
observation time and an availability time in US Eastern, from the rules in
availability_rule(). Tokyo rules are written in JST and converted, because a
fixed ET time would be an hour early in the US summer.

Placeholder times. The observation files written by the collection notebooks
still carry provisional available_time values (ETFs next day 12:00 UTC, FRED
+10 days, CFTC +8 days, J-Quants next session 09:00 JST). They are not used:
load_long_observations() reads only series_id, observation_date and value, and
retime() stamps the rules below. tests/test_timing.py checks this.

As-of join. For each decision time the latest observation with
available_time <= decision_time is matched. A match older than the series'
staleness cap (7 days; 35 for weekly CFTC and broad-dollar data), measured from
the observation time, is blanked and recorded as 'stale (blanked)'.

CFTC release dates. The development schedule and special announcements are
fixed tables below. The reserved-year extension is passed explicitly as
arguments; there is no shared state, so calling the reserved build cannot
change any development result.
"""

import gzip
import html
import json
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy
import pandas
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import END, ET, JST, PRIVATE_DIR, PRIVATE_SCRATCH, RAW, START, is_inside

# ----------------------------------------------------------------------------
# Clocks
# ----------------------------------------------------------------------------


def at_et(day, hour, minute=0):
    """Wall-clock time on a date in US Eastern."""
    return pandas.Timestamp(datetime.combine(day, time(hour, minute), tzinfo=ET))


def at_jst(day, hour, minute=0):
    """Wall-clock time on a Tokyo date, converted to US Eastern."""
    return pandas.Timestamp(datetime.combine(day, time(hour, minute), tzinfo=JST)).tz_convert(ET)


def iso_text(timestamps):
    """ISO-8601 text with offset for a tz-aware column; blank for missing."""
    return timestamps.map(lambda stamp: stamp.isoformat() if pandas.notna(stamp) else "")


# ----------------------------------------------------------------------------
# FMP FX cleaning
# ----------------------------------------------------------------------------
HOLIDAY_STUB_DAYS = [(12, 25), (1, 1)]
BASKET_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"]
EXPECTED_DEVELOPMENT_SUNDAYS = 38   # audited Sunday-row count of the development USDJPY cache


def load_fmp_fx_raw(symbol, raw=RAW, start=START, end=END, extra_raw=()):
    """Raw FMP 'full' rows for one FX symbol with start <= date <= end, sorted by date.

    The date column holds datetime.date objects. extra_raw lists further raw
    folders (the reserved-year cache) whose rows are appended before the
    window filter; a date may not appear in two folders.
    """
    frames = [
        pandas.DataFrame(json.loads((folder / f"shared_fmp_{symbol.lower()}_full.json").read_text()))
        for folder in [raw, *extra_raw]
    ]
    prices = pandas.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    prices["date"] = pandas.to_datetime(prices["date"]).dt.date
    in_window = (prices["date"] >= start) & (prices["date"] <= end)
    prices = prices[in_window].sort_values("date").reset_index(drop=True)
    assert prices["date"].is_unique, f"{symbol}: a date repeats across raw folders"
    return prices


def artifact_masks(raw_prices):
    """Boolean masks for vendor artifacts: Sunday, weekend, holiday stub and zero-volume rows.

    A stub is a weekday row on 25 December or 1 January, or a weekday row with
    zero volume. USDJPY has Sunday rows only; the basket pairs also have
    Saturday rows. One rule (drop weekends and stubs) covers both.
    """
    weekday = pandas.to_datetime(raw_prices["date"]).dt.dayofweek          # Monday = 0, Sunday = 6
    weekend = weekday >= 5
    holiday = raw_prices["date"].map(lambda day: (day.month, day.day) in HOLIDAY_STUB_DAYS)
    if "volume" in raw_prices:
        zero_volume = raw_prices["volume"] == 0
    else:
        zero_volume = pandas.Series(False, index=raw_prices.index)
    stub = (holiday | zero_volume) & ~weekend
    return {"sunday": weekday == 6, "weekend": weekend, "holiday": holiday,
            "zero_volume": zero_volume, "stub": stub}


def clean_fmp_fx(raw_prices):
    """Drop weekend rows and stubs; return ['date', 'close'] with a fresh index."""
    masks = artifact_masks(raw_prices)
    kept = raw_prices[~(masks["weekend"] | masks["stub"])].reset_index(drop=True)
    closes = kept[["date", "close"]].copy()
    assert (pandas.to_datetime(closes["date"]).dt.dayofweek < 5).all(), "weekend row survived cleaning"
    assert closes["date"].is_unique, "duplicate close date after cleaning"
    return closes


def usdjpy_closes(raw=RAW, start=START, end=END, extra_raw=(), expected_sundays=EXPECTED_DEVELOPMENT_SUNDAYS):
    """The cleaned USD/JPY close series (1,298 development closes).

    expected_sundays is asserted for the development cache; pass None for
    another window, where the count is not fixed in advance.
    """
    raw_prices = load_fmp_fx_raw("USDJPY", raw, start, end, extra_raw)
    masks = artifact_masks(raw_prices)
    if expected_sundays is not None:
        assert masks["sunday"].sum() == expected_sundays, int(masks["sunday"].sum())
    assert not (masks["weekend"] & ~masks["sunday"]).any(), "USDJPY should have no Saturday rows"
    return clean_fmp_fx(raw_prices)


def basket_closes(raw=RAW, start=START, end=END, extra_raw=()):
    """Cleaned closes of the five basket pairs, one float Series per pair indexed by Timestamp."""
    closes_by_pair = {}
    for pair in BASKET_PAIRS:
        closes = clean_fmp_fx(load_fmp_fx_raw(pair, raw, start, end, extra_raw))
        closes_by_pair[pair] = pandas.Series(
            closes["close"].astype(float).to_numpy(), index=pandas.to_datetime(closes["date"])
        )
    return closes_by_pair


def weekdays_without_close(closes, start=START, end=END):
    """Monday-Friday dates in the window without a cleaned close (holiday stubs)."""
    return sorted(set(pandas.bdate_range(start, end).date) - set(closes["date"]))


# ----------------------------------------------------------------------------
# US and Tokyo calendars
# ----------------------------------------------------------------------------
# pandas calendar: includes Juneteenth from 2021, excludes one-off closures such as 2025-01-09.
US_HOLIDAYS = set(USFederalHolidayCalendar().holidays("2020-01-01", "2026-12-31").date)


def is_us_business_day(day):
    return day.weekday() < 5 and day not in US_HOLIDAYS


def next_us_business_day(day):
    day = day + timedelta(days=1)
    while not is_us_business_day(day):
        day += timedelta(days=1)
    return day


def h10_weekly_release_day(day):
    """First Monday strictly after the date; Tuesday when that Monday is a US federal holiday."""
    monday = day + timedelta(days=7 - day.weekday())
    return monday + timedelta(days=1) if monday in US_HOLIDAYS else monday


def tokyo_close(day):
    """TSE close in JST: 15:00, or 15:30 from 2024-11-05 when the session was extended."""
    return (15, 30) if day >= date(2024, 11, 5) else (15, 0)


def next_tokyo_business_day(day, tokyo_days):
    """Next date with a BOJ fix after the date; after the data end, the next weekday."""
    position = bisect_right(tokyo_days, day)
    if position < len(tokyo_days):
        return tokyo_days[position]
    day = day + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


# ----------------------------------------------------------------------------
# CFTC release dates
# ----------------------------------------------------------------------------
# Release dates (day numbers) by year-month, from the archived CFTC schedule pages, December 2020 to October 2025.
CFTC_SCHEDULE_DEVELOPMENT = {
    "2020-12": "4 11 18 28",
    "2021-01": "4 8 15 22 29", "2021-02": "5 12 19 26", "2021-03": "5 12 19 26", "2021-04": "2 9 16 23 30",
    "2021-05": "7 14 21 28", "2021-06": "4 11 18 25", "2021-07": "2 9 16 23 30", "2021-08": "6 13 20 27",
    "2021-09": "3 10 17 24", "2021-10": "1 8 15 22 29", "2021-11": "5 15 19 29", "2021-12": "3 10 17 27",
    "2022-01": "3 7 14 21 28", "2022-02": "4 11 18 25", "2022-03": "4 11 18 25", "2022-04": "1 8 15 22 29",
    "2022-05": "6 13 20 27", "2022-06": "3 10 17 24", "2022-07": "1 8 15 22 29", "2022-08": "5 12 19 26",
    "2022-09": "2 9 16 23 30", "2022-10": "7 14 21 28", "2022-11": "4 14 18 28", "2022-12": "2 9 16 23 30",
    "2023-01": "6 13 20 27", "2023-02": "3 10 17 24", "2023-03": "3 10 17 24 31", "2023-04": "7 14 21 28",
    "2023-05": "5 12 19 26", "2023-06": "2 9 16 23 30", "2023-07": "7 14 21 28", "2023-08": "4 11 18 25",
    "2023-09": "1 8 15 22 29", "2023-10": "6 13 20 27", "2023-11": "3 13 17 27", "2023-12": "1 8 15 22 29",
    "2024-01": "5 12 19 26", "2024-02": "2 9 16 23", "2024-03": "1 8 15 22 29", "2024-04": "5 12 19 26",
    "2024-05": "3 10 17 24 31", "2024-06": "7 14 24 28", "2024-07": "8 12 19 26", "2024-08": "2 9 16 23 30",
    "2024-09": "6 13 20 27", "2024-10": "4 11 18 25", "2024-11": "1 8 15 22", "2024-12": "2 6 13 20 30",
    "2025-01": "6 10 17 24 31", "2025-02": "7 14 21 28", "2025-03": "7 14 21 28", "2025-04": "4 11 18 25",
    "2025-05": "2 9 16 23 30", "2025-06": "6 13 23 27", "2025-07": "7 11 18 25", "2025-08": "1 8 15 22 29",
    "2025-09": "5 12 19 26", "2025-10": "3 10 17 24 31",
}
# Report date -> actual release date (CFTC Historical Special Announcements).
CFTC_EXCEPTIONS_DEVELOPMENT = {
    date(2021, 6, 15): date(2021, 6, 21),                                              # Juneteenth 2021
    date(2023, 1, 31): date(2023, 2, 24), date(2023, 2, 7): date(2023, 3, 3), date(2023, 2, 14): date(2023, 3, 8),
    date(2023, 2, 21): date(2023, 3, 10), date(2023, 2, 28): date(2023, 3, 14), date(2023, 3, 7): date(2023, 3, 16),
    date(2023, 3, 14): date(2023, 3, 21),                                              # ION cyber incident
    date(2025, 1, 7): date(2025, 1, 13),                                               # day of mourning
    date(2025, 9, 30): date(2025, 11, 19), date(2025, 10, 7): date(2025, 11, 21),
    date(2025, 10, 14): date(2025, 11, 25), date(2025, 10, 21): date(2025, 12, 2),    # federal shutdown
}


def validate_cftc_extension(extra_schedule, extra_exceptions):
    """A reserved-year extension may only add months after October 2025 and reports after 2025-10-21."""
    for month in extra_schedule:
        assert month not in CFTC_SCHEDULE_DEVELOPMENT and month >= "2025-11", month
    for report_date in extra_exceptions:
        assert report_date not in CFTC_EXCEPTIONS_DEVELOPMENT and report_date > date(2025, 10, 21), report_date


def scheduled_release_dates(extra_schedule=None):
    """Sorted scheduled release dates from the development schedule plus an optional extension."""
    tables = [CFTC_SCHEDULE_DEVELOPMENT, extra_schedule or {}]
    return sorted(
        date(int(month[:4]), int(month[5:]), int(day))
        for table in tables for month, days in table.items() for day in days.split()
    )


def cftc_release(report_date, scheduled=None, extra_exceptions=None):
    """Actual release date of a Tuesday report and which source gave it.

    Order: special announcement, then the archived schedule (first scheduled
    date within ten days), then the fallback for reports before December 2020
    (Friday, plus four days when the report week holds a US holiday).
    scheduled defaults to the development schedule.
    """
    extra_exceptions = extra_exceptions or {}
    if report_date in CFTC_EXCEPTIONS_DEVELOPMENT:
        return CFTC_EXCEPTIONS_DEVELOPMENT[report_date], "special announcement"
    if report_date in extra_exceptions:
        return extra_exceptions[report_date], "special announcement"
    if report_date >= date(2020, 12, 1):
        scheduled = scheduled if scheduled is not None else scheduled_release_dates()
        following = [day for day in scheduled if report_date < day <= report_date + timedelta(days=10)]
        assert following, f"no scheduled CFTC release within ten days of {report_date}"
        return following[0], "archived schedule"
    friday = report_date + timedelta(days=(4 - report_date.weekday()) % 7)
    report_week = {report_date - timedelta(days=report_date.weekday() - offset) for offset in range(5)}
    return (friday + timedelta(days=4) if report_week & US_HOLIDAYS else friday), "fallback rule"


def cftc_release_table(report_dates, extra_schedule=None, extra_exceptions=None):
    """report_date, release_date, source and days_after_report for each report."""
    if extra_schedule or extra_exceptions:
        validate_cftc_extension(extra_schedule or {}, extra_exceptions or {})
    scheduled = scheduled_release_dates(extra_schedule)
    table = pandas.DataFrame(
        [(report, *cftc_release(report, scheduled, extra_exceptions)) for report in report_dates],
        columns=["report_date", "release_date", "source"],
    )
    table["days_after_report"] = [(released - report).days
                                  for report, released in zip(table["report_date"], table["release_date"])]
    assert table.loc[table["source"] == "archived schedule", "release_date"].is_unique, \
        "two reports mapped to one scheduled release"
    return table


# ----------------------------------------------------------------------------
# Availability rules per series
# ----------------------------------------------------------------------------
@dataclass
class Calendars:
    """The two data-driven calendars the availability rules need."""
    tokyo_days: list            # dates with a BOJ 17:00 JST fix
    cftc_release: dict          # report date -> actual release date


def calendars_from_long(jpy_long, extra_schedule=None, extra_exceptions=None):
    """Tokyo business days and CFTC release dates from a Japanese-sources long frame."""
    series = jpy_long["series_id"]
    tokyo_days = sorted(pandas.to_datetime(jpy_long.loc[series == "jpy.boj_usdjpy_17_mid", "observation_date"]).dt.date)
    reports = sorted(pandas.to_datetime(jpy_long.loc[series == "jpy.cftc_lf_long", "observation_date"]).dt.date)
    table = cftc_release_table(reports, extra_schedule, extra_exceptions)
    return Calendars(tokyo_days, dict(zip(table["report_date"], table["release_date"])))


def calendars_from_file(path, extra_schedule=None, extra_exceptions=None):
    """calendars_from_long for a jpy_sources.csv file."""
    return calendars_from_long(pandas.read_csv(path), extra_schedule, extra_exceptions)


def availability_rule(series_id, calendars):
    """(observation_time(d), available_time(d), rule text, staleness cap in days) for one series."""
    if series_id == "fmp.usdjpy.close":
        return (lambda d: at_et(d, 17)), (lambda d: at_et(d, 17)), "17:00 ET d (label anchor)", 7
    if series_id.startswith("fmp."):
        return (lambda d: at_et(d, 16)), (lambda d: at_et(d, 16, 5)), "16:05 ET d", 7
    if series_id == "fred.vixcls":
        return (lambda d: at_et(d, 16, 15)), (lambda d: at_et(d, 16, 20)), "16:20 ET d", 7
    if series_id in ("fred.dgs2", "fred.dgs10", "fred.dfii5", "fred.dfii10"):
        return (lambda d: at_et(d, 15, 30)), (lambda d: at_et(d, 16, 30)), "16:30 ET d", 7
    if series_id == "fred.sofr":
        return ((lambda d: at_et(d, 17)), (lambda d: at_et(next_us_business_day(d), 8, 5)),
                "08:05 ET next US business day", 7)
    if series_id == "fred.dcoilwtico":
        return (lambda d: at_et(d, 14, 30)), (lambda d: at_et(d, 15)), "15:00 ET d (posting time unverified)", 7
    if series_id == "fred.dtwexbgs":
        return ((lambda d: at_et(d, 12)), (lambda d: at_et(h10_weekly_release_day(d), 16, 20)),
                "16:20 ET first Monday after d (Tue if holiday)", 35)
    if series_id.startswith("fred.dex"):
        return (lambda d: at_et(d, 12)), (lambda d: at_et(d, 12, 5)), "12:05 ET d (H.10 noon)", 7
    if series_id.startswith("jpy.cftc_"):
        return ((lambda d: at_et(d, 17)), (lambda d: at_et(calendars.cftc_release[d], 15, 35)),
                "CFTC release date 15:35 ET", 35)
    if series_id.startswith("jpy.mof_jgb_"):
        return (lambda d: at_jst(d, 15)), (lambda d: at_jst(d, 15, 5)), "15:05 JST d", 7
    if series_id == "jpy.boj_usdjpy_17_mid":
        return (lambda d: at_jst(d, 17)), (lambda d: at_jst(d, 17, 5)), "17:05 JST d", 7
    if series_id == "jpy.boj_tona_final":
        return ((lambda d: at_jst(d, 17)),
                (lambda d: at_jst(next_tokyo_business_day(d, calendars.tokyo_days), 10, 5)),
                "10:05 JST next Tokyo business day", 7)
    if series_id == "jpy.nikkei225" or series_id.startswith("jpy.topix"):
        # Tokyo-close rule; not the next-session placeholder the collection notebook wrote for J-Quants.
        return (lambda d: at_jst(d, *tokyo_close(d))), (lambda d: at_jst(d, 15, 35)), "15:35 JST d", 7
    raise KeyError(series_id)


def available_time(series_id, day, calendars):
    """When the observation of a series dated day becomes usable, in ET."""
    return availability_rule(series_id, calendars)[1](day)


def observation_time(series_id, day, calendars):
    """When the observation of a series dated day was made, in ET."""
    return availability_rule(series_id, calendars)[0](day)


# ----------------------------------------------------------------------------
# Long observations -> re-timed observations -> as-of table and audit
# ----------------------------------------------------------------------------
def load_long_observations(path, start=START, end=END):
    """series_id, observation_date, value of a long file with start <= date <= end.

    The file's own observation_time and available_time columns are ignored on
    purpose (see the module docstring); retime() supplies the rules.
    """
    observations = pandas.read_csv(path, usecols=["series_id", "observation_date", "value"])
    observations["observation_date"] = pandas.to_datetime(observations["observation_date"]).dt.date
    in_window = (observations["observation_date"] >= start) & (observations["observation_date"] <= end)
    return observations[in_window]


def closes_as_long(closes):
    """The cleaned USD/JPY close as a long frame with series_id 'fmp.usdjpy.close'."""
    return closes.rename(columns={"date": "observation_date", "close": "value"}).assign(series_id="fmp.usdjpy.close")


def public_long_frame(closes, observations_dir):
    """The cleaned close plus the ETF, FRED and Japanese long files, in that order."""
    frame = pandas.concat([
        closes_as_long(closes),
        load_long_observations(observations_dir / "shared_fmp_etf.csv"),
        load_long_observations(observations_dir / "shared_fred.csv"),
        load_long_observations(observations_dir / "jpy_sources.csv"),
    ], ignore_index=True)
    frame["licensed"] = False
    return frame


def jquants_long_frame(path, start=START, end=END):
    """The licensed J-Quants long file (values are never printed)."""
    frame = load_long_observations(path, start, end)
    assert frame["series_id"].str.startswith("jpy.topix").all(), "unexpected series in the J-Quants file"
    return frame.assign(licensed=True)


def retime(observations, calendars):
    """Stamp observation_time, available_time, rule and cap_days on every row from availability_rule()."""
    parts = []
    for series_id, group in observations.groupby("series_id", sort=False):
        observed_at, available_at, rule_text, cap_days = availability_rule(series_id, calendars)
        days = sorted(set(group["observation_date"]))
        observation_times = dict(zip(days, (observed_at(day) for day in days)))
        available_times = dict(zip(days, (available_at(day) for day in days)))
        parts.append(group.assign(
            observation_time=group["observation_date"].map(observation_times),
            available_time=group["observation_date"].map(available_times),
            rule=rule_text, cap_days=cap_days,
        ))
    timed = pandas.concat(parts, ignore_index=True)
    for column in ("observation_time", "available_time"):
        timed[column] = pandas.to_datetime(timed[column]).dt.tz_convert(ET).dt.as_unit("ns")
    assert timed["value"].notna().all(), "missing observation value"
    assert not timed.duplicated(["series_id", "observation_date"]).any(), "duplicate observation"
    assert (timed["available_time"] >= timed["observation_time"]).all(), "available before observed"
    return timed


def decision_frame(label):
    """decision_date and tz-aware decision_time (17:00 ET) from the label table."""
    decisions = label[["date", "decision_time"]].rename(columns={"date": "decision_date"}).reset_index(drop=True)
    decisions["decision_time"] = pandas.to_datetime(decisions["decision_time"]).dt.tz_convert(ET).dt.as_unit("ns")
    return decisions


def asof_match(timed_observations, decisions):
    """Latest observation with available_time <= decision_time per series, with a staleness status.

    Status is 'filled' when the observation is at most cap_days old (from its
    observation time), 'stale (blanked)' when older, and 'no prior
    observation' when nothing was available yet.
    """
    matches = []
    for series_id, group in timed_observations.groupby("series_id", sort=False):
        group = group.sort_values(["available_time", "observation_date"])
        assert group["observation_date"].is_monotonic_increasing, f"{series_id}: a later release carries older data"
        group = group.drop_duplicates("available_time", keep="last")      # same release time: keep the newest date
        matched = pandas.merge_asof(
            decisions, group.drop(columns="series_id"), left_on="decision_time", right_on="available_time",
            direction="backward", allow_exact_matches=True,
        )
        matched["series_id"] = series_id
        matches.append(matched)
    audit = pandas.concat(matches, ignore_index=True)
    has_match = audit["available_time"].notna()
    assert (audit.loc[has_match, "available_time"] <= audit.loc[has_match, "decision_time"]).all(), "look-ahead"
    audit["age_obs_h"] = (audit["decision_time"] - audit["observation_time"]).dt.total_seconds() / 3600
    audit["age_avail_h"] = (audit["decision_time"] - audit["available_time"]).dt.total_seconds() / 3600
    within_cap = audit["age_obs_h"] <= audit["cap_days"] * 24
    audit["status"] = numpy.where(~has_match, "no prior observation",
                                  numpy.where(within_cap, "filled", "stale (blanked)"))
    return audit


def public_column_order(timed_public):
    """fmp.usdjpy.close first, then the fmp.*, fred.* and jpy.* series, each sorted."""
    series = timed_public["series_id"].unique()
    return (["fmp.usdjpy.close"]
            + sorted(s for s in series if s.startswith("fmp.") and s != "fmp.usdjpy.close")
            + sorted(s for s in series if s.startswith("fred."))
            + sorted(s for s in series if s.startswith("jpy.")))


def widen(audit, decisions, columns):
    """Filled cells as a wide table, one row per decision date."""
    filled = audit[audit["status"] == "filled"]
    wide = filled.pivot(index="decision_date", columns="series_id", values="value")
    return wide.reindex(index=decisions["decision_date"], columns=columns).reset_index()


def wide_table(decisions, audit, columns):
    """decision_date, decision_time and the as-of columns."""
    wide = decisions.merge(widen(audit, decisions, columns), on="decision_date")
    assert len(wide) == len(decisions), "decision rows lost in widening"
    return wide


def wide_for_csv(wide):
    """Copy with decision_time as ISO text, ready for to_csv."""
    out = wide.copy()
    out["decision_time"] = iso_text(out["decision_time"])
    return out


AUDIT_COLUMNS = ["decision_date", "decision_time", "series_id", "status", "observation_date", "observation_time",
                 "available_time", "age_obs_h", "age_avail_h", "cap_days", "rule"]


def audit_for_csv(audit):
    """Audit rows that matched something, times as text, ages to 3 decimals. No values."""
    rows = audit[audit["status"] != "no prior observation"][AUDIT_COLUMNS].rename(columns={
        "observation_date": "matched_observation_date",
        "observation_time": "matched_observation_time",
        "available_time": "matched_available_time",
    }).copy()
    for column in ("decision_time", "matched_observation_time", "matched_available_time"):
        rows[column] = iso_text(rows[column])
    rows[["age_obs_h", "age_avail_h"]] = rows[["age_obs_h", "age_avail_h"]].round(3)
    return rows


def write_public_audit(audit, path):
    """Write the public audit CSV (no values, no J-Quants rows)."""
    rows = audit_for_csv(audit)
    assert not rows["series_id"].str.startswith("jpy.topix").any(), "licensed rows in the public audit"
    rows.to_csv(path, index=False)
    return path


def write_private_csv(frame, path):
    """Write a licensed table with mode 0600, only inside the private cache or the private scratch folder."""
    if not (is_inside(path, PRIVATE_DIR) or is_inside(path, PRIVATE_SCRATCH)):
        raise ValueError("Licensed data must stay in the private folder.")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", newline="") as handle:
        frame.to_csv(handle, index=False)
    os.chmod(path, 0o600)
    return path


# ----------------------------------------------------------------------------
# Reserved-year CFTC extension (parsed from cached pages; relocated from data_review/reserved_year_build.py)
# ----------------------------------------------------------------------------
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
RESERVED_DECEMBER_2025_RELEASE = {"2025-12": "29"}   # added by the reserved build beside the 2026 page


def cached_page_text(path):
    """Visible text of a cached HTML page (regular expressions only; the page is untrusted data)."""
    raw_bytes = path.read_bytes()
    try:
        text = gzip.decompress(raw_bytes).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        text = raw_bytes.decode("utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))   # unescape before collapsing


def parse_cftc_schedule_2026(path):
    """{'2026-MM': 'd d d ...'} from the cached CFTC release-schedule page (fetched 2026-10-27)."""
    text = cached_page_text(path)
    begin = text.find("2026 Release Schedule")
    assert begin >= 0, "2026 schedule heading not found"
    segment = text[begin:text.find("*Delayed release date", begin)]
    schedule = {}
    for match in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                             r"((?: \d{2}\*?)+)", segment):
        month = MONTH_NAMES.index(match.group(1)) + 1
        days = [int(day.rstrip("*")) for day in match.group(2).split()]
        schedule[f"2026-{month:02d}"] = " ".join(str(day) for day in days)
    assert len(schedule) == 12, schedule
    return schedule


def parse_cftc_catch_up_2025(path):
    """Report date -> new publication date from the 2025-12-09 announcement (supersedes 2025-11-18)."""
    text = cached_page_text(path)
    begin = text.find("December 9, 2025:")
    stop = text.find("November 18, 2025:", begin)
    assert 0 <= begin < stop, "catch-up announcement not found"
    rows = re.findall(r"(\d{2}/\d{2}/\d{4}) (\d{2}/\d{2}/\d{4}) (\d{2}/\d{2}/\d{4})\+*", text[begin:stop])
    assert rows, "no catch-up rows parsed"
    return {datetime.strptime(report, "%m/%d/%Y").date(): datetime.strptime(new, "%m/%d/%Y").date()
            for report, _original, new in rows}


def reserved_cftc_extension(calendar_dir):
    """(extra_schedule, extra_exceptions) for reserved-year reports, as the reserved build applied them.

    Only reports after the development end take a catch-up date; the
    development table already holds the shutdown dates of earlier reports.
    """
    schedule = {**RESERVED_DECEMBER_2025_RELEASE, **parse_cftc_schedule_2026(calendar_dir / "cftc_schedule.html")}
    catch_up = parse_cftc_catch_up_2025(calendar_dir / "cftc_special.html")
    exceptions = {report: released for report, released in catch_up.items() if report > END}
    validate_cftc_extension(schedule, exceptions)
    return schedule, exceptions


# ----------------------------------------------------------------------------
# Cache-only rebuild of the 17:00 ET aligned tables (from data_review/run_all.py)
# ----------------------------------------------------------------------------
PUBLIC_ALIGNED_FILES = ['usdjpy_5pm_et_observables.csv', 'usdjpy_5pm_et_label.csv', 'usdjpy_5pm_et_asof_audit.csv']
PRIVATE_ALIGNED_FILES = ['usdjpy_5pm_et_observables_with_jquants.csv', 'usdjpy_5pm_et_asof_audit_jquants.csv']


def write_aligned_tables(observations_dir, out_dir, private_observations=None, private_out=None, raw=RAW):
    """Clean, label, re-time, as-of join and write the aligned tables and audits.

    Public files go to out_dir. When private_observations (a J-Quants long
    file) is given, the two licensed tables go to private_out with mode 0600.
    """
    from . import target

    closes = usdjpy_closes(raw)
    label = target.build_label(closes)
    target.label_for_csv(label).to_csv(out_dir / "usdjpy_5pm_et_label.csv", index=False)

    calendars = calendars_from_file(observations_dir / "jpy_sources.csv")
    decisions = decision_frame(label)
    timed_public = retime(public_long_frame(closes, observations_dir), calendars)
    audit_public = asof_match(timed_public, decisions)
    wide_public = wide_table(decisions, audit_public, public_column_order(timed_public))
    assert not any(column.startswith("jpy.topix") for column in wide_public.columns), "licensed column in public table"
    wide_for_csv(wide_public).to_csv(out_dir / "usdjpy_5pm_et_observables.csv", index=False)
    write_public_audit(audit_public, out_dir / "usdjpy_5pm_et_asof_audit.csv")
    if private_observations is None:
        return
    timed_jquants = retime(jquants_long_frame(private_observations), calendars)
    audit_jquants = asof_match(timed_jquants, decisions)
    jquants_columns = sorted(timed_jquants["series_id"].unique())
    wide_full = wide_public.merge(widen(audit_jquants, decisions, jquants_columns), on="decision_date")
    assert len(wide_full) == len(decisions), "decision rows lost joining licensed columns"
    write_private_csv(wide_for_csv(wide_full), private_out / "usdjpy_5pm_et_observables_with_jquants.csv")
    write_private_csv(audit_for_csv(audit_jquants), private_out / "usdjpy_5pm_et_asof_audit_jquants.csv")
