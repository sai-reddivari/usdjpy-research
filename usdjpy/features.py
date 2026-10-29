"""The 39 USD/JPY candidate features of protocol v2.1, on native calendars, aligned to decision rows.

Every feature is computed on its source's own observation calendar (FX
sessions, US sessions, Tokyo sessions, the EIA oil calendar or the CFTC report
sequence) and then aligned to the 17:00 ET decision rows with the availability
rules of timing.py. The matched observation date and the age at decision are
kept for every aligned feature. Features defined on decision rows (the gap2y
changes, vol_z, short_yen_stress, gapchg_x_vixchg) are derived from aligned
inputs. Trailing z-scores are causal (ddof 1, missing until the window fills).

Missing data: a stale match is blanked, never carried; warm-up rows stay
missing. Nothing is imputed.

Licensed J-Quants values never leave the private folder. The public feature
file and the public audit omit the four sector features entirely. A build that
needs them (private=True) raises config.PrivateDataUnavailable when they are
absent instead of returning the 35 public columns.

Formulas, calendars and caps are relocated unchanged from
data_review/exam3/features.py; tests/test_data_equivalence.py compares the
build with the saved feature files.

Entry points
- build_features(...)        -> FeatureBuild (features, audits, basket flags, ablation aux)
- reserved_inputs(...)       -> Inputs over warm-up + development + reserved window
- write_feature_tables(...)  -> the public and private feature files, into an explicit output folder
- event_timing_exceptions    -> the protocol section-7 exception table
- usable_start, missing_after_warmup, age_summary, spearman_pairs, reference_row
"""
from __future__ import annotations

import gzip
import html as html_lib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy
import pandas

from . import target, timing
from .config import (ALIGNED, END, ET, OBSERVATIONS, PRIVATE_DIR, RAW, RESERVED_ALIGNED, RESERVED_END,
                     RESERVED_RAW, RESERVED_START, START, require_private_inputs)
from .timing import Calendars, at_et, iso_text

# ----------------------------------------------------------------------------
# Column lists (protocol section 3, in dictionary order)
# ----------------------------------------------------------------------------
OWN_MARKET = ['ret_1d', 'ret_5d', 'ret_20d', 'abs_ret_1d', 'vol_20', 'vol_5_over_20', 'dist_ma20', 'dist_ma60',
              'tokyo_reference_to_ny_close', 'weekend_in_window']
RATES = ['gap2y', 'gap2y_chg_1d', 'gap2y_chg_5d', 'us_slope_chg_5d', 'jp_slope_chg_5d']
SECTORS = ['banks_rel_5d', 'banks_rel_20d', 'elec_rel_5d', 'topix_5d']                       # licensed, private only
RISK = ['vix', 'vix_chg_1d', 'vix_chg_5d', 'spy_1d', 'spy_5d', 'qqq_minus_spy_5d', 'qqq_minus_spy_20d', 'smh_minus_qqq_5d']
DOLLAR = ['usd_basket_5d', 'yen_relative_5d']
COMMODITIES = ['wti_5d_lagged', 'gld_5d']
POSITIONING = ['cftc_net', 'cftc_net_chg_4reports', 'cftc_release_age_days', 'pos_z', 'vol_z', 'short_yen_stress']
EVENTS = ['event_in_window']
INTERACTION = ['gapchg_x_vixchg']
FEATURES = OWN_MARKET + RATES + SECTORS + RISK + DOLLAR + COMMODITIES + POSITIONING + EVENTS + INTERACTION
LICENSED = list(SECTORS)
PUBLIC = [c for c in FEATURES if c not in LICENSED]
assert len(FEATURES) == 39 and len(set(FEATURES)) == 39 and len(PUBLIC) == 35

FAMILY = {**{c: 'own_market' for c in OWN_MARKET}, **{c: 'rates' for c in RATES}, **{c: 'sectors' for c in SECTORS},
          **{c: 'risk' for c in RISK}, **{c: 'dollar' for c in DOLLAR}, **{c: 'commodities' for c in COMMODITIES},
          **{c: 'positioning' for c in POSITIONING}, **{c: 'events' for c in EVENTS}, **{c: 'interaction' for c in INTERACTION}}

# The six renames of protocol section 4 (new -> notebook-07 row-calendar name) and the replaced interaction.
RENAMES = {'tokyo_reference_to_ny_close': 'ny_session_move', 'weekend_in_window': 'fri_to_mon',
           'yen_relative_5d': 'yen_specific_5d', 'wti_5d_lagged': 'wti_5d',
           'cftc_net_chg_4reports': 'cftc_net_chg_4w', 'cftc_release_age_days': 'cftc_age_days'}
REPLACED = {'short_yen_stress': 'fragility'}          # new definition, not a rename

AUX_COLS = ['us2y', 'us2y_chg_1d', 'us2y_chg_5d', 'z260_us2y_chg_5d', 'us_only_gapchg_x_vixchg']

Z_WINDOW = 260            # trailing decision rows for z260
CFTC_Z_REPORTS = 52       # distinct reports for pos_z
SD_FLOOR = 1e-12
BASKET_SIGN = {'EURUSD': -1, 'GBPUSD': -1, 'AUDUSD': -1, 'USDCAD': +1, 'USDCHF': +1}   # + = stronger dollar
REFERENCE_DATE = '2024-07-30'

PREV_PUBLIC_DIR = ALIGNED / '_prev_rowcal'
PREV_PRIVATE_DIR = PRIVATE_DIR / '_prev_rowcal'

AUDIT_COLS = ['feature', 'decision_date', 'source_series', 'matched_observation_date', 'matched_observation_time',
              'matched_available_time', 'age_obs_h', 'age_avail_h', 'status']


# ----------------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------------
def logret(s: pandas.Series, k: int) -> pandas.Series:
    """k-observation log return on the series' own index; missing until k observations exist."""
    return numpy.log(s / s.shift(k))


def z_trailing(s: pandas.Series, n: int) -> pandas.Series:
    """(x_t - mean) / sd over the n observations ending at t, ddof 1; missing until n exist or when sd < SD_FLOOR."""
    m = s.rolling(n, min_periods=n)
    sd = m.std(ddof=1)
    return ((s - m.mean()) / sd).where(sd >= SD_FLOOR)


def z260(s: pandas.Series) -> pandas.Series:
    return z_trailing(s, Z_WINDOW)


def native(long_df: pandas.DataFrame, sid: str) -> pandas.Series:
    """One series on its own observation calendar: float values indexed by observation_date (date objects)."""
    g = long_df[long_df['series_id'] == sid].sort_values('observation_date')
    assert len(g) and g['observation_date'].is_unique, sid
    return pandas.Series(g['value'].astype(float).to_numpy(), index=pandas.Index(list(g['observation_date']), name='observation_date'))


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
@dataclass
class Inputs:
    spot: pandas.DataFrame                 # cleaned USD/JPY ['date', 'close'] (1,298 rows)
    label: pandas.DataFrame                # notebook 06 label table (1,297 rows)
    dec: pandas.DataFrame                  # decision_date, decision_time (17:00 ET)
    cal: Calendars
    etf: pandas.DataFrame                  # long frames (series_id, observation_date, value)
    fred: pandas.DataFrame
    jpy: pandas.DataFrame
    jq: pandas.DataFrame | None            # licensed, or None when the private folder is absent
    basket: dict[str, pandas.Series]       # cleaned basket closes indexed by Timestamp
    events: pandas.DataFrame


def load_inputs(private: bool = True, raw: Path = RAW, obs_dir: Path = OBSERVATIONS,
                private_root: Path = PRIVATE_DIR) -> Inputs:
    """Read everything from the caches and observation files over the development window; nothing is fetched.

    private=True requires the licensed J-Quants file and raises
    PrivateDataUnavailable without it. private=False is the explicit
    public-only build (35 features).
    """
    spot = timing.usdjpy_closes(raw)
    label = target.build_label(spot)
    dec = timing.decision_frame(label)
    cal = timing.calendars_from_file(obs_dir / 'jpy_sources.csv')
    jq = None
    if private:
        (jq_path,) = require_private_inputs(private_root / 'jquants_observations.csv')
        jq = timing.jquants_long_frame(jq_path)
    events = pandas.read_csv(obs_dir / 'scheduled_events_2020_2025.csv')
    return Inputs(spot, label, dec, cal, timing.load_long_observations(obs_dir / 'shared_fmp_etf.csv'),
                  timing.load_long_observations(obs_dir / 'shared_fred.csv'), timing.load_long_observations(obs_dir / 'jpy_sources.csv'), jq,
                  timing.basket_closes(raw), events)


# ----------------------------------------------------------------------------
# Native-calendar alignment
# ----------------------------------------------------------------------------
def _asof_one(obs: pandas.DataFrame, dec: pandas.DataFrame, feature: str, sources: str) -> tuple[pandas.Series, pandas.DataFrame]:
    """As-of join of one re-timed series to the decision rows: (values, audit rows). Blanked when stale."""
    m = timing.asof_match(obs, dec)
    assert len(m) == len(dec) and (m['decision_date'].to_numpy() == dec['decision_date'].to_numpy()).all()
    values = pandas.Series(numpy.where(m['status'] == 'filled', m['value'], numpy.nan), index=dec.index, name=feature)
    a = pandas.DataFrame({'feature': feature, 'decision_date': m['decision_date'], 'source_series': sources,
                      'matched_observation_date': m['observation_date'], 'matched_observation_time': iso_text(m['observation_time']),
                      'matched_available_time': iso_text(m['available_time']), 'age_obs_h': m['age_obs_h'].round(3),
                      'age_avail_h': m['age_avail_h'].round(3), 'status': m['status']})
    return values, a[AUDIT_COLS]


def align_native(values: pandas.Series, feature: str, rule_sid: str, cal: Calendars, dec: pandas.DataFrame,
                 sources: str | None = None) -> tuple[pandas.Series, pandas.DataFrame]:
    """Align a series computed on a native calendar using the availability rule of series ``rule_sid``.

    ``values`` is indexed by observation_date (date objects); warm-up rows
    (missing) are dropped before the join, so early decision rows match nothing.
    """
    s = values.dropna()
    df = pandas.DataFrame({'series_id': rule_sid, 'observation_date': list(s.index), 'value': s.to_numpy()})
    obs = timing.retime(df, cal)
    return _asof_one(obs, dec, feature, sources or rule_sid)


def align_lagged_one_observation(values: pandas.Series, feature: str, rule_sid: str, cal: Calendars, dec: pandas.DataFrame,
                                 sources: str | None = None) -> tuple[pandas.Series, pandas.DataFrame]:
    """Align a native-calendar series lagged by one observation.

    The value whose last observation is d_{k-1} is released to the decision
    rows only when observation d_k becomes available (rule of ``rule_sid``),
    so the audit records d_{k-1} as the matched observation (and its age) and
    the availability time of d_k. Used for wti_5d_lagged.
    """
    s = values.dropna()
    full_dates = list(values.index)                              # the whole native calendar, including warm-up dates
    pos = {d: i for i, d in enumerate(full_dates)}
    rows = []
    _, _, text, cap = timing.availability_rule(rule_sid, cal)
    for d_prev, v in s.items():
        i = pos[d_prev] + 1
        if i >= len(full_dates):
            continue                                             # no later observation inside the window: never released
        d_next = full_dates[i]
        rows.append((rule_sid, d_prev, v, timing.observation_time(rule_sid, d_prev, cal),
                     timing.available_time(rule_sid, d_next, cal), f'{text}, released with the next observation', cap))
    obs = pandas.DataFrame(rows, columns=['series_id', 'observation_date', 'value', 'observation_time', 'available_time', 'rule', 'cap_days'])
    for c in ('observation_time', 'available_time'):
        obs[c] = pandas.to_datetime(obs[c]).dt.tz_convert(ET).dt.as_unit('ns')
    assert (obs['available_time'] >= obs['observation_time']).all()
    return _asof_one(obs, dec, feature, sources or rule_sid)


# ----------------------------------------------------------------------------
# Feature families
# ----------------------------------------------------------------------------
def own_market_features(inp: Inputs) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """ret_*, abs_ret_1d, vol_20, vol_5_over_20, dist_ma*, tokyo_reference_to_ny_close, weekend_in_window on the FX calendar."""
    spot = inp.spot
    px = spot['close'].astype(float)
    r1 = logret(px, 1)
    r1_simple = px / px.shift(1) - 1                                    # the label's return definition
    f = pandas.DataFrame({'decision_date': spot['date'], 'ret_1d': r1, 'ret_5d': logret(px, 5), 'ret_20d': logret(px, 20),
                      'abs_ret_1d': r1.abs(),
                      'vol_20': r1_simple.rolling(20, min_periods=20).std(ddof=1),
                      'vol_5_over_20': r1_simple.rolling(5, min_periods=5).std(ddof=1)
                      / r1_simple.rolling(20, min_periods=20).std(ddof=1),
                      'dist_ma20': px / px.rolling(20, min_periods=20).mean() - 1,
                      'dist_ma60': px / px.rolling(60, min_periods=60).mean() - 1})
    f = inp.dec[['decision_date']].merge(f, on='decision_date', how='left')
    assert len(f) == len(inp.dec)

    fix, a_fix = align_native(native(inp.jpy, 'jpy.boj_usdjpy_17_mid'), 'tokyo_reference_to_ny_close',
                              'jpy.boj_usdjpy_17_mid', inp.cal, inp.dec)
    close = inp.dec[['decision_date']].merge(spot, left_on='decision_date', right_on='date', how='left')['close'].astype(float)
    f['tokyo_reference_to_ny_close'] = close.to_numpy() / fix.to_numpy() - 1

    d = pandas.to_datetime(inp.label['date'])
    gap_days = (pandas.to_datetime(inp.label['next_date']) - d).dt.days
    f['weekend_in_window'] = (d.dt.dayofweek + gap_days >= 5).astype(int).to_numpy()   # (t, t+1] passes a Saturday
    return f, a_fix


def rates_features(inp: Inputs) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """gap2y from the latest available legs; its changes over decision rows; slopes on their native calendars.

    Returns (features, audit, ablation aux with the US-only columns).
    """
    us2 = native(inp.fred, 'fred.dgs2'); us10 = native(inp.fred, 'fred.dgs10')
    jp2 = native(inp.jpy, 'jpy.mof_jgb_2y'); jp10 = native(inp.jpy, 'jpy.mof_jgb_10y')
    us2_al, a1 = align_native(us2, 'gap2y', 'fred.dgs2', inp.cal, inp.dec)
    jp2_al, a2 = align_native(jp2, 'gap2y', 'jpy.mof_jgb_2y', inp.cal, inp.dec)
    gap = us2_al - jp2_al
    f = pandas.DataFrame({'decision_date': inp.dec['decision_date'], 'gap2y': gap, 'gap2y_chg_1d': gap.diff(1), 'gap2y_chg_5d': gap.diff(5)})
    us_slope, a3 = align_native((us10 - us2).diff(5), 'us_slope_chg_5d', 'fred.dgs2', inp.cal, inp.dec, 'fred.dgs10 - fred.dgs2')
    jp_slope, a4 = align_native((jp10 - jp2).diff(5), 'jp_slope_chg_5d', 'jpy.mof_jgb_2y', inp.cal, inp.dec, 'jpy.mof_jgb_10y - jpy.mof_jgb_2y')
    f['us_slope_chg_5d'] = us_slope
    f['jp_slope_chg_5d'] = jp_slope
    aux = pandas.DataFrame({'decision_date': inp.dec['decision_date'], 'us2y': us2_al, 'us2y_chg_1d': us2_al.diff(1), 'us2y_chg_5d': us2_al.diff(5)})
    return f, pandas.concat([a1, a2, a3, a4], ignore_index=True), aux


def sector_features(inp: Inputs) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Licensed J-Quants sector features on Tokyo sessions, then aligned at 15:35 JST."""
    assert inp.jq is not None
    topix = native(inp.jq, 'jpy.topix_pr.c'); banks = native(inp.jq, 'jpy.topix17_banks_pr.c')
    elec = native(inp.jq, 'jpy.topix17_electric_precision_pr.c')
    assert topix.index.equals(banks.index) and topix.index.equals(elec.index)
    spec = {'banks_rel_5d': (logret(banks, 5) - logret(topix, 5), 'jpy.topix17_banks_pr.c - jpy.topix_pr.c'),
            'banks_rel_20d': (logret(banks, 20) - logret(topix, 20), 'jpy.topix17_banks_pr.c - jpy.topix_pr.c'),
            'elec_rel_5d': (logret(elec, 5) - logret(topix, 5), 'jpy.topix17_electric_precision_pr.c - jpy.topix_pr.c'),
            'topix_5d': (logret(topix, 5), 'jpy.topix_pr.c')}
    f = pandas.DataFrame({'decision_date': inp.dec['decision_date']})
    audits = []
    for name, (s, src) in spec.items():
        f[name], a = align_native(s, name, 'jpy.topix_pr.c', inp.cal, inp.dec, src)
        audits.append(a)
    return f, pandas.concat(audits, ignore_index=True)


def risk_features(inp: Inputs) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """VIX level and changes on the VIX calendar; ETF returns on the US session calendar; then aligned."""
    vix = native(inp.fred, 'fred.vixcls')
    spy = native(inp.etf, 'fmp.spy.close'); qqq = native(inp.etf, 'fmp.qqq.close'); smh = native(inp.etf, 'fmp.smh.close')
    assert spy.index.equals(qqq.index) and spy.index.equals(smh.index)
    spec = {'vix': (vix, 'fred.vixcls', 'fred.vixcls'),
            'vix_chg_1d': (vix.diff(1), 'fred.vixcls', 'fred.vixcls'),
            'vix_chg_5d': (vix.diff(5), 'fred.vixcls', 'fred.vixcls'),
            'spy_1d': (logret(spy, 1), 'fmp.spy.close', 'fmp.spy.close'),
            'spy_5d': (logret(spy, 5), 'fmp.spy.close', 'fmp.spy.close'),
            'qqq_minus_spy_5d': (logret(qqq, 5) - logret(spy, 5), 'fmp.spy.close', 'fmp.qqq.close - fmp.spy.close'),
            'qqq_minus_spy_20d': (logret(qqq, 20) - logret(spy, 20), 'fmp.spy.close', 'fmp.qqq.close - fmp.spy.close'),
            'smh_minus_qqq_5d': (logret(smh, 5) - logret(qqq, 5), 'fmp.spy.close', 'fmp.smh.close - fmp.qqq.close')}
    f = pandas.DataFrame({'decision_date': inp.dec['decision_date']})
    audits = []
    for name, (s, rule_sid, src) in spec.items():
        f[name], a = align_native(s, name, rule_sid, inp.cal, inp.dec, src)
        audits.append(a)
    return f, pandas.concat(audits, ignore_index=True)


def basket_features(inp: Inputs, ret_5d: pandas.Series) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """usd_basket_5d with common endpoints [t-5, t] on the USD/JPY FX calendar, and yen_relative_5d.

    Each pair's close on the two endpoint dates is used; where a pair has no
    bar on an endpoint date the latest earlier close (at most 7 days old) is
    carried and the decision date is flagged. Returns (features, flags).
    """
    fx_dates = pandas.to_datetime(inp.spot['date'])                       # the FX calendar, 1,298 dates
    n = len(fx_dates)
    carried: dict[str, pandas.Series] = {}
    from_date: dict[str, pandas.Series] = {}
    for pair, ser in inp.basket.items():
        pos = ser.index.searchsorted(fx_dates, side='right') - 1
        ok = pos >= 0
        matched = pandas.Series(pandas.NaT, index=range(n), dtype='datetime64[ns]')
        matched[ok] = ser.index[pos[ok]]
        age = (fx_dates - matched).dt.days
        vals = pandas.Series(numpy.where(ok & (age <= 7), ser.to_numpy()[numpy.clip(pos, 0, None)], numpy.nan), index=range(n))
        carried[pair] = vals
        from_date[pair] = matched
    rows, flags = {}, []
    dec_idx = {d: i for i, d in enumerate(fx_dates.dt.date)}
    for pair, vals in carried.items():
        r = numpy.log(vals / vals.shift(5))
        rows[pair] = r
        for i in range(5, n):
            for endpoint, j in (('t', i), ('t-5', i - 5)):
                if from_date[pair][j] != fx_dates[j]:
                    flags.append({'decision_date': fx_dates[i].date(), 'pair': pair, 'endpoint': endpoint,
                                  'endpoint_date': fx_dates[j].date(), 'carried_from': from_date[pair][j].date()})
    basket = sum(BASKET_SIGN[p] * rows[p] for p in BASKET_SIGN) / 5
    f = pandas.DataFrame({'decision_date': inp.spot['date'], 'usd_basket_5d': basket.to_numpy()})
    f = inp.dec[['decision_date']].merge(f, on='decision_date', how='left')
    f['yen_relative_5d'] = ret_5d.to_numpy() - f['usd_basket_5d'].to_numpy()
    flag_df = pandas.DataFrame(flags, columns=['decision_date', 'pair', 'endpoint', 'endpoint_date', 'carried_from'])
    flag_df = flag_df[flag_df['decision_date'].isin(set(inp.dec['decision_date']))].sort_values(['decision_date', 'pair', 'endpoint']).reset_index(drop=True)
    return f, flag_df


def commodity_features(inp: Inputs) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """wti_5d_lagged on the EIA/FRED oil calendar, lagged one observation; gld_5d on US sessions."""
    wti = native(inp.fred, 'fred.dcoilwtico')
    gld = native(inp.etf, 'fmp.gld.close')
    f = pandas.DataFrame({'decision_date': inp.dec['decision_date']})
    f['wti_5d_lagged'], a1 = align_lagged_one_observation(logret(wti, 5), 'wti_5d_lagged', 'fred.dcoilwtico', inp.cal, inp.dec)
    f['gld_5d'], a2 = align_native(logret(gld, 5), 'gld_5d', 'fmp.spy.close', inp.cal, inp.dec, 'fmp.gld.close')
    return f, pandas.concat([a1, a2], ignore_index=True)


def cftc_report_table(inp: Inputs) -> pandas.DataFrame:
    """The CFTC report sequence: report_date, net, net_chg_4reports, pos_z (52 reports)."""
    lng = native(inp.jpy, 'jpy.cftc_lf_long'); sht = native(inp.jpy, 'jpy.cftc_lf_short'); oi = native(inp.jpy, 'jpy.cftc_open_interest')
    assert lng.index.equals(sht.index) and lng.index.equals(oi.index)
    net = (lng - sht) / oi
    return pandas.DataFrame({'net': net, 'net_chg_4reports': net - net.shift(4), 'pos_z': z_trailing(net, CFTC_Z_REPORTS)})


def positioning_features(inp: Inputs, vol_20: pandas.Series) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """CFTC features on the report sequence then aligned by actual release; vol_z and short_yen_stress on decision rows."""
    rep = cftc_report_table(inp)
    f = pandas.DataFrame({'decision_date': inp.dec['decision_date']})
    f['cftc_net'], a1 = align_native(rep['net'], 'cftc_net', 'jpy.cftc_lf_long', inp.cal, inp.dec,
                                     '(jpy.cftc_lf_long - jpy.cftc_lf_short) / jpy.cftc_open_interest')
    f['cftc_net_chg_4reports'], a2 = align_native(rep['net_chg_4reports'], 'cftc_net_chg_4reports', 'jpy.cftc_lf_long', inp.cal, inp.dec,
                                                  'cftc_net, four distinct reports back')
    f['cftc_release_age_days'] = numpy.where(a1['status'] == 'filled', a1['age_avail_h'] / 24, numpy.nan)
    f['pos_z'], a3 = align_native(rep['pos_z'], 'pos_z', 'jpy.cftc_lf_long', inp.cal, inp.dec, f'cftc_net, z over {CFTC_Z_REPORTS} reports')
    f['vol_z'] = z260(vol_20).to_numpy()
    f['short_yen_stress'] = numpy.maximum(0, -f['cftc_net']) * numpy.maximum(0, f['vol_z'])
    a_age = a1.assign(feature='cftc_release_age_days')
    return f, pandas.concat([a1, a2, a_age, a3], ignore_index=True)


def event_in_window(inp: Inputs) -> pandas.Series:
    """1 if a scheduled event falls in (17:00 ET t, 17:00 ET t+1], with t+1 the next decision date."""
    ev_t = numpy.sort(pandas.to_datetime(inp.events['event_time_et'], utc=True).dt.tz_convert(ET).to_numpy())
    lo = (pandas.to_datetime(inp.label['date']).dt.tz_localize(ET) + pandas.Timedelta(hours=17)).to_numpy()
    hi = (pandas.to_datetime(inp.label['next_date']).dt.tz_localize(ET) + pandas.Timedelta(hours=17)).to_numpy()
    n_in = numpy.searchsorted(ev_t, hi, side='right') - numpy.searchsorted(ev_t, lo, side='right')
    return pandas.Series((n_in > 0).astype(int), index=inp.dec.index, name='event_in_window')


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
@dataclass
class FeatureBuild:
    features: pandas.DataFrame                 # decision_date + the 39 (or 35 without the private folder)
    audit: pandas.DataFrame                    # public aligned-feature audit (no values, no licensed rows)
    audit_private: pandas.DataFrame            # the four sector features' audit rows
    basket_flags: pandas.DataFrame
    aux: pandas.DataFrame                      # decision_date + AUX_COLS
    inputs: Inputs = field(repr=False)

    @property
    def has_private(self) -> bool:
        return all(c in self.features.columns for c in LICENSED)

    @property
    def columns(self) -> list[str]:
        return [c for c in FEATURES if c in self.features.columns]


def build_features(private: bool = True, raw: Path = RAW, obs_dir: Path = OBSERVATIONS, private_root: Path = PRIVATE_DIR,
                   inputs: Inputs | None = None) -> FeatureBuild:
    """Build the feature table and its audits from the caches. Nothing is written.

    ``inputs`` (default None = load_inputs over the development window) lets the
    reserved-year driver supply an Inputs object assembled over a longer window;
    every computation below is then identical.
    """
    inp = inputs if inputs is not None else load_inputs(private, raw, obs_dir, private_root)
    own, a_own = own_market_features(inp)
    rates, a_rates, aux = rates_features(inp)
    risk, a_risk = risk_features(inp)
    basket, flags = basket_features(inp, own['ret_5d'])
    comm, a_comm = commodity_features(inp)
    posn, a_pos = positioning_features(inp, own['vol_20'])
    parts = [own, rates, risk, basket, comm, posn]
    audits = [a_own, a_rates, a_risk, a_comm, a_pos]
    audit_private = pandas.DataFrame(columns=AUDIT_COLS)
    if inp.jq is not None:
        sect, audit_private = sector_features(inp)
        parts.append(sect)
    F = parts[0]
    for p in parts[1:]:
        F = F.merge(p, on='decision_date', how='left')
    assert len(F) == len(inp.dec)
    F['event_in_window'] = event_in_window(inp).to_numpy()
    F['gapchg_x_vixchg'] = (z260(F['gap2y_chg_5d']) * z260(F['vix_chg_5d'])).to_numpy()
    cols = [c for c in FEATURES if c in F.columns]
    F = F[['decision_date'] + cols].copy()
    F['decision_date'] = F['decision_date'].astype(str)

    aux['z260_us2y_chg_5d'] = z260(aux['us2y_chg_5d'])
    aux['us_only_gapchg_x_vixchg'] = aux['z260_us2y_chg_5d'] * z260(F['vix_chg_5d'])
    aux = aux[['decision_date'] + AUX_COLS].copy()
    aux['decision_date'] = aux['decision_date'].astype(str)

    audit = pandas.concat(audits, ignore_index=True)
    assert not audit['feature'].isin(LICENSED).any() and 'value' not in audit.columns
    for a in (audit, audit_private):
        if len(a):
            a['decision_date'] = a['decision_date'].astype(str)
            a['matched_observation_date'] = a['matched_observation_date'].map(lambda d: '' if pandas.isna(d) else str(d))
    return FeatureBuild(F, audit, audit_private, flags, aux, inp)


# ----------------------------------------------------------------------------
# Verification helpers (used by the notebook and tests/test_features.py)
# ----------------------------------------------------------------------------
def usable_start(F: pandas.DataFrame) -> str:
    """First decision date with every feature column present."""
    cols = [c for c in FEATURES if c in F.columns]
    ok = F[cols].notna().all(axis=1)
    assert ok.any()
    return str(F.loc[ok.idxmax(), 'decision_date'])


def missing_after_warmup(F: pandas.DataFrame) -> pandas.DataFrame:
    """Per column: warm-up rows, first value date, and missing values after the first value (should be zero)."""
    rows = []
    for c in [c for c in FEATURES if c in F.columns]:
        s = F[c]
        first = s.first_valid_index()
        after = int(s.loc[first:].isna().sum()) if first is not None else len(s)
        dates_after = list(F.loc[first:, 'decision_date'][s.loc[first:].isna()]) if first is not None else []
        rows.append({'feature': c, 'family': FAMILY[c], 'warm_up_rows': int(s.isna().sum()) - after,
                     'first_value': F.loc[first, 'decision_date'] if first is not None else '',
                     'missing_after_warm_up': after, 'missing_dates': ', '.join(dates_after[:5])})
    return pandas.DataFrame(rows)


def age_summary(audit: pandas.DataFrame) -> pandas.DataFrame:
    """Per feature and leg: age distribution at decision and rows whose match is older than one session.

    ``stale_rows`` counts decision rows whose matched availability date is
    earlier than the decision date (a holiday on the source's calendar). For
    the weekly CFTC features the count is the number of rows not on a release
    day and is shown for completeness only.
    """
    a = audit[audit['status'] == 'filled'].copy()
    a['avail_date'] = a['matched_available_time'].str[:10]
    a['stale'] = a['avail_date'] < a['decision_date']
    g = a.groupby(['feature', 'source_series'], sort=False)
    out = pandas.DataFrame({'rows_filled': g.size(), 'age_obs_h_min': g['age_obs_h'].min(), 'age_obs_h_median': g['age_obs_h'].median(),
                        'age_obs_h_p95': g['age_obs_h'].quantile(0.95), 'age_obs_h_max': g['age_obs_h'].max(),
                        'stale_rows': g['stale'].sum()}).reset_index()
    out.insert(1, 'family', out['feature'].map(FAMILY))
    return out.round(2)


def stale_examples(audit: pandas.DataFrame, feature: str, k: int = 4, source: str | None = None) -> pandas.DataFrame:
    """The first k decision rows of a feature whose matched availability date is before the decision date."""
    a = audit[(audit['feature'] == feature) & (audit['status'] == 'filled')]
    if source is not None:
        a = a[a['source_series'] == source]
    a = a[a['matched_available_time'].str[:10] < a['decision_date']]
    return a[['decision_date', 'source_series', 'matched_observation_date', 'matched_available_time', 'age_obs_h']].head(k)


def spearman_pairs(F: pandas.DataFrame, threshold: float = 0.9, top: int = 3) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """(pairs above the threshold, the top-k pairs) by absolute Spearman correlation on pairwise-complete rows."""
    cols = [c for c in FEATURES if c in F.columns]
    rho = F[cols].corr(method='spearman')
    pairs = pandas.DataFrame([(a, b, rho.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]],
                         columns=['feature_a', 'feature_b', 'spearman'])
    pairs = pairs.reindex(pairs['spearman'].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return pairs[pairs['spearman'].abs() > threshold], pairs.head(top)


def reference_row(F: pandas.DataFrame, d: str = REFERENCE_DATE) -> pandas.Series:
    r = F[F['decision_date'] == d]
    assert len(r) == 1
    return r.iloc[0].drop('decision_date')


def previous_reference_row(d: str = REFERENCE_DATE, private: bool = True) -> pandas.Series:
    """The row-calendar values of the backed-up previous build, renamed to the protocol names."""
    paths = [PREV_PUBLIC_DIR / 'usdjpy_features_public.csv']
    if private and (PREV_PRIVATE_DIR / 'usdjpy_features_full.csv').exists():
        paths = [PREV_PRIVATE_DIR / 'usdjpy_features_full.csv']
    prev = pandas.read_csv(paths[0])
    r = prev[prev['decision_date'] == d].iloc[0].drop('decision_date')
    r = r.rename({old: new for new, old in RENAMES.items()})
    return r


def compare_reference(new: pandas.Series, prev: pandas.Series, decimals: int = 6) -> pandas.DataFrame:
    rows = []
    for c in FEATURES:
        if c not in new.index:
            continue
        old_name = RENAMES.get(c, c)
        p = prev.get(c, numpy.nan)
        rows.append({'feature': c, 'previous_name': old_name if old_name != c else '', 'previous': p, 'new': new[c],
                     'difference': new[c] - p if pandas.notna(p) else numpy.nan})
    return pandas.DataFrame(rows)


# ----------------------------------------------------------------------------
# Event-timing exceptions (protocol section 7)
# ----------------------------------------------------------------------------
def read_archived_empsit_schedule(path: Path = RAW / 'bls_schedule_empsit_archive.html') -> dict[str, date]:
    """Reference month -> originally scheduled release date, from the archived BLS Employment Situation schedule page.

    The cached response is gzip-encoded; it is read with regular expressions
    only (untrusted data). Returns {} when the cache is absent.
    """
    if not path.exists():
        return {}
    b = path.read_bytes()
    try:
        s = gzip.decompress(b).decode('utf-8', errors='replace')
    except (OSError, EOFError):
        s = b.decode('utf-8', errors='replace')
    t = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', s, flags=re.S)
    t = html_lib.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', t)))
    out = {}
    for ref, mon, day, year in re.findall(r'([A-Z][a-z]+ 20\d\d) ([A-Z][a-z]{2})\.? (\d{1,2}), (20\d\d) 08:30 AM', t):
        out[ref] = date(int(year), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(mon) + 1, int(day))
    return out


def event_timing_exceptions(build: FeatureBuild) -> tuple[pandas.DataFrame, dict]:
    """Decision dates whose event_in_window value depends on a rescheduled or cancelled release.

    Verified part: the archived Employment Situation schedule (snapshot
    2025-10-06, data/raw/bls_schedule_empsit_archive.html) lists the original
    date of the September 2025 report; the actual calendar has it on
    2025-11-20, after the sample. Conservative part: the CPI schedule page was
    not retrieved (the four-GET budget was spent on redirects), so every
    decision date from 2025-10-01 to 2025-10-21 whose window covers an October
    day inside the historical CPI release-day range of the calendar itself is
    flagged. Fed and BOJ dates in October 2025 come from primary pages and fall
    after the sample (2025-10-29, 2025-10-30).
    """
    inp = build.inputs
    F = build.features
    label = inp.label
    windows = pandas.DataFrame({'decision_date': label['date'], 'next_date': label['next_date']})
    windows['lo'] = pandas.to_datetime(windows['decision_date']).dt.tz_localize(ET) + pandas.Timedelta(hours=17)
    windows['hi'] = pandas.to_datetime(windows['next_date']).dt.tz_localize(ET) + pandas.Timedelta(hours=17)
    ev = inp.events
    actual = {t: set(pandas.to_datetime(ev.loc[ev['event_type'] == t, 'event_date_local']).dt.date) for t in ev['event_type'].unique()}
    sched = read_archived_empsit_schedule()
    rows = []

    def flag(d_event: date, event_type: str, reason: str, announced: str, source: str):
        t = pandas.Timestamp(d_event).tz_localize(ET) + pandas.Timedelta(hours=8, minutes=30)
        hit = windows[(windows['lo'] < t) & (t <= windows['hi'])]
        for _, w in hit.iterrows():
            dd = str(w['decision_date'])
            rows.append({'decision_date': dd, 'event_type': event_type, 'original_date': str(d_event),
                         'event_in_window_actual_calendar': int(F.loc[F['decision_date'] == dd, 'event_in_window'].iloc[0]),
                         'reason': reason, 'announcement_date': announced, 'source': source})

    info = {'empsit_schedule_rows_parsed': len(sched), 'empsit_original_dates_missed': [], 'empsit_schedule_vs_actual_2025_mismatches': []}
    if sched:
        for ref, d0 in sched.items():
            if d0.year != 2025 or d0 > END:
                continue
            if d0 not in actual['US_EMPSIT']:
                info['empsit_original_dates_missed'].append(str(d0))
                flag(d0, 'US_EMPSIT', f'{ref} Employment Situation originally scheduled {d0}, not released on that date '
                     '(federal shutdown from 2025-10-01); actual release after the sample', '',
                     'web.archive.org snapshot 20251006223843 of bls.gov/schedule/news_release/empsit.htm (no notice on the page)')
        in_win = {d for d in actual['US_EMPSIT'] if date(2025, 1, 1) <= d <= END}
        info['empsit_schedule_vs_actual_2025_mismatches'] = sorted(str(d) for d in in_win ^ {d for d in sched.values() if date(2025, 1, 1) <= d <= END})
    # CPI: conservative rule (original page not retrieved). Use the calendar's own CPI release-day range.
    cpi_days = sorted(pandas.to_datetime(ev.loc[ev['event_type'] == 'US_CPI', 'event_date_local']).dt.day)
    lo_day, hi_day = cpi_days[0], cpi_days[-1]
    info['cpi_rule'] = f'conservative: October 2025 days {lo_day}-{hi_day} (historical CPI release-day range in the calendar)'
    for day in range(lo_day, hi_day + 1):
        d0 = date(2025, 10, day)
        if d0.weekday() >= 5 or d0 > END or d0 in actual['US_CPI']:
            continue
        flag(d0, 'US_CPI', 'September 2025 CPI not released on its original date (federal shutdown); original schedule page not '
             f'retrieved, so every weekday {lo_day}-{hi_day} October is treated as a possible original date', '',
             'conservative rule (protocol v2.1 section 7); actual release 2025-10-24 is after the sample')
    out = pandas.DataFrame(rows, columns=['decision_date', 'event_type', 'original_date', 'event_in_window_actual_calendar',
                                      'reason', 'announcement_date', 'source'])
    out = out[(out['decision_date'] >= '2025-10-01') | (out['event_type'] == 'US_EMPSIT')]
    return out.sort_values(['decision_date', 'event_type']).reset_index(drop=True), info


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def _write_private(df: pandas.DataFrame, path: Path) -> Path:
    return timing.write_private_csv(df, path)


def write_feature_tables(build: FeatureBuild, aligned: Path, private_root: Path) -> dict[str, str]:
    """Write the public and private feature tables, audits, basket flags and the ablation aux file.

    Both folders are required arguments: callers pass config.output_dir() and
    config.private_scratch_dir(), so the reference files cannot be overwritten.
    """
    F = build.features
    out = {}
    pub = F[['decision_date'] + PUBLIC]
    assert not set(LICENSED) & set(pub.columns)
    pub.to_csv(aligned / 'usdjpy_features_public.csv', index=False); out['public'] = str(aligned / 'usdjpy_features_public.csv')
    build.audit.to_csv(aligned / 'usdjpy_features_audit.csv', index=False); out['audit'] = str(aligned / 'usdjpy_features_audit.csv')
    build.basket_flags.to_csv(aligned / 'usdjpy_basket_flags.csv', index=False); out['basket_flags'] = str(aligned / 'usdjpy_basket_flags.csv')
    build.aux.to_csv(aligned / 'usdjpy_features_ablation_aux.csv', index=False); out['aux'] = str(aligned / 'usdjpy_features_ablation_aux.csv')
    if build.has_private:
        assert list(F.columns) == ['decision_date'] + FEATURES
        out['private'] = str(_write_private(F, private_root / 'usdjpy_features_full.csv'))
        out['private_audit'] = str(_write_private(build.audit_private, private_root / 'usdjpy_features_audit_jquants.csv'))
    return out


def write_event_exceptions(build: FeatureBuild, obs_dir: Path) -> tuple[Path, pandas.DataFrame, dict]:
    """Write event_timing_exceptions.csv into an explicit output folder."""
    table, info = event_timing_exceptions(build)
    path = obs_dir / 'event_timing_exceptions.csv'
    table.to_csv(path, index=False)
    return path, table, info


# ----------------------------------------------------------------------------
# Reserved-year inputs (relocated from data_review/exam3/reserved.py)
# ----------------------------------------------------------------------------
def concat_long_windows(development_path: Path, reserved_path: Path, reserved_start=RESERVED_START,
                        reserved_end=RESERVED_END) -> pandas.DataFrame:
    """Development observations (development dates) followed by reserved observations (reserved dates)."""
    development = timing.load_long_observations(development_path)
    reserved = timing.load_long_observations(reserved_path, reserved_start, reserved_end)
    both = pandas.concat([development, reserved], ignore_index=True)
    assert not both.duplicated(['series_id', 'observation_date']).any(), 'an observation appears in both windows'
    assert development['observation_date'].max() <= END < reserved['observation_date'].min(), 'windows overlap'
    return both


def reserved_inputs(cftc_extension: tuple[dict, dict], private: bool = True, reserved_obs: Path = RESERVED_ALIGNED,
                    reserved_raw: Path = RESERVED_RAW, private_root: Path = PRIVATE_DIR,
                    events: pandas.DataFrame | None = None) -> Inputs:
    """Inputs over warm-up + development + reserved window (2020-10-22 to 2026-10-21); nothing is fetched.

    Development observation files supply development dates and the reserved
    files supply reserved dates, so trailing windows of reserved rows reach
    back into development history, as in real time. cftc_extension is
    timing.reserved_cftc_extension(...): the reserved-year CFTC schedule and
    catch-up dates, passed explicitly so no development calendar changes.
    """
    extra_schedule, extra_exceptions = cftc_extension
    spot = timing.usdjpy_closes(RAW, START, RESERVED_END, extra_raw=[reserved_raw], expected_sundays=None)
    label = target.build_label(spot, end=RESERVED_END)
    dec = timing.decision_frame(label)
    jpy = concat_long_windows(OBSERVATIONS / 'jpy_sources.csv', reserved_obs / 'jpy_sources_reserved.csv')
    etf = concat_long_windows(OBSERVATIONS / 'shared_fmp_etf.csv', reserved_obs / 'shared_fmp_etf_reserved.csv')
    fred = concat_long_windows(OBSERVATIONS / 'shared_fred.csv', reserved_obs / 'shared_fred_reserved.csv')
    cal = timing.calendars_from_long(jpy, extra_schedule, extra_exceptions)
    jq = None
    if private:
        development_jq, reserved_jq = require_private_inputs(
            private_root / 'jquants_observations.csv', private_root / 'reserved' / 'jquants_observations_reserved.csv')
        jq = concat_long_windows(development_jq, reserved_jq)
        assert jq['series_id'].str.startswith('jpy.topix').all(), 'unexpected series in the J-Quants files'
        jq = jq.assign(licensed=True)
    if events is None:
        events = pandas.read_csv(reserved_obs / 'scheduled_events_2020_2026.csv')
    basket = timing.basket_closes(RAW, START, RESERVED_END, extra_raw=[reserved_raw])
    return Inputs(spot, label, dec, cal, etf, fred, jpy, jq, basket, events)
