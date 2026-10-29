"""Cache access and acquisition for every USD/JPY source.

Default is offline. Every fetch function is cache-only unless its switch is
set: acquire (FMP, FRED), allow_download (MOF, BOJ, CFTC, Nikkei) or refresh
(J-Quants, also CQF_JQUANTS_REFRESH=1). With the switch off, a missing cache
raises instead of contacting a provider. Importing this module makes no
request and writes nothing.

Credentials are read from the macOS Keychain with /usr/bin/security inside the
fetch functions, held in memory only and never written, logged or printed.
Request budgets and the FMP ledger are those of the collection notebooks.

Parsing produces the long observation schema (config.OBSERVATION_COLUMNS).
The availability_time values written here are the collection notebooks'
provisional placeholders; timing.retime() replaces them before any as-of join.

The network functions are relocated verbatim from data_review/exam3/sources.py
and were not re-run (no acquisition in this refactor). The readers and parsers
are checked byte for byte against the saved observation files
(tests/test_data_equivalence.py).

Window arguments start / end default to the development window; the
reserved-year build passes its own bounds and raw folder.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import http.client
import io
import json
import math
import os
import ssl
import subprocess
import tempfile
import time
import urllib.parse
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

from .config import (END, FMP_LEDGER, FMP_LOCK, JST, LEGACY_CACHE, OBSERVATION_COLUMNS, PRIVATE_DIR,
                     PRIVATE_SCRATCH, RAW, START, UTC, in_development, is_inside)

try:
    import certifi
except ImportError:            # the notebook also tolerates a missing certifi
    certifi = None

Record = dict


# ----------------------------------------------------------------------------
# Small helpers shared by notebooks 01 and 02
# ----------------------------------------------------------------------------
def sha(b: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(b).hexdigest()


def atomic_json(path: Path, data) -> None:
    """Write JSON through a temporary file so a crash never leaves a half file."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def in_window(d: date, start: date = START, end: date = END) -> bool:
    """True when d lies inside [start, end] (defaults: the development window, as in_development)."""
    return start <= d <= end


def bounded_json(path: Path, start: date = START, end: date = END) -> list:
    """Load a list-of-rows JSON and keep only dates inside [start, end] (default: development dates)."""
    rows = json.loads(path.read_text())
    assert isinstance(rows, list)
    return [r for r in rows if start <= date.fromisoformat(r['date']) <= end]


def save_observations(path: Path, records: list[Record]) -> None:
    """Write long observation records as notebook 01 did (csv module, CRLF rows)."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OBSERVATION_COLUMNS)
        w.writeheader()
        w.writerows(records)
    tmp.replace(path)


def numeric(x) -> Optional[float]:
    """Float value of x, or None when x is blank, non-numeric, NaN or infinite."""
    try:
        y = float(str(x).strip())
        return y if y == y and abs(y) != float('inf') else None
    except (ValueError, TypeError):
        return None


def keychain_key(service: str, account: str, timeout: int = 20) -> str:
    """Read one Keychain item. The value is returned to the caller and never stored."""
    p = subprocess.run(['/usr/bin/security', 'find-generic-password', '-s', service, '-a', account, '-w'],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode or not p.stdout.strip():
        raise RuntimeError('Keychain credential unavailable; no request sent')
    return p.stdout.strip()


# ----------------------------------------------------------------------------
# Notebook 01: FMP (ETFs and USDJPY) via Keychain, ledger and caps
# ----------------------------------------------------------------------------
FMP_HOST = 'financialmodelingprep.com'
FMP_ENDPOINT = {'adjusted': '/stable/historical-price-eod/dividend-adjusted',
                'full': '/stable/historical-price-eod/full'}
FMP_SYMBOLS = {'SPY': 'adjusted', 'QQQ': 'adjusted', 'GLD': 'adjusted', 'TLT': 'adjusted',
               'ARKK': 'adjusted', 'SMH': 'adjusted', 'USDJPY': 'full'}
FMP_MAX_REQUESTS = 50
FMP_MAX_BYTES = 100_000_000
FMP_MIN_INTERVAL_S = 1.2


def _tls_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()


def fmp_keychain_key() -> str:
    """The FMP API key from the Keychain item the notebooks use."""
    return keychain_key('fmp-market-data', 'fmp-api-key')


def fmp_raw_path(symbol: str, kind: str, raw: Path = RAW) -> Path:
    return raw / f'shared_fmp_{symbol.lower()}_{kind}.json'


def fmp_get(symbol: str, kind: str, credential: Optional[str], acquire: bool = False,
            raw: Path = RAW, start: date = START, end: date = END, purpose: Optional[str] = None,
            request_cap: int = FMP_MAX_REQUESTS) -> Path:
    """Return the cached rows for one symbol and window; fetch only when acquire is True.

    The fetch holds the shared ledger under flock, enforces the cumulative
    50-request / 100 MB ceiling (or the smaller ``request_cap`` the caller
    sets for its own run) and the 1.2 s spacing, filters the payload to the
    window [start, end] before writing the cache, and never records the URL,
    the response body on error or the exception detail. ``purpose`` is written
    to the ledger event as a plain label.
    """
    dest = fmp_raw_path(symbol, kind, raw)
    if dest.exists():
        return dest
    if not acquire:
        raise RuntimeError('Missing raw cache; set acquire=True to fetch explicitly')
    with FMP_LOCK.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        led = json.loads(FMP_LEDGER.read_text())
        if led['requests'] >= min(FMP_MAX_REQUESTS, request_cap) or led['bytes'] >= FMP_MAX_BYTES:
            raise RuntimeError('FMP cumulative budget exhausted')
        delay = FMP_MIN_INTERVAL_S - (time.time() - led['last_started_epoch'])
        if delay > 0:
            time.sleep(delay)
        params = {'symbol': symbol, 'from': start.isoformat(), 'to': end.isoformat()}
        ev = {'endpoint': FMP_ENDPOINT[kind], 'parameters': params,
              'started_utc': datetime.now(UTC).isoformat(), 'status': 'started', 'bytes': 0}
        if purpose:
            ev['purpose'] = purpose
        led['requests'] += 1
        led['last_started_epoch'] = time.time()
        led['events'].append(ev)
        atomic_json(FMP_LEDGER, led)
        conn = http.client.HTTPSConnection(FMP_HOST, timeout=45, context=_tls_context())
        try:
            request_path = FMP_ENDPOINT[kind] + '?' + urllib.parse.urlencode({**params, 'apikey': credential})
            conn.request('GET', request_path, headers={'Accept': 'application/json', 'Accept-Encoding': 'identity'})
            res = conn.getresponse()
            ev['http_status'] = res.status
            if res.status != 200:
                ev['status'] = 'http_failure_stopped'
                raise RuntimeError('FMP HTTP rejected; safe status in ledger')
            chunks = []
            while True:
                remaining = FMP_MAX_BYTES - led['bytes']
                if remaining <= 0:
                    ev['status'] = 'byte_cap_stopped'
                    raise RuntimeError('FMP byte budget exhausted')
                chunk = res.read(min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                led['bytes'] += len(chunk)
                ev['bytes'] += len(chunk)
                atomic_json(FMP_LEDGER, led)
            payload = json.loads(b''.join(chunks))
            if not isinstance(payload, list) or not payload:
                ev['status'] = 'schema_failure_stopped'
                raise RuntimeError('FMP unexpected response')
            fields = ({'date', 'open', 'high', 'low', 'close', 'volume'} if kind == 'full'
                      else {'date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume'})
            rows = []
            for r in payload:
                if not isinstance(r, dict) or not fields.issubset(r) or r.get('symbol', symbol) != symbol:
                    raise RuntimeError('FMP row schema invalid')
                if start <= date.fromisoformat(r['date']) <= end:
                    rows.append(r)
            if not rows:
                ev['status'] = 'empty_development_stopped' if (start, end) == (START, END) else 'empty_window_stopped'
                raise RuntimeError('FMP empty response for the requested window')
            # Persist only rows inside the window, though the request itself is date bounded.
            blob = (json.dumps(rows, separators=(',', ':')) + '\n').encode()
            with dest.open('xb') as f:
                f.write(blob)
            ev.update(status='cached_development_only' if (start, end) == (START, END) else 'cached_window_only',
                      rows=len(rows), cache_sha256=sha(blob))
            return dest
        except RuntimeError:
            if ev['status'] == 'started':
                ev['status'] = 'validation_failure_stopped'
            raise
        except Exception:
            ev['status'] = 'transport_or_parse_failure_stopped'
            raise RuntimeError('FMP transport or parse failure; details suppressed') from None
        finally:
            conn.close()
            atomic_json(FMP_LEDGER, led)


def existing_filtered(symbol: str, kind: str, raw: Path = RAW) -> Optional[Path]:
    """Development-only copy of a legacy full-range cache, made before any inspection."""
    dest = fmp_raw_path(symbol, kind, raw)
    if dest.exists():
        return dest
    src = LEGACY_CACHE / f'{symbol}_{kind}_2020-10-22_2026-10-21.json'
    if not src.exists():
        return None
    rows = bounded_json(src)   # filter immediately, before summary/inspection
    assert rows and all(START <= date.fromisoformat(r['date']) <= END for r in rows)
    blob = (json.dumps(rows, separators=(',', ':')) + '\n').encode()
    with dest.open('xb') as f:
        f.write(blob)
    return dest


def fmp_raw_paths(acquire: bool = False, raw: Path = RAW) -> dict[str, Path]:
    """Raw development caches per symbol, in FMP_SYMBOLS order (notebook 01 loop)."""
    paths: dict[str, Path] = {}
    credential = None
    blocked = False
    for sym, kind in FMP_SYMBOLS.items():
        p = existing_filtered(sym, kind, raw)
        if p is None and not blocked:
            try:
                if acquire and credential is None:
                    credential = fmp_keychain_key()
                p = fmp_get(sym, kind, credential, acquire, raw)
            except (RuntimeError, subprocess.TimeoutExpired):
                blocked = True
                p = None
        if p:
            paths[sym] = p
    credential = None   # drop the key as soon as the loop ends
    return paths


# ----------------------------------------------------------------------------
# Notebook 01: FRED graph CSVs
# ----------------------------------------------------------------------------
FRED_IDS = ['DGS2', 'DGS10', 'DFII5', 'DFII10', 'VIXCLS', 'DCOILWTICO', 'DTWEXBGS', 'SOFR',
            'DEXUSEU', 'DEXUSUK', 'DEXCAUS', 'DEXSZUS', 'DEXUSAL', 'DEXJPUS']
FRED_HOST = 'fred.stlouisfed.org'
FRED_PATH = '/graph/fredgraph.csv'
FRED_UNITS = {'DGS2': 'percent per annum', 'DGS10': 'percent per annum', 'DFII5': 'percent per annum',
              'DFII10': 'percent per annum', 'VIXCLS': 'index points', 'DCOILWTICO': 'USD per barrel',
              'DTWEXBGS': 'index points', 'SOFR': 'percent per annum', 'DEXUSEU': 'USD per EUR',
              'DEXUSUK': 'USD per GBP', 'DEXCAUS': 'CAD per USD', 'DEXSZUS': 'CHF per USD',
              'DEXUSAL': 'USD per AUD', 'DEXJPUS': 'JPY per USD'}


class PublicBudget:
    """Shared-owner ceiling for public sources: 60 requests and 20 MB per run."""

    def __init__(self, max_calls: int = 60, max_bytes: int = 20_000_000):
        self.max_calls, self.max_bytes = max_calls, max_bytes
        self.attempts: list[dict] = []
        self.bytes = 0

    def exhausted(self) -> bool:
        return len(self.attempts) >= self.max_calls or self.bytes >= self.max_bytes


def fred_raw_path(sid: str, raw: Path = RAW) -> Path:
    return raw / f'shared_fred_{sid.lower()}.csv'


def fred_fetch(sid: str, acquire: bool = False, budget: Optional[PublicBudget] = None,
               raw: Path = RAW, start: date = START, end: date = END) -> Optional[Path]:
    """Cached FRED CSV for one series; curl only when acquire is True.

    The response is capped at 100 kB, filtered to the window [start, end]
    (default: development dates) and written once. Failures record a status
    word only, never the error text.
    """
    dest = fred_raw_path(sid, raw)
    if dest.exists():
        return dest
    if not acquire:
        raise RuntimeError('Missing FRED cache; acquisition disabled')
    budget = budget or PublicBudget()
    if budget.exhausted():
        raise RuntimeError('Public budget reached')
    params = {'id': sid, 'cosd': start.isoformat(), 'coed': end.isoformat()}
    url = 'https://' + FRED_HOST + FRED_PATH + '?' + urllib.parse.urlencode(params)
    event = {'series': sid, 'url': url, 'fetched_at': datetime.now(UTC).isoformat()}
    budget.attempts.append(event)
    try:
        proc = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--max-redirs', '0',
                               '--max-time', '15', '--max-filesize', '100000', url],
                              capture_output=True, timeout=20)
        if proc.returncode:
            event['status'] = 'http_or_transport_failure'
            return None
        body = proc.stdout
        event['http_status'] = 200
        event['body_bytes'] = len(body)
        budget.bytes += len(body)
        if len(body) > 100_000:
            event['status'] = 'byte_cap_failure'
            return None
        rd = csv.DictReader(io.StringIO(body.decode('utf-8-sig')))
        records = list(rd)
        if not records or sid not in rd.fieldnames or not ({'DATE', 'observation_date'} & set(rd.fieldnames)):
            event['status'] = 'schema_failure'
            return None
        date_field = 'DATE' if 'DATE' in rd.fieldnames else 'observation_date'
        kept = [r for r in records if start <= date.fromisoformat(r[date_field]) <= end]
        if not kept:
            event['status'] = 'empty_development' if (start, end) == (START, END) else 'empty_window'
            return None
        output = io.StringIO()
        w = csv.DictWriter(output, fieldnames=rd.fieldnames, lineterminator='\n')
        w.writeheader()
        w.writerows(kept)
        filtered = output.getvalue().encode()
        event.update(status='cached_development_only' if (start, end) == (START, END) else 'cached_window_only',
                     rows=len(kept), raw_sha256=sha(filtered))
        with dest.open('xb') as f:
            f.write(filtered)
        return dest
    except Exception:
        event['status'] = 'transport_or_parse_failure'
        return None


def fred_raw_paths(acquire: bool = False, raw: Path = RAW) -> dict[str, Path]:
    """Raw FRED caches per series id, in FRED_IDS order."""
    budget = PublicBudget()
    paths = {}
    for sid in FRED_IDS:
        p = fred_fetch(sid, acquire, budget, raw)
        if p:
            paths[sid] = p
    return paths


# ----------------------------------------------------------------------------
# Notebook 01: parse FMP and FRED caches into long observations
# ----------------------------------------------------------------------------
def iso_midnight_utc(d: date) -> str:
    """'YYYY-MM-DDT00:00:00+00:00' for date d (notebook 01 spelling)."""
    return datetime.combine(d, datetime.min.time(), UTC).isoformat()


def one(sid: str, d: date, obs_time: str, avail: str, value, unit: str, source: str, basis: str) -> Record:
    """One long-schema record."""
    return dict(zip(OBSERVATION_COLUMNS, [sid, d.isoformat(), obs_time, avail, value, unit, source, basis]))


def fmp_observations(fmp_paths: dict[str, Path], start: date = START, end: date = END) -> tuple[list[Record], list[Record]]:
    """(ETF records, USDJPY records) exactly as notebook 01 wrote them.

    ETF availability is the old placeholder rule 'next UTC day 12:00'. USDJPY
    rows carry a blank available_time: these are the pre-cleaning observations
    and are deliberately excluded from any as-of join until notebook 06 cleans
    them and anchors the close at 17:00 ET.
    """
    fmp_etf: list[Record] = []
    fmp_fx: list[Record] = []
    for sym, p in fmp_paths.items():
        for r in bounded_json(p, start, end):
            d = date.fromisoformat(r['date'])
            if sym == 'USDJPY':
                avail = ''
                basis = ('FMP FX EOD date only; boundary/quote side/publication unresolved; '
                         'latest vendor history, not point-in-time')
                target = fmp_fx
            else:
                avail = iso_midnight_utc(d + timedelta(days=1)).replace('T00:00:00', 'T12:00:00')
                basis = 'ASSUMED next UTC day 12:00; FMP adjusted ETF EOD latest-vintage/revision risk'
                target = fmp_etf
            for field in ['open', 'high', 'low', 'close', 'volume']:
                adj = 'adj' + field.title()
                key = adj if (sym != 'USDJPY' and field != 'volume' and adj in r) else field
                val = r[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    continue
                if field != 'volume':
                    unit = 'JPY per USD' if sym == 'USDJPY' else 'USD per share'
                else:
                    unit = 'vendor FX volume unit unverified' if sym == 'USDJPY' else 'shares reported by vendor'
                source = 'fmp_' + ('full' if sym == 'USDJPY' else 'adjusted')
                target.append(one(f'fmp.{sym.lower()}.{field}', d, '', avail, val, unit, source, basis))
    return fmp_etf, fmp_fx


def fred_observations(fred_paths: dict[str, Path], start: Optional[date] = None, end: Optional[date] = None) -> list[Record]:
    """FRED records with notebook 01's blanket '+10 calendar days 00:00 UTC' placeholder.

    No date filter by default (the development caches are already bounded); a
    [start, end] window may be given for a cache that holds other dates.
    """
    out: list[Record] = []
    basis = ('ASSUMED observation+10 calendar days 00:00 UTC; FRED latest vintage; replace with actual '
             'publication/revision history. H10 noon observation is not noon publication.')
    for sid in FRED_IDS:
        p = fred_paths.get(sid)
        if not p:
            continue
        with p.open(newline='') as f:
            rd = csv.DictReader(f)
            records = list(rd)
            date_field = 'DATE' if 'DATE' in rd.fieldnames else 'observation_date'
        for r in records:
            d = date.fromisoformat(r[date_field])
            if start is not None and end is not None and not (start <= d <= end):
                continue
            try:
                v = float(r[sid])
            except (ValueError, TypeError):
                continue
            if not math.isfinite(v):
                continue
            out.append(one('fred.' + sid.lower(), d, '', iso_midnight_utc(d + timedelta(days=10)), v,
                           FRED_UNITS[sid], 'fred_' + sid.lower(), basis))
    return out


# ----------------------------------------------------------------------------
# Notebook 02: MOF, BOJ API, CFTC Socrata, Nikkei (FRED) via curl
# ----------------------------------------------------------------------------
_CFTC_QUERY = {
    '$select': ('report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,open_interest_all,'
                'lev_money_positions_long,lev_money_positions_short,lev_money_positions_spread'),
    '$where': ("cftc_contract_market_code='097741' AND report_date_as_yyyy_mm_dd >= '2020-10-22T00:00:00.000' "
               "AND report_date_as_yyyy_mm_dd <= '2025-10-21T00:00:00.000'"),
    '$order': 'report_date_as_yyyy_mm_dd ASC', '$limit': '1000'}
def cftc_query(start: date = START, end: date = END) -> dict:
    """The Socrata query of notebook 02 for a window (default: the development window, byte-identical)."""
    q = dict(_CFTC_QUERY)
    q['$where'] = (f"cftc_contract_market_code='097741' AND report_date_as_yyyy_mm_dd >= '{start.isoformat()}T00:00:00.000' "
                   f"AND report_date_as_yyyy_mm_dd <= '{end.isoformat()}T00:00:00.000'")
    return q


def jpy_source_url(key: str, start: date = START, end: date = END) -> str:
    """Request URL of one notebook 02 public source for a window (defaults reproduce JPY_SOURCE_URLS exactly)."""
    if key == 'mof':
        return 'https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv'
    if key == 'tona':      # BOJ API month bounds (YYYYMM): a daily YYYYMMDD form returned HTTP 400.
        return ('https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=en&db=FM01'
                f'&code=STRDCLUCON&startDate={start:%Y%m}&endDate={end:%Y%m}')
    if key == 'boj_fx':
        return ('https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=en&db=FM08'
                f'&code=FXERD04&startDate={start:%Y%m}&endDate={end:%Y%m}')
    if key == 'nikkei':
        return f'https://fred.stlouisfed.org/graph/fredgraph.csv?id=NIKKEI225&cosd={start.isoformat()}&coed={end.isoformat()}'
    if key == 'cftc':
        return 'https://publicreporting.cftc.gov/resource/gpe5-46if.csv?' + urllib.parse.urlencode(cftc_query(start, end))
    raise KeyError(key)


JPY_SOURCE_URLS = {key: jpy_source_url(key) for key in ('mof', 'tona', 'boj_fx', 'nikkei', 'cftc')}
assert JPY_SOURCE_URLS['cftc'] == 'https://publicreporting.cftc.gov/resource/gpe5-46if.csv?' + urllib.parse.urlencode(_CFTC_QUERY)
JPY_FILES = {'mof': 'jpy_mof_jgb.csv', 'tona': 'jpy_boj_tona.csv', 'boj_fx': 'jpy_boj_usdjpy_17.csv',
             'cftc': 'jpy_cftc_tff_yen.csv', 'nikkei': 'jpy_fred_nikkei225.csv'}
JPY_RESPONSE_CAPS = [('mof', 3_000_000), ('tona', 2_000_000), ('boj_fx', 2_000_000),
                     ('cftc', 500_000), ('nikkei', 500_000)]


def jpy_raw_path(key: str, raw: Path = RAW) -> Path:
    return raw / JPY_FILES[key]


def iso_z(dt: datetime) -> str:
    """ISO-8601 UTC with a 'Z' suffix (notebook 02 spelling)."""
    return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def jst_at(d: date, hh: int, mm: int = 0) -> datetime:
    """Wall-clock time on Tokyo date d, tz-aware in JST."""
    return datetime.combine(d, dtime(hh, mm), JST)


def bounded_fetch_filtered(key: str, cap: int, allow_download: bool = False,
                           budget: Optional[PublicBudget] = None, raw: Path = RAW,
                           start: date = START, end: date = END) -> None:
    """Make sure one public source cache exists; curl only when allow_download is True.

    Whole-history responses (MOF, BOJ) are filtered to the window [start, end]
    (default: development dates) before the cache is written, so a
    development cache never stores a reserved-period row.
    """
    def keep_date(d: date) -> bool:
        return start <= d <= end

    dest = jpy_raw_path(key, raw)
    if dest.exists():
        return
    if not allow_download:
        raise FileNotFoundError(f'missing cached source: {key}')
    budget = budget or PublicBudget()
    if len(budget.attempts) >= budget.max_calls:
        raise RuntimeError('request cap')
    fd, tmp = tempfile.mkstemp(prefix='jpy_download_', dir=raw)
    os.close(fd)
    try:
        p = subprocess.run(['curl', '--silent', '--show-error', '--location', '--compressed', '--max-time', '30',
                            '--max-filesize', str(cap), '--output', tmp, '--write-out', '%{http_code}',
                            jpy_source_url(key, start, end)], capture_output=True, text=True, timeout=35)
        if p.returncode or p.stdout.strip() != '200':
            raise RuntimeError(f'public source {key}: HTTP/transport failure; no error body retained')
        b = Path(tmp).read_bytes()
        budget.attempts.append({'source': key, 'bytes': len(b), 'sha256': sha(b)})
        budget.bytes += len(b)
        if budget.bytes > budget.max_bytes:
            raise RuntimeError('byte cap')
        rows = list(csv.reader(io.StringIO(b.decode('utf-8-sig'))))
        if key == 'mof':
            keep = rows[:2]
            for r in rows[2:]:
                try:
                    d = datetime.strptime(r[0], '%Y/%m/%d').date()
                except (ValueError, IndexError):
                    continue
                if keep_date(d):
                    keep.append(r)
            rows = keep
        elif key in ('tona', 'boj_fx'):
            if not rows or rows[0][:2] != ['STATUS', '200']:
                raise RuntimeError('BOJ unsuccessful response')
            h = next(i for i, r in enumerate(rows) if r and r[0] == 'SERIES_CODE')
            rows = rows[:h + 1] + [r for r in rows[h + 1:]
                                   if len(r) > 6 and r[6].isdigit() and keep_date(datetime.strptime(r[6], '%Y%m%d').date())]
        elif key == 'nikkei':
            rows = rows[:1] + [r for r in rows[1:] if r and keep_date(date.fromisoformat(r[0]))]
        elif key == 'cftc':
            rows = rows[:1] + [r for r in rows[1:] if r and keep_date(date.fromisoformat(r[0][:10]))]
        fd2, tmp2 = tempfile.mkstemp(prefix='jpy_filtered_', dir=raw, text=True)
        with os.fdopen(fd2, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        os.replace(tmp2, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def jpy_raw_paths(allow_download: bool = False, raw: Path = RAW, start: date = START, end: date = END,
                  budget: Optional[PublicBudget] = None) -> dict[str, Path]:
    """Make sure the five Japanese/CFTC caches exist and return their paths."""
    budget = budget or PublicBudget()
    for key, cap in JPY_RESPONSE_CAPS:
        bounded_fetch_filtered(key, cap, allow_download, budget, raw, start, end)
    return {key: jpy_raw_path(key, raw) for key in JPY_FILES}


def _record_02(sid: str, d: date, ot: datetime, at: Optional[datetime], value, unit: str,
               source: str, basis: str, start: date = START, end: date = END) -> Optional[Record]:
    """Notebook 02's `add`: skip None values and dates outside the window."""
    if value is None or not in_window(d, start, end):
        return None
    return {'series_id': sid, 'observation_date': d.isoformat(), 'observation_time': iso_z(ot),
            'available_time': iso_z(at) if at else '', 'value': value, 'unit': unit,
            'source_id': source, 'availability_basis': basis}


def mof_observations(path: Path, start: date = START, end: date = END) -> list[Record]:
    """MOF 2Y/10Y JGB yields; available 09:30 JST on the next observed MOF date.

    The last date has no next date, so its available_time is blank.
    """
    with path.open(newline='', encoding='utf-8-sig') as f:
        rr = csv.reader(f)
        next(rr)
        hh = next(rr)
        rows = list(rr)
    assert hh[0] == 'Date' and '2Y' in hh and '10Y' in hh
    parsed = []
    for r in rows:
        try:
            d = datetime.strptime(r[0], '%Y/%m/%d').date()
        except (ValueError, IndexError):
            continue
        if in_window(d, start, end):
            parsed.append((d, r))
    parsed.sort(key=lambda z: z[0])
    assert len({d for d, _ in parsed}) == len(parsed)
    out = []
    for i, (d, r) in enumerate(parsed):
        nextd = parsed[i + 1][0] if i + 1 < len(parsed) else None
        at = jst_at(nextd, 9, 30) if nextd else None
        for tenor in ('2Y', '10Y'):
            rec = _record_02('jpy.mof_jgb_' + tenor.lower(), d, jst_at(d, 15), at, numeric(r[hh.index(tenor)]),
                             'percent annual yield', 'mof_jgb',
                             'MOF next-observed Japan business date 09:30 JST; latest-vintage curve; '
                             'last date availability unknown', start, end)
            if rec:
                out.append(rec)
    return out


def read_boj(path: Path, code: str, start: date = START, end: date = END) -> list[tuple[date, Optional[float], str]]:
    """(date, value, unit) rows of one series from a BOJ API CSV, window dates only (default: development)."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    assert rows[:1] == [['STATUS', '200']]
    h = next(i for i, r in enumerate(rows) if r and r[0] == 'SERIES_CODE')
    hdr = rows[h]
    out = []
    for r in rows[h + 1:]:
        if not r or r[0] != code:
            continue
        o = dict(zip(hdr, r))
        d = datetime.strptime(o['SURVEY_DATES'], '%Y%m%d').date()
        if in_window(d, start, end):
            out.append((d, numeric(o['VALUES']), o['UNIT']))
    out.sort(key=lambda z: z[0])
    assert len({d for d, _, _ in out}) == len(out)
    return out


BOJ_SERIES = [  # key, code, series_id, observation hour, availability hour, minute, unit
    ('tona', 'STRDCLUCON', 'jpy.boj_tona_final', 17, 10, 5, 'percent overnight rate'),
    ('boj_fx', 'FXERD04', 'jpy.boj_usdjpy_17_mid', 17, 9, 0, 'JPY per USD')]


def boj_observations(paths: dict[str, Path], start: date = START, end: date = END) -> list[Record]:
    """BOJ TONA and 17:00 JST USD/JPY midpoint; available on the next valid date (notebook 02 rule)."""
    out = []
    for key, code, sid, obs_hour, av_hour, av_min, unit in BOJ_SERIES:
        valid = [(d, v) for d, v, _ in read_boj(paths[key], code, start, end) if v is not None]
        basis = ('BOJ final overnight result normally next business ~10:00 JST; 10:05 conservative rule; latest vintage'
                 if key == 'tona'
                 else 'BOJ 17:00 JST bid/offer midpoint; API ~08:50 next business, 09:00 conservative rule; latest vintage')
        for i, (d, v) in enumerate(valid):
            nextd = valid[i + 1][0] if i + 1 < len(valid) else None
            rec = _record_02(sid, d, jst_at(d, obs_hour), jst_at(nextd, av_hour, av_min) if nextd else None,
                             v, unit, 'boj_' + key, basis, start, end)
            if rec:
                out.append(rec)
    return out


def cftc_observations(path: Path, start: date = START, end: date = END) -> list[Record]:
    """CFTC TFF yen positions; notebook 02's conservative '+8 days 00:00 UTC' placeholder."""
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = date.fromisoformat(r['report_date_as_yyyy_mm_dd'][:10])
        assert r['cftc_contract_market_code'] == '097741'
        # Tuesday report date is not release date. +8 days is deliberately conservative;
        # notebook 06 replaces it with the actual release calendar.
        ot = datetime.combine(d, dtime(23, 59), UTC)
        at = datetime.combine(d + timedelta(days=8), dtime(0), UTC)
        for sid, field in [('jpy.cftc_lf_long', 'lev_money_positions_long'),
                           ('jpy.cftc_lf_short', 'lev_money_positions_short'),
                           ('jpy.cftc_open_interest', 'open_interest_all')]:
            rec = _record_02(sid, d, ot, at, numeric(r[field]), 'JPY futures contracts', 'cftc_tff_097741',
                             'TFF futures-only Tuesday position; assumed available by following Wednesday 00:00 UTC '
                             '(+8d), not actual release log; latest vintage', start, end)
            if rec:
                out.append(rec)
    return out


def nikkei_observations(path: Path, start: date = START, end: date = END) -> list[Record]:
    """Nikkei 225 close; available 09:00 JST on the next observed Japan day."""
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    valid = [(date.fromisoformat(r['observation_date']), numeric(r['NIKKEI225'])) for r in rows]
    valid = [z for z in valid if z[1] is not None and in_window(z[0], start, end)]
    out = []
    for i, (d, v) in enumerate(valid):
        nextd = valid[i + 1][0] if i + 1 < len(valid) else None
        rec = _record_02('jpy.nikkei225', d, jst_at(d, 16), jst_at(nextd, 9) if nextd else None, v,
                         'JPY index points', 'fred_nikkei225',
                         'Local-currency Nikkei 225 close; 16:00 JST observation proxy and next observed Japan day '
                         '09:00 JST conservative market-public rule, not FRED ingestion', start, end)
        if rec:
            out.append(rec)
    return out


def jpy_observations(paths: dict[str, Path], start: date = START, end: date = END) -> list[Record]:
    """All notebook 02 public records, sorted by (series_id, observation_date) as the file is."""
    records = (mof_observations(paths['mof'], start, end) + boj_observations(paths, start, end)
               + cftc_observations(paths['cftc'], start, end) + nikkei_observations(paths['nikkei'], start, end))
    records.sort(key=lambda r: (r['series_id'], r['observation_date']))
    assert all(in_window(date.fromisoformat(r['observation_date']), start, end) for r in records)
    assert len({(r['series_id'], r['observation_date']) for r in records}) == len(records)
    return records


def write_records_csv(path: Path, records: list[Record]) -> None:
    """Plain csv.DictWriter write (notebook 02 style: no temp file, CRLF rows)."""
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OBSERVATION_COLUMNS)
        w.writeheader()
        w.writerows(records)


# ----------------------------------------------------------------------------
# Notebook 02: J-Quants V2 price indices, private cache only
# ----------------------------------------------------------------------------
JQ_CODES = {'0000': 'jpy.topix_pr', '008E': 'jpy.topix17_banks_pr', '0088': 'jpy.topix17_electric_precision_pr'}
JQ_URL = 'https://api.jquants.com/v2/indices/bars/daily'
JQ_MAX_REQUESTS = 100
JQ_MAX_BYTES = 20_000_000
JQ_PAGE_CAP = 5_500_000
JQ_START, JQ_END = START.isoformat(), END.isoformat()


def jquants_refresh_requested() -> bool:
    """The notebook's explicit acquire switch: CQF_JQUANTS_REFRESH=1."""
    return os.environ.get('CQF_JQUANTS_REFRESH') == '1'


def jq_write(path: Path, data: bytes) -> None:
    """Write bytes with mode 0600 from the first byte; licensed data stay private."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    os.chmod(path, 0o600)


def jq_json(path: Path, obj) -> None:
    jq_write(path, (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode())


def jq_key() -> str:
    """The J-Quants API key from the Keychain; returned to the caller, never stored."""
    p = subprocess.run(['/usr/bin/security', 'find-generic-password', '-s', 'jquant-market-data',
                        '-a', 'jquant-api-key', '-w'], capture_output=True, timeout=30)
    if p.returncode or not p.stdout.strip():
        raise RuntimeError('J-Quants Keychain item unavailable')
    return p.stdout.strip().decode('utf-8')


class JQBudget:
    """Run-wide J-Quants request and byte counters."""

    def __init__(self):
        self.attempts = 0
        self.bytes = 0


def jq_manifest_path(code: str, private_root: Path = PRIVATE_DIR, start: str = JQ_START, end: str = JQ_END) -> Path:
    """Raw-manifest path: the notebook 02 name for the development window, a window-suffixed name otherwise."""
    if (start, end) == (JQ_START, JQ_END):
        return private_root / f'jquants_{code}_raw_manifest.json'
    return private_root / f'jquants_{code}_{start}_{end}_raw_manifest.json'


def jq_fetch(code: str, key: str, budget: JQBudget, private_root: Path = PRIVATE_DIR,
             start: str = JQ_START, end: str = JQ_END, max_requests: int = JQ_MAX_REQUESTS) -> dict:
    """Download all pages for one index code into the private root and write its raw manifest.

    The key goes to curl through a stdin config, never on the command line.
    """
    pages = []
    page_key = None
    seen = set()
    total_bytes = 0
    attempts = 0
    while True:
        params = {'code': code, 'from': start, 'to': end}
        if page_key:
            params['pagination_key'] = page_key
        url = JQ_URL + '?' + urllib.parse.urlencode(params)
        for retry in range(2):
            if budget.attempts >= min(JQ_MAX_REQUESTS, max_requests):
                raise RuntimeError('J-Quants request budget exceeded')
            attempts += 1
            budget.attempts += 1
            config = 'header = "x-api-key: ' + key + '"\nheader = "Accept: application/json"\n'
            fd, tmp = tempfile.mkstemp(prefix='jq_transfer_', dir=private_root)
            os.close(fd)
            cmd = ['/usr/bin/curl', '--silent', '--show-error', '--compressed', '--proto', '=https',
                   '--max-time', '20', '--max-filesize', str(JQ_PAGE_CAP), '--output', tmp,
                   '--write-out', '%{http_code}', url, '--config', '-']
            try:
                response = subprocess.run(cmd, input=config.encode(), capture_output=True, timeout=25)
                # Some local curl builds return 23 after writing complete HTTP 200 JSON; the schema
                # and date checks below must still pass.
                ok_23 = response.returncode == 23 and response.stdout.strip() == b'200' and Path(tmp).stat().st_size
                if response.returncode and not ok_23:
                    raise RuntimeError('J-Quants curl exit ' + str(response.returncode))
                try:
                    status = int(response.stdout.strip())
                except Exception:
                    raise RuntimeError('J-Quants response framing failed') from None
                body = Path(tmp).read_bytes()
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            if status == 429 and retry == 0:
                time.sleep(2)
                continue
            if status != 200:
                raise RuntimeError('J-Quants HTTP status ' + str(status))
            break
        total_bytes += len(body)
        budget.bytes += len(body)
        if budget.bytes > JQ_MAX_BYTES:
            raise RuntimeError('J-Quants byte budget exceeded')
        try:
            obj = json.loads(body)
        except Exception:
            raise RuntimeError('J-Quants JSON parse failed') from None
        if not isinstance(obj, dict) or not isinstance(obj.get('data'), list):
            raise RuntimeError('J-Quants response schema changed')
        pages.append(body)
        nxt = obj.get('pagination_key')
        if not nxt:
            break
        if not isinstance(nxt, str) or nxt in seen:
            raise RuntimeError('J-Quants pagination invalid')
        seen.add(nxt)
        page_key = nxt
    del key
    page_meta = []
    for i, body in enumerate(pages):
        name = f'jquants_{code}_{start}_{end}_page{i + 1:02}.json'
        jq_write(private_root / name, body)
        page_meta.append({'file': name, 'bytes': len(body), 'sha256': sha(body)})
    manifest = {'code': code, 'start': start, 'end': end, 'pages': page_meta, 'attempts': attempts,
                'response_bytes': total_bytes, 'fetched_at_utc': iso_z(datetime.now(UTC)), 'endpoint': JQ_URL}
    jq_json(jq_manifest_path(code, private_root, start, end), manifest)
    return manifest


def jq_cached(code: str, refresh: bool = False, key: Optional[str] = None,
              budget: Optional[JQBudget] = None, private_root: Path = PRIVATE_DIR,
              start: str = JQ_START, end: str = JQ_END, max_requests: int = JQ_MAX_REQUESTS) -> dict:
    """Raw manifest for one code and window: the private cache by default, a refresh only on request."""
    mp = jq_manifest_path(code, private_root, start, end)
    if refresh and not mp.exists():
        return jq_fetch(code, key, budget or JQBudget(), private_root, start, end, max_requests)
    if not mp.exists():
        raise FileNotFoundError('Private J-Quants cache missing for ' + code + '; run an authorized bounded refresh')
    m = json.loads(mp.read_text())
    assert m['code'] == code and m['start'] == start and m['end'] == end and m['pages']
    return m


def jquants_raw_manifests(refresh: Optional[bool] = None, private_root: Path = PRIVATE_DIR,
                          start: str = JQ_START, end: str = JQ_END, max_requests: int = JQ_MAX_REQUESTS) -> dict[str, dict]:
    """Raw manifests for all three codes. refresh=None reads CQF_JQUANTS_REFRESH.

    A refresh fetches only the codes whose manifest for this window is absent
    (a cached window is never requested again); for the development window
    this is the notebook 02 behaviour.
    """
    if refresh is None:
        refresh = jquants_refresh_requested()
    if refresh:                                        # a cache-only read leaves the private folder untouched
        private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(private_root, 0o700)
    need_key = refresh and any(not jq_manifest_path(c, private_root, start, end).exists() for c in JQ_CODES)
    key = jq_key() if need_key else None
    budget = JQBudget()
    try:
        return {code: jq_cached(code, refresh, key, budget, private_root, start, end, max_requests) for code in JQ_CODES}
    finally:
        del key


def jquants_rows(manifests: dict[str, dict], private_root: Path = PRIVATE_DIR,
                 start: date = START, end: date = END) -> list[dict]:
    """Validated OHLC rows from the cached pages: {'code', 'date', 'O', 'H', 'L', 'C'}."""
    jq_rows = []
    total_bytes = 0
    requests = 0
    counts = {}
    for code, m in manifests.items():
        rows = []
        total_bytes += m['response_bytes']
        requests += m['attempts']
        for p in m['pages']:
            b = (private_root / p['file']).read_bytes()
            assert len(b) == p['bytes'] and sha(b) == p['sha256']
            rows.extend(json.loads(b)['data'])
        seen_dates = set()
        for r in rows:
            assert r['Code'] == code
            d = date.fromisoformat(r['Date'])
            assert in_window(d, start, end) and d not in seen_dates
            seen_dates.add(d)
            prices = [numeric(r.get(k)) for k in ('O', 'H', 'L', 'C')]
            assert all(v is not None and v > 0 for v in prices)
            op, hi, lo, cl = prices
            assert lo <= min(op, cl) <= max(op, cl) <= hi
            jq_rows.append({'code': code, 'date': d, 'O': op, 'H': hi, 'L': lo, 'C': cl})
        counts[code] = len(rows)
    assert total_bytes <= JQ_MAX_BYTES and requests <= JQ_MAX_REQUESTS
    assert len({(r['code'], r['date']) for r in jq_rows}) == len(jq_rows)
    assert len({r['date'] for r in jq_rows}) == counts['0000']
    return jq_rows


def jquants_observations(jq_rows: list[dict]) -> list[Record]:
    """Notebook 02 rule: available 09:00 JST on the next observed Japan session.

    The 16:30 JST update is approximate, not a historical release timestamp.
    The last development-day observation has a blank available_time because
    its next session is outside the window. (Notebook 06 uses a different,
    Tokyo-close rule for the 17:00 ET table; see align.py.)
    """
    out = []
    for code, sid in JQ_CODES.items():
        rr = sorted((r for r in jq_rows if r['code'] == code), key=lambda r: r['date'])
        for i, r in enumerate(rr):
            d = r['date']
            nextd = rr[i + 1]['date'] if i + 1 < len(rr) else None
            for field in ('O', 'H', 'L', 'C'):
                out.append({'series_id': sid + '.' + field.lower(), 'observation_date': d.isoformat(),
                            'observation_time': iso_z(jst_at(d, 16)),
                            'available_time': iso_z(jst_at(nextd, 9)) if nextd else '', 'value': r[field],
                            'unit': 'JPY price-index points', 'source_id': 'jquants_v2_' + code,
                            'availability_basis': ('Official daily PR index OHLC; ~16:30 JST update documented; '
                                                   'assumed available by 09:00 next observed Japan session; '
                                                   'latest-vintage, not archived release time')})
    assert len({(r['series_id'], r['observation_date']) for r in out}) == len(out)
    return out


def write_private_records_csv(path: Path, records: list[Record]) -> None:
    """Write licensed records with mode 0600, only inside the private cache or the private scratch folder."""
    if not (is_inside(path, PRIVATE_DIR) or is_inside(path, PRIVATE_SCRATCH)):
        raise ValueError('Licensed data must stay in the private folder.')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OBSERVATION_COLUMNS)
        w.writeheader()
        w.writerows(records)
    os.chmod(path, 0o600)


# ----------------------------------------------------------------------------
# Cache-only rebuild of the long observation files (from data_review/run_all.py)
# ----------------------------------------------------------------------------
PUBLIC_OBSERVATION_FILES = ['shared_fmp_etf.csv', 'shared_fmp_usdjpy.csv', 'shared_fred.csv', 'jpy_sources.csv']


def write_observation_files(out_dir: Path, raw: Path = RAW) -> list[Path]:
    """Raw caches -> the four public long observation files in out_dir. No request is made."""
    fmp_etf, fmp_fx = fmp_observations(fmp_raw_paths(acquire=False, raw=raw))
    save_observations(out_dir / 'shared_fmp_etf.csv', fmp_etf)
    save_observations(out_dir / 'shared_fmp_usdjpy.csv', fmp_fx)
    save_observations(out_dir / 'shared_fred.csv', fred_observations(fred_raw_paths(acquire=False, raw=raw)))
    write_records_csv(out_dir / 'jpy_sources.csv', jpy_observations(jpy_raw_paths(allow_download=False, raw=raw)))
    return [out_dir / name for name in PUBLIC_OBSERVATION_FILES]


def write_private_jquants_observations(private_out: Path) -> Path:
    """Private J-Quants raw pages -> jquants_observations.csv (mode 0600) in a private folder. No request is made."""
    manifests = jquants_raw_manifests(refresh=False)
    path = private_out / 'jquants_observations.csv'
    write_private_records_csv(path, jquants_observations(jquants_rows(manifests)))
    return path
