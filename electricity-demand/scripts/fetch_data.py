"""Build the committed data snapshot: NYISO hourly demand (EIA-930) + NOAA hourly temperature.

    cd electricity-demand && python scripts/fetch_data.py

Sources (both listed on data.gov):
  - EIA-930 Hourly Electric Grid Monitor, balancing authority NYIS (New York ISO): hourly
    demand and the grid operator's own day-ahead demand forecast, six-month BALANCE files.
    https://www.eia.gov/electricity/gridmonitor/
  - NOAA Global Historical Climatology Network-hourly (GHCNh), air temperature at three
    New York airports. https://catalog.data.gov/dataset/global-historical-climatology-network-hourly-ghcnh-version-1

Window: 2021-01-01 to 2025-12-31 (UTC), hour-ending timestamps. Nothing is imputed here:
gaps stay gaps and are counted in the SOURCE file; the analysis decides how to treat them.

Outputs: data/nyiso_load_weather.csv.gz and data/nyiso_load_weather.SOURCE.json
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BA = "NYIS"
YEARS = range(2021, 2026)
EIA_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_{half}.csv"
GHCNH_URLS = [
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/{year}/psv/GHCNh_{station}_{year}.psv",
]
STATIONS = {"nyc_lga": "USW00014732", "albany": "USW00014735", "buffalo": "USW00014733"}
START, END = datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)
OUT = Path("data") / "nyiso_load_weather.csv.gz"
SOURCE = Path("data") / "nyiso_load_weather.SOURCE.json"
UA = {"User-Agent": "ml-portfolio-data-snapshot"}


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def get(url: str) -> requests.Response:
    r = requests.get(url, timeout=300, headers=UA)
    if r.status_code != 200:
        raise SystemExit(f"{r.status_code} for {url}: {r.text[:300]}")
    return r


def col(header: list[str], *names: str) -> str:
    by = {norm(h): h for h in header}
    for n in names:
        if norm(n) in by:
            return by[norm(n)]
    raise SystemExit(f"none of {names} in header {header}")


def parse_num(v: str) -> float | None:
    v = (v or "").replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def fetch_eia(meta: dict) -> dict:
    load = {}
    for year in YEARS:
        for half in ("Jan_Jun", "Jul_Dec"):
            r = get(EIA_URL.format(year=year, half=half))
            text = r.content.decode("utf-8-sig")
            rows = csv.DictReader(io.StringIO(text))
            h = rows.fieldnames or []
            c_ba, c_utc = col(h, "Balancing Authority"), col(h, "UTC Time at End of Hour")
            c_dem, c_fc = col(h, "Demand (MW)"), col(h, "Demand Forecast (MW)")
            meta.setdefault("eia_header", h)
            meta.setdefault("eia_files", []).append({"url": r.url, "sha256": hashlib.sha256(r.content).hexdigest()})
            for row in rows:
                if row[c_ba] != BA:
                    continue
                t = None
                for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %I:%M %p"):
                    try:
                        t = datetime.strptime(row[c_utc].strip(), fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        pass
                if t is None:
                    raise SystemExit(f"unparsed UTC time {row[c_utc]!r}")
                load[t] = (parse_num(row[c_dem]), parse_num(row[c_fc]))
            print(f"EIA {year} {half}: {len(load)} NYIS hours so far", flush=True)
    return load


def fetch_ghcnh(station: str, meta: dict) -> dict:
    temps = defaultdict(list)
    for year in list(YEARS) + [2020]:  # Dec 31 2020 local hours fall inside the UTC window edges
        r = None
        for pattern in GHCNH_URLS:
            url = pattern.format(year=year, station=station)
            rr = requests.get(url, timeout=300, headers=UA)
            if rr.status_code == 200:
                r = rr
                break
        if r is None:
            raise SystemExit(f"GHCNh {station} {year}: not found at {GHCNH_URLS}")
        meta.setdefault("ghcnh_files", []).append({"url": r.url, "sha256": hashlib.sha256(r.content).hexdigest()})
        rows = csv.DictReader(io.StringIO(r.content.decode("utf-8-sig")), delimiter="|")
        h = rows.fieldnames or []
        meta.setdefault("ghcnh_header", h)
        c_t, c_q = col(h, "temperature"), col(h, "temperature_Quality_Code")
        cy, cm, cd, ch = col(h, "Year"), col(h, "Month"), col(h, "Day"), col(h, "Hour")
        for row in rows:
            v = parse_num(row[c_t])
            if v is None or row[c_q].strip() not in ("", "0", "1", "4", "5", "9"):  # keep unflagged / passed values
                continue
            obs = datetime(int(row[cy]), int(row[cm]), int(row[cd]), int(row[ch]), tzinfo=timezone.utc)
            # Assign each observation to the hour it falls in, labelled by the hour END (EIA convention).
            temps[obs + timedelta(hours=1)].append(v)
    return {t: sum(v) / len(v) for t, v in temps.items()}


def main() -> int:
    meta = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    load = fetch_eia(meta)
    weather = {name: fetch_ghcnh(sid, meta) for name, sid in STATIONS.items()}
    rows, counts = [], Counter()
    t = START + timedelta(hours=1)
    while t <= END:
        dem, fc = load.get(t, (None, None))
        row = {"utc_hour_end": t.strftime("%Y-%m-%dT%H:00Z"), "demand_mw": dem, "eia_demand_forecast_mw": fc}
        for name in STATIONS:
            row[f"temp_c_{name}"] = weather[name].get(t)
        counts["hours"] += 1
        counts["demand_missing"] += dem is None
        counts["forecast_missing"] += fc is None
        for name in STATIONS:
            counts[f"temp_missing_{name}"] += row[f"temp_c_{name}"] is None
        rows.append(row)
        t += timedelta(hours=1)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else (f"{v:.2f}" if isinstance(v, float) else v)) for k, v in r.items()})
    data = buf.getvalue().encode()
    OUT.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(data)
    SOURCE.write_text(json.dumps({
        "datasets": {"EIA-930 Hourly Electric Grid Monitor (BALANCE)": "https://www.eia.gov/electricity/gridmonitor/",
                     "NOAA GHCNh": "https://catalog.data.gov/dataset/global-historical-climatology-network-hourly-ghcnh-version-1"},
        "balancing_authority": BA, "stations": STATIONS, "window_utc": [START.isoformat(), END.isoformat()],
        "timestamp_convention": "hour ending, UTC", "counts": dict(counts), **meta,
        "snapshot_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(json.dumps(dict(counts)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
