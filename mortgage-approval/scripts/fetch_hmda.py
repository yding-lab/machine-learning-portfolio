"""Build the committed data snapshot from public HMDA data (CFPB / FFIEC).

    cd mortgage-approval && python scripts/fetch_hmda.py

Source: Home Mortgage Disclosure Act (HMDA) public loan/application register, listed on data.gov
(https://catalog.data.gov/dataset/hmda-public-data-starting-in-2017), downloaded through the
HMDA Data Browser API (https://ffiec.cfpb.gov/documentation/api/data-browser/).

Snapshot definition (fixed):
  - New York State, applications with a credit decision: originated (1), approved but not
    accepted (2), denied (3). Withdrawn, incomplete, purchased and pre-approval records are out.
  - home purchase (loan_purpose 1); kept after download only if first lien, 1-4 units,
    owner-occupied principal residence, not business/commercial, not reverse mortgage,
    not an open-end line of credit (the API takes at most two data filters)
  - train/validation = applications acted on in 2023, test = 2024 (a time split)
  - seeded random sample of at most TRAIN_N / TEST_N records; full counts recorded
  - kept fields: application characteristics an underwriter would see, census-tract context,
    and the demographic fields (for the fairness audit only). Lender identifiers, census tract
    codes and every post-decision field (interest rate, costs, purchaser, denial reasons) are
    dropped.

Outputs: data/hmda_ny_home_purchase.csv.gz and data/hmda_ny_home_purchase.SOURCE.json
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
# The Data Browser API accepts at most two data filters per query, so lien status and units
# are applied after download (POST_FILTERS) together with the fields the API can't filter on.
QUERY = {"states": "NY", "actions_taken": "1,2,3", "loan_purposes": "1"}
YEARS = {"train": 2023, "test": 2024}
SAMPLE = {"train": 80000, "test": 40000}
SEED = 2023
OUT = Path("data") / "hmda_ny_home_purchase.csv.gz"
SOURCE = Path("data") / "hmda_ny_home_purchase.SOURCE.json"

KEEP = [
    # decision (target) and year
    "activity_year", "action_taken",
    # application characteristics
    "loan_type", "loan_amount", "combined_loan_to_value_ratio", "loan_term", "property_value",
    "construction_method", "total_units", "income", "debt_to_income_ratio", "applicant_credit_score_type",
    "co_applicant_credit_score_type", "conforming_loan_limit", "derived_dwelling_category",
    "interest_only_payment", "balloon_payment", "negative_amortization", "other_nonamortizing_features",
    "prepayment_penalty_term", "intro_rate_period",
    # census-tract context (also see the README on proxies)
    "county_code", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
    # demographics: fairness audit only, never model inputs
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
]
FILTERS = {"lien_status": "1", "occupancy_type": "1", "business_or_commercial_purpose": "2",
           "reverse_mortgage": "2", "open_end_line_of_credit": "2"}
UNITS = {"1", "2", "3", "4"}


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def normalize_rows(rows: list[dict]) -> list[dict]:
    """The API header mixes hyphens and underscores (e.g. 'open-end_line_of_credit',
    'derived_msa-md'); map every column to its underscore name used here."""
    if not rows:
        return rows
    wanted = {_norm(c): c for c in KEEP + list(FILTERS) + ["total_units"]}
    mapping = {h: wanted.get(_norm(h), h) for h in rows[0]}
    return [{mapping[k]: v for k, v in r.items()} for r in rows]


def fetch(year: int) -> tuple[list[dict], str, dict]:
    params = {**QUERY, "years": str(year)}
    r = requests.get(API, params=params, timeout=600, headers={"User-Agent": "hmda-portfolio-snapshot"})
    if r.status_code != 200:
        raise SystemExit(f"{r.status_code} for {r.url}: {r.text[:500]}")
    sha = hashlib.sha256(r.content).hexdigest()
    rows = normalize_rows(list(csv.DictReader(io.StringIO(r.content.decode("utf-8")))))
    return rows, sha, {"url": r.url, "bytes": len(r.content)}


def main() -> int:
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rng = random.Random(SEED)
    out, counts, downloads = [], {}, {}
    for split, year in YEARS.items():
        rows, sha, meta = fetch(year)
        downloads[split] = {**meta, "sha256": sha, "rows": len(rows)}
        missing = [c for c in KEEP + list(FILTERS) if rows and c not in rows[0]]
        if missing:
            raise SystemExit(f"{year}: columns not found {missing}; header: {list(rows[0])}")
        c = Counter()
        kept = []
        for row in rows:
            c["downloaded"] += 1
            if any(row[k] != v for k, v in FILTERS.items()) or row["total_units"] not in UNITS:
                c["dropped_filters"] += 1
                continue
            kept.append({k: row[k] for k in KEEP} | {"split": split})
        c["eligible"] = len(kept)
        c["action_taken"] = dict(Counter(r["action_taken"] for r in kept))
        sample = kept if len(kept) <= SAMPLE[split] else rng.sample(kept, SAMPLE[split])
        c["sampled"] = len(sample)
        counts[split] = dict(c)
        out += sample
        print(split, year, dict(c), flush=True)

    OUT.parent.mkdir(exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["split"] + KEEP)
    w.writeheader()
    w.writerows(out)
    data = buf.getvalue().encode("utf-8")
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(data)
    SOURCE.write_text(json.dumps({
        "dataset": "HMDA public loan/application register",
        "publisher": "Consumer Financial Protection Bureau / FFIEC",
        "data_gov": "https://catalog.data.gov/dataset/hmda-public-data-starting-in-2017",
        "api": API, "query": QUERY, "post_filters": {**FILTERS, "total_units": sorted(UNITS)}, "years": YEARS, "sample": SAMPLE, "seed": SEED,
        "retrieved_utc": retrieved, "downloads": downloads, "counts": counts, "rows": len(out),
        "snapshot_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
