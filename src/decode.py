"""Obfuscation forensics: what can be learned about the hashed identifiers.

Four columns arrive as 32-character lowercase hex strings: ``domain``, ``url``,
``ad_slot`` and ``publisher_properties``. ``user_id`` is a version 4 UUID. This
module asks two questions and answers both reproducibly.

1. Can the hashes be reversed? We try the cheap attacks a careful analyst would
   try: hashing popular domains under common URL spellings with several digest
   functions, hashing small integers (a common way to obfuscate internal ids),
   hashing twice, and chaining one column's hash into another. Every attempt this
   project ran came back empty (see ``attack_ledger``), which is what a salted or
   keyed hash looks like from the outside.

2. What does the structure leak anyway? Functional dependencies between the hashed
   columns reveal the hidden schema: a page belongs to exactly one domain, an ad
   slot is a site-level placement reused across pages, and ``publisher_properties``
   is a publisher-level configuration shared across domains.

The Tranco top-1M wordlist is not committed. Fetch it with::

    curl -L https://tranco-list.eu/top-1m.csv.zip -o data/top-1m.csv.zip
    unzip data/top-1m.csv.zip -d data/      # gives data/top-1m.csv (git-ignored)
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

OBFUSCATED = ["domain", "url", "ad_slot", "publisher_properties"]
HEX32 = re.compile(r"^[0-9a-f]{32}$")
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

ALGORITHMS = {
    "md5": lambda b: hashlib.md5(b).hexdigest(),
    "sha1_first32": lambda b: hashlib.sha1(b).hexdigest()[:32],
    "sha256_first32": lambda b: hashlib.sha256(b).hexdigest()[:32],
    "sha256_last32": lambda b: hashlib.sha256(b).hexdigest()[-32:],
}

WORDLIST_CANDIDATES = ("data/top-1m.csv", "../data/top-1m.csv", "/tmp/top-1m.csv")


# --------------------------------------------------------------------------- format
def hash_format_report(df: pd.DataFrame) -> pd.DataFrame:
    """Length, charset and first-digit uniformity for each obfuscated column."""
    rows = []
    for c in OBFUSCATED:
        s = df[c].dropna().astype(str)
        distinct = s.drop_duplicates()
        first = distinct.str[0].value_counts(normalize=True)
        rows.append(
            {
                "column": c,
                "n_distinct": int(distinct.size),
                "lengths": ", ".join(str(v) for v in sorted(s.str.len().unique())),
                "share_32_lower_hex": float(s.str.fullmatch(HEX32.pattern).mean()),
                "first_digit_max_share": float(first.max()),
                "first_digit_min_share": float(first.min()),
                "reads_as": "32-hex digest (MD5-shaped)",
            }
        )
    u = df["user_id"].dropna().astype(str)
    rows.append(
        {
            "column": "user_id",
            "n_distinct": int(u.nunique()),
            "lengths": ", ".join(str(v) for v in sorted(u.str.len().unique())),
            "share_32_lower_hex": 0.0,
            "first_digit_max_share": np.nan,
            "first_digit_min_share": np.nan,
            "reads_as": f"UUID v4 ({u.str.fullmatch(UUID4.pattern).mean():.1%} match)",
        }
    )
    return pd.DataFrame(rows)


def nibble_frequencies(df: pd.DataFrame, cols=OBFUSCATED, position: int = 0) -> pd.DataFrame:
    """Share of distinct values whose hex digit at ``position`` is each of 0..f."""
    digits = list("0123456789abcdef")
    out = {}
    for c in cols:
        distinct = df[c].dropna().astype(str).drop_duplicates()
        out[c] = distinct.str[position].value_counts(normalize=True).reindex(digits).fillna(0.0)
    return pd.DataFrame(out, index=pd.Index(digits, name="hex_digit"))


# --------------------------------------------------------------------------- structure
def functional_dependency_matrix(df: pd.DataFrame, cols=OBFUSCATED) -> pd.DataFrame:
    """Cell (X, Y) = share of X values that co-occur with exactly one Y value.

    1.0 means X determines Y (every page has one domain). Lower values mean the X
    identifier is reused across several Y values.
    """
    m = pd.DataFrame(np.nan, index=pd.Index(cols, name="given"), columns=pd.Index(cols, name="fixed"))
    for a in cols:
        for b in cols:
            if a == b:
                m.loc[a, b] = 1.0
                continue
            sub = df[[a, b]].dropna()
            m.loc[a, b] = float((sub.groupby(a)[b].nunique() == 1).mean())
    return m.astype(float)


def hash_sets(df: pd.DataFrame, cols=OBFUSCATED) -> dict[str, set]:
    return {c: set(df[c].dropna().astype(str).unique()) for c in cols}


# --------------------------------------------------------------------------- attacks
def url_variants(domain: str) -> list[str]:
    d = domain.strip().lower()
    return [
        d, "www." + d, "http://" + d, "https://" + d, "http://www." + d, "https://www." + d,
        d.upper(), "https://" + d + "/", "https://www." + d + "/",
    ]


def find_wordlist(candidates=WORDLIST_CANDIDATES) -> Path | None:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def _lookup(h: str, sets: dict[str, set]) -> list[str]:
    return [c for c, s in sets.items() if h in s]


def dictionary_attack(sets: dict[str, set], wordlist_path=None, variants=url_variants, algorithms=ALGORITHMS, limit=None) -> dict:
    """Hash every candidate string under every variant and algorithm; report hits.

    ``wordlist_path`` is a CSV with the domain in the last column (Tranco format
    ``rank,domain``) or one domain per line. When it is missing the function returns
    ``status="skipped"`` instead of failing, so notebooks run anywhere.
    """
    path = Path(wordlist_path) if wordlist_path else find_wordlist()
    if path is None or not path.exists():
        return {"status": "skipped", "note": "wordlist not found; see module docstring for the Tranco download", "hits": [], "digests": 0}
    t0 = time.time()
    hits, digests, n_words = [], 0, 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            word = line.strip().split(",")[-1]
            if not word:
                continue
            n_words += 1
            for v in variants(word):
                b = v.encode()
                for name, fn in algorithms.items():
                    digests += 1
                    cols = _lookup(fn(b), sets)
                    if cols:
                        hits.append({"word": word, "variant": v, "algorithm": name, "columns": cols})
            if limit and n_words >= limit:
                break
    return {"status": "ran", "wordlist": str(path), "n_words": n_words, "digests": digests, "hits": hits, "seconds": time.time() - t0}


def integer_bruteforce(sets: dict[str, set], n: int = 3_000_000, double: bool = False) -> dict:
    """md5(str(i)) for i in [0, n), optionally hashed twice (md5 of the hex digest)."""
    t0 = time.time()
    hits = []
    for i in range(n):
        h = hashlib.md5(str(i).encode()).hexdigest()
        if double:
            h = hashlib.md5(h.encode()).hexdigest()
        cols = _lookup(h, sets)
        if cols:
            hits.append({"i": i, "columns": cols})
    return {"status": "ran", "n": n, "double": double, "hits": hits, "seconds": time.time() - t0}


def chained_hash_test(df: pd.DataFrame, sample: int = 20_000) -> pd.DataFrame:
    """Is one column the md5 of another? (e.g. domain = md5(url)). Counts matches."""
    sets = hash_sets(df)
    pairs = [("url", "domain"), ("ad_slot", "url"), ("domain", "publisher_properties"), ("ad_slot", "domain"), ("publisher_properties", "domain")]
    rows = []
    for src, dst in pairs:
        values = df[src].dropna().astype(str).drop_duplicates().head(sample)
        n_hits = sum(hashlib.md5(v.encode()).hexdigest() in sets[dst] for v in values)
        rows.append({"test": f"md5({src}) found in {dst}", "n_tried": int(len(values)), "hits": int(n_hits)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ledger
RECORDED_LEDGER = [
    {"method": "Popular-domain dictionary (about 280 domains, 9 URL spellings) with md5, sha1, sha256", "search_space": "6,939 digests", "hits": 0},
    {"method": "Tranco top 1,000,000 domains, 9 URL spellings, 4 digest functions", "search_space": "36,000,000 digests", "hits": 0},
    {"method": "md5 of the integers 0 to 3,000,000 (internal-id obfuscation)", "search_space": "3,000,000 digests", "hits": 0},
    {"method": "double md5 of the integers 0 to 200,000", "search_space": "200,000 digests", "hits": 0},
    {"method": "chained hashes: md5(url) in domain, md5(ad_slot) in url, md5(domain) in publisher_properties, md5(ad_slot) in domain", "search_space": "4 x 20,000 values", "hits": 0},
]


def attack_ledger(live: list[dict] | None = None) -> pd.DataFrame:
    """The attacks run for this project (recorded 2026-09-09) plus any live results."""
    rows = [dict(r, run="recorded") for r in RECORDED_LEDGER]
    for r in live or []:
        rows.append(dict(r, run="this session"))
    return pd.DataFrame(rows)[["method", "search_space", "hits", "run"]]
