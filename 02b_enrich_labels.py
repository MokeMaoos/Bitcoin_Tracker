"""
02b_enrich_labels.py — enrich the transaction feature matrix with confirmed
external labels from multiple sources of deanonymised Bitcoin addresses.

Sits between 02_feature.py and 03_label.py in the pipeline.  External data
is used here only to validate the detection mechanism and supply ground-truth
labels for supervised training; the deployed model itself operates entirely
offline once trained.

External sources (all read from local files in the repository — no network
fetches at run time):

    1. Elliptic++ Actors Dataset
       Path: ../drive-download-20260426T124634Z-3-001/wallets_classes.csv
             (≈29 MB, populated copy)
       NOT  ../EllipticPlusPlus-main/Actors Dataset/wallets_classes.csv —
       that location can contain Git LFS placeholder files which silently
       collapse to ~0 confirmed addresses (root cause of the prior bug).
       Labels: 1 = illicit, 2 = licit, 3 = unknown (excluded).
       Reference: Elmougy and Liu, KDD 2023.

    2. BABD-13 Bitcoin Address Behavior Dataset
       Path: data/external_labels/BABD-13.csv
       Auto-extracted from ../archive.zip on first run if not present.
       13 categories; the mapping to {illicit, licit, neutral} is below
       (see _BABD13_CATEGORY_MAP).  Reference: Xiang et al., IEEE TIFS 2023.

    3. Bitcoin Abuse Database (deprecated public dump)
       Path: ../DatasetBitcoinAbuse.csv
       All entries are illicit by definition (community-reported abuse).

    4. OpenSanctions consolidated targets
       Path: ../OpenSanctions/targets.simple.csv
       Bitcoin addresses are extracted from the ``identifiers`` column —
       tokens (semicolon-separated) that match a Bitcoin address regex
       (legacy 1.../3... or bech32 bc1...).  All matches are illicit
       (sanctions-listed entities).

Produces: outputs/enriched_features.csv

New columns added to the feature matrix:
    confirmed_label      — 1 = confirmed illicit, 0 = confirmed licit,
                           NaN = no confirmed match
    confirmation_source  — semicolon-joined source list when multiple
                           sources confirm the same address
                           (e.g. "BitcoinAbuse;EllipticPlusPlus")
    confirmation_strength — "direct" if the matched address appears in
                            an input or output address field of the
                            transaction; NaN if no match

Conflict resolution: if two sources disagree on the same address, illicit
wins over licit (the more conservative classification for AML purposes).

Usage:
    python 02b_enrich_labels.py            # use local files
    python 02b_enrich_labels.py --refresh  # re-extract BABD-13 from the zip
"""

import argparse
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

# ── Path configuration ─────────────────────────────────────────────────────────

OUTPUT_DIR        = "outputs"
DATA_DIR          = "data"
EXTERNAL_DIR      = os.path.join(DATA_DIR, "external_labels")
FEATURES_CSV      = os.path.join(OUTPUT_DIR, "transaction_features.csv")
TRANS_DETAIL_CSV  = os.path.join(DATA_DIR, "Trans_Detail.csv")
OUT_CSV           = os.path.join(OUTPUT_DIR, "enriched_features.csv")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

# Local source paths (anchored on the repo root, one level above this script).
_ELLIPTIC_CSV     = os.path.join(_REPO, "drive-download-20260426T124634Z-3-001",
                                 "wallets_classes.csv")
_BITCOIN_ABUSE_CSV = os.path.join(_REPO, "DatasetBitcoinAbuse.csv")
_OPENSANCTIONS_CSV = os.path.join(_REPO, "OpenSanctions", "targets.simple.csv")
_ARCHIVE_ZIP      = os.path.join(_REPO, "archive.zip")
_BABD13_CSV       = os.path.join(EXTERNAL_DIR, "BABD-13.csv")

# Source name constants — used in the confirmation_source column.
SRC_ELLIPTIC      = "EllipticPlusPlus"
SRC_BABD13        = "BABD-13"
SRC_BITCOIN_ABUSE = "BitcoinAbuse"
SRC_OPENSANCTIONS = "OpenSanctions"

# BABD-13 numeric labels → (illicit/licit, human-readable category).
# Counts in the published dataset (Xiang et al., 2023):
#   0 Blackmail 8,686 / 1 Cyber-Security Service 91,617 /
#   2 Darknet Market 13,861 / 3 Centralized Exchange 300,000 /
#   4 P2P Financial Infrastructure 180 / 5 P2P Financial Service 9,309 /
#   6 Gambling 105,257 / 7 Government Criminal Blocklist 27 /
#   8 Money Laundering 16 / 9 Ponzi Scheme 15 /
#   10 Mining Pool 1,580 / 11 Tumbler 12,412 / 12 Individual Wallet 1,502
_BABD13_CATEGORY_MAP = {
    0:  (1, "blackmail"),
    1:  (0, "cyber_security_service"),
    2:  (1, "darknet_market"),
    3:  (0, "centralized_exchange"),
    4:  (0, "p2p_financial_infrastructure"),
    5:  (0, "p2p_financial_service"),
    6:  (1, "gambling"),
    7:  (1, "government_criminal_blocklist"),
    8:  (1, "money_laundering"),
    9:  (1, "ponzi_scheme"),
    10: (0, "mining_pool"),
    11: (1, "tumbler"),
    # 12 = individual wallet — neutral, intentionally excluded from the lookup.
}

# Bitcoin address regex used for OpenSanctions identifier scanning.
# Matched against whole semicolon-separated tokens (anchored) to avoid
# false positives where a base58-looking substring is embedded inside an
# Ethereum hex address or other identifier.
#   Legacy / P2SH (1... or 3...) — base58, total length 26-35
#   Bech32 (bc1...) — total length 42-62 (i.e. 39-59 chars after the bc1)
_BTC_ADDRESS_RE = re.compile(
    r"^(?:bc1[ac-hj-np-z02-9]{39,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})$"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(27) == b"version https://git-lfs.git"
    except OSError:
        return False


def _empty_lookup() -> pd.DataFrame:
    return pd.DataFrame(columns=["address", "confirmed_label"])


def _ensure_babd13_extracted() -> bool:
    """
    Make sure data/external_labels/BABD-13.csv exists.  If absent, extract it
    from ../archive.zip.  Returns True iff the file is available afterwards.
    """
    if os.path.exists(_BABD13_CSV) and os.path.getsize(_BABD13_CSV) > 1_000_000:
        return True

    if not os.path.exists(_ARCHIVE_ZIP):
        print(f"  [WARN] BABD-13 source zip not found: {_ARCHIVE_ZIP}")
        return False

    print(f"  Extracting BABD-13.csv from {_ARCHIVE_ZIP} (this can take a few minutes)…")
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    t0 = time.time()
    try:
        with zipfile.ZipFile(_ARCHIVE_ZIP) as zf:
            with zf.open("BABD-13.csv") as src, open(_BABD13_CSV, "wb") as dst:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        print(f"  [WARN] Failed to extract BABD-13.csv: {exc}")
        return False

    size_mb = os.path.getsize(_BABD13_CSV) / 1e6
    print(f"    Extracted {size_mb:.1f} MB → {_BABD13_CSV}  ({time.time()-t0:.1f} s)")
    return True


# ── Source 1: Elliptic++ ───────────────────────────────────────────────────────

def load_elliptic() -> pd.DataFrame:
    """
    Load confirmed illicit / licit labels from the local Elliptic++ Actors
    Dataset.  Uses the populated copy in drive-download-… ; the
    EllipticPlusPlus-main mirror is intentionally not consulted because it
    can contain Git-LFS placeholder files.

    Returns DataFrame[address, confirmed_label].
    """
    if not os.path.exists(_ELLIPTIC_CSV):
        print(f"  [WARN] Elliptic++ wallets_classes.csv not found: {_ELLIPTIC_CSV}")
        print(f"         (Expected the populated drive-download copy.)")
        return _empty_lookup()

    if _is_lfs_pointer(_ELLIPTIC_CSV) or os.path.getsize(_ELLIPTIC_CSV) < 1_000_000:
        print(f"  [WARN] Elliptic++ file looks like an LFS pointer or stub: "
              f"{_ELLIPTIC_CSV}")
        return _empty_lookup()

    print(f"  Loading Elliptic++ from {_ELLIPTIC_CSV}")
    cls = pd.read_csv(_ELLIPTIC_CSV)
    cls.columns = cls.columns.str.strip()

    addr_col  = cls.columns[0]
    label_col = [c for c in cls.columns if c != addr_col][0]

    cls[label_col] = pd.to_numeric(
        cls[label_col].astype(str).str.strip(), errors="coerce"
    ).fillna(3).astype(int)

    n1 = int((cls[label_col] == 1).sum())
    n2 = int((cls[label_col] == 2).sum())
    n3 = int((cls[label_col] == 3).sum())
    print(f"    Class breakdown — illicit(1)={n1:,}  licit(2)={n2:,}  unknown(3)={n3:,}")
    if n1 < 5_000:
        print(f"    [WARN] Illicit count ({n1:,}) is far below the documented "
              f"~14k — wrong file?")

    confirmed = cls[cls[label_col].isin([1, 2])].copy()
    confirmed["confirmed_label"] = (confirmed[label_col] == 1).astype(int)
    result = confirmed[[addr_col, "confirmed_label"]].rename(
        columns={addr_col: "address"}
    )
    print(f"    {len(result):,} confirmed addresses  "
          f"(illicit={int((result['confirmed_label']==1).sum()):,}, "
          f"licit={int((result['confirmed_label']==0).sum()):,})")
    return result


# ── Source 2: BABD-13 ─────────────────────────────────────────────────────────

def load_babd13(refresh: bool = False) -> pd.DataFrame:
    """
    Load Bitcoin addresses with category labels from BABD-13.

    Categories with ``illicit==1`` in _BABD13_CATEGORY_MAP map to
    confirmed_label=1; categories with ``illicit==0`` map to confirmed_label=0;
    label 12 (individual wallet) is intentionally dropped — neutral by design.

    Returns DataFrame[address, confirmed_label].
    """
    if refresh and os.path.exists(_BABD13_CSV):
        try:
            os.remove(_BABD13_CSV)
        except OSError as exc:
            print(f"  [WARN] Could not remove existing BABD-13.csv: {exc}")

    if not _ensure_babd13_extracted():
        return _empty_lookup()

    print(f"  Loading BABD-13 from {_BABD13_CSV}")
    df = pd.read_csv(_BABD13_CSV, usecols=["account", "label"],
                     dtype={"account": str, "label": "Int64"})
    df = df.dropna(subset=["account", "label"])

    print("    BABD-13 raw label distribution:")
    for lbl, n in df["label"].value_counts().sort_index().items():
        cat = _BABD13_CATEGORY_MAP.get(int(lbl))
        cat_name = cat[1] if cat else "individual_wallet"
        cls = "illicit" if cat and cat[0] == 1 else (
              "licit"   if cat and cat[0] == 0 else "neutral")
        print(f"      {int(lbl):>2}  {cat_name:<32s} {cls:<7s}  {int(n):>8,}")

    df["label"] = df["label"].astype(int)
    df["category_class"] = df["label"].map(
        {k: v[0] for k, v in _BABD13_CATEGORY_MAP.items()}
    )
    df = df.dropna(subset=["category_class"])
    df["confirmed_label"] = df["category_class"].astype(int)

    result = df[["account", "confirmed_label"]].rename(columns={"account": "address"})
    result = result.drop_duplicates(subset=["address"])
    n_ill = int((result["confirmed_label"] == 1).sum())
    n_lit = int((result["confirmed_label"] == 0).sum())
    print(f"    {len(result):,} addresses kept  "
          f"(illicit={n_ill:,}, licit={n_lit:,})")
    return result


# ── Source 3: Bitcoin Abuse Database ──────────────────────────────────────────

def load_bitcoin_abuse() -> pd.DataFrame:
    """
    Load reported addresses from the Bitcoin Abuse Database public dump.
    All addresses are illicit by definition.

    Returns DataFrame[address, confirmed_label].
    """
    if not os.path.exists(_BITCOIN_ABUSE_CSV):
        print(f"  [WARN] Bitcoin Abuse dump not found: {_BITCOIN_ABUSE_CSV}")
        return _empty_lookup()

    print(f"  Loading Bitcoin Abuse from {_BITCOIN_ABUSE_CSV}")
    try:
        df = pd.read_csv(_BITCOIN_ABUSE_CSV, usecols=["address"], dtype=str)
    except (ValueError, OSError) as exc:
        print(f"  [WARN] Failed to load Bitcoin Abuse CSV: {exc}")
        return _empty_lookup()

    df = df.dropna(subset=["address"])
    df["address"] = df["address"].str.strip()
    df = df[df["address"] != ""].drop_duplicates(subset=["address"])
    df["confirmed_label"] = 1
    print(f"    {len(df):,} unique addresses (all illicit)")
    return df


# ── Source 4: OpenSanctions ───────────────────────────────────────────────────

def load_opensanctions() -> pd.DataFrame:
    """
    Extract Bitcoin addresses from the OpenSanctions consolidated targets file.
    Streams the CSV in chunks so the 436 MB file is not held in memory.
    All extracted addresses are treated as illicit.

    Returns DataFrame[address, confirmed_label].
    """
    if not os.path.exists(_OPENSANCTIONS_CSV):
        print(f"  [WARN] OpenSanctions dump not found: {_OPENSANCTIONS_CSV}")
        return _empty_lookup()

    print(f"  Scanning OpenSanctions identifiers in {_OPENSANCTIONS_CSV}")
    t0 = time.time()
    addresses: set = set()
    rows_seen = 0
    try:
        for chunk in pd.read_csv(_OPENSANCTIONS_CSV,
                                 usecols=["identifiers"],
                                 dtype=str, chunksize=200_000):
            rows_seen += len(chunk)
            ids = chunk["identifiers"].dropna()
            for raw in ids:
                for tok in raw.split(";"):
                    tok = tok.strip()
                    if _BTC_ADDRESS_RE.match(tok):
                        addresses.add(tok)
    except (ValueError, OSError) as exc:
        print(f"  [WARN] Failed to scan OpenSanctions CSV: {exc}")
        return _empty_lookup()

    print(f"    Scanned {rows_seen:,} rows in {time.time()-t0:.1f} s — "
          f"{len(addresses):,} unique BTC addresses extracted")
    if not addresses:
        return _empty_lookup()

    return pd.DataFrame({
        "address": list(addresses),
        "confirmed_label": 1,
    })


# ── Master lookup table ────────────────────────────────────────────────────────

def build_lookup(refresh: bool = False) -> pd.DataFrame:
    """
    Construct a deduplicated master address → (label, source-list) lookup.
    Conflict resolution: illicit (1) wins over licit (0); sources are
    concatenated in the ``source`` field so we can attribute matches.

    Returns DataFrame[address, confirmed_label, source].
    """
    print("\n── Loading external label sources ───────────────────────")

    sources = {
        SRC_ELLIPTIC:      load_elliptic(),
        SRC_BABD13:        load_babd13(refresh=refresh),
        SRC_BITCOIN_ABUSE: load_bitcoin_abuse(),
        SRC_OPENSANCTIONS: load_opensanctions(),
    }

    frames = []
    for src_name, df in sources.items():
        if df.empty:
            continue
        df = df.copy()
        df["source"] = src_name
        frames.append(df[["address", "confirmed_label", "source"]])

    if not frames:
        print("\n[WARN] No external label data available from any source.")
        return pd.DataFrame(columns=["address", "confirmed_label", "source"])

    combined = pd.concat(frames, ignore_index=True)
    combined["address"] = combined["address"].astype(str).str.strip()
    combined = combined[combined["address"] != ""]

    # Per-source counts before dedup (for the final summary).
    src_pre = (
        combined.groupby("source")["confirmed_label"]
        .agg(illicit=lambda s: int((s == 1).sum()),
             licit  =lambda s: int((s == 0).sum()),
             total  ="count")
        .reset_index()
    )

    # Deduplicate: keep illicit (1) over licit (0) on conflict.
    def _resolve(grp: pd.DataFrame) -> pd.Series:
        label = 1 if (grp["confirmed_label"] == 1).any() else 0
        srcs  = ";".join(sorted(grp["source"].unique()))
        return pd.Series({"confirmed_label": label, "source": srcs})

    lookup = (
        combined
        .groupby("address", sort=False)
        .apply(_resolve, include_groups=False)
        .reset_index()
    )

    print("\n── Per-source contributions (pre-dedup) ────────────────")
    for _, row in src_pre.iterrows():
        print(f"  {row['source']:<18s}  illicit={int(row['illicit']):>8,}  "
              f"licit={int(row['licit']):>8,}  total={int(row['total']):>8,}")

    print("\n── Master lookup table (post-dedup) ────────────────────")
    print(f"  Total unique addresses : {len(lookup):,}")
    print(f"  Confirmed illicit      : {int((lookup['confirmed_label']==1).sum()):,}")
    print(f"  Confirmed licit        : {int((lookup['confirmed_label']==0).sum()):,}")
    return lookup


# ── Transaction matching ───────────────────────────────────────────────────────

def match_transactions(feat_df: pd.DataFrame, td: pd.DataFrame,
                       lookup: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction in *feat_df*, check whether any of its input or output
    addresses appear in *lookup*.

    Adds three columns:
        confirmed_label       — int (1/0) or NaN
        confirmation_source   — str (";"-joined source list) or NaN
        confirmation_strength — "direct" or NaN
    """
    if lookup.empty:
        feat_df = feat_df.copy()
        feat_df["confirmed_label"]       = np.nan
        feat_df["confirmation_source"]   = np.nan
        feat_df["confirmation_strength"] = np.nan
        return feat_df

    lookup_map = lookup.set_index("address")
    lookup_set = set(lookup_map.index)

    print("  Building per-transaction address index…")
    keys = ["Block_No", "Trans_No"]
    in_addrs  = (
        td.dropna(subset=["Input_Address"])
        .groupby(keys, sort=False)["Input_Address"]
        .apply(set).rename("in_addrs")
    )
    out_addrs = (
        td.dropna(subset=["Output_Address"])
        .groupby(keys, sort=False)["Output_Address"]
        .apply(set).rename("out_addrs")
    )
    tx_addrs = pd.concat([in_addrs, out_addrs], axis=1).reset_index()
    in_clean  = [x if isinstance(x, set) else set() for x in tx_addrs["in_addrs"]]
    out_clean = [x if isinstance(x, set) else set() for x in tx_addrs["out_addrs"]]
    tx_addrs["all_addrs"] = [a | b for a, b in zip(in_clean, out_clean)]

    tx_addrs_dict = {
        (row["Block_No"], row["Trans_No"]):
            (row["all_addrs"] if isinstance(row["all_addrs"], set) else set())
        for _, row in tx_addrs.iterrows()
    }

    print(f"  Matching {len(feat_df):,} transactions against "
          f"{len(lookup_set):,} known addresses…")

    confirmed_labels:  list = []
    confirmation_srcs: list = []
    confirmation_strs: list = []

    for _, row in feat_df[["Block_No", "Trans_No"]].iterrows():
        addrs   = tx_addrs_dict.get((row["Block_No"], row["Trans_No"]), set())
        matched = addrs & lookup_set

        if not matched:
            confirmed_labels.append(np.nan)
            confirmation_srcs.append(np.nan)
            confirmation_strs.append(np.nan)
            continue

        matched_rows = lookup_map.loc[list(matched)]
        label = 1 if (matched_rows["confirmed_label"] == 1).any() else 0
        # Source field is already ";"-joined; collect every source attributed
        # to any matched address, keep them deduped and sorted.
        all_srcs = set()
        for s in matched_rows["source"]:
            all_srcs.update(s.split(";"))
        src_str = ";".join(sorted(all_srcs))

        confirmed_labels.append(label)
        confirmation_srcs.append(src_str)
        confirmation_strs.append("direct")

    feat_df = feat_df.copy()
    feat_df["confirmed_label"]       = confirmed_labels
    feat_df["confirmation_source"]   = confirmation_srcs
    feat_df["confirmation_strength"] = confirmation_strs
    return feat_df


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    total       = len(df)
    if total == 0:
        print("\n[WARN] Feature matrix is empty — nothing to summarise.")
        return

    n_confirmed = int(df["confirmed_label"].notna().sum())
    n_illicit   = int((df["confirmed_label"] == 1).sum())
    n_licit     = int((df["confirmed_label"] == 0).sum())

    print(f"\n── Enrichment summary ───────────────────────────────────")
    print(f"  Total transactions         : {total:>10,}")
    print(f"  With confirmed label       : {n_confirmed:>10,}  "
          f"({n_confirmed/total:.2%})")
    print(f"    Confirmed illicit (1)    : {n_illicit:>10,}")
    print(f"    Confirmed licit   (0)    : {n_licit:>10,}")
    print(f"  No confirmed match         : {total-n_confirmed:>10,}")

    if n_confirmed == 0:
        return

    # Per-source breakdown (illicit / licit / total transactions matched).
    # Each transaction's source list is split, so the same transaction can
    # contribute to multiple source rows.
    matched = df.dropna(subset=["confirmation_source"]).copy()
    matched["src_list"] = matched["confirmation_source"].str.split(";")
    by_src_pairs = (
        matched[["src_list", "confirmed_label"]]
        .explode("src_list")
        .rename(columns={"src_list": "source"})
    )
    by_src = pd.crosstab(by_src_pairs["source"], by_src_pairs["confirmed_label"])
    print(f"\n── Transactions matched by source ──────────────────────")
    for src in sorted(by_src.index):
        ill = int(by_src.loc[src].get(1.0, 0))
        lit = int(by_src.loc[src].get(0.0, 0))
        print(f"  {src:<18s}  illicit={ill:>8,}  licit={lit:>8,}  "
              f"total={ill+lit:>8,}")

    # Specific Elliptic++ illicit-count diagnostic (was the prior-run bug).
    ell_ill = int(((df["confirmation_source"]
                    .fillna("")
                    .str.split(";")
                    .apply(lambda lst: SRC_ELLIPTIC in lst)) &
                   (df["confirmed_label"] == 1)).sum())
    print(f"\n[DIAG] Elliptic++ illicit transaction matches: {ell_ill:,}  "
          f"({'OK' if ell_ill > 1 else 'STILL BROKEN — investigate'})")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich transaction features with confirmed external labels."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-extract BABD-13.csv from archive.zip even if a copy is cached.",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Bitcoin Illicit Detection — External Label Enrichment")
    print("=" * 55 + "\n")

    if not os.path.exists(FEATURES_CSV):
        print(f"[ERROR] Feature matrix not found: {FEATURES_CSV}")
        print("  Run 02_feature.py first.")
        sys.exit(1)

    print(f"Loading feature matrix from {FEATURES_CSV}…")
    feat_df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(feat_df):,} transactions loaded\n")

    if not os.path.exists(TRANS_DETAIL_CSV):
        print(f"[ERROR] Transaction detail file not found: {TRANS_DETAIL_CSV}")
        sys.exit(1)

    print(f"Loading transaction detail from {TRANS_DETAIL_CSV}…")
    td = pd.read_csv(
        TRANS_DETAIL_CSV,
        usecols=["Block_No", "Trans_No", "Input_Address", "Output_Address"],
    )
    print(f"  {len(td):,} detail rows loaded\n")

    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    lookup = build_lookup(refresh=args.refresh)

    print("\n── Matching transactions ────────────────────────────────")
    t0       = time.time()
    enriched = match_transactions(feat_df, td, lookup)
    print(f"  Matching completed in {time.time()-t0:.1f} s")

    print_summary(enriched)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enriched.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Enriched features saved → {OUT_CSV}")
    print(f"     Shape: {enriched.shape[0]:,} rows × {enriched.shape[1]} cols")
