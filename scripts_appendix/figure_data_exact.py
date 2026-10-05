"""
figure_data_exact.py — produce exact numerical CSV blocks for Chapter 4 figures.

Reads from:
    outputs/labeled_features.csv          (rule_* flags, label_type, confirmed_label)
    outputs/enriched_features.csv         (confirmation_source per tx)
    outputs/model.pkl                     (trained LightGBM booster)
    data/Block_Header.csv                 (Block_No → snapshot date)
    data/Trans_Detail.csv                 (per-tx input/output addresses)
    data/external_labels/BABD-13.csv      (account → category label)

Writes outputs/figure_data_exact.txt with three labelled blocks:
    BLOCK 1 — Figure 4.2 per-quarter rule firing rates (24 × 6, 1 dp)
    BLOCK 2 — Figure 4.4 Panel B top-18 LightGBM feature gains (integer)
    BLOCK 3 — Figure 4.8 Panel B illicit-category composition per source
"""

from __future__ import annotations

import datetime as dt
import io
import os
import time

import joblib
import pandas as pd

OUT_TXT = "outputs/figure_data_exact.txt"

# BABD-13 numeric label → (illicit/licit, category name)
# Mirrors _BABD13_CATEGORY_MAP in 02b_enrich_labels.py.
BABD13_MAP = {
    0:  (1, "Blackmail"),
    1:  (0, "Cyber-Security Service"),
    2:  (1, "Darknet Market"),
    3:  (0, "Centralized Exchange"),
    4:  (0, "P2P Financial Infrastructure"),
    5:  (0, "P2P Financial Service"),
    6:  (1, "Gambling"),
    7:  (1, "Government Criminal Blocklist"),
    8:  (1, "Money Laundering"),
    9:  (1, "Ponzi Scheme"),
    10: (0, "Mining Pool"),
    11: (1, "Tumbler"),
    # 12 = Individual Wallet — neutral, excluded from the lookup
}
ILLICIT_LABELS = {k for k, v in BABD13_MAP.items() if v[0] == 1}


def block1_rule_firing_per_quarter() -> str:
    print("[1/3] Per-quarter rule firing rates…")
    bh = pd.read_csv("data/Block_Header.csv",
                     usecols=["Block_No", "DT"], dtype=str)
    bh["Block_No"] = pd.to_numeric(bh["Block_No"], errors="coerce").astype("Int64")
    bh["dt"] = pd.to_datetime(bh["DT"], errors="coerce", format="mixed")
    bh = bh.dropna(subset=["Block_No", "dt"])
    bh["period"] = bh["dt"].dt.year.astype(str) + "-Q" + (
        ((bh["dt"].dt.month - 1) // 3 + 1).astype(str))
    period_map = dict(zip(bh["Block_No"].astype(int), bh["period"]))

    rule_cols = ["rule_equal_value", "rule_mixing", "rule_peeling",
                 "rule_high_fan_out", "rule_high_entropy",
                 "rule_fresh_outputs"]
    df = pd.read_csv("outputs/labeled_features.csv",
                     usecols=["Block_No"] + rule_cols)
    df["period"] = pd.to_numeric(df["Block_No"],
                                  errors="coerce").map(period_map)
    df = df.dropna(subset=["period"])

    grp = df.groupby("period")
    n_per = grp.size().rename("n_tx")
    rates = grp[rule_cols].mean() * 100  # firing rate as %

    out = io.StringIO()
    out.write("period,equal_value,mixing,peeling,high_fan_out,"
              "high_entropy,fresh_outputs\n")
    for p in sorted(rates.index):
        row = rates.loc[p]
        out.write(",".join([
            p,
            f"{row['rule_equal_value']:.1f}",
            f"{row['rule_mixing']:.1f}",
            f"{row['rule_peeling']:.1f}",
            f"{row['rule_high_fan_out']:.1f}",
            f"{row['rule_high_entropy']:.1f}",
            f"{row['rule_fresh_outputs']:.1f}",
        ]) + "\n")
    print(f"      {len(rates)} periods, "
          f"sum of n_tx = {n_per.sum():,}")
    return out.getvalue()


def block2_feature_gains_top18() -> str:
    print("[2/3] Top-18 LightGBM feature gains…")
    bundle = joblib.load("outputs/model.pkl")
    booster = bundle["model"].named_steps["clf"].booster_
    gains = booster.feature_importance(importance_type="gain")
    feats = bundle["features"]
    df = pd.DataFrame({"feature": feats, "gain": gains})
    df = df.sort_values("gain", ascending=False).head(18)

    out = io.StringIO()
    out.write("feature,gain\n")
    for _, r in df.iterrows():
        # LightGBM's gain is float64, not int. Emit full-precision repr so
        # the value round-trips exactly; user can format/round at display
        # time if desired.
        out.write(f"{r['feature']},{repr(float(r['gain']))}\n")
    print(f"      top-18 head: {df.head(3)['feature'].tolist()}")
    return out.getvalue()


def block3_illicit_category_per_source() -> str:
    print("[3/3] Per-source illicit-category composition…")

    # ── Elliptic++ ──────────────────────────────────────────────────────────
    # The Elliptic++ wallets_classes.csv carries only a binary class label
    # (1 = illicit, 2 = licit, 3 = unknown). It does not publish a
    # finer-grained illicit-category taxonomy; the original Weber et al.
    # (2019) Elliptic dataset documentation stated that illicit covered
    # categories such as scams, ransomware, dark-market activity, and
    # exchange theft, but the public release does not record which row
    # belongs to which category. We therefore report Elliptic++ as a single
    # "illicit (unspecified)" group and state the limitation explicitly.
    enriched = pd.read_csv(
        "outputs/enriched_features.csv",
        usecols=["Block_No", "confirmed_label", "confirmation_source"])
    enriched = enriched.dropna(subset=["confirmation_source"])
    enriched["sources"] = enriched["confirmation_source"].str.split(";")

    elliptic_ill_count = int((
        (enriched["confirmed_label"] == 1)
        & enriched["sources"].apply(lambda lst: "EllipticPlusPlus" in lst)
    ).sum())

    # ── BABD-13 ─────────────────────────────────────────────────────────────
    # Identify confirmed-illicit transactions whose match list includes
    # BABD-13.  Then walk Trans_Detail to find every BABD-13-illicit-category
    # address that appears as an input or output of those transactions.  A
    # transaction is allocated to the BABD-13 category of its first matching
    # address (sorted by category id) so that per-category counts sum to
    # the total without double-counting; the breakdown is otherwise
    # agnostic to the order of categories.

    babd_il_keys = enriched[
        (enriched["confirmed_label"] == 1)
        & enriched["sources"].apply(lambda lst: "BABD-13" in lst)
    ][["Block_No"]].copy()
    babd_il_total = len(babd_il_keys)
    print(f"      confirmed-illicit txs matched by BABD-13: {babd_il_total:,}")

    # Need (Block_No, Trans_No) keys for filtering Trans_Detail
    babd_il_keys = pd.read_csv(
        "outputs/enriched_features.csv",
        usecols=["Block_No", "Trans_No", "confirmed_label",
                 "confirmation_source"],
    )
    babd_il_keys = babd_il_keys.dropna(subset=["confirmation_source"])
    mask = (
        (babd_il_keys["confirmed_label"] == 1)
        & babd_il_keys["confirmation_source"].str.split(";").apply(
            lambda lst: "BABD-13" in lst)
    )
    babd_il_keys = babd_il_keys.loc[mask, ["Block_No", "Trans_No"]].copy()
    babd_il_keys["Block_No"] = babd_il_keys["Block_No"].astype(int)
    babd_il_keys["Trans_No"] = babd_il_keys["Trans_No"].astype(str)
    keys = set(zip(babd_il_keys["Block_No"], babd_il_keys["Trans_No"]))
    print(f"      target tx keys: {len(keys):,}")

    # Load BABD-13 illicit addresses → category label
    babd = pd.read_csv("data/external_labels/BABD-13.csv",
                       usecols=["account", "label"],
                       dtype={"account": str, "label": "Int64"})
    babd = babd.dropna(subset=["account", "label"])
    babd = babd[babd["label"].isin(ILLICIT_LABELS)]
    babd_addr_to_label = dict(zip(babd["account"], babd["label"].astype(int)))
    print(f"      BABD-13 illicit-category addresses: "
          f"{len(babd_addr_to_label):,}")

    # Stream Trans_Detail and gather per-tx category memberships
    tx_cats: dict[tuple[int, str], set[int]] = {}
    t0 = time.time()
    rows_seen = 0
    for chunk in pd.read_csv(
            "data/Trans_Detail.csv",
            usecols=["Block_No", "Trans_No",
                     "Input_Address", "Output_Address"],
            dtype={"Block_No": "int64", "Trans_No": str,
                   "Input_Address": str, "Output_Address": str},
            chunksize=1_000_000, low_memory=False):
        rows_seen += len(chunk)
        # Quick filter: only rows whose (Block_No, Trans_No) is in target keys
        ch_keys = list(zip(chunk["Block_No"].astype(int),
                            chunk["Trans_No"].astype(str)))
        keep_mask = pd.Series(
            [k in keys for k in ch_keys], index=chunk.index)
        sub = chunk[keep_mask]
        for _, row in sub.iterrows():
            k = (int(row["Block_No"]), row["Trans_No"])
            for col in ("Input_Address", "Output_Address"):
                addr = row[col]
                if pd.isna(addr):
                    continue
                cat = babd_addr_to_label.get(addr)
                if cat is not None:
                    tx_cats.setdefault(k, set()).add(cat)
    print(f"      streamed {rows_seen:,} detail rows "
          f"in {time.time()-t0:.1f}s")
    print(f"      txs with at least one BABD-13 illicit category match: "
          f"{len(tx_cats):,}")

    # Allocate each tx to the lowest-numbered illicit category it matched
    # (deterministic, stable across reruns, reproducible).
    primary_cat = {k: min(v) for k, v in tx_cats.items()}
    counts: dict[int, int] = {lbl: 0 for lbl in ILLICIT_LABELS}
    for k, c in primary_cat.items():
        counts[c] += 1
    matched_total = sum(counts.values())
    unmatched = babd_il_total - matched_total
    print(f"      categorised: {matched_total:,}; "
          f"unmatched: {unmatched:,}")

    # ── Build the CSV block ─────────────────────────────────────────────────
    out = io.StringIO()
    out.write("source,category,count,percentage\n")

    # Elliptic++ — single unspecified illicit row
    out.write(f"Elliptic++,Illicit (unspecified — Elliptic++ does not "
              f"publish illicit sub-categories),{elliptic_ill_count},100.0\n")

    # BABD-13 — per-category breakdown sorted by count descending
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    babd_total = sum(counts.values()) + (unmatched if unmatched > 0 else 0)
    # Percentages should sum to 100 across the BABD-13 rows.
    for lbl, n in items:
        cat_name = BABD13_MAP[lbl][1]
        pct = 100.0 * n / max(babd_total, 1)
        out.write(f"BABD-13,{cat_name} (label={lbl}),{n},{pct:.1f}\n")
    if unmatched > 0:
        pct = 100.0 * unmatched / babd_total
        out.write(
            f"BABD-13,Unresolved (matched at the address level via "
            f"BABD-13's licit-class addresses or via Trans_Detail rows "
            f"with null addresses; primary illicit category not "
            f"recoverable),{unmatched},{pct:.1f}\n")
    return out.getvalue()


def main():
    block1 = block1_rule_firing_per_quarter()
    block2 = block2_feature_gains_top18()
    block3 = block3_illicit_category_per_source()

    today = dt.date.today().isoformat()
    header = (
        f"# figure_data_exact.txt — exact numerical values for "
        f"Chapter 4 figures\n"
        f"# Generated: {today} from outputs/labeled_features.csv, "
        f"outputs/enriched_features.csv, outputs/model.pkl, "
        f"data/Block_Header.csv, data/Trans_Detail.csv, "
        f"data/external_labels/BABD-13.csv "
        f"(produced by 02_feature.py / 02b_enrich_labels.py / "
        f"03_label.py / 04_train.py).\n\n"
    )

    body = (
        header
        + "## BLOCK 1 — Figure 4.2 per-quarter rule firing rates "
          "(% of transactions in that snapshot, 1 dp)\n"
        + block1 + "\n"
        + "## BLOCK 2 — Figure 4.4 Panel B top-18 LightGBM feature gains "
          "(raw float64 gain from booster.feature_importance(type='gain'); "
          "values are full-precision repr — round at display time if needed)"
          "\n"
        + block2 + "\n"
        + "## BLOCK 3 — Figure 4.8 Panel B illicit-category composition by "
          "source (confirmed-illicit transactions, primary BABD-13 "
          "category = lowest-numbered illicit label among matched "
          "addresses; Elliptic++ does not publish illicit sub-categories "
          "in wallets_classes.csv and is reported as a single row)\n"
        + block3 + "\n"
    )

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(body)
    print(f"\n[OK] wrote {os.path.abspath(OUT_TXT)}")


if __name__ == "__main__":
    main()
