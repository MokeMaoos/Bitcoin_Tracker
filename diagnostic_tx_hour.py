"""
diagnostic_tx_hour.py — read-only analysis of the tx_hour feature.

Question: is tx_hour's high gain in the LightGBM model a real
behavioural signal (illicit txs cluster at specific hours) or a snapshot
artefact (block-extraction times bias the broadcast-hour distribution)?

Sources:
    outputs/labeled_features.csv  — tx_hour, label_type, confirmed_label
    data/Block_Header.csv         — Block_No → DT for snapshot bucketing

Writes: outputs/tx_hour_diagnostic.md
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

OUT_MD = "outputs/tx_hour_diagnostic.md"


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt_pct(x):
    return f"{x*100:.2f}%" if x == x else "—"


def main():
    print("Loading data…")
    df = pd.read_csv("outputs/labeled_features.csv",
                     usecols=["Block_No", "tx_hour", "label_type",
                              "confirmed_label"])
    df = df.dropna(subset=["tx_hour"])
    df["tx_hour"] = df["tx_hour"].astype(int)

    bh = pd.read_csv("data/Block_Header.csv",
                     usecols=["Block_No", "DT", "TM"], dtype=str)
    bh["Block_No"] = pd.to_numeric(bh["Block_No"], errors="coerce").astype("Int64")
    bh = bh.dropna(subset=["Block_No"])
    bh["block_dt"]   = pd.to_datetime(bh["DT"], errors="coerce", format="mixed")
    bh["block_hour"] = pd.to_numeric(bh["TM"].str.slice(0, 2), errors="coerce")
    block_dt_map  = dict(zip(bh["Block_No"].astype(int),
                             bh["block_dt"].dt.strftime("%Y-%m-%d")))
    block_hr_map  = dict(zip(bh["Block_No"].astype(int),
                             bh["block_hour"].astype("Int64")))

    df["snapshot"]   = pd.to_numeric(df["Block_No"],
                                     errors="coerce").map(block_dt_map)
    df["block_hour"] = pd.to_numeric(df["Block_No"],
                                     errors="coerce").map(block_hr_map)
    df = df.dropna(subset=["snapshot"])

    illicit_mask = (df["label_type"] == "confirmed") & (df["confirmed_label"] == 1)
    licit_mask   = (df["label_type"] == "confirmed") & (df["confirmed_label"] == 0)
    df_illicit = df[illicit_mask]
    df_licit   = df[licit_mask]

    n_all = len(df)
    n_il  = len(df_illicit)
    n_lt  = len(df_licit)
    print(f"  Loaded: {n_all:,} rows")
    print(f"  Confirmed illicit: {n_il:,}")
    print(f"  Confirmed licit:   {n_lt:,}")

    # ── Task 1: Hour distribution per class ──────────────────────────────────
    rows_dist = []
    for h in range(24):
        c_all = int((df["tx_hour"] == h).sum())
        c_il  = int((df_illicit["tx_hour"] == h).sum())
        c_lt  = int((df_licit["tx_hour"] == h).sum())
        rows_dist.append([
            h,
            f"{c_all:,}", fmt_pct(c_all / n_all),
            f"{c_il:,}",  fmt_pct(c_il / max(n_il, 1)),
            f"{c_lt:,}",  fmt_pct(c_lt / max(n_lt, 1)),
        ])
    dist_table = md_table(
        rows_dist,
        ["hour", "all (n)", "all (%)",
         "illicit (n)", "illicit (%)",
         "licit (n)", "licit (%)"])

    # ── Task 2: Chi-square test ──────────────────────────────────────────────
    contingency = np.zeros((2, 24), dtype=int)
    for h in range(24):
        contingency[0, h] = int((df_illicit["tx_hour"] == h).sum())
        contingency[1, h] = int((df_licit["tx_hour"] == h).sum())
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    chi_text = (
        f"- chi-square statistic: **{chi2:.4f}**\n"
        f"- degrees of freedom : **{dof}**\n"
        f"- p-value            : **{p:.6e}**\n"
        f"- Cramér's V (effect): **{cramers_v:.4f}**\n"
        f"- n (illicit + licit): **{n:,}**\n\n"
        f"Significance threshold p < 0.001 — "
        f"{'**rejects null** (distributions differ)' if p < 0.001 else '**fails to reject null** (no detectable difference)'}"
    )
    print(f"  chi2 = {chi2:.2f}, p = {p:.4e}, Cramér's V = {cramers_v:.4f}")

    # ── Task 3: Per-snapshot hour range ──────────────────────────────────────
    snap_rows = []
    for snap in sorted(df["snapshot"].unique()):
        sub = df[df["snapshot"] == snap]
        bh_sub = bh[bh["block_dt"].dt.strftime("%Y-%m-%d") == snap]
        snap_rows.append([
            snap,
            f"{len(sub):,}",
            int(bh_sub["block_hour"].min()),
            int(bh_sub["block_hour"].max()),
            int(bh_sub["block_hour"].astype(int).std() or 0),
            int(sub["tx_hour"].min()),
            int(sub["tx_hour"].max()),
            f"{sub['tx_hour'].astype(int).std():.2f}",
        ])
    snap_table = md_table(
        snap_rows,
        ["snapshot", "n_tx", "block hr min", "block hr max", "block hr std",
         "tx_hour min", "tx_hour max", "tx_hour std"])

    # ── Task 4: Within-snapshot illicit vs licit (3 representative days) ────
    chosen = ["2020-03-08", "2022-06-08", "2024-12-08"]
    within_blocks = []
    for snap in chosen:
        sub_il = df_illicit[df_illicit["snapshot"] == snap]
        sub_lt = df_licit  [df_licit  ["snapshot"] == snap]
        n_si = len(sub_il); n_sl = len(sub_lt)
        rows = []
        for h in range(24):
            c_il = int((sub_il["tx_hour"] == h).sum())
            c_lt = int((sub_lt["tx_hour"] == h).sum())
            rows.append([
                h,
                f"{c_il:,}", fmt_pct(c_il / max(n_si, 1)),
                f"{c_lt:,}", fmt_pct(c_lt / max(n_sl, 1)),
            ])
        snap_chi_table = md_table(
            rows,
            ["hour", "illicit (n)", "illicit (%)", "licit (n)", "licit (%)"])

        # Per-snapshot chi-square
        if n_si > 0 and n_sl > 0:
            ct = np.zeros((2, 24), dtype=int)
            for h in range(24):
                ct[0, h] = int((sub_il["tx_hour"] == h).sum())
                ct[1, h] = int((sub_lt["tx_hour"] == h).sum())
            mask = (ct.sum(axis=0) > 0)
            ct_clean = ct[:, mask]
            try:
                chi2_s, p_s, dof_s, _ = chi2_contingency(ct_clean)
                ns = ct_clean.sum()
                v_s = np.sqrt(chi2_s / (ns * (min(ct_clean.shape) - 1)))
                chi_line = (
                    f"chi-square = {chi2_s:.4f}, dof = {dof_s}, "
                    f"p = {p_s:.4e}, Cramér's V = {v_s:.4f}, "
                    f"n_illicit = {n_si:,}, n_licit = {n_sl:,}")
            except ValueError as exc:
                chi_line = f"chi-square not computable ({exc})"
        else:
            chi_line = (f"Skipped — illicit n = {n_si}, licit n = {n_sl}; "
                        "need both > 0.")
        within_blocks.append(
            f"### {snap}\n\n{chi_line}\n\n{snap_chi_table}\n")

    # ── Conclusion logic ─────────────────────────────────────────────────────
    real_signal_per_snap = 0
    for snap in chosen:
        sub_il = df_illicit[df_illicit["snapshot"] == snap]
        sub_lt = df_licit  [df_licit  ["snapshot"] == snap]
        if len(sub_il) > 100 and len(sub_lt) > 100:
            ct = np.zeros((2, 24), dtype=int)
            for h in range(24):
                ct[0, h] = int((sub_il["tx_hour"] == h).sum())
                ct[1, h] = int((sub_lt["tx_hour"] == h).sum())
            mask = (ct.sum(axis=0) > 0)
            ct_clean = ct[:, mask]
            chi2_s, p_s, _, _ = chi2_contingency(ct_clean)
            ns = ct_clean.sum()
            v_s = np.sqrt(chi2_s / (ns * (min(ct_clean.shape) - 1)))
            if p_s < 0.001 and v_s > 0.05:
                real_signal_per_snap += 1

    if cramers_v < 0.05:
        verdict = ("**Ambiguous / weak.** Even though the chi-square p-value "
                   "is significant on the 8 M-row pool, Cramér's V is small "
                   "(< 0.05). The class-wise hour distributions are nearly "
                   "identical at the population level. The model's high "
                   "feature gain on tx_hour likely reflects per-snapshot "
                   "variation rather than a stable behavioural signal.")
    elif real_signal_per_snap >= 2:
        verdict = ("**Real signal.** Within individual snapshots the illicit "
                   "and licit hour distributions differ at p < 0.001 with "
                   f"Cramér's V > 0.05 in {real_signal_per_snap} of "
                   f"{len(chosen)} sampled snapshots. The hour-of-day pattern "
                   "is replicated within single days, not just across them.")
    elif real_signal_per_snap == 0:
        verdict = ("**Snapshot artefact.** The whole-population chi-square "
                   "is significant, but within individual snapshots the "
                   "illicit and licit hour distributions are statistically "
                   "indistinguishable. The high LightGBM gain on tx_hour is "
                   "therefore driven by *cross-snapshot* drift in the hour "
                   "distribution (e.g. confirmation-time bias of certain "
                   "snapshot days), not a within-day behavioural difference.")
    else:
        verdict = ("**Mixed / inconclusive.** Some snapshots show a real "
                   "within-day signal, others do not. Recommend an additional "
                   "robustness check before reporting tx_hour as a behavioural "
                   "feature in the dissertation.")

    # ── Write report ─────────────────────────────────────────────────────────
    md = []
    md.append("# tx_hour diagnostic — leakage vs real signal\n")
    md.append(f"**Generated**: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
    md.append(f"\n**Source files**:\n"
              f"- `outputs/labeled_features.csv` "
              f"(8.34 M rows; tx_hour from broadcast TM)\n"
              f"- `data/Block_Header.csv` "
              f"(snapshot date and block-confirmation hour)\n")
    md.append(f"\n**Sample sizes**:\n"
              f"- All transactions: {n_all:,}\n"
              f"- Confirmed illicit: {n_il:,}\n"
              f"- Confirmed licit:   {n_lt:,}\n")
    md.append("\n## 1. Hour distribution comparison\n")
    md.append(dist_table)
    md.append("\n*tx_hour is taken from the broadcast/origination TM column; "
              "it is unaffected by the earlier dayfirst-parsing bug, which "
              "only scrambled the date portion.*\n")
    md.append("\n## 2. Chi-square test (illicit vs licit, whole sample)\n")
    md.append(chi_text + "\n")
    md.append("\n## 3. Per-snapshot hour range\n")
    md.append("**block hr** = hour of confirmation per Block_Header.TM. "
              "**tx_hour** = broadcast hour from Transactions.TM "
              "(distribution within the snapshot's transactions).\n\n")
    md.append(snap_table)
    md.append("\n## 4. Within-snapshot illicit vs licit comparison\n")
    md.append("Three representative snapshots — checking whether the "
              "illicit/licit hour distributions differ *within a single "
              "day*. If they do, the signal is real; if they don't, the "
              "feature is exploiting cross-snapshot drift.\n")
    md.extend(within_blocks)
    md.append("\n## 5. Conclusion\n")
    md.append(verdict + "\n")
    md.append("\n---\n")
    md.append(f"**Saved**: `{os.path.abspath(OUT_MD)}`\n")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))

    print(f"\n[OK] Diagnostic written to: {os.path.abspath(OUT_MD)}")


if __name__ == "__main__":
    main()
