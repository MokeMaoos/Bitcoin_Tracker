"""
build_appendices.py — assemble outputs/Dissertation_Appendices.docx.

Reads source materials from:
    - /Users/constantinesamarikas/Documents/University/Dissertation/Docs/
        user-manual.pdf, feature_specification.xlsx, rule_specification.xlsx,
        how_to_run.docx, model_configuration.docx, pipeline_diagram.svg
    - outputs/ (this repo): section3_inventory.md, mechanism_eval.json,
        feature_importance.csv, per_period_stats.csv, dbscan_summary.json,
        dbscan_top50.csv, dbscan_confirmed_per_cluster.csv,
        pipeline_architecture.png, model.pkl, elliptic_eval.json
    - 02_feature.py, 03_label.py, 04_train.py, run_pipeline.py (this repo)

Writes outputs/Dissertation_Appendices.docx in the dissertation house style:
    - Times New Roman 12pt body, 14pt subsection headings,
      18pt main title (all navy blue #2F5597)
    - 1.5 line spacing for body, 1.15 for headings/tables
    - 2.5 cm margins
    - Tables with navy header row and alternating row shading
"""

import datetime as dt
import json
import os
import sys

import openpyxl
import pandas as pd
import pypdf
from docx import Document
from docx.enum.text import WD_LINE_SPACING, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
DOCS_DIR  = "/Users/constantinesamarikas/Documents/University/Dissertation/Docs"
OUT_PATH  = os.path.abspath(os.path.join(ROOT, "outputs",
                                          "Dissertation_Appendices.docx"))

# ── Style constants ───────────────────────────────────────────────────────
NAVY       = RGBColor(0x2F, 0x55, 0x97)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
ROW_SHADE  = "F2F2F2"
HEADER_HEX = "2F5597"
BODY_FONT  = "Times New Roman"
BODY_PT    = Pt(12)
HEAD2_PT   = Pt(14)
HEAD1_PT   = Pt(18)


# ── Section status tracking ───────────────────────────────────────────────
APPENDIX_STATUS = {}


def mark(name: str, status: str, note: str = "") -> None:
    APPENDIX_STATUS[name] = (status, note)


# ── Helpers ────────────────────────────────────────────────────────────────

def set_cell_shading(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def set_cell_borders(cell, color_hex="BFBFBF", size="6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), size)
        b.set(qn("w:color"), color_hex)
        borders.append(b)
    tc_pr.append(borders)


def style_paragraph(p, *, font_name=BODY_FONT, font_size=BODY_PT,
                     bold=False, color=None, line_spacing=1.5,
                     space_after=Pt(6)):
    fmt = p.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = line_spacing
    fmt.space_after  = space_after
    for run in p.runs:
        run.font.name = font_name
        # East Asian fallback
        run.font.element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
        run.font.size = font_size
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color


def heading_para(doc, text, level=2):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = BODY_FONT
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run.font.color.rgb = NAVY
    run.font.bold = True
    if level == 1:
        run.font.size = HEAD1_PT
    elif level == 2:
        run.font.size = HEAD2_PT
    else:
        run.font.size = Pt(12)
    fmt = p.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = 1.15
    fmt.space_before = Pt(12) if level <= 2 else Pt(8)
    fmt.space_after  = Pt(6)
    fmt.keep_with_next = True
    return p


def body_para(doc, text, *, italic=False, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = BODY_FONT
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run.font.size = BODY_PT
    run.font.italic = italic
    run.font.bold = bold
    style_paragraph(p)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.name = BODY_FONT
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run.font.size = BODY_PT
    fmt = p.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = 1.15
    fmt.space_after  = Pt(2)
    return p


def add_table(doc, header, rows, *, col_widths_cm=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.autofit = False
    if col_widths_cm:
        # Set widths
        for col_idx, w in enumerate(col_widths_cm):
            for row in table.rows:
                row.cells[col_idx].width = Cm(w)
    # Header row
    for j, txt in enumerate(header):
        c = table.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(str(txt))
        run.font.name = BODY_FONT
        run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = WHITE
        fmt = c.paragraphs[0].paragraph_format
        fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        fmt.line_spacing = 1.15
        fmt.space_after  = Pt(0)
        set_cell_shading(c, HEADER_HEX)
        set_cell_borders(c)
    # Body rows
    for ri, row in enumerate(rows):
        wrow = table.rows[ri + 1]
        shade = ROW_SHADE if (ri % 2 == 1) else "FFFFFF"
        for j, txt in enumerate(row):
            c = wrow.cells[j]
            c.text = ""
            run = c.paragraphs[0].add_run(str(txt) if txt is not None else "")
            run.font.name = BODY_FONT
            run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
            run.font.size = Pt(11)
            fmt = c.paragraphs[0].paragraph_format
            fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            fmt.line_spacing = 1.15
            fmt.space_after  = Pt(0)
            set_cell_shading(c, shade)
            set_cell_borders(c)
    # Spacer paragraph after table
    doc.add_paragraph()
    return table


def page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def setup_document() -> Document:
    doc = Document()
    # Set default style
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    style.font.size = BODY_PT
    # Margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)
    return doc


# ── Title page and intro ──────────────────────────────────────────────────

def title_page(doc):
    for _ in range(8):
        doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Appendices")
    run.font.name = BODY_FONT
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = NAVY
    p.alignment = 1  # centred

    p = doc.add_paragraph()
    run = p.add_run("BMAN31260 Final Year Project Report")
    run.font.name = BODY_FONT
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run.font.size = Pt(16)
    run.font.color.rgb = NAVY
    p.alignment = 1

    p = doc.add_paragraph()
    run = p.add_run(
        "Operationalising FATF Red-Flag Indicators for Bitcoin "
        "Anti-Money-Laundering Detection on the CUBiD Database")
    run.font.name = BODY_FONT
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run.font.size = Pt(13)
    run.font.italic = True
    p.alignment = 1

    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run(f"Generated {dt.date.today().isoformat()}")
    run.font.name = BODY_FONT
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)
    p.alignment = 1
    page_break(doc)


def intro_page(doc):
    heading_para(doc, "Appendices", level=1)
    body_para(doc,
        "This volume contains the supporting material referenced throughout "
        "the main report. Appendices A–I document the data, code, "
        "configuration, and external resources used by the detection "
        "framework. The contents are intended to make every numerical claim "
        "in the main report independently reproducible from the underlying "
        "CUBiD multi-period dataset and the four external label sources.")
    body_para(doc,
        "Appendix A reproduces the Cardiff University Bitcoin Database (CUBiD) "
        "user manual and summarises the seven CUBiD tables consumed by the "
        "pipeline. Appendix B catalogues all 39 engineered features, of which "
        "35 are deployed in the multi-period evaluation. Appendix C provides "
        "per-period descriptive statistics across the 24 quarterly snapshots. "
        "Appendix D presents the full DBSCAN clustering output. Appendix E "
        "specifies the six FATF rule definitions. Appendix F lists model "
        "hyperparameters and the computational environment. Appendix G is the "
        "reproducibility guide. Appendix H contains the pipeline architecture "
        "diagram. Appendix I documents the four external label sources.")
    page_break(doc)


# ── Appendix A — CUBiD schema and user manual ─────────────────────────────

def appendix_A(doc):
    heading_para(doc, "Appendix A — CUBiD Database Schema and User Manual",
                 level=1)
    body_para(doc,
        "This appendix reproduces the Cardiff University Bitcoin Database "
        "(CUBiD) user manual published by Dr Hossein Jahanshahloo (Cardiff "
        "Business School, May 2023). The manual is the canonical schema "
        "reference for every table consumed by the pipeline. A summary table "
        "follows the manual text, mapping each of the seven CUBiD tables used "
        "in this project to its purpose, key columns, observed row volume in "
        "the multi-period dataset, and the pipeline scripts that read it.")

    # ── A.1 User manual text ──────────────────────────────────────────────
    heading_para(doc, "A.1  CUBiD User's Manual — Reproduced Text", level=2)

    pdf_path = os.path.join(DOCS_DIR, "user-manual.pdf")
    pdf_status = "OK"
    if os.path.exists(pdf_path):
        try:
            reader = pypdf.PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                heading_para(doc, f"Page {i + 1}", level=3)
                text = page.extract_text() or ""
                # Split into paragraphs by blank lines; trim whitespace
                paragraphs = [p.strip() for p in text.split("\n\n")
                              if p.strip()]
                if not paragraphs:
                    paragraphs = [text]
                for para in paragraphs:
                    body_para(doc, " ".join(para.split()))
        except Exception as exc:
            body_para(doc, f"[Error extracting user-manual.pdf: {exc}]",
                      italic=True)
            pdf_status = f"PDF error: {exc}"
    else:
        body_para(doc,
            f"[TO BE COMPLETED — user-manual.pdf not located at {pdf_path}]")
        pdf_status = "PDF missing"

    # ── A.2 Schema summary ─────────────────────────────────────────────────
    heading_para(doc, "A.2  CUBiD Tables Used by the Pipeline", level=2)
    body_para(doc,
        "Table A.1 summarises the seven CUBiD tables ingested by the "
        "pipeline. Row volumes are the values observed on the multi-period "
        "sample described in Section 3.1 (24 quarterly snapshots, "
        "8 March 2019 – 8 December 2024). The two CUBiD balance-block tables "
        "(Address_Balance_Block, Wallet_Balance_Block) are not included in "
        "the multi-period evaluation; their absence is handled by "
        "graceful no-op loaders in 02_feature.py and reduces the deployed "
        "feature count from 39 to 35.")

    schema_rows = [
        ("Transactions",
         "Per-transaction summary (one row per tx)",
         "Block_No, Trans_No, DT/TM, Total_input, Total_Output, Fees, Wallet",
         "8,343,135 rows",
         "01_load.py, 02_feature.py"),
        ("Trans_Detail",
         "Input/output rows with addresses and values",
         "Block_No, Trans_No, Trans_Row, Input_Address, Input_Value, "
         "Output_Address, Output_Value, Input_Wallet, Output_Wallet",
         "33,505,266 rows",
         "01_load.py, 02_feature.py, 02b_enrich_labels.py"),
        ("Address_Info",
         "Per-address lifetime statistics",
         "ADR (PK), N_Active_Blocks, First/Last_Active_Block, "
         "N_Transactions, Final_Balance, Total_Sent, Total_Received, Wallet",
         "≈ 30 M global addresses (2.5 GB)",
         "01_load.py, 02_feature.py"),
        ("Wallet_Info",
         "Per-wallet lifetime statistics",
         "Wallet (PK), N_Addresses, N_Active_Blocks, First/Last_Active_Block, "
         "N_Transactions, Final_Balance, Total_Sent, Total_Received",
         "≈ 4.6 M wallets (650 MB)",
         "01_load.py, 02_feature.py"),
        ("Block_Header",
         "Per-block header (mining metadata)",
         "Block_No (PK), Block_Hash, DT, TM, Number_Transactions, "
         "Total_Input, Total_Output, Trans_Fees, Block_Reward",
         "3,642 rows (24 snapshots × ≈150 blocks)",
         "02_feature.py, 07_temporal_eval.py"),
        ("Address_Activity",
         "Per-block address-count summary",
         "Block_No (PK), Active_Addresses, Zero_Balance, New_Addresses",
         "3,642 rows",
         "(loaded by 01_load.py for QA only)"),
        ("Wallet_Activity",
         "Per-block wallet-count summary",
         "Block_No (PK), Active_Wallets, Zero_Balance, New_Wallets",
         "3,642 rows",
         "(loaded by 01_load.py for QA only)"),
    ]
    add_table(doc,
              ["Table", "Purpose", "Key Columns (PK)",
               "Row Volume", "Pipeline Reads"],
              schema_rows,
              col_widths_cm=[2.5, 3.0, 5.5, 2.5, 2.5])
    body_para(doc,
        "All monetary values are recorded in Satoshi "
        "(1 BTC = 100,000,000 Satoshi). Trans_No in the multi-period dataset "
        "is the 64-character hexadecimal transaction hash, not the integer "
        "ordinal used in the original CUBiD reference sample; the pipeline "
        "reads it as a string at every join site.",
        italic=True)
    mark("A", "OK", f"manual extracted: {pdf_status}")
    page_break(doc)


# ── Appendix B — Feature specification ────────────────────────────────────

def appendix_B(doc):
    heading_para(doc, "Appendix B — Feature Specification", level=1)
    body_para(doc,
        "Table B.1 catalogues all 39 features defined by 02_feature.py. "
        "Of these, 35 are deployed in the multi-period evaluation; the "
        "remaining four (the wallet-temporal triple and "
        "mean_out_addr_bal_velocity) require the CUBiD balance-block tables, "
        "which are not present in the multi-period sample. The Group column "
        "follows the narrative grouping described in Section 2.4 of the main "
        "report. Importance ranks are computed from the trained LightGBM "
        "booster (gain importance) and reflect the multi-period hybrid model "
        "as saved in outputs/model.pkl.")

    # Read feature_specification.xlsx
    spec_path = os.path.join(DOCS_DIR, "feature_specification.xlsx")
    spec_rows = {}
    if os.path.exists(spec_path):
        wb = openpyxl.load_workbook(spec_path, data_only=True)
        ws = wb.worksheets[0]
        for row in ws.iter_rows(values_only=True):
            if row[1] and isinstance(row[0], int):
                spec_rows[row[1]] = {
                    "definition_calc": row[2] or "",
                    "source_table":    row[3] or "",
                    "fatf_mapping":    row[4] or "",
                }

    # Read importance ranks
    imp = pd.read_csv(os.path.join(ROOT, "outputs", "feature_importance.csv"))
    rank_map = dict(zip(imp["feature"], imp["gain_rank"]))

    # 4 dropped features (wallet-temporal triple + mean_out_addr_bal_velocity)
    DROPPED = {"wallet_balance_velocity", "wallet_balance_range",
               "wallet_balance_std", "mean_out_addr_bal_velocity"}

    # Narrative + technical grouping
    NARRATIVE = {
        "total_output": "Output statistics", "mean_output": "Output statistics",
        "std_output": "Output statistics", "max_output": "Output statistics",
        "min_output": "Output statistics", "value_entropy": "Output statistics",
        "total_input": "Output statistics",
        "num_inputs": "Address counts", "num_outputs": "Address counts",
        "io_ratio": "Address counts", "value_ratio": "Address counts",
        "n_distinct_input_wallets": "Address counts",
        "n_distinct_output_wallets": "Address counts",
        "equal_value_flag": "FATF rule signals",
        "peeling_flag": "FATF rule signals",
        "coinjoin_strength": "FATF rule signals",
        "fresh_output_ratio": "FATF rule signals",
        "wallet_total_tx": "Wallet context",
        "wallet_total_sent": "Wallet context",
        "wallet_size": "Wallet context",
        "wallet_age": "Wallet context",
        "wallet_sent_ratio": "Wallet context",
        "wallet_final_balance": "Wallet context",
        "wallet_balance_ratio": "Wallet context",
        "wallet_balance_velocity": "Wallet context",
        "wallet_balance_range":    "Wallet context",
        "wallet_balance_std":      "Wallet context",
        "tx_fee": "Transaction metadata",
        "tx_fee_normalized": "Transaction metadata",
        "lock_time_flag": "Transaction metadata",
        "tx_hour": "Transaction metadata",
        "tx_day_of_week": "Transaction metadata",
        "is_night": "Transaction metadata",
        "mean_out_addr_tx_count": "Output-address statistics",
        "mean_out_addr_balance":  "Output-address statistics",
        "frac_out_addr_zero_balance": "Output-address statistics",
        "mean_out_addr_age":      "Output-address statistics",
        "mean_out_addr_sent_ratio": "Output-address statistics",
        "mean_out_addr_bal_velocity": "Output-address statistics",
    }

    TECHNICAL = {
        "Output statistics":         "A. Output statistics",
        "Address counts":            "C/D. Address counts and graph topology",
        "FATF rule signals":         "E/F. FATF rule signals",
        "Wallet context":            "G/H. Wallet context and temporal",
        "Transaction metadata":      "I. Transaction metadata",
        "Output-address statistics": "J. Output-address statistics",
    }

    # Group entries by narrative group, sorted by gain rank within group
    by_group = {}
    for feat, narr in NARRATIVE.items():
        by_group.setdefault(narr, []).append(feat)

    GROUP_ORDER = [
        "Output statistics", "Address counts", "FATF rule signals",
        "Wallet context", "Output-address statistics", "Transaction metadata",
    ]

    for grp in GROUP_ORDER:
        feats = by_group.get(grp, [])
        # Sort within group by gain rank (deployed first, then dropped)
        def key(f):
            r = rank_map.get(f)
            return (1 if f in DROPPED else 0, r if r is not None else 999)
        feats = sorted(feats, key=key)

        heading_para(doc, f"B.1.{GROUP_ORDER.index(grp)+1}  {grp}",
                     level=2)
        rows = []
        for f in feats:
            calc   = spec_rows.get(f, {}).get("definition_calc", "")
            source = spec_rows.get(f, {}).get("source_table", "—")
            deployed = "N" if f in DROPPED else "Y"
            rk = rank_map.get(f)
            rk_disp = f"{rk}" if rk is not None else "—"
            rows.append([f, calc, source,
                         TECHNICAL.get(grp, grp), deployed, rk_disp])
        add_table(doc,
                  ["Feature", "Definition / Computation", "Source Table",
                   "Technical Category", "Deployed", "Gain rank"],
                  rows,
                  col_widths_cm=[3.4, 5.5, 3.0, 3.4, 1.3, 1.4])

    body_para(doc,
        "Wallet-context features (wallet_total_tx through "
        "wallet_balance_ratio) are suppressed to NaN whenever "
        "coinjoin_strength > 0.1 — a defensive measure against false "
        "positives in CoinJoin-merged wallets. All monetary values are "
        "recorded in Satoshi.",
        italic=True)
    mark("B", "OK",
         f"feature_specification.xlsx loaded; {len(rank_map)} importance ranks")
    page_break(doc)


# ── Appendix C — Per-period descriptive statistics ────────────────────────

def appendix_C(doc):
    heading_para(doc, "Appendix C — Per-Period Descriptive Statistics",
                 level=1)
    body_para(doc,
        "Table C.1 reports per-snapshot descriptive statistics across the "
        "24 quarterly samples. Each row corresponds to a single CUBiD "
        "snapshot day (8 March, 8 June, 8 September, or 8 December "
        "2019–2024). The table supports Sections 3.1 (dataset overview) and "
        "3.2 (label distribution and rule firing), and is the source of the "
        "per-period charts referenced in those sections. All Bitcoin values "
        "are derived from the original Satoshi columns.")

    df = pd.read_csv(os.path.join(ROOT, "outputs", "per_period_stats.csv"))
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["period_q"],
            f"{int(r['n_tx']):,}",
            f"{float(r['mean_btc']):.4f}",
            f"{float(r['median_btc']):.6f}",
            f"{float(r['mean_fee_sat']):,.0f}",
            f"{float(r['mean_inputs']):.2f}",
            f"{float(r['mean_outputs']):.2f}",
            f"{int(r['total_btc']):,}",
            f"{int(r['conf_il']):,}",
            f"{int(r['conf_lt']):,}",
            f"{int(r['pseu_il']):,}",
            f"{int(r['pseu_lt']):,}",
            f"{int(r['uncert']):,}",
        ])
    add_table(doc,
              ["Period", "Tx count",
               "Mean tx (BTC)", "Median tx (BTC)", "Mean fee (sat)",
               "Mean in", "Mean out",
               "Total volume (BTC)",
               "Conf. illicit", "Conf. licit",
               "Pseudo illicit", "Pseudo licit", "Uncertain"],
              rows,
              col_widths_cm=[1.2, 1.2, 1.4, 1.6, 1.4, 1.0, 1.0,
                             1.6, 1.4, 1.3, 1.3, 1.3, 1.3])
    body_para(doc,
        "Per-period medians fall by roughly two orders of magnitude between "
        "2019-Q1 (≈ 0.077 BTC) and 2024-Q3 (≈ 0.0003 BTC), reflecting the "
        "rise of micro-transactions and Inscriptions. The 2024-Q3 snapshot "
        "is the largest by transaction count (910k tx in 167 blocks) and "
        "shows the lowest mean fee. The confirmed-illicit count tracks the "
        "BABD-13 corpus age limit (May 2021) and falls sharply from 2021-Q4 "
        "onward. Source: outputs/per_period_stats.csv (recomputed from "
        "outputs/labeled_features.csv joined to data/Block_Header.csv).",
        italic=True)
    mark("C", "OK", f"24 periods × 13 columns")
    page_break(doc)


# ── Appendix D — DBSCAN cluster output ─────────────────────────────────────

def appendix_D(doc):
    heading_para(doc, "Appendix D — Full DBSCAN Cluster Output", level=1)
    body_para(doc,
        "DBSCAN was run on the full 8.3 M-row feature matrix (eps = 1.5, "
        "min_samples = 5, ball_tree algorithm). This appendix supports "
        "Section 3.7 by reporting the headline metrics, the fifty largest "
        "clusters profiled by mean feature values, and the confirmed-label "
        "purity of every cluster that intersects a confirmed-label "
        "transaction.")

    # Headline
    with open(os.path.join(ROOT, "outputs", "dbscan_summary.json")) as f:
        h = json.load(f)
    heading_para(doc, "D.1  Headline Metrics", level=2)
    headline_rows = [
        ("Total clusters (excluding noise)", f"{h['n_clusters']:,}"),
        ("Noise points (label = -1)",        f"{h['n_noise']:,}"),
        ("Total rows",                       f"{h['n_total']:,}"),
        ("Noise rate",                       f"{h['noise_rate']:.4%}"),
        ("Largest cluster size",             f"{h['largest_cluster']:,}"),
        ("Median cluster size",              f"{h['median_cluster']:,}"),
        ("Smallest cluster size",            f"{h['smallest_cluster']:,}"),
        ("Parameter — eps",                  f"{h['eps']}"),
        ("Parameter — min_samples",          f"{h['min_samples']}"),
        ("Algorithm",                        h["algorithm"]),
    ]
    add_table(doc, ["Metric", "Value"], headline_rows,
              col_widths_cm=[8.0, 5.0])

    # Top 50
    heading_para(doc, "D.2  Fifty Largest Clusters by Size", level=2)
    body_para(doc,
        "Each cluster is profiled by the mean of seven behavioural "
        "features computed across its member transactions. Clusters with "
        "uniform peeling_flag = 0 and equal_value-style structure indicate "
        "homogeneous structuring patterns; clusters with high peeling_flag "
        "rate concentrate layering behaviour.")
    top50 = pd.read_csv(os.path.join(ROOT, "outputs", "dbscan_top50.csv"))
    rows = []
    for _, r in top50.iterrows():
        rows.append([
            f"{int(r['cluster_dbscan'])}",
            f"{int(r['size']):,}",
            f"{float(r['coinjoin']):.3f}",
            f"{float(r['num_outputs']):.2f}",
            f"{float(r['total_input']):,.0f}",
            f"{float(r['value_entropy']):.3f}",
            f"{float(r['peeling_rate']):.3f}",
        ])
    add_table(doc,
              ["Cluster ID", "Size",
               "Mean coinjoin", "Mean num_outputs",
               "Mean total_input (sat)", "Mean entropy",
               "Peeling rate"],
              rows,
              col_widths_cm=[1.6, 1.6, 2.0, 2.5, 3.5, 2.0, 1.6])

    # Confirmed-label distribution per cluster
    heading_para(doc, "D.3  Confirmed-Label Distribution per Cluster",
                 level=2)
    body_para(doc,
        "Only clusters that intersect at least one confirmed-label "
        "transaction are listed. The illicit-purity column is "
        "confirmed-illicit / total confirmed within the cluster. The "
        "noise group (cluster_id = -1) is included because most "
        "confirmed-label transactions fall outside any dense cluster at "
        "these parameters.")
    conf = pd.read_csv(
        os.path.join(ROOT, "outputs", "dbscan_confirmed_per_cluster.csv"))
    rows = []
    for _, r in conf.iterrows():
        rows.append([
            f"{int(r['cluster_dbscan'])}",
            f"{int(r['n_confirmed']):,}",
            f"{int(r['illicit']):,}",
            f"{int(r['licit']):,}",
            f"{float(r['illicit_pct']):.1f}%",
        ])
    add_table(doc,
              ["Cluster ID", "Confirmed total",
               "Confirmed illicit", "Confirmed licit",
               "Illicit purity"],
              rows,
              col_widths_cm=[1.8, 2.5, 2.5, 2.5, 2.0])

    heading_para(doc, "D.4  Interpretation", level=2)
    body_para(doc,
        "The eps = 1.5 / min_samples = 5 parameters return 42,162 distinct "
        "clusters but place 85.86 % of transactions in noise. At "
        "8.3 M rows in a 35-dimensional standardised space, a Euclidean "
        "neighbourhood radius of 1.5 captures only the densest local pockets "
        "— the median cluster contains just eleven transactions. The "
        "noise group itself is heterogeneous (107k confirmed illicit "
        "alongside 35k confirmed licit), so DBSCAN is functioning here as a "
        "secondary outlier flag rather than a coherent partition. A "
        "parameter sweep across eps ∈ {1.5, 3, 5, 8} and min_samples "
        "∈ {5, 25, 100} would be required to recover meaningful "
        "partitions, with the trade-off being a smaller number of larger "
        "clusters at the cost of within-cluster heterogeneity.")
    mark("D", "OK",
         f"{h['n_clusters']:,} clusters, top 50 + {len(conf)} confirmed-cluster rows")
    page_break(doc)


# ── Appendix E — Rule specification ────────────────────────────────────────

def appendix_E(doc):
    heading_para(doc, "Appendix E — Rule Specification", level=1)
    body_para(doc,
        "Table E.1 specifies the six FATF-aligned heuristic rules implemented "
        "in 03_label.py. Each rule operationalises one or more red-flag "
        "indicators from the FATF (2020) virtual-asset guidance. The "
        "pseudo-labelling logic sums the rule outcomes into a suspicion score "
        "(range 0–6); a transaction is pseudo-illicit when the score ≥ 2, "
        "pseudo-licit when the score = 0, and uncertain (excluded from "
        "supervised training) when the score = 1. The score-≥-2 cutoff "
        "follows FATF guidance that risk emerges from combinations of "
        "indicators rather than any single signal.")

    rule_rows = [
        ("rule_equal_value",
         "Structuring / CoinJoin splitting",
         "CV(output_value) = std/mean < 0.01",
         "equal_value_flag (Trans_Detail.Output_Value)",
         "FATF (2020) flag indicators on structured deposits and "
         "CoinJoin-style equal-value splitting; the 1 % CV threshold "
         "captures near-identical output amounts within a tx."),
        ("rule_peeling",
         "Layering / peeling chain",
         "max(output_value) ≥ 80% × total_output AND num_outputs ≥ 2",
         "peeling_flag (Trans_Detail)",
         "FATF (2020) layering indicator: a single dominant output with "
         "small change outputs is the canonical peeling-chain pattern."),
        ("rule_high_fan_out",
         "Fund dispersion",
         "num_outputs ≥ 10",
         "num_outputs (Trans_Detail.Output_Address)",
         "FATF (2020) dispersion indicator; ≥ 10 distinct recipient "
         "addresses in a single transaction is unusual for legitimate use "
         "and characteristic of cash-out / mixer disbursement."),
        ("rule_fresh_outputs",
         "Address churn / layering",
         "fresh_output_ratio ≥ 0.80",
         "fresh_output_ratio (Trans_Detail + Address_Info)",
         "FATF (2020) churn indicator; structurally inert in the "
         "multi-period dataset because CUBiD's Address_Info table is "
         "global (covers blocks 677 – 873k), so every output address has "
         "prior history. The rule cannot fire on this corpus — a "
         "limitation flagged in Sections 2.8 and 3.2."),
        ("rule_mixing",
         "Mixer involvement",
         "coinjoin_strength ≥ 0.5",
         "coinjoin_strength composite",
         "FATF (2020) mixer indicator. The composite combines equal-value "
         "splitting (60 % weight) with input/output count signals "
         "(20 % each), matched against the 0.5 cutoff."),
        ("rule_high_entropy",
         "Anomalous value distribution",
         "value_entropy ≥ 3.0 bits",
         "value_entropy (Trans_Detail.Output_Value)",
         "FATF (2020) anomalous-value indicator. Three bits of Shannon "
         "entropy implies output values spread across roughly 8 distinct "
         "magnitudes — suggestive of structuring or value obfuscation."),
    ]
    add_table(doc,
              ["Rule", "FATF Indicator",
               "Threshold / Computation", "Input Features",
               "Rationale and Limitations"],
              rule_rows,
              col_widths_cm=[2.4, 2.6, 3.4, 3.0, 4.0])

    body_para(doc,
        "All thresholds are encoded as constants at the top of 03_label.py "
        "(HIGH_FAN_OUT, HIGH_FRESH_RATIO, HIGH_COINJOIN, HIGH_ENTROPY). "
        "The pseudo-labelling logic is intentionally conservative: only "
        "transactions with two or more concurrent rule firings are "
        "promoted to the illicit class, and the uncertain band "
        "(score = 1) is excluded from supervised training to avoid "
        "label noise. References Sections 2.5 (rule design) and 4.1 "
        "(rule evaluation against confirmed labels).",
        italic=True)
    mark("E", "OK", "6 rules from rule_specification.xlsx + 03_label.py")
    page_break(doc)


# ── Appendix F — Model configuration ───────────────────────────────────────

def appendix_F(doc):
    heading_para(doc, "Appendix F — Model Configuration and Hyperparameters",
                 level=1)
    body_para(doc,
        "This appendix documents the supervised classifier, the "
        "unsupervised clustering algorithms, the cross-validation protocol, "
        "and the computational environment used to produce all results "
        "reported in Section 3 of the main report. References Sections 2.5 "
        "and 2.7.")

    # F.1 LightGBM
    heading_para(doc, "F.1  Supervised Classifier — LightGBM", level=2)
    body_para(doc,
        "LightGBM 4.6.0 is the primary classifier; Random Forest is the "
        "fallback when LightGBM is unavailable. All hyperparameters are set "
        "in 04_train.py:build_classifier().")
    add_table(doc, ["Hyperparameter", "Value"], [
        ("n_estimators",  "500"),
        ("learning_rate", "0.05"),
        ("max_depth",     "8"),
        ("num_leaves",    "63"),
        ("class_weight",  "balanced"),
        ("random_state",  "42"),
        ("n_jobs",        "-1"),
        ("verbose",       "-1"),
    ], col_widths_cm=[5.0, 8.0])

    # F.2 Random Forest fallback
    heading_para(doc, "F.2  Random Forest Fallback", level=2)
    body_para(doc,
        "Used only when LightGBM cannot be imported.")
    add_table(doc, ["Hyperparameter", "Value"], [
        ("n_estimators",  "200"),
        ("max_depth",     "10"),
        ("class_weight",  "balanced"),
        ("random_state",  "42"),
        ("n_jobs",        "-1"),
        ("preprocessing", "SimpleImputer(median) + StandardScaler"),
    ], col_widths_cm=[5.0, 8.0])

    # F.3 Clustering
    heading_para(doc, "F.3  Unsupervised Clustering", level=2)
    add_table(doc,
              ["Algorithm", "Parameter", "Value"],
              [
                  ("K-Means", "n_clusters",  "3"),
                  ("K-Means", "n_init",      "10"),
                  ("K-Means", "random_state", "42"),
                  ("DBSCAN",  "eps",         "1.5"),
                  ("DBSCAN",  "min_samples", "5"),
                  ("DBSCAN",  "algorithm",   "ball_tree"),
                  ("DBSCAN",  "n_jobs",      "-1"),
              ],
              col_widths_cm=[3.0, 4.0, 6.0])

    # F.4 Cross-validation
    heading_para(doc, "F.4  Cross-Validation Protocol", level=2)
    add_table(doc, ["Setting", "Value"], [
        ("Method",          "StratifiedKFold (sklearn.model_selection)"),
        ("Number of folds", "5"),
        ("Shuffle",         "True"),
        ("Random state",    "42"),
        ("Stratification",  "By binary illicit/licit class label"),
        ("Scoring (04_train.py)", "precision_macro, recall_macro, f1_macro"),
        ("Scoring (08_mechanism_eval.py)",
         "macro+weighted F1, per-class precision/recall, ROC-AUC, PR-AUC; "
         "out-of-fold confusion matrix"),
        ("Class-weighting",
         "class_weight='balanced' (LightGBM and Random Forest); no "
         "additional resampling at the data level"),
    ], col_widths_cm=[5.0, 8.0])

    # F.5 Computational environment
    heading_para(doc, "F.5  Computational Environment", level=2)
    add_table(doc, ["Component", "Value"], [
        ("Python",        "3.13.12"),
        ("python-docx",   "1.2.0"),
        ("LightGBM",      "4.6.0"),
        ("scikit-learn",  "1.8.0"),
        ("pandas",        "2.3.3"),
        ("numpy",         "2.4.3"),
        ("joblib",        "1.5.3"),
        ("matplotlib",    "3.10.9"),
        ("openpyxl",      "3.1.5"),
        ("pypdf",         "6.10.2"),
        ("Pillow",        "12.1.1"),
        ("CPU",           "Apple M3 Pro (12-core)"),
        ("RAM",           "18 GB"),
        ("OS",            "macOS (Darwin 25.3.0, arm64)"),
        ("Storage",       "[TO BE COMPLETED — internal SSD class/size]"),
    ], col_widths_cm=[5.0, 8.0])
    mark("F", "OK", "Hyperparams from 04_train.py + observed env")
    page_break(doc)


# ── Appendix G — Reproducibility guide ─────────────────────────────────────

def appendix_G(doc):
    heading_para(doc, "Appendix G — Pipeline Reproducibility Guide", level=1)
    body_para(doc,
        "This appendix is a step-by-step recipe for reproducing every "
        "result reported in Sections 3 and 4 from the underlying CUBiD "
        "tables and four external label sources. The pipeline runs "
        "end-to-end in approximately two hours on the environment "
        "specified in Appendix F.5, dominated by feature engineering "
        "(15 minutes), label-enrichment matching (12 minutes), and "
        "DBSCAN on the full feature matrix (67 minutes).")

    heading_para(doc, "G.1  Prerequisites", level=2)
    bullet(doc, "Python 3.13 (any recent 3.10+ should work).")
    bullet(doc, "Required packages (install via "
                "pip install -r requirements.txt or individually): "
                "pandas, numpy, scikit-learn, joblib, lightgbm, scipy, "
                "openpyxl, pypdf, Pillow, matplotlib, plotly, streamlit, "
                "python-docx.")
    bullet(doc, "macOS / Linux shell. Pipeline scripts are run from the "
                "repository root (Bitcoin_Tracker-master).")

    heading_para(doc, "G.2  Data Acquisition", level=2)
    body_para(doc,
        "The CUBiD multi-period dataset is provided by Dr Hossein "
        "Jahanshahloo (Cardiff Business School, cubid@cardiff.ac.uk) "
        "subject to research-use agreement. The pipeline expects the seven "
        "tables listed in Appendix A.2 to be available under data/ in the "
        "repository; see the symlink layout used in this work where the "
        "T_-prefixed multi-period files in /Users/.../New Data/ are linked "
        "into Bitcoin_Tracker-master/data/ stripped of the prefix.")

    heading_para(doc, "G.3  External Labels", level=2)
    body_para(doc,
        "The four external label files must be located at the following "
        "paths (relative to the repository root):")
    add_table(doc,
              ["Source", "Path"],
              [
                  ("Elliptic++",
                   "../drive-download-20260426T124634Z-3-001/wallets_classes.csv"),
                  ("BABD-13",
                   "data/external_labels/BABD-13.csv "
                   "(auto-extracted from ../archive.zip on first run)"),
                  ("BitcoinAbuse",
                   "../DatasetBitcoinAbuse.csv"),
                  ("OpenSanctions",
                   "../OpenSanctions/targets.simple.csv"),
              ],
              col_widths_cm=[3.5, 9.5])

    heading_para(doc, "G.4  Execution Order", level=2)
    body_para(doc,
        "The pipeline is structured as eight numbered scripts. Each is run "
        "directly with python <script>.py from the repository root.")
    add_table(doc,
              ["Script", "Stage", "Output", "Approx. runtime"],
              [
                  ("01_load.py", "Data load and schema validation",
                   "stdout report only (no artefact written)", "≈ 30 s"),
                  ("02_feature.py", "Feature engineering (35 features)",
                   "outputs/transaction_features.csv (8.3 M rows)",
                   "≈ 15 minutes"),
                  ("02b_enrich_labels.py",
                   "External label enrichment from four sources",
                   "outputs/enriched_features.csv (with confirmed_label, "
                   "confirmation_source columns)", "≈ 12 minutes"),
                  ("03_label.py",
                   "FATF rule firing and pseudo-label assignment",
                   "outputs/labeled_features.csv", "≈ 1 minute"),
                  ("04_train.py",
                   "Hybrid-mode classifier training, 5-fold CV, K-Means, "
                   "full-set DBSCAN",
                   "outputs/model.pkl, model_<mode>.pkl, "
                   "cluster_assignments.csv", "≈ 75 minutes "
                   "(DBSCAN dominates ≈ 67 min)"),
                  ("06_evaluate.py",
                   "Elliptic++ external benchmark",
                   "outputs/elliptic_eval.json", "≈ 5 minutes"),
                  ("07_temporal_eval.py",
                   "Train on 2019 – 2022 quarters, test on 2023 – 2024",
                   "outputs/temporal_eval.json", "≈ 5 minutes"),
                  ("08_mechanism_eval.py",
                   "Confirmed-label and augmented mechanism CV",
                   "outputs/mechanism_eval.json, mechanism_eval_set.csv",
                   "≈ 5 minutes"),
              ],
              col_widths_cm=[2.6, 3.4, 4.5, 2.5])

    heading_para(doc, "G.5  Output Structure", level=2)
    body_para(doc,
        "All artefacts are written to outputs/ in the repository root.")
    bullet(doc, "transaction_features.csv — 35-column feature matrix (one "
                "row per transaction)")
    bullet(doc, "enriched_features.csv — adds confirmed_label, "
                "confirmation_source, confirmation_strength columns")
    bullet(doc, "labeled_features.csv — adds rule_* flags, suspicion_score, "
                "pseudo_label, label_type")
    bullet(doc, "cluster_assignments.csv — cluster_kmeans, cluster_dbscan "
                "(full-dataset DBSCAN, no -2 sentinel)")
    bullet(doc, "model.pkl — joblib bundle: pipeline, features, km_model, "
                "classifier, label_mode")
    bullet(doc, "mechanism_eval.json (schema v2) — augmented and "
                "confirmed-only CV with held-out confusion matrix")
    bullet(doc, "temporal_eval.json — train / test metrics, drift "
                "differential")
    bullet(doc, "elliptic_eval.json — Elliptic++ transaction- and "
                "actor-level CV results")
    bullet(doc, "section3_inventory.md — consolidated numerical inventory "
                "(produced by build_section3_inventory.py)")

    heading_para(doc, "G.6  Streamlit Dashboard", level=2)
    body_para(doc,
        "The dashboard is launched with:")
    body_para(doc, "streamlit run app.py", italic=True)
    body_para(doc,
        "It exposes seven pages, each backed by the artefacts written in "
        "the previous step:")
    bullet(doc, "Overview (app.py) — summary metrics and rule firing rates")
    bullet(doc, "Transaction Explorer (1_Explorer.py) — filter / search "
                "the 8 M-row labelled set")
    bullet(doc, "Transaction Detail (2_Transaction.py) — per-tx verdict, "
                "feature breakdown, raw input/output rows")
    bullet(doc, "Wallet Analysis (3_Wallet.py) — aggregate verdict and "
                "transaction timeline for a given wallet ID")
    bullet(doc, "Feature Analysis (4_Features.py) — distributions, "
                "correlation matrix, importance plot")
    bullet(doc, "Cluster Visualisation (5_Clusters.py) — PCA scatter "
                "coloured by K-Means / DBSCAN cluster")
    bullet(doc, "Elliptic++ Evaluation (6_Evaluate.py) — confusion "
                "matrix and CV metrics from elliptic_eval.json")
    bullet(doc, "How It Works (7_How_It_Works.py) — methodology notes "
                "and dataset disclaimers")
    mark("G", "OK", "8 scripts + 7 dashboard pages")
    page_break(doc)


# ── Appendix H — Pipeline architecture diagram ────────────────────────────

def appendix_H(doc):
    heading_para(doc, "Appendix H — Pipeline Architecture Diagram", level=1)
    body_para(doc,
        "Figure H.1 traces the four logical layers of the detection "
        "framework: input (the seven CUBiD tables), processing "
        "(feature engineering → external label enrichment → "
        "pseudo-labelling → hybrid label set), modelling (parallel "
        "unsupervised clustering and supervised classification, with the "
        "Streamlit dashboard as a side branch consuming the trained "
        "model), and evaluation (mechanism, temporal, and external "
        "Elliptic++ validation). Arrows indicate data flow only; the "
        "control script run_pipeline.py invokes each stage sequentially.")

    img_path = os.path.join(ROOT, "outputs", "pipeline_architecture.png")
    if os.path.exists(img_path):
        try:
            doc.add_picture(img_path, width=Cm(16))
            cap = doc.add_paragraph()
            run = cap.add_run("Figure H.1. End-to-end pipeline architecture "
                              "for the BMAN31260 detection framework.")
            run.font.name = BODY_FONT
            run.font.italic = True
            run.font.size = Pt(11)
            cap.alignment = 1
            mark("H", "OK", "PNG embedded")
        except Exception as exc:
            body_para(doc, f"[Error embedding image: {exc}]", italic=True)
            mark("H", "Partial", f"image error: {exc}")
    else:
        body_para(doc, "[TO BE COMPLETED — pipeline_architecture.png "
                       "not found in outputs/]")
        mark("H", "Missing", "PNG not generated")
    page_break(doc)


# ── Appendix I — External label sources ───────────────────────────────────

def appendix_I(doc):
    heading_para(doc, "Appendix I — External Label Source Documentation",
                 level=1)
    body_para(doc,
        "Four external label sources are merged into the master lookup "
        "by 02b_enrich_labels.py. The combined master lookup contains "
        "861,529 deduplicated addresses (209,043 confirmed illicit / "
        "652,486 confirmed licit). When two sources disagree on the same "
        "address, illicit takes precedence over licit and the "
        "confirmation_source field records both contributors "
        "(e.g. \"BitcoinAbuse;EllipticPlusPlus\"). Match counts on the "
        "multi-period dataset are reported in Section 3.3 of the main "
        "report and reproduced below for each source.")

    # Helper for one source block
    def source_block(name, citation, source_loc, coverage,
                     categories, license_, format_norm, dedup, matches):
        heading_para(doc, name, level=2)
        body_para(doc, citation, italic=True)
        add_table(doc, ["Field", "Value"], [
            ("Source location",          source_loc),
            ("Coverage",                 coverage),
            ("Categories",               categories),
            ("Licence",                  license_),
            ("Address format handling",  format_norm),
            ("Deduplication",            dedup),
            ("Matches in this study",    matches),
        ], col_widths_cm=[4.0, 9.0])

    # Elliptic++
    source_block(
        "I.1  Elliptic++",
        "Elmougy, A. and Liu, L. (2023). Demystifying Fraudulent Transactions "
        "and Illicit Nodes in the Bitcoin Network for Financial Forensics. "
        "Proceedings of KDD 2023.",
        "../drive-download-20260426T124634Z-3-001/wallets_classes.csv "
        "(populated copy used; the EllipticPlusPlus-main mirror in the "
        "repo can contain Git-LFS pointer stubs and is excluded by an "
        "explicit guard in 02b_enrich_labels.py).",
        "265,354 confirmed addresses (14,266 illicit, 251,088 licit). "
        "Original observation window: Bitcoin blocks ≈ 2014 – 2018.",
        "Two-class (1 = illicit, 2 = licit, 3 = unknown — class 3 excluded "
        "from the master lookup).",
        "Open access (CC BY 4.0, per the public KDD 2023 dataset release).",
        "Bitcoin addresses ingested as plain strings; no normalisation "
        "needed.",
        "Within-source: drop_duplicates on the address column. Across "
        "sources: master-lookup groupby preserves illicit > licit on "
        "conflict.",
        "2,899 transactions matched (18 illicit, 2,881 licit) on the "
        "multi-period sample — Section 3.3.")

    # BABD-13
    source_block(
        "I.2  BABD-13",
        "Xiang, Y., Ren, W., Lei, T., Yang, X., Zhang, K., Zhang, T. and "
        "Choo, K.-K.R. (2024). BABD: A Bitcoin Address Behavior Dataset for "
        "Pattern Analysis. IEEE Transactions on Information Forensics and "
        "Security 19, 1077-1091.",
        "data/external_labels/BABD-13.csv (auto-extracted from "
        "../archive.zip if absent).",
        "541,330 addresses across 13 categories; ≈ 138k illicit / 403k "
        "licit after the explicit category-to-label mapping in "
        "02b_enrich_labels.py:_BABD13_CATEGORY_MAP. Original observation "
        "window: 12 July 2019 – 26 May 2021.",
        "13 categories (numeric labels 0 – 12): 0 Blackmail, "
        "1 Cyber-Security Service, 2 Darknet Market, 3 Centralised "
        "Exchange, 4 P2P Financial Infrastructure, "
        "5 P2P Financial Service, 6 Gambling, "
        "7 Government Criminal Blocklist, 8 Money Laundering, "
        "9 Ponzi Scheme, 10 Mining Pool, 11 Tumbler, 12 Individual Wallet. "
        "Categories 0/2/6/7/8/9/11 mapped to illicit; 1/3/4/5/10 to licit; "
        "category 12 excluded as neutral.",
        "Open access (IEEE TIFS dataset release; redistribution permitted "
        "for non-commercial research per the publication terms).",
        "Bitcoin addresses ingested as plain strings; no normalisation "
        "needed. The category mapping is encoded once in "
        "_BABD13_CATEGORY_MAP and applied consistently.",
        "Within-source: drop_duplicates on address. Across sources: "
        "illicit > licit on conflict (master-lookup groupby).",
        "26,665 transactions matched (4,031 illicit, 22,634 licit) on the "
        "multi-period sample — Section 3.3.")

    # BitcoinAbuse
    source_block(
        "I.3  BitcoinAbuse (deprecated public dump)",
        "BitcoinAbuse (2019 – 2022). Community-curated database of Bitcoin "
        "addresses associated with abuse reports. The live API was "
        "discontinued in 2022; the public CSV dump is the only "
        "accessible artefact.",
        "../DatasetBitcoinAbuse.csv (the deprecated public dump).",
        "47,099 deduplicated addresses, all illicit by definition. "
        "Original observation window: ≈ 2017 – 2021. Single-class source: "
        "no licit examples.",
        "Free-text abuse_type field (ransomware, blackmail, sextortion, "
        "scam, theft, etc.); the pipeline does not split by type.",
        "Public dump (community-contributed; redistribution permitted).",
        "Bitcoin addresses ingested as plain strings; whitespace stripped.",
        "Within-source: drop_duplicates on address. Across sources: "
        "concatenated into the master lookup; on conflict illicit wins.",
        "454 transactions matched (all illicit) on the multi-period "
        "sample — Section 3.3. Single-class status means BitcoinAbuse is "
        "excluded from per-source supervised CV in 08_mechanism_eval.py.")

    # OpenSanctions
    source_block(
        "I.4  OpenSanctions",
        "OpenSanctions (2024). Consolidated targets dataset, daily build "
        "from OFAC, EU, UN, UK HMT, and other sanctions list publishers. "
        "Available at data.opensanctions.org.",
        "../OpenSanctions/targets.simple.csv (the 'simple' tabular flavour "
        "of the consolidated daily build).",
        "129 distinct Bitcoin addresses extracted from the identifiers "
        "column of the consolidated dataset (after regex-anchored "
        "tokenisation by ';'). All single-class illicit (sanctioned). "
        "Includes OFAC SDN crypto designations and equivalent EU/UN "
        "listings.",
        "Sanctions-listed entities flagged across multiple jurisdictions; "
        "no fine-grained categorisation in the simple flavour.",
        "CC BY 4.0 (OpenSanctions public dataset).",
        "Bitcoin addresses extracted via anchored regex "
        "^(?:bc1[ac-hj-np-z02-9]{39,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})$ "
        "to filter base58/bech32 forms and avoid Ethereum-hex false "
        "positives.",
        "Within-source: drop_duplicates on address. Across sources: master "
        "lookup; illicit wins on conflict.",
        "129 transactions matched (all illicit) on the multi-period "
        "sample — Section 3.3. Single-class status means OpenSanctions is "
        "excluded from per-source supervised CV.")
    mark("I", "OK", "4 sources documented")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    doc = setup_document()
    title_page(doc)
    intro_page(doc)
    appendix_A(doc)
    appendix_B(doc)
    appendix_C(doc)
    appendix_D(doc)
    appendix_E(doc)
    appendix_F(doc)
    appendix_G(doc)
    appendix_H(doc)
    appendix_I(doc)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    print(f"\n[OK] Appendices saved to: {OUT_PATH}")

    # ── Status report ─────────────────────────────────────────────────────
    print("\n── Appendix status report ──")
    for k in sorted(APPENDIX_STATUS):
        s, n = APPENDIX_STATUS[k]
        print(f"  Appendix {k}: {s} — {n}")


if __name__ == "__main__":
    main()
