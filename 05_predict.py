"""
05_predict.py — predict whether a wallet or transaction is illicit.
Usage:
    python 05_predict.py --wallet  <wallet_id>
    python 05_predict.py --tx      <block_no> <trans_no>
"""
import pandas as pd
import numpy as np
import joblib
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import is_equal_value, compute_value_entropy, is_peeling, compute_fresh_output_ratio

MODEL_PATH   = "outputs/model.pkl"
FEATURES_CSV = "outputs/transaction_features.csv"
DATA_DIR     = "data"

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]

def load_data():
    td = pd.read_csv(os.path.join(DATA_DIR, "Trans_Detail.csv"))
    tx = pd.read_csv(os.path.join(DATA_DIR, "Transactions.csv"))
    ai = pd.read_csv(os.path.join(DATA_DIR, "Address_Info.csv"))
    wi = pd.read_csv(os.path.join(DATA_DIR, "Wallet_Info.csv"))
    return td, tx, ai, wi

def get_features_for_tx(block_no, trans_no, td, tx, ai, wi):
    """Extract the feature vector for a single transaction."""
    feat_df = pd.read_csv(FEATURES_CSV)
    row = feat_df[(feat_df["Block_No"] == block_no) & (feat_df["Trans_No"] == trans_no)]
    if not row.empty:
        return row
    # If not cached, compute on-the-fly
    grp = td[(td["Block_No"] == block_no) & (td["Trans_No"] == trans_no)]
    if grp.empty:
        print(f"[ERROR] Transaction Block {block_no} / Trans {trans_no} not found in data.")
        return None
    ov = grp["Output_Value"].values
    oa = grp["Output_Address"].values
    tx_row = tx[(tx["Block_No"] == block_no) & (tx["Trans_No"] == trans_no)]
    wtx = wsent = wsize = np.nan
    if not tx_row.empty:
        wid = tx_row["Wallet"].values[0]
        w = wi[wi["Wallet"] == wid]
        if not w.empty:
            wtx   = w["N_Transactions"].values[0]
            wsent = w["Total_Sent"].values[0]
            wsize = w["N_Addresses"].values[0]
    return pd.DataFrame([{
        "num_inputs":          grp["Input_Address"].nunique(),
        "num_outputs":         grp["Output_Address"].nunique(),
        "total_input":         grp["Input_Value"].sum(),
        "total_output":        ov.sum(),
        "mean_output":         ov.mean(),
        "std_output":          ov.std() if len(ov) > 1 else 0,
        "max_output":          ov.max(),
        "min_output":          ov.min(),
        "io_ratio":            grp["Input_Address"].nunique() / max(grp["Output_Address"].nunique(), 1),
        "value_ratio":         grp["Input_Value"].sum() / max(ov.sum(), 1e-9),
        "equal_value_flag":    is_equal_value(ov),
        "value_entropy":       compute_value_entropy(ov),
        "peeling_flag":        is_peeling(ov),
        "fresh_output_ratio":  compute_fresh_output_ratio(oa, ai),
        "wallet_total_tx":     wtx,
        "wallet_total_sent":   wsent,
        "wallet_size":         wsize,
    }])

def predict_tx(block_no, trans_no):
    model, feature_cols = load_model()
    td, tx, ai, wi = load_data()
    feat = get_features_for_tx(int(block_no), int(trans_no), td, tx, ai, wi)
    if feat is None: return
    X = feat[feature_cols].fillna(feat[feature_cols].median())
    prob = model.predict_proba(X)[0][1]
    label = "ILLICIT" if prob >= 0.5 else "LICIT"
    print(f"\nTransaction  Block {block_no} / Trans {trans_no}")
    print(f"Prediction   : {label}")
    print(f"Confidence   : {prob:.1%} probability of being illicit")
    if prob >= 0.5:
        print("⚠  This transaction has been flagged as potentially illicit.")

def predict_wallet(wallet_id):
    model, feature_cols = load_model()
    td, tx, ai, wi = load_data()
    wallet_txs = tx[tx["Wallet"] == wallet_id][["Block_No","Trans_No"]]
    if wallet_txs.empty:
        print(f"[ERROR] Wallet '{wallet_id}' not found.")
        return
    probs = []
    for _, r in wallet_txs.iterrows():
        feat = get_features_for_tx(r["Block_No"], r["Trans_No"], td, tx, ai, wi)
        if feat is None: continue
        X = feat[feature_cols].fillna(feat[feature_cols].median())
        probs.append(model.predict_proba(X)[0][1])
    if not probs:
        print("No transactions found for this wallet.")
        return
    avg_prob = np.mean(probs)
    flagged  = sum(p >= 0.5 for p in probs)
    print(f"\nWallet: {wallet_id}")
    print(f"Transactions analysed : {len(probs)}")
    print(f"Flagged as illicit    : {flagged} / {len(probs)}")
    print(f"Average illicit prob  : {avg_prob:.1%}")
    verdict = "HIGH RISK" if avg_prob >= 0.5 else ("MEDIUM RISK" if avg_prob >= 0.25 else "LOW RISK")
    print(f"Overall verdict       : {verdict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", type=str, help="Wallet ID to analyse")
    parser.add_argument("--tx",     nargs=2,  help="Block_No Trans_No")
    args = parser.parse_args()
    if args.wallet:
        predict_wallet(args.wallet)
    elif args.tx:
        predict_tx(args.tx[0], args.tx[1])
    else:
        print("Usage:\n  python 05_predict.py --wallet <id>\n  python 05_predict.py --tx <block> <trans>")