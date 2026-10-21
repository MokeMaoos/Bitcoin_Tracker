# Operationalising Regulatory Red Flags for Bitcoin AML Detection

Code for my final-year dissertation (BSc Information Technology Management for Business, Alliance Manchester Business School, University of Manchester, 2026; awarded First Class).

Regulators spot money laundering on Bitcoin with qualitative red flags that rely on context and judgment. Automated detectors use quantitative features but are held back by the lack of labelled data. This project tests how far the six FATF virtual-asset red flags can be turned into measurable features and used to detect illicit activity without large hand-labelled datasets.

## Results

- **35 features** derived from the six FATF red flags, computed over **8.3M transactions** in **24 quarterly snapshots (2019–2024)**.
- **0.96 macro F1** on a balanced evaluation set of **225,722 transactions**: the **149,128 confirmed labels** plus a sample of unmatched transactions presumed licit. **0.93** on the confirmed labels alone. Roughly 12 points above the strongest combined rule baseline.
- Matched or exceeded published results on the **Elliptic++** benchmark.
- **The rules tracked enforcement history.** Mixing-related flags rose from 13.5% in early 2019 to 58.6% in Q3 2023, about a year after the Tornado Cash sanctions, then fell to 7.5% by Q3 2024 as Samourai and Wasabi faced enforcement.
- **Scope limit:** the method is strongest on obfuscation-heavy laundering and weaker on illicit activity that leaves no structural pattern, such as gambling or blackmail.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_load.py` | Load and validate the raw transaction, input/output, address and wallet tables |
| 2 | `02_feature.py` | Engineer structural, value, address and wallet features plus the FATF rule signals |
| 2b | `02b_enrich_labels.py` | Add confirmed labels from four external sources (Elliptic++, BABD-13, BitcoinAbuse, OpenSanctions) |
| 3 | `03_label.py` | FATF rule-based pseudo-labelling: illicit (2+ red flags), licit (none), uncertain (one); confirmed labels take precedence |
| 4 | `04_train.py` | K-Means and DBSCAN clustering, then LightGBM (Random Forest fallback) |
| 5 | `05_predict.py` | Score a wallet or transaction |
| 6 | `06_evaluate.py` | External validation against Elliptic++ |
| 7 | `07_temporal_eval.py` | Train on 2019–2022, test on 2023–2024 |
| 8 | `08_mechanism_eval.py` | Main ground-truth benchmark: out-of-fold evaluation against the confirmed external labels |

`run_pipeline.py` runs the steps in order. `app.py` is a Streamlit dashboard (explorer, transaction and wallet views, features, clusters, evaluation, and a how-it-works page).

## Data

Not included. The main dataset is the Cardiff University Bitcoin Database, used with access arranged by my supervisor. Elliptic++ is public (see `06_evaluate.py` for where to place it). Raw data and model outputs are excluded by `.gitignore`.

## Run

```bash
pip install pandas numpy scipy scikit-learn lightgbm networkx joblib streamlit matplotlib
python run_pipeline.py
streamlit run app.py
```

## Contact

Constantine Samarikas · constantinesamarikas@gmail.com · [LinkedIn](https://www.linkedin.com/in/constantine-samarikas-b3763a23a/)
