# Parcl buyer segmentation and investment profiling

This project groups real-estate buyers by their client attributes and completed purchase behaviour. It produces a reproducible K-means model, a hierarchical-clustering validation, a concise findings report, and an interactive Streamlit dashboard.

## What the model uses

The training data joins `clients.csv` to **sold** records in `properties.csv`. Each client receives purchase-behaviour measures: number of completed purchases, total spend, average purchase price, average floor area, and office-purchase share. It also uses age, satisfaction score, client type, country, region, acquisition purpose, loan status, and referral channel.

Gender, names, IDs, raw dates of birth, and latest purchase date are not used for clustering. In particular, gender is excluded so marketing segments are not driven by a protected attribute.

Available property listings have no buyer reference, so they are excluded from buyer profiling and reported separately in the dashboard.

## Project structure

```text
Project 1/
├── clients.csv                     # supplied client data
├── properties.csv                  # supplied property data
├── app.py                          # Streamlit dashboard
├── requirements.txt                # required Python packages
├── src/
│   ├── data_preparation.py         # cleaning, validation, feature engineering
│   └── train_clustering.py          # training, evaluation, exports
├── tests/
│   └── test_data_preparation.py    # regression checks
├── models/                         # saved K-means pipeline after training
├── outputs/                        # CSV/JSON dashboard inputs after training
└── reports/                        # generated research paper
```

## Step-by-step setup

Run these commands in PowerShell from this project folder.

1. Check that Python 3.11 or later is installed:

   ```powershell
   py --version
   ```

2. Create a project-only virtual environment:

   ```powershell
   py -m venv .venv
   ```

3. Install the project dependencies into that environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Run the data checks:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

5. Train the recommended four-segment version for the project submission:

   ```powershell
   .\.venv\Scripts\python.exe src\train_clustering.py --data-dir . --n-clusters 4
   ```

   Alternatively, allow the highest silhouette score to select the number of clusters:

   ```powershell
   .\.venv\Scripts\python.exe src\train_clustering.py --data-dir . --n-clusters auto
   ```

6. Start the interactive dashboard:

   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```

   Open the local address Streamlit prints, usually `http://localhost:8501`.

## Outputs after training

- `models/buyer_segmentation_kmeans.joblib` — the trained preprocessing and K-means model.
- `outputs/segmented_clients.csv` — client-level segments and derived purchase measures.
- `outputs/cluster_profiles.csv` — segment statistics for business interpretation.
- `outputs/cluster_selection.csv` — elbow inputs and silhouette scores for K = 2–8.
- `outputs/model_metadata.json` — training date, feature list, validation metrics, and record counts.
- `reports/research_paper.md` — EDA, methodology, segment profiles, recommendations, and implementation cautions.

## Dashboard guide

Use the sidebar to filter country, region, acquisition purpose, and client type. The dashboard includes:

- **Overview:** cluster distribution and top-level buyer metrics.
- **Investor behaviour:** acquisition purpose, financing behaviour, and portfolio scale by segment.
- **Geographic analysis:** global country map and regional buyer-segment distribution.
- **Segment insights:** descriptive statistics for the current filter plus full-population profiles.
- **Model quality:** source coverage and silhouette diagnostics supporting the selected K.

## How to use the results responsibly

Segments support marketing and service design; they do not decide an individual’s eligibility, value, or access to offers. Before running a campaign, review segment results for fairness and re-train when substantial new client or sales records arrive.
