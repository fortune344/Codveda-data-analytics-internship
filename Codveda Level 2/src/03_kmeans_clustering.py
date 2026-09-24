"""
Task 3 - K-Means clustering.

Every stock becomes one point described by 4 numbers (average daily return, volatility,
average volume, average price). I standardize them, pick k with the elbow method and
group the stocks.

Run it from the project folder, after 01_clean_data.py:
    python src/03_kmeans_clustering.py

Figures go to figures/, tables go to results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "stock_prices_clean.csv"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
MIN_DAYS = 250     # ignore stocks with less than about a year of data
K = 3              # chosen after looking at figures/km_1_elbow_silhouette.png

sns.set_theme(style="whitegrid")

# 1. one row per stock -------------------------------------------------------------
df = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values(["symbol", "date"])
df["daily_return"] = df.groupby("symbol")["close"].pct_change()

stocks = df.groupby("symbol").agg(
    avg_daily_return=("daily_return", "mean"),
    volatility=("daily_return", "std"),
    avg_volume=("volume", "mean"),
    avg_price=("close", "mean"),
    n_days=("close", "size"),
)
stocks = stocks[stocks["n_days"] >= MIN_DAYS].copy()

# volume and price are very skewed (a few huge values), so I use their log10
stocks["log_volume"] = np.log10(stocks["avg_volume"])
stocks["log_price"] = np.log10(stocks["avg_price"])

FEATURES = ["avg_daily_return", "volatility", "log_volume", "log_price"]
print(f"{len(stocks)} stocks kept (at least {MIN_DAYS} trading days)")
print(stocks[FEATURES].describe().round(4))


# 2. standardize -------------------------------------------------------------------
# without this, volatility (around 0.015) would count for nothing next to log_volume (around 6)
X = StandardScaler().fit_transform(stocks[FEATURES])


# 3. elbow method (+ silhouette as a second opinion) -------------------------------
ks = range(1, 11)
inertia = []
silhouette = {}
for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inertia.append(model.inertia_)
    if k > 1:
        silhouette[k] = silhouette_score(X, model.labels_)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(ks), inertia, marker="o")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(silhouette), list(silhouette.values()), marker="o", color="crimson")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average silhouette")
fig.tight_layout()
fig.savefig(FIG_DIR / "km_1_elbow_silhouette.png", dpi=130)
plt.close(fig)

print("\nInertia by k:", [round(v) for v in inertia])
print("Silhouette by k:", {k: round(v, 3) for k, v in silhouette.items()})


# 4. final model -------------------------------------------------------------------
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
stocks["cluster"] = km.labels_

# K-Means numbers its clusters randomly, so I renumber them from the calmest (0) to the
# most volatile (K-1). That way the numbers in the README always mean the same thing.
order = stocks.groupby("cluster")["volatility"].mean().sort_values().index
stocks["cluster"] = stocks["cluster"].map({old: new for new, old in enumerate(order)})

print(f"\nk = {K}, silhouette = {silhouette_score(X, stocks['cluster']):.3f}")
print(stocks["cluster"].value_counts().sort_index().to_string())


# 5. scatter plots -----------------------------------------------------------------
colors = sns.color_palette("Set2", K)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=stocks, x="volatility", y="avg_daily_return", hue="cluster",
                palette=colors, alpha=0.8, ax=ax)
ax.set_title("Risk vs return")
ax.set_xlabel("Volatility (std of daily returns)")
ax.set_ylabel("Average daily return")
fig.tight_layout()
fig.savefig(FIG_DIR / "km_2_risk_vs_return.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=stocks, x="log_volume", y="log_price", hue="cluster",
                palette=colors, alpha=0.8, ax=ax)
ax.set_title("Trading volume vs price level")
ax.set_xlabel("log10 of average daily volume")
ax.set_ylabel("log10 of average closing price")
fig.tight_layout()
fig.savefig(FIG_DIR / "km_3_volume_vs_price.png", dpi=130)
plt.close(fig)

# the 4 features squeezed into 2 axes, just to see the clusters in one picture
pca = PCA(n_components=2, random_state=SEED)
pcs = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=stocks["cluster"], palette=colors, alpha=0.8, ax=ax)
ax.set_title(f"PCA view of the clusters ({pca.explained_variance_ratio_.sum():.0%} of the variance)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(FIG_DIR / "km_4_pca.png", dpi=130)
plt.close(fig)


# 6. describe each cluster ---------------------------------------------------------
summary = stocks.groupby("cluster").agg(
    n_stocks=("avg_price", "size"),
    avg_daily_return_pct=("avg_daily_return", lambda s: s.mean() * 100),
    volatility_pct=("volatility", lambda s: s.mean() * 100),
    avg_volume_millions=("avg_volume", lambda s: s.mean() / 1e6),
    avg_price_usd=("avg_price", "mean"),
).round(3)
print("\nCluster summary:")
print(summary.to_string())

# tickers closest to each cluster centre, as examples
dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
stocks["distance_to_centre"] = dist
print()
for c in range(K):
    examples = stocks[stocks["cluster"] == c].nsmallest(8, "distance_to_centre").index
    print(f"cluster {c}: {', '.join(examples)}")

summary.to_csv(RESULTS_DIR / "cluster_summary.csv")
stocks[FEATURES + ["cluster"]].round(6).to_csv(RESULTS_DIR / "stock_clusters.csv")
print(f"\nFigures in {FIG_DIR}, tables in {RESULTS_DIR}")
