"""
Task 2 - Time series analysis.

I work on one stock (AAPL, but you can change SYMBOL below) and look at its closing price:
plot, decomposition (trend / seasonality / residuals), moving averages.

Run it from the project folder, after 01_clean_data.py:
    python src/02_time_series_analysis.py

Figures go to the figures/ folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no window, we only save the pictures
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)   # statsmodels warns about adfuller

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "stock_prices_clean.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SYMBOL = "AAPL"
TRADING_DAYS_PER_YEAR = 252

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
close = df.loc[df["symbol"] == SYMBOL].set_index("date")["close"].sort_index()

if len(close) < 2 * TRADING_DAYS_PER_YEAR:
    raise SystemExit(f"Not enough data for {SYMBOL} ({len(close)} days) to decompose a yearly pattern")

print(f"{SYMBOL}: {len(close)} trading days, {close.index.min().date()} to {close.index.max().date()}")


# 1. plot the series ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(close.index, close.values, color="steelblue")
ax.set_title(f"{SYMBOL} - daily closing price")
ax.set_xlabel("Date")
ax.set_ylabel("Closing price (USD)")
fig.tight_layout()
fig.savefig(FIG_DIR / "ts_1_closing_price.png", dpi=130)
plt.close(fig)

# quick summary per year
by_year = close.groupby(close.index.year).agg(["first", "last", "min", "max"])
by_year["change_pct"] = (by_year["last"] / by_year["first"] - 1) * 100
print("\nPrice by year:")
print(by_year.round(1))


# 2. decomposition -----------------------------------------------------------------
# multiplicative because the ups and downs get bigger when the price is higher.
# period = 252 trading days, roughly one year.
parts = seasonal_decompose(close, model="multiplicative", period=TRADING_DAYS_PER_YEAR)

fig = parts.plot()
fig.set_size_inches(12, 9)
fig.tight_layout()
fig.savefig(FIG_DIR / "ts_2_decomposition.png", dpi=130)
plt.close(fig)

print("\nDecomposition:")
print(f"  seasonal factor goes from {parts.seasonal.min():.3f} to {parts.seasonal.max():.3f}")
print(f"  residual std = {parts.resid.std():.3f}")

# does the seasonality show up by calendar month too?
month_end = close.resample("ME").last()
monthly_return = month_end.pct_change().dropna() * 100
avg_by_month = monthly_return.groupby(monthly_return.index.month).mean()

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=avg_by_month.index, y=avg_by_month.values, color="steelblue", ax=ax)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(f"{SYMBOL} - average monthly return for each calendar month")
ax.set_xlabel("Month")
ax.set_ylabel("Average return (%)")
fig.tight_layout()
fig.savefig(FIG_DIR / "ts_3_monthly_returns.png", dpi=130)
plt.close(fig)

print("\nAverage return by calendar month (%):")
print(avg_by_month.round(1).to_string())


# 3. moving averages ---------------------------------------------------------------
ma_30 = close.rolling(30).mean()
ma_90 = close.rolling(90).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(close.index, close.values, color="lightgray", label="Closing price")
ax.plot(ma_30.index, ma_30.values, color="steelblue", label="30-day moving average")
ax.plot(ma_90.index, ma_90.values, color="crimson", label="90-day moving average")
ax.set_title(f"{SYMBOL} - moving averages")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "ts_4_moving_averages.png", dpi=130)
plt.close(fig)


# 4. stationarity check (extra) ------------------------------------------------------
def adf(series, label):
    stat, p_value = adfuller(series.dropna())[:2]
    verdict = "stationary" if p_value < 0.05 else "not stationary"
    print(f"  {label:<14} ADF = {stat:7.2f}   p = {p_value:.4f}   -> {verdict}")


print("\nAugmented Dickey-Fuller test:")
adf(close, "closing price")
adf(close.pct_change(), "daily returns")

print(f"\nFigures saved in {FIG_DIR}")
