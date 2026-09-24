"""
Step 1 - clean the raw stock prices file.

Run it from the project folder:
    python src/01_clean_data.py

Reads  data/raw/2__Stock_Prices_Data_Set.csv
Writes data/stock_prices_clean.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "2__Stock_Prices_Data_Set.csv"
CLEAN_FILE = ROOT / "data" / "stock_prices_clean.csv"

PRICE_COLS = ["open", "high", "low", "close"]

df = pd.read_csv(RAW_FILE, parse_dates=["date"])
start_rows = len(df)
print(f"Raw file: {start_rows:,} rows, {df['symbol'].nunique()} symbols")

# make sure tickers have no stray spaces or lowercase letters
df["symbol"] = df["symbol"].str.strip().str.upper()

# 1. duplicates (same stock, same day)
n_dup = df.duplicated(subset=["symbol", "date"]).sum()
df = df.drop_duplicates(subset=["symbol", "date"])
print(f"Duplicates removed: {n_dup}")

# 2. missing prices
# 11 rows have no open price (8 of them also miss high and low). That is 0.002% of the
# file and I didn't want to make up prices, so these rows are simply dropped.
missing = df[PRICE_COLS].isna().any(axis=1)
print(f"Rows with a missing price: {missing.sum()}")
df = df[~missing]

# 3. prices that can't be right
# - a high lower than the low is impossible
# - an open or close more than 20% outside the day's low-high range is a bad tick
#   (for example CHD opening at 18.77 on a day it traded around 34)
bad_range = df["high"] < df["low"]
far_open = (df["open"] < df["low"] * 0.8) | (df["open"] > df["high"] * 1.2)
far_close = (df["close"] < df["low"] * 0.8) | (df["close"] > df["high"] * 1.2)
bad = bad_range | far_open | far_close

print(f"Rows with impossible prices: {bad.sum()}")
print(df.loc[bad, ["symbol", "date"] + PRICE_COLS].to_string(index=False))
df = df[~bad]

df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# things I checked but decided to leave in the file ------------------------------
# a few rows have the close a little outside the low-high range (spin-offs, ex-dividend
# days...). Not obviously wrong, so they stay.
slightly_off = (
    (df["close"] > df["high"]) | (df["close"] < df["low"])
    | (df["open"] > df["high"]) | (df["open"] < df["low"])
)
print(f"\nKept, but open/close slightly outside the low-high range: {slightly_off.sum()} rows")

# the prices are not adjusted for splits or spin-offs, so some days look like a crash
daily_move = df.groupby("symbol")["close"].pct_change()
big_moves = df.loc[daily_move.abs() > 0.4, ["symbol", "date", "close"]]
big_moves = big_moves.assign(move_pct=(daily_move[big_moves.index] * 100).round(1))
print(f"\nKept, one-day moves above 40% ({len(big_moves)} rows):")
print(big_moves.to_string(index=False))

df.to_csv(CLEAN_FILE, index=False, date_format="%Y-%m-%d")
print(f"\nClean file: {len(df):,} rows ({start_rows - len(df)} removed) -> {CLEAN_FILE}")
