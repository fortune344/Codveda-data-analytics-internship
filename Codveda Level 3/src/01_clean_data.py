"""
Step 1 - clean the two raw datasets of Level 3.

Run it from the project folder:
    python src/01_clean_data.py

Reads   data/raw/churn-bigml-80.csv, churn-bigml-20.csv, 3__Sentiment_dataset.csv
Writes  data/churn_train_clean.csv, data/churn_test_clean.csv, data/sentiment_clean.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"


def snake(name):
    return name.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------- churn (Task 1)
print("=== Churn files ===")
train = pd.read_csv(RAW / "churn-bigml-80.csv")
test = pd.read_csv(RAW / "churn-bigml-20.csv")
print(f"train: {train.shape}, test: {test.shape}")

for name, d in (("train", train), ("test", test)):
    print(f"{name}: missing = {int(d.isna().sum().sum())}, duplicated rows = {int(d.duplicated().sum())}")

# same customer in both files would be a leak, so check it
overlap = pd.concat([train, test]).duplicated().sum()
print(f"rows that appear in both files: {overlap}")


def clean_churn(d):
    d = d.copy()
    d.columns = [snake(c) for c in d.columns]
    d["churn"] = d["churn"].astype(int)                      # True/False -> 1/0
    for col in ("state", "international_plan", "voice_mail_plan"):
        d[col] = d[col].str.strip()
    return d


train_clean = clean_churn(train)
test_clean = clean_churn(test)

# minutes and charges are the same information (correlation of 1.00), just noting it here.
# The dropping is done in the modelling script, not in the file.
print("corr minutes/charge (day):", round(train_clean["total_day_minutes"].corr(train_clean["total_day_charge"]), 4))
print("churn rate: train {:.1%}, test {:.1%}".format(train_clean["churn"].mean(), test_clean["churn"].mean()))

train_clean.to_csv(OUT / "churn_train_clean.csv", index=False)
test_clean.to_csv(OUT / "churn_test_clean.csv", index=False)


# ------------------------------------------------------------ sentiment (Task 3)
print("\n=== Sentiment file ===")
s = pd.read_csv(RAW / "3__Sentiment_dataset.csv")
start = len(s)
print(f"raw: {s.shape}")

# two leftover index columns from a previous export
s = s.drop(columns=[c for c in s.columns if c.startswith("Unnamed")])

# almost every text column is padded with spaces ("USA       ", " Joy ")
before = {c: s[c].nunique() for c in ("Country", "Platform", "Sentiment")}
for col in s.columns:
    if pd.api.types.is_string_dtype(s[col]):
        s[col] = s[col].str.strip()
after = {c: s[c].nunique() for c in ("Country", "Platform", "Sentiment")}
print("distinct values before/after stripping:", {c: (before[c], after[c]) for c in before})

s["Timestamp"] = pd.to_datetime(s["Timestamp"])
s[["Retweets", "Likes"]] = s[["Retweets", "Likes"]].astype(int)   # they were stored as 15.0, 30.0...

# the same text can appear twice
n_dup = s.duplicated(subset="Text").sum()
s = s.drop_duplicates(subset="Text").reset_index(drop=True)
print(f"duplicated texts removed: {n_dup}")

# the file calls the emotion "Sentiment" but it holds 190+ different emotions, not
# positive/negative/neutral. I rename it so it doesn't get mixed up with the 3-class result.
s = s.rename(columns={"Sentiment": "emotion"})
s.columns = [snake(c) for c in s.columns]

print(f"missing values: {int(s.isna().sum().sum())}")
print(f"clean: {s.shape} ({start - len(s)} rows removed)")
s.to_csv(OUT / "sentiment_clean.csv", index=False)

print("\nFiles written in", OUT)
