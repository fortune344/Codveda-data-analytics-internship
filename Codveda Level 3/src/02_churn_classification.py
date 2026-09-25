"""
Task 1 - Predictive modelling (classification): will a telecom customer churn?

Three models (Logistic Regression, Decision Tree, Random Forest) are trained on the
80% file, compared, tuned with grid search, and finally evaluated on the 20% file.

Run it from the project folder, after 01_clean_data.py:
    python src/02_churn_classification.py

Figures go to figures/, tables to results/.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
sns.set_theme(style="whitegrid")

train = pd.read_csv(ROOT / "data" / "churn_train_clean.csv")
test = pd.read_csv(ROOT / "data" / "churn_test_clean.csv")
print(f"train: {train.shape}, test: {test.shape}")
print(f"churn rate: train {train['churn'].mean():.1%}, test {test['churn'].mean():.1%}")


# 1. a first look at who churns -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

by_calls = train.groupby("customer_service_calls")["churn"].agg(["mean", "size"])
axes[0].bar(by_calls.index, by_calls["mean"] * 100, color="steelblue")
axes[0].set_title("Churn rate by number of customer service calls")
axes[0].set_xlabel("Customer service calls")
axes[0].set_ylabel("Customers who left (%)")

by_plan = train.groupby("international_plan")["churn"].mean() * 100
axes[1].bar(by_plan.index, by_plan.values, color=["steelblue", "crimson"])
axes[1].set_title("Churn rate by international plan")
axes[1].set_xlabel("International plan")
axes[1].set_ylabel("Customers who left (%)")
fig.tight_layout()
fig.savefig(FIG_DIR / "churn_1_who_leaves.png", dpi=130)
plt.close(fig)

print("\nChurn rate by customer service calls (%):")
print((by_calls["mean"] * 100).round(1).to_string())
print("Churn rate by international plan (%):", by_plan.round(1).to_dict())


# 2. preprocessing ------------------------------------------------------------------
# the charge columns are the minutes columns multiplied by a fixed price (correlation 1.00),
# so I keep the minutes and drop the charges
DROP = ["total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"]
CATEGORICAL = ["state", "area_code", "international_plan", "voice_mail_plan"]
TARGET = "churn"

X_train = train.drop(columns=DROP + [TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=DROP + [TARGET])
y_test = test[TARGET]
NUMERIC = [c for c in X_train.columns if c not in CATEGORICAL]

# area_code is a number in the file but it is a label (415, 408, 510), so it is encoded
X_train["area_code"] = X_train["area_code"].astype(str)
X_test["area_code"] = X_test["area_code"].astype(str)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])


def make_pipeline(model):
    return Pipeline([("prep", preprocess), ("model", model)])


# 3. three models, default settings -------------------------------------------------
# only 14.6% of the customers churn, so class_weight="balanced" is used everywhere
# to stop the models from just predicting "stays" all the time
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200,
                                            random_state=SEED, n_jobs=-1),
}


def scores(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


rows = []
for name, model in models.items():
    pipe = make_pipeline(model).fit(X_train, y_train)
    rows.append({"model": name, "version": "default", **scores(pipe, X_test, y_test)})

default_table = pd.DataFrame(rows)
print("\nDefault settings, evaluated on the test file:")
print(default_table.round(3).to_string(index=False))


# 4. grid search --------------------------------------------------------------------
# 5-fold cross validation on the training file only, scored with the F1 of the churn class
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, 12, 20, None],
        "model__min_samples_leaf": [1, 2, 5],
    },
}

tuned = {}
cv_scores = {}
rows = []
best_params = []
for name, model in models.items():
    search = GridSearchCV(make_pipeline(model), grids[name], scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    cv_scores[name] = search.best_score_
    rows.append({"model": name, "version": "tuned", **scores(search.best_estimator_, X_test, y_test)})
    clean_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    best_params.append({"model": name, "cv_f1": round(search.best_score_, 3), "best_params": clean_params})
    print(f"{name}: best CV F1 = {search.best_score_:.3f}  {clean_params}")

tuned_table = pd.DataFrame(rows)
print("\nAfter grid search, evaluated on the test file:")
print(tuned_table.round(3).to_string(index=False))

comparison = pd.concat([default_table, tuned_table]).round(4)
comparison.to_csv(RESULTS_DIR / "churn_model_comparison.csv", index=False)
pd.DataFrame(best_params).to_csv(RESULTS_DIR / "churn_best_params.csv", index=False)


# 5. charts -------------------------------------------------------------------------
# comparison of the tuned models
metrics = ["accuracy", "precision", "recall", "f1"]
plot_df = tuned_table.set_index("model")[metrics]
ax = plot_df.plot(kind="bar", figsize=(10, 5), width=0.8, colormap="Set2")
ax.set_title("Tuned models on the test file")
ax.set_xlabel("")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4)
plt.tight_layout()
plt.savefig(FIG_DIR / "churn_2_model_comparison.png", dpi=130)
plt.close()

# the final model is chosen with the cross-validation score (training file only),
# so the test file stays a fair, untouched exam
best_name = max(cv_scores, key=cv_scores.get)
best_model = tuned[best_name]
print(f"\nBest tuned model according to cross-validation: {best_name}")
print(f"Churners in the test file: {int(y_test.sum())} out of {len(y_test)}")
print(f"Accuracy of a model that always says 'stays': {1 - y_test.mean():.3f}")

tn, fp, fn, tp = confusion_matrix(y_test, best_model.predict(X_test)).ravel()
print(f"Confusion matrix: {tp} churners found, {fn} missed, {fp} false alarms, {tn} correct 'stays'")

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Stays", "Churns"],
                                      cmap="Blues", colorbar=False, ax=ax)
ax.set_title(f"{best_name} - confusion matrix (test file)")
fig.tight_layout()
fig.savefig(FIG_DIR / "churn_3_confusion_matrix.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, model in tuned.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_title("ROC curves (test file)")
fig.tight_layout()
fig.savefig(FIG_DIR / "churn_4_roc_curves.png", dpi=130)
plt.close(fig)

# which variables matter? (only works for tree-based models)
if best_name != "Logistic Regression":
    feature_names = best_model.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(best_model.named_steps["model"].feature_importances_, index=feature_names)
    importances.index = [n.replace("num__", "").replace("cat__", "") for n in importances.index]

    # one-hot encoding splits a variable in several columns (international_plan_Yes / _No,
    # 51 columns for the states...), so I add them back together
    def original_name(col):
        for cat in CATEGORICAL:
            if col.startswith(cat + "_"):
                return cat
        return col

    top = importances.groupby(original_name).sum().sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, color="steelblue", ax=ax)
    ax.set_title(f"{best_name} - 12 most important variables")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_5_feature_importance.png", dpi=130)
    plt.close(fig)
    print("\nTop variables:")
    print(top.round(3).to_string())

print(f"\nFigures in {FIG_DIR}, tables in {RESULTS_DIR}")
