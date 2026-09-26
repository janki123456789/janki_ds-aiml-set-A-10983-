from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(404)
n = 300
cols = ['visits', 'recency', 'engagement', 'spend']
x = rng.normal(size=(n, 4))
group = rng.choice(["G1", "G2"], size=n)
score = x @ np.array([0.6, -0.9, 1.1, 0.4])
score += 0.4 * (group == "G2") + rng.normal(0, 1, n)
y = (score > 0).astype(int)
df = pd.DataFrame(np.round(50 + 10*x, 2), columns=cols)
df.insert(0, "record_id", np.arange(1, n+1))
df["group"] = group
df["response"] = y
for col in cols[:2]:
    df.loc[rng.choice(n, 15, replace=False), col] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/set_d.csv", index=False)




















































"""
Data Science & AI/ML Practical Exam - Set D: Campaign Response
Run from repository root:
    python campaign_response_exam.py

This script creates the required outputs and saved models.
"""
from pathlib import Path
import json, sys, platform, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.exceptions import UndefinedMetricWarning
import joblib

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SEED = 42
np.random.seed(SEED)

ROOT = Path(".")
DATA = ROOT / "data/raw/set_d.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SPLITS = OUT / "splits"
METRICS = OUT / "metrics"
PREDS = OUT / "predictions"
MODELS = ROOT / "models"
for d in [FIG, SPLITS, METRICS, PREDS, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

NUMERIC = ["visits", "recency", "engagement", "spend"]
TARGET = "response"
ID = "record_id"
CAT = "group"
ENGINEERED = "engineered_feature"

def metric_row(name, y_true, y_pred, prob=None):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

print("=" * 70)
print("SET D - CAMPAIGN RESPONSE PRACTICAL")
print("=" * 70)

# ------------------------------------------------------------------
# 0. DATA SETUP
# ------------------------------------------------------------------
if not DATA.exists():
    raise FileNotFoundError("Run: python src/generate_data.py first")

raw = pd.read_csv(DATA)
print("\nRAW DATA SHAPE:", raw.shape)
print("EXACT DUPLICATES:", raw.duplicated().sum())
print("MISSING COUNTS:\n", raw[NUMERIC].isna().sum())
print("TARGET COUNTS:\n", raw[TARGET].value_counts().sort_index())

clean = raw.drop_duplicates().reset_index(drop=True)
assert len(clean) == 300, f"Expected 300 unique records, got {len(clean)}"

# First: 80/20 train/test, then 80/20 fit/validation inside training.
train_df, test_df = train_test_split(
    clean, test_size=0.20, stratify=clean[TARGET], random_state=SEED
)
fit_df, val_df = train_test_split(
    train_df, test_size=0.20, stratify=train_df[TARGET], random_state=SEED
)

# Preserve IDs.
pd.DataFrame({"record_id": fit_df[ID]}).to_csv(SPLITS/"fit_ids.csv", index=False)
pd.DataFrame({"record_id": val_df[ID]}).to_csv(SPLITS/"validation_ids.csv", index=False)
pd.DataFrame({"record_id": test_df[ID]}).to_csv(SPLITS/"test_ids.csv", index=False)

fit_ids, val_ids, test_ids = set(fit_df[ID]), set(val_df[ID]), set(test_df[ID])
assert len(fit_ids) == 192 and len(val_ids) == 48 and len(test_ids) == 60
assert not fit_ids & val_ids and not fit_ids & test_ids and not val_ids & test_ids

pd.DataFrame({
    "partition": ["fit", "validation", "test"],
    "rows": [len(fit_df), len(val_df), len(test_df)]
}).to_csv(SPLITS/"partition_sizes.csv", index=False)

# ------------------------------------------------------------------
# TASK 1 - MATHS & ADVANCED STATISTICS
# ------------------------------------------------------------------
# M1: observed engagement only, fit rows, no imputation.
eng = fit_df["engagement"].dropna()
summary = {
    "observed_n": int(eng.count()),
    "mean": float(eng.mean()),
    "median": float(eng.median()),
    "sample_std_ddof1": float(eng.std(ddof=1)),
}
pd.DataFrame([summary]).to_csv(METRICS/"statistics_summary.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.hist(eng, bins=12, edgecolor="black")
plt.title("Engagement distribution - fit rows")
plt.xlabel("Engagement")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIG/"engagement_histogram.png", dpi=150)
plt.close()

# M2: Welch two-sided G1 vs G2 using observed values.
g1 = fit_df.loc[fit_df[CAT] == "G1", "engagement"].dropna()
g2 = fit_df.loc[fit_df[CAT] == "G2", "engagement"].dropna()
welch = stats.ttest_ind(g1, g2, equal_var=False, alternative="two-sided")

overall = eng
mean = overall.mean()
sem = stats.sem(overall)
ci_low, ci_high = stats.t.interval(0.95, df=len(overall)-1, loc=mean, scale=sem)

inference = {
    "G1_n_observed": int(len(g1)),
    "G2_n_observed": int(len(g2)),
    "G1_mean": float(g1.mean()),
    "G2_mean": float(g2.mean()),
    "welch_t": float(welch.statistic),
    "welch_p_value": float(welch.pvalue),
    "alpha": 0.05,
    "overall_n_observed": int(len(overall)),
    "overall_mean": float(mean),
    "CI95_low": float(ci_low),
    "CI95_high": float(ci_high),
    "assumptions": "independent groups; approximate normality/robustness for Welch test"
}
pd.DataFrame([inference]).to_csv(METRICS/"inference_results.csv", index=False)

# M3: covariance and eigendecomposition on complete fit rows.
complete = fit_df[["engagement", "visits"]].dropna()
X = complete.to_numpy(dtype=float)
Xc = X - X.mean(axis=0)
cov = (Xc.T @ Xc) / (len(Xc) - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
variance_share = float(eigvals[0] / eigvals.sum())
cov_df = pd.DataFrame(cov, index=["engagement", "visits"], columns=["engagement", "visits"])
cov_df.to_csv(METRICS/"covariance_matrix.csv")
pd.DataFrame({
    "eigenvalue": eigvals,
    "variance_share": eigvals / eigvals.sum()
}).to_csv(METRICS/"eigenvalues.csv", index=False)

# ------------------------------------------------------------------
# TASK 2 - PREPROCESSING & FEATURE ENGINEERING
# ------------------------------------------------------------------
# Fit imputation only on the 192 fit records.
imputer = SimpleImputer(strategy="median")
fit_num_imp = pd.DataFrame(
    imputer.fit_transform(fit_df[NUMERIC]),
    columns=NUMERIC, index=fit_df.index
)
val_num_imp = pd.DataFrame(
    imputer.transform(val_df[NUMERIC]),
    columns=NUMERIC, index=val_df.index
)
test_num_imp = pd.DataFrame(
    imputer.transform(test_df[NUMERIC]),
    columns=NUMERIC, index=test_df.index
)

# Required feature after numeric imputation, before scaling.
for frame in [fit_num_imp, val_num_imp, test_num_imp]:
    frame[ENGINEERED] = frame["engagement"] / (frame["recency"] + 1)

NUMERIC_ENGINEERED = NUMERIC + [ENGINEERED]

# One-hot group. Fit encoder only on fit.
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

fit_cat = encoder.fit_transform(fit_df[[CAT]])
val_cat = encoder.transform(val_df[[CAT]])
test_cat = encoder.transform(test_df[[CAT]])
cat_names = list(encoder.get_feature_names_out([CAT]))

# Scale numeric + engineered features using fit only.
scaler = StandardScaler()
fit_scaled = scaler.fit_transform(fit_num_imp[NUMERIC_ENGINEERED])
val_scaled = scaler.transform(val_num_imp[NUMERIC_ENGINEERED])
test_scaled = scaler.transform(test_num_imp[NUMERIC_ENGINEERED])

# Group one-hot columns stay unscaled.
X_fit = np.hstack([fit_scaled, fit_cat])
X_val = np.hstack([val_scaled, val_cat])
X_test = np.hstack([test_scaled, test_cat])
FEATURE_NAMES = NUMERIC_ENGINEERED + cat_names

assert np.isfinite(X_fit).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

pd.DataFrame({
    "feature": FEATURE_NAMES,
    "index": range(len(FEATURE_NAMES))
}).to_csv(METRICS/"feature_names.csv", index=False)

pd.DataFrame({
    "fit_shape": [str(X_fit.shape)],
    "validation_shape": [str(X_val.shape)],
    "test_shape": [str(X_test.shape)]
}).to_csv(METRICS/"transformed_shapes.csv", index=False)

joblib.dump({
    "imputer": imputer,
    "encoder": encoder,
    "scaler": scaler,
    "numeric": NUMERIC,
    "engineered": ENGINEERED,
    "features": FEATURE_NAMES,
}, MODELS/"preprocessing.joblib")

# ------------------------------------------------------------------
# TASK 3 - SUPERVISED LEARNING
# ------------------------------------------------------------------
y_fit, y_val, y_test = fit_df[TARGET], val_df[TARGET], test_df[TARGET]

dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy.fit(X_fit, y_fit)
dummy_pred = dummy.predict(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=SEED)
logreg.fit(X_fit, y_fit)
log_pred = logreg.predict(X_test)
log_prob = logreg.predict_proba(X_test)[:, 1]

metrics = pd.DataFrame([
    metric_row("DummyClassifier", y_test, dummy_pred),
    metric_row("LogisticRegression", y_test, log_pred, log_prob)
])
metrics.to_csv(METRICS/"baseline_logistic_metrics.csv", index=False)

test_predictions = pd.DataFrame({
    "record_id": test_df[ID].to_numpy(),
    "true_label": y_test.to_numpy(),
    "predicted_label": log_pred,
    "class_1_probability": log_prob
})
test_predictions.to_csv(PREDS/"logistic_test_predictions.csv", index=False)

cm = confusion_matrix(y_test, log_pred, labels=[0, 1])
pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
    METRICS/"logistic_confusion_matrix.csv"
)

# ------------------------------------------------------------------
# TASK 4 - UNSUPERVISED LEARNING
# ------------------------------------------------------------------
# Required: transformed numeric predictors + engineered feature; no group one-hot.
k_rows = []
models_k = {}
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(fit_scaled)
    sil = silhouette_score(fit_scaled, labels)
    k_rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    models_k[k] = km

k_scores = pd.DataFrame(k_rows)
k_scores.to_csv(METRICS/"k_selection_scores.csv", index=False)

best_sil = k_scores["silhouette"].max()
best_k = int(k_scores.loc[np.isclose(k_scores["silhouette"], best_sil), "k"].min())
best_km = models_k[best_k]
fit_cluster = best_km.labels_

profile = fit_num_imp.copy()
profile["cluster"] = fit_cluster
profile = profile.groupby("cluster")[NUMERIC_ENGINEERED].mean()
profile.to_csv(METRICS/"cluster_profiles.csv")

# ------------------------------------------------------------------
# TASK 5 - ANN (TensorFlow/Keras)
# ------------------------------------------------------------------
ann_available = True
try:
    import tensorflow as tf
except Exception as exc:
    ann_available = False
    tf = None
    print("\nTensorFlow is not available. ANN section will be skipped.")
    print("Install it with: python -m pip install tensorflow")

ann_metrics = None
if ann_available:
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_fit.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_fit, y_fit.to_numpy(),
        validation_data=(X_val, y_val.to_numpy()),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(METRICS/"ann_history.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["loss"], label="Training loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("ANN training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG/"ann_loss_curve.png", dpi=150)
    plt.close()

    ann_prob = model.predict(X_test, verbose=0).ravel()
    ann_pred = (ann_prob >= 0.5).astype(int)
    ann_metrics = metric_row("ANN", y_test, ann_pred, ann_prob)
    pd.DataFrame([ann_metrics]).to_csv(METRICS/"ann_metrics.csv", index=False)

    pd.DataFrame({
        "record_id": test_df[ID].to_numpy(),
        "true_label": y_test.to_numpy(),
        "predicted_label": ann_pred,
        "class_1_probability": ann_prob
    }).to_csv(PREDS/"ann_test_predictions.csv", index=False)

    model.save(MODELS/"campaign_ann.keras")
    with open(MODELS/"ann_architecture.json", "w", encoding="utf-8") as f:
        f.write(model.to_json())

# ------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------
summary = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": __import__("scipy").__version__,
    "sklearn": __import__("sklearn").__version__,
    "joblib": joblib.__version__,
    "tensorflow": None if not ann_available else tf.__version__,
    "raw_rows": int(len(raw)),
    "clean_unique_rows": int(len(clean)),
    "fit_rows": int(len(fit_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "best_k_by_silhouette": best_k,
    "welch_p_value": float(welch.pvalue),
    "engagement_ci95_low": float(ci_low),
    "engagement_ci95_high": float(ci_high),
    "pc1_variance_share": variance_share,
}
if ann_metrics:
    summary["logistic_f1"] = float(metrics.loc[metrics.model=="LogisticRegression", "f1"].iloc[0])
    summary["ann_f1"] = float(ann_metrics["f1"])
with open(METRICS/"run_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDONE.")
print("Fit / validation / test:", len(fit_df), len(val_df), len(test_df))
print("Welch p-value:", round(welch.pvalue, 6))
print("95% CI for overall observed engagement mean:", round(ci_low, 4), "to", round(ci_high, 4))
print("PC1 variance share:", round(variance_share, 4))
print("Best k:", best_k)
print("\nOpen outputs/metrics and outputs/figures to inspect your evidence.")















