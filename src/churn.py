import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT           = Path(__file__).resolve().parents[1]
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
FIGURES_DIR    = ROOT / "reports" / "figures"

AUC_ROC_GATE         = 0.88
PRECISION_TOP20_GATE = 0.75
MLFLOW_RUN_NAME      = "churn_xgboost_v1"

SNAPSHOT        = pd.Timestamp("2011-06-01")
_CHURN_SEGMENTS = {"At Risk", "Lost"}


def load_and_preprocess(retail_path: Path = DATA_PROCESSED / "retail_clean.csv"):
    df = pd.read_csv(
        retail_path,
        usecols=["Customer ID", "Invoice", "InvoiceDate", "Revenue", "StockCode", "Quantity"],
        parse_dates=["InvoiceDate"],
    )

    obs      = df[df["InvoiceDate"] < SNAPSHOT]
    cutoff90 = SNAPSHOT - pd.Timedelta(days=90)
    cutoff30 = SNAPSHOT - pd.Timedelta(days=30)

    records = []
    for cid, grp in obs.groupby("Customer ID"):
        inv_dates = grp.groupby("Invoice")["InvoiceDate"].min().sort_values()
        last_dt   = inv_dates.max()
        freq      = len(inv_dates)
        pos_rev   = grp.loc[grp["Revenue"] > 0, "Revenue"]
        monetary  = pos_rev.sum()
        recency   = (SNAPSHOT - last_dt).days
        inter     = inv_dates.diff().dt.days.dropna()  # type: ignore[union-attr]
        recent90  = grp[grp["InvoiceDate"] >= cutoff90]
        inv_qty   = grp.groupby("Invoice")["Quantity"].sum()

        records.append({
            "Customer ID":             cid,
            "recency":                 recency,
            "frequency":               freq,
            "monetary":                monetary,
            "tenure":                  max((last_dt - inv_dates.min()).days, 1),
            "avg_order_value":         monetary / max(freq, 1),
            "avg_qty_per_order":       inv_qty.mean(),
            "n_unique_products":       grp["StockCode"].nunique(),
            "avg_days_between_orders": float(inter.mean()) if freq > 1 else float(recency),
            "purchase_regularity":     float(inter.std(ddof=0)) if len(inter) > 0 else 0.0,
            "recent_freq_90d":         recent90["Invoice"].nunique(),
            "revenue_last_30d": (
                grp.loc[(grp["InvoiceDate"] >= cutoff30) & (grp["Revenue"] > 0), "Revenue"].sum()
            ),
            "n_months_active":         grp["InvoiceDate"].dt.to_period("M").nunique(),  # type: ignore[attr-defined]
            "spend_trend_90d": (
                grp.loc[(grp["InvoiceDate"] >= cutoff90) & (grp["Revenue"] > 0), "Revenue"].sum()
                / (grp.loc[(grp["InvoiceDate"] < cutoff90) & (grp["Revenue"] > 0), "Revenue"].sum() + 1.0)
            ),
        })

    feat_df = pd.DataFrame(records)

    rfm = pd.read_csv(DATA_PROCESSED / "rfm_scores.csv", usecols=["Customer ID", "Segment"])
    seg  = rfm.set_index("Customer ID")["Segment"]
    feat_df["churned"] = feat_df["Customer ID"].map(
        lambda cid: int(seg.get(cid, "Unknown") in _CHURN_SEGMENTS)
    )
    feat_df = feat_df[feat_df["Customer ID"].isin(seg.index)].reset_index(drop=True)

    customer_ids = feat_df["Customer ID"]
    y            = feat_df["churned"]
    X            = feat_df.drop(columns=["Customer ID", "churned"])
    return X, y, customer_ids


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    n_top = max(1, int(len(y_true) * k))
    idx   = np.argsort(y_prob)[::-1][:n_top]
    return float(y_true[idx].mean())


def _tune(X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> dict:
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": spw,
            "tree_method":      "hist",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        scores = []
        for tr, val in cv.split(X_train, y_train):
            m = XGBClassifier(**params)
            m.fit(X_train[tr], y_train[tr])
            scores.append(roc_auc_score(y_train[val], m.predict_proba(X_train[val])[:, 1]))
        return float(np.mean(scores))  # type: ignore[no-matching-overload]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["scale_pos_weight"] = spw
    return best


def _compute_metrics(model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray):
    y_prob        = model.predict_proba(X_test)[:, 1]
    y_pred        = (y_prob >= 0.5).astype(int)
    pr_p, pr_r, _ = precision_recall_curve(y_test, y_prob)
    metrics = {
        "auc_roc":         roc_auc_score(y_test, y_prob),
        "auc_pr":          auc(pr_r, pr_p),
        "f1":              f1_score(y_test, y_pred),
        "precision":       precision_score(y_test, y_pred),
        "recall":          recall_score(y_test, y_pred),
        "precision_top20": precision_at_top_k(y_test, y_prob, k=0.20),
    }
    return metrics, y_prob


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["No Churn", "Churn"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    threshold = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > threshold else "black")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_shap(model: XGBClassifier, X_df: pd.DataFrame, y_test: np.ndarray, out_dir: Path) -> None:
    explainer  = shap.TreeExplainer(model)
    sv         = explainer.shap_values(X_df)
    shap_exp   = explainer(X_df)
    feat_names = X_df.columns.tolist()

    shap.summary_plot(sv, X_df, feature_names=feat_names, show=False)
    plt.savefig(out_dir / "churn_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(sv, X_df, feature_names=feat_names, plot_type="bar", show=False)
    plt.savefig(out_dir / "churn_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    for label, mask in [("churn", y_test == 1), ("no_churn", y_test == 0)]:
        idx = int(np.where(mask)[0][0])
        shap.plots.waterfall(shap_exp[idx], show=False)
        plt.savefig(out_dir / f"churn_shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    top_idx = int(np.abs(sv).mean(axis=0).argmax())
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(top_idx, sv, X_df, feature_names=feat_names, ax=ax, show=False)
    fig.tight_layout()
    fig.savefig(out_dir / "churn_shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(n_trials: int = 50) -> dict:
    X, y, cids = load_and_preprocess()

    X_tr, X_te, y_tr, y_te, _, cids_te = train_test_split(
        X.values, y.values, cids.values,
        test_size=0.20, stratify=y.values, random_state=42,  # type: ignore[arg-type]
    )

    print("Tuning hyperparameters ...")
    best = _tune(X_tr, y_tr, n_trials)

    print("Training final model ...")
    model = XGBClassifier(**best, tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr)

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr, val in cv.split(X_tr, y_tr):
        m = XGBClassifier(**best, tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr[tr], y_tr[tr])
        cv_scores.append(roc_auc_score(y_tr[val], m.predict_proba(X_tr[val])[:, 1]))

    metrics, y_prob = _compute_metrics(model, X_te, y_te)
    fpr, tpr, _     = roc_curve(y_te, y_prob)
    X_te_df         = pd.DataFrame(X_te, columns=X.columns.tolist())

    print("\n--- Evaluation ---")
    for k, v in metrics.items():
        print(f"  {k:<22} {v:.4f}")
    print(f"  {'cv_auc_mean':<22} {np.mean(cv_scores):.4f}")  # type: ignore[no-matching-overload]
    print(f"  {'cv_auc_std':<22} {np.std(cv_scores):.4f}")  # type: ignore[no-matching-overload]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "churn_xgboost.pkl"
    joblib.dump(model, model_path)

    _plot_roc(fpr, tpr, FIGURES_DIR / "churn_roc_curve.png")
    _plot_confusion_matrix(y_te, (y_prob >= 0.5).astype(int), FIGURES_DIR / "churn_confusion_matrix.png")
    _plot_shap(model, X_te_df, y_te, FIGURES_DIR)

    pred_path = DATA_PROCESSED / "churn_predictions.csv"
    pd.DataFrame({
        "Customer_ID":       cids_te,
        "churn_probability": y_prob,
        "predicted_churn":   (y_prob >= 0.5).astype(int),
        "actual_churn":      y_te,
    }).to_csv(pred_path, index=False)

    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.log_params({**best, "cv_folds": 5, "test_size": 0.20, "n_trials": n_trials})
        mlflow.log_metrics({
            **metrics,
            "cv_auc_mean": float(np.mean(cv_scores)),  # type: ignore[no-matching-overload]
            "cv_auc_std":  float(np.std(cv_scores)),   # type: ignore[no-matching-overload]
        })
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(pred_path))
        for png in sorted(FIGURES_DIR.glob("churn_*.png")):
            mlflow.log_artifact(str(png))

    failures = []
    if metrics["auc_roc"] < AUC_ROC_GATE:
        failures.append(f"AUC-ROC {metrics['auc_roc']:.4f} < {AUC_ROC_GATE}")
    if metrics["precision_top20"] < PRECISION_TOP20_GATE:
        failures.append(f"precision@top20 {metrics['precision_top20']:.4f} < {PRECISION_TOP20_GATE}")

    if failures:
        for msg in failures:
            print(f"WARNING: acceptance criterion not met — {msg}", file=sys.stderr)
        sys.exit(1)

    return metrics
