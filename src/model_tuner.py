"""Multi-model feature importance analysis and Optuna hyperparameter tuning for churn prediction."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.churn import load_and_preprocess, precision_at_top_k

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT           = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
FIGURES_DIR    = ROOT / "reports" / "figures"

FEATURE_SUBSET_SIZES = [5, 8, 10, 13]
N_CV_FOLDS           = 5


def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc_roc":         float(roc_auc_score(y_test, y_prob)),
        "f1":              float(f1_score(y_test, y_pred, zero_division=0)),
        "precision":       float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":          float(recall_score(y_test, y_pred, zero_division=0)),
        "precision_top20": precision_at_top_k(y_test, y_prob, k=0.20),
    }


def compute_feature_importance(
    model: XGBClassifier,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compute feature importance via 5 methods and return unified rank table."""
    feature_names = X_train_df.columns.tolist()
    n             = len(feature_names)
    scores: dict[str, np.ndarray] = {}

    # XGBoost uses f0..fN internally when trained on numpy arrays
    booster            = model.get_booster()
    xgb_gain           = booster.get_score(importance_type="gain")
    xgb_weight         = booster.get_score(importance_type="weight")
    scores["xgb_gain"]   = np.array([xgb_gain.get(f"f{i}", 0.0)   for i in range(n)])
    scores["xgb_weight"] = np.array([xgb_weight.get(f"f{i}", 0.0) for i in range(n)])

    explainer      = shap.TreeExplainer(model)
    sv             = explainer.shap_values(X_test_df)
    scores["shap"] = np.abs(sv).mean(axis=0)

    perm = permutation_importance(
        model, X_test_df.values, y_test,
        n_repeats=5, random_state=42, scoring="roc_auc",
    )
    scores["permutation"] = np.maximum(perm.importances_mean, 0.0)  # type: ignore[attr-defined]

    scores["mutual_info"] = mutual_info_classif(
        X_train_df.values, y_train, random_state=42
    )

    df      = pd.DataFrame(scores, index=feature_names)
    rank_df = df.rank(ascending=False, method="min").astype(int)
    rank_df.columns   = [f"rank_{c}" for c in rank_df.columns]
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    rank_df = (
        rank_df.sort_values("mean_rank")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    for col, vals in scores.items():
        rank_df[f"score_{col}"] = [vals[feature_names.index(f)] for f in rank_df["feature"]]

    return rank_df


def plot_feature_importance_heatmap(importance_df: pd.DataFrame, path: Path) -> None:
    rank_cols = [c for c in importance_df.columns if c.startswith("rank_")]
    plot_df   = importance_df.set_index("feature")[rank_cols]

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(plot_df.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(rank_cols)))
    ax.set_xticklabels(
        [c.replace("rank_", "").replace("_", " ").title() for c in rank_cols],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=9)

    n_rows = len(plot_df)
    for i in range(n_rows):
        for j in range(len(rank_cols)):
            val   = int(plot_df.values[i, j])
            color = "white" if val > n_rows * 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Rank (1 = most important)")
    ax.set_title("Feature Importance: Rank Comparison Across Methods")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def feature_selection_experiment(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    feature_names: list[str],
    ranked_features: list[str],
    base_params: dict,
) -> dict[int, float]:
    """Retrain XGBoost with top-K features; return {K: cv_auc}. Logs each K to MLflow."""
    cv      = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
    results: dict[int, float] = {}

    mlflow.set_experiment("model_tuning")
    for k in FEATURE_SUBSET_SIZES:
        top_k   = ranked_features[:k]
        X_tr_k  = X_tr_df[top_k].values
        cv_aucs = []
        for tr, val in cv.split(X_tr_k, y_tr):
            m = XGBClassifier(**base_params, tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_tr_k[tr], y_tr[tr])
            cv_aucs.append(roc_auc_score(y_tr[val], m.predict_proba(X_tr_k[val])[:, 1]))
        auc = float(np.mean(cv_aucs))  # type: ignore[no-matching-overload]
        results[k] = auc

        with mlflow.start_run(run_name=f"xgb_top{k}_features"):
            mlflow.log_param("k", k)
            mlflow.log_param("features", str(top_k))
            mlflow.log_metric("auc_roc", auc)

    return results


def plot_auc_vs_feature_count(k_results: dict[int, float], path: Path) -> None:
    ks   = sorted(k_results.keys())
    aucs = [k_results[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, aucs, marker="o", linewidth=2, markersize=8, color="steelblue")
    for k, a in zip(ks, aucs):
        ax.annotate(f"{a:.4f}", (k, a), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Number of Features (Top-K)")
    ax.set_ylabel("CV AUC-ROC (5-fold)")
    ax.set_title("Feature Selection: AUC-ROC vs Feature Count")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _tune_xgboost(X_tr: np.ndarray, y_tr: np.ndarray, n_trials: int) -> tuple[dict, float]:
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv  = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": spw,
            "tree_method":      "hist",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        cv_aucs = []
        for tr, val in cv.split(X_tr, y_tr):
            m = XGBClassifier(**params)
            m.fit(X_tr[tr], y_tr[tr])
            cv_aucs.append(roc_auc_score(y_tr[val], m.predict_proba(X_tr[val])[:, 1]))
        return float(np.mean(cv_aucs))  # type: ignore[no-matching-overload]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["scale_pos_weight"] = spw
    return best, study.best_value


def _tune_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, n_trials: int) -> tuple[dict, float]:
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv  = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":  spw,
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        cv_aucs = []
        for tr, val in cv.split(X_tr, y_tr):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr[tr], y_tr[tr])
            cv_aucs.append(roc_auc_score(y_tr[val], m.predict_proba(X_tr[val])[:, 1]))
        return float(np.mean(cv_aucs))  # type: ignore[no-matching-overload]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["scale_pos_weight"] = spw
    return best, study.best_value


def _tune_rf(X_tr: np.ndarray, y_tr: np.ndarray, n_trials: int) -> tuple[dict, float]:
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":      "balanced",
            "random_state":      42,
            "n_jobs":            -1,
        }
        cv_aucs = []
        for tr, val in cv.split(X_tr, y_tr):
            m = RandomForestClassifier(**params)  # type: ignore[arg-type]
            m.fit(X_tr[tr], y_tr[tr])
            cv_aucs.append(roc_auc_score(y_tr[val], m.predict_proba(X_tr[val])[:, 1]))
        return float(np.mean(cv_aucs))  # type: ignore[no-matching-overload]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study.best_value


def plot_model_comparison(comparison: dict[str, dict], path: Path) -> None:
    models  = list(comparison.keys())
    metrics = ["auc_roc", "f1", "precision_top20"]
    labels  = ["AUC-ROC", "F1", "Precision@Top20"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [comparison[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: XGBoost vs LightGBM vs RandomForest")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(n_trials: int = 75) -> dict:
    print("Loading data ...")
    X, y, cids    = load_and_preprocess()
    feature_names = X.columns.tolist()

    X_tr, X_te, y_tr, y_te, _, _ = train_test_split(
        X.values, y.values, cids.values,
        test_size=0.20, stratify=y.values, random_state=42,  # type: ignore[arg-type]
    )

    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
    X_te_df = pd.DataFrame(X_te, columns=feature_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Feature Importance ---
    print("\nPhase 1: Feature importance analysis ...")
    baseline      = joblib.load(MODELS_DIR / "churn_xgboost.pkl")
    importance_df = compute_feature_importance(baseline, X_tr_df, y_tr, X_te_df, y_te)
    importance_df.to_csv(DATA_PROCESSED / "feature_importance_ranks.csv", index=False)
    plot_feature_importance_heatmap(importance_df, FIGURES_DIR / "feature_importance_comparison.png")
    ranked_features = importance_df["feature"].tolist()
    print(f"  Top 5 features: {ranked_features[:5]}")

    # --- Phase 2: Feature Selection Experiment ---
    print("\nPhase 2: Feature selection experiment ...")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    base_params = {
        "max_depth": 5, "learning_rate": 0.1, "n_estimators": 200,
        "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": spw,
    }
    k_results = feature_selection_experiment(X_tr, y_tr, X_te, y_te, feature_names, ranked_features, base_params)
    for k, auc_val in sorted(k_results.items()):
        print(f"  Top-{k:2d} features: CV AUC = {auc_val:.4f}")
    plot_auc_vs_feature_count(k_results, FIGURES_DIR / "auc_vs_feature_count.png")

    # --- Phase 3: Multi-model Optuna tuning ---
    mlflow.set_experiment("model_tuning")
    tuned_models: dict[str, object] = {}
    comparison:   dict[str, dict]   = {}

    print(f"\nPhase 3a: Tuning XGBoost ({n_trials} trials) ...")
    xgb_params, xgb_cv_auc = _tune_xgboost(X_tr, y_tr, n_trials)
    xgb_model = XGBClassifier(**xgb_params, tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_tr, y_tr)
    xgb_metrics = _evaluate(xgb_model, X_te, y_te)
    tuned_models["XGBoost"] = xgb_model
    comparison["XGBoost"]   = xgb_metrics
    print(f"  AUC-ROC: {xgb_metrics['auc_roc']:.4f}  F1: {xgb_metrics['f1']:.4f}  P@20: {xgb_metrics['precision_top20']:.4f}")

    with mlflow.start_run(run_name="xgb_tuned"):
        mlflow.log_params({**xgb_params, "n_trials": n_trials, "model": "XGBoost"})
        mlflow.log_metrics({**xgb_metrics, "cv_auc": xgb_cv_auc})
        mlflow.set_tag("model_type", "XGBoost")

    print(f"\nPhase 3b: Tuning LightGBM ({n_trials} trials) ...")
    lgbm_params, lgbm_cv_auc = _tune_lgbm(X_tr, y_tr, n_trials)
    lgbm_model = lgb.LGBMClassifier(**lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_model.fit(X_tr, y_tr)
    lgbm_metrics = _evaluate(lgbm_model, X_te, y_te)
    tuned_models["LightGBM"] = lgbm_model
    comparison["LightGBM"]   = lgbm_metrics
    print(f"  AUC-ROC: {lgbm_metrics['auc_roc']:.4f}  F1: {lgbm_metrics['f1']:.4f}  P@20: {lgbm_metrics['precision_top20']:.4f}")

    with mlflow.start_run(run_name="lgbm_tuned"):
        mlflow.log_params({**lgbm_params, "n_trials": n_trials, "model": "LightGBM"})
        mlflow.log_metrics({**lgbm_metrics, "cv_auc": lgbm_cv_auc})
        mlflow.set_tag("model_type", "LightGBM")

    print(f"\nPhase 3c: Tuning RandomForest ({n_trials} trials) ...")
    rf_params, rf_cv_auc = _tune_rf(X_tr, y_tr, n_trials)
    rf_model = RandomForestClassifier(**rf_params, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_model.fit(X_tr, y_tr)
    rf_metrics = _evaluate(rf_model, X_te, y_te)
    tuned_models["RandomForest"] = rf_model
    comparison["RandomForest"]   = rf_metrics
    print(f"  AUC-ROC: {rf_metrics['auc_roc']:.4f}  F1: {rf_metrics['f1']:.4f}  P@20: {rf_metrics['precision_top20']:.4f}")

    with mlflow.start_run(run_name="rf_tuned"):
        mlflow.log_params({**rf_params, "n_trials": n_trials, "model": "RandomForest"})
        mlflow.log_metrics({**rf_metrics, "cv_auc": rf_cv_auc})
        mlflow.set_tag("model_type", "RandomForest")

    # --- Phase 4: Best model selection ---
    print("\nPhase 4: Best model selection ...")
    best_name    = max(comparison, key=lambda m: comparison[m]["auc_roc"])
    best_model   = tuned_models[best_name]
    best_metrics = comparison[best_name]
    print(f"  Winner: {best_name}")

    best_path = MODELS_DIR / "best_churn_model.pkl"
    joblib.dump(best_model, best_path)
    plot_model_comparison(comparison, FIGURES_DIR / "model_comparison.png")

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params({"model_type": best_name, "n_trials": n_trials})
        mlflow.log_metrics(best_metrics)
        mlflow.set_tag("best_model", "True")
        mlflow.set_tag("winner", best_name)
        mlflow.log_artifact(str(best_path))
        for png in ["feature_importance_comparison.png", "auc_vs_feature_count.png", "model_comparison.png"]:
            mlflow.log_artifact(str(FIGURES_DIR / png))

    print("\n--- Final Comparison ---")
    print(f"  {'Model':<15} {'AUC-ROC':>8}  {'F1':>7}  {'P@Top20':>9}")
    print("  " + "-" * 44)
    for name, m in comparison.items():
        marker = "  <- best" if name == best_name else ""
        print(f"  {name:<15} {m['auc_roc']:>8.4f}  {m['f1']:>7.4f}  {m['precision_top20']:>9.4f}{marker}")

    return {"best_model": best_name, "metrics": best_metrics, "comparison": comparison}
