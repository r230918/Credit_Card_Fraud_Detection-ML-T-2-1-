import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, silhouette_score, brier_score_loss
)

STATIC_IMG_DIR = "Static/images"
MODELS_DIR = "Models"
os.makedirs(STATIC_IMG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_unsupervised_and_evaluation():
    print("--- Starting Unsupervised Learning & Advanced Evaluation Pipeline ---")
    print("1. Loading and preparing data...")
    df = pd.read_csv("Data/creditcard.csv")
    df = df.drop_duplicates()

    # Fill nulls if any
    numeric_cols = df.select_dtypes(include=np.number).columns
    for c in numeric_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())

    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols].copy()
    y = df['Class'].copy()

    scaler = RobustScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    # Stratified Train/Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Representative sample for computationally intensive visual methods (TSNE, UMAP, Dendrogram)
    # Include all test frauds and a sample of genuine transactions
    fraud_indices = df[df['Class'] == 1].index
    genuine_sample_indices = df[df['Class'] == 0].sample(n=2500, random_state=42).index
    sample_indices = np.concatenate([fraud_indices, genuine_sample_indices])
    np.random.RandomState(42).shuffle(sample_indices)
    
    X_sample = X.loc[sample_indices]
    y_sample = y.loc[sample_indices]

    results = {}

    # ==========================================
    # 1. CLUSTERING TECHNIQUES
    # ==========================================
    print("2. Running Clustering Algorithms (K-Means, Hierarchical, DBSCAN)...")
    
    # K-Means Elbow & Silhouette
    k_range = list(range(2, 7))
    inertias = []
    sil_scores = []
    
    # Subsample for silhouette to keep runtime snappy
    sil_sample_idx = np.random.RandomState(42).choice(len(X_sample), size=min(1000, len(X_sample)), replace=False)
    X_sil_sample = X_sample.iloc[sil_sample_idx]
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_sample)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_sil_sample, km.predict(X_sil_sample)))
        sil_scores.append(round(sil, 4))
    
    # Plot K-Means Elbow & Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(k_range, inertias, marker='o', linewidth=2.5, color='#3b82f6')
    ax1.set_title('K-Means: Elbow Method (Inertia)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(k_range, sil_scores, marker='s', linewidth=2.5, color='#10b981')
    ax2.set_title('K-Means: Silhouette Score by k', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Coefficient', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "kmeans_elbow_silhouette.png"), dpi=150)
    plt.close()

    # Optimal K-Means (k=3)
    best_k = 3
    kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(X_sample)
    
    # Cluster profile analysis
    cluster_profiles = []
    for c_id in range(best_k):
        c_mask = (kmeans_labels == c_id)
        total_pts = int(c_mask.sum())
        fraud_pts = int(y_sample[c_mask].sum())
        fraud_pct = round((fraud_pts / total_pts) * 100, 2) if total_pts > 0 else 0.0
        avg_amt = round(float(df.loc[sample_indices[c_mask], 'Amount'].mean()), 2)
        cluster_profiles.append({
            "cluster_id": c_id,
            "total_transactions": total_pts,
            "fraud_count": fraud_pts,
            "fraud_rate": fraud_pct,
            "avg_amount": avg_amt
        })

    # Hierarchical Clustering (Dendrogram)
    print("   -> Hierarchical Clustering & Dendrogram...")
    dendro_subsample_idx = np.random.RandomState(42).choice(len(X_sample), size=120, replace=False)
    X_dendro = X_sample.iloc[dendro_subsample_idx]
    
    plt.figure(figsize=(11, 5))
    linked = sch.linkage(X_dendro, method='ward')
    sch.dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, no_labels=True)
    plt.title('Hierarchical Clustering: Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Transaction Clusters / Branches', fontsize=11)
    plt.ylabel('Euclidean Cophenetic Distance', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "hierarchical_dendrogram.png"), dpi=150)
    plt.close()

    # Agglomerative clustering model
    agg_model = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg_model.fit_predict(X_sample)
    agg_sil = round(float(silhouette_score(X_sil_sample, agg_model.fit_predict(X_sil_sample))), 4)

    # DBSCAN (Density-Based)
    print("   -> DBSCAN Density-Based Outlier Detection...")
    dbscan = DBSCAN(eps=3.5, min_samples=15)
    db_labels = dbscan.fit_predict(X_sample)
    
    n_noise = int((db_labels == -1).sum())
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    noise_frauds = int(y_sample[db_labels == -1].sum())
    noise_fraud_rate = round((noise_frauds / n_noise) * 100, 2) if n_noise > 0 else 0.0

    # ==========================================
    # 2. DIMENSIONALITY REDUCTION (PCA, t-SNE, UMAP)
    # ==========================================
    print("3. Dimensionality Reduction (PCA, t-SNE, UMAP)...")
    
    # 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    pca_proj = pca_2d.fit_transform(X_sample)
    pca_var = [round(float(v) * 100, 2) for v in pca_2d.explained_variance_ratio_]

    # 2D t-SNE
    print("   -> Computing 2D t-SNE...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=800)
    tsne_proj = tsne.fit_transform(X_sample)

    # 2D UMAP
    print("   -> Computing 2D UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_proj = reducer.fit_transform(X_sample)

    # Combined 3-Panel Dimensionality Reduction Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # PCA Plot
    axes[0].scatter(pca_proj[y_sample == 0, 0], pca_proj[y_sample == 0, 1], c='#3b82f6', alpha=0.35, s=20, label='Genuine')
    axes[0].scatter(pca_proj[y_sample == 1, 0], pca_proj[y_sample == 1, 1], c='#ef4444', alpha=0.85, s=30, edgecolors='black', linewidth=0.5, label='Fraud')
    axes[0].set_title(f'PCA 2D Projection\n(Var: PC1={pca_var[0]}%, PC2={pca_var[1]}%)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1', fontsize=10)
    axes[0].set_ylabel('Principal Component 2', fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # t-SNE Plot
    axes[1].scatter(tsne_proj[y_sample == 0, 0], tsne_proj[y_sample == 0, 1], c='#3b82f6', alpha=0.35, s=20, label='Genuine')
    axes[1].scatter(tsne_proj[y_sample == 1, 0], tsne_proj[y_sample == 1, 1], c='#ef4444', alpha=0.85, s=30, edgecolors='black', linewidth=0.5, label='Fraud')
    axes[1].set_title('t-SNE 2D Manifold Projection\n(Perplexity = 35)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=10)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # UMAP Plot
    axes[2].scatter(umap_proj[y_sample == 0, 0], umap_proj[y_sample == 0, 1], c='#3b82f6', alpha=0.35, s=20, label='Genuine')
    axes[2].scatter(umap_proj[y_sample == 1, 0], umap_proj[y_sample == 1, 1], c='#ef4444', alpha=0.85, s=30, edgecolors='black', linewidth=0.5, label='Fraud')
    axes[2].set_title('UMAP 2D Manifold Projection\n(n_neighbors = 15, min_dist = 0.1)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('UMAP Dimension 1', fontsize=10)
    axes[2].set_ylabel('UMAP Dimension 2', fontsize=10)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Manifold Dimensionality Reduction Comparison (PCA vs t-SNE vs UMAP)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(STATIC_IMG_DIR, "dim_reduction_comparison.png"), dpi=150)
    plt.close()

    # Save individual plots too for dynamic dashboard tabs
    for name, proj, title, fname in [
        ('PCA', pca_proj, f'PCA 2D Projection (Expl. Var: {sum(pca_var)}%)', 'pca_2d.png'),
        ('t-SNE', tsne_proj, 't-SNE 2D Manifold Projection', 'tsne_2d.png'),
        ('UMAP', umap_proj, 'UMAP 2D Projection (Preserves Global & Local Topology)', 'umap_2d.png')
    ]:
        plt.figure(figsize=(9, 6))
        plt.scatter(proj[y_sample == 0, 0], proj[y_sample == 0, 1], c='#3b82f6', alpha=0.4, s=25, label='Genuine')
        plt.scatter(proj[y_sample == 1, 0], proj[y_sample == 1, 1], c='#ef4444', alpha=0.9, s=40, edgecolors='black', linewidth=0.6, label='Fraud')
        plt.title(title, fontsize=13, fontweight='bold', pad=12)
        plt.xlabel(f"{name} Dimension 1", fontsize=11)
        plt.ylabel(f"{name} Dimension 2", fontsize=11)
        plt.legend(frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_IMG_DIR, fname), dpi=150)
        plt.close()

    # ==========================================
    # 3. ANOMALY DETECTION (Isolation Forest & One-Class SVM - Syllabus M4)
    # ==========================================
    print("4. Training Anomaly Detection Models (Isolation Forest, One-Class SVM)...")
    
    # Calculate dataset contamination rate (~0.0017)
    fraud_contamination = float(y.mean())
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.005,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_full)
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    
    # Isolation forest test scoring: -1 is anomaly, 1 is normal
    iso_raw_pred = iso_forest.predict(X_test)
    iso_pred = np.where(iso_raw_pred == -1, 1, 0)
    iso_scores = -iso_forest.score_samples(X_test) # Higher score = more anomalous

    # One-Class SVM (Syllabus M4)
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    # Fit on genuine samples
    oc_train_sample = X_train_full[y_train_full == 0].sample(n=6000, random_state=42)
    oc_svm.fit(oc_train_sample)
    joblib.dump(oc_svm, os.path.join(MODELS_DIR, "one_class_svm.pkl"))

    oc_raw_pred = oc_svm.predict(X_test)
    oc_pred = np.where(oc_raw_pred == -1, 1, 0)
    oc_scores = -oc_svm.score_samples(X_test)

    anomaly_results = {}
    anomaly_models = {
        "Isolation Forest": (iso_pred, iso_scores),
        "One-Class SVM": (oc_pred, oc_scores)
    }

    for name, (pred, scores) in anomaly_models.items():
        acc = round(accuracy_score(y_test, pred) * 100, 2)
        prec = round(precision_score(y_test, pred, zero_division=0) * 100, 2)
        rec = round(recall_score(y_test, pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, pred, zero_division=0) * 100, 2)
        auc = round(roc_auc_score(y_test, scores) * 100, 2)
        pr_auc = round(average_precision_score(y_test, scores) * 100, 2)
        cm = confusion_matrix(y_test, pred).tolist()

        anomaly_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "detected_frauds": cm[1][1],
            "total_frauds": int(y_test.sum())
        }

    # Plot Anomaly Detection Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    ano_names = list(anomaly_results.keys())
    x_pos = np.arange(len(ano_names))
    width = 0.2
    
    plt.bar(x_pos - 1.5 * width, [anomaly_results[m]["precision"] for m in ano_names], width, label='Precision (%)', color='#3b82f6')
    plt.bar(x_pos - 0.5 * width, [anomaly_results[m]["recall"] for m in ano_names], width, label='Recall (%)', color='#10b981')
    plt.bar(x_pos + 0.5 * width, [anomaly_results[m]["f1_score"] for m in ano_names], width, label='F1-Score (%)', color='#f59e0b')
    plt.bar(x_pos + 1.5 * width, [anomaly_results[m]["pr_auc"] for m in ano_names], width, label='PR-AUC (%)', color='#8b5cf6')

    plt.xlabel('Unsupervised Anomaly Detectors', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
    plt.title('Unsupervised Anomaly Detection Performance on Unlabelled Test Data', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x_pos, ano_names, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "anomaly_detection_comparison.png"), dpi=150)
    plt.close()

    # ==========================================
    # 4. ADVANCED EVALUATION (Row 2 of Rubric): PR-AUC, CV, CALIBRATION, TUNING
    # ==========================================
    print("5. Evaluating Supervised Models with PR-AUC, Calibration, and K-Fold CV...")

    # Load existing trained supervised models
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "AdaBoost": "adaboost.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "XGBoost": "xgboost.pkl"
    }

    sup_models = {}
    for name, fname in model_files.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            sup_models[name] = joblib.load(fpath)

    # 1. Precision-Recall AUC (Crucial for class imbalance)
    pr_data = {}
    pr_metrics = {}
    plt.figure(figsize=(9, 6.5))
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#ef4444']

    for (name, model), color in zip(sup_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)
        
        pr_score = average_precision_score(y_test, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        
        # Subsample points for plot clarity
        idx = np.linspace(0, len(prec_curve) - 1, min(100, len(prec_curve))).astype(int)
        pr_data[name] = {
            "precision": prec_curve[idx].tolist(),
            "recall": rec_curve[idx].tolist(),
            "pr_auc": round(pr_score * 100, 2)
        }
        pr_metrics[name] = round(pr_score * 100, 2)
        plt.plot(rec_curve, prec_curve, label=f"{name} (PR-AUC = {pr_score:.3f})", color=color, linewidth=2)

    no_skill = float(y_test.mean())
    plt.plot([0, 1], [no_skill, no_skill], 'k--', linewidth=1.5, label=f'Baseline Chance ({no_skill:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (True Positive Rate)', fontsize=11, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    plt.title('Precision-Recall Curves (Gold Standard for Extreme Imbalance)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "pr_curves_comparison.png"), dpi=150)
    plt.close()

    # 2. Probability Calibration Curves (Reliability Diagram & Brier Score)
    print("   -> Calculating Probability Calibration & Brier Scores...")
    calib_models = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
    calib_scores = {}

    plt.figure(figsize=(9, 6.5))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')

    for name in calib_models:
        if name in sup_models:
            model = sup_models[name]
            y_proba = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
            brier = brier_score_loss(y_test, y_proba)
            calib_scores[name] = {
                "brier_score": round(float(brier), 5),
                "calibration_quality": "Well-Calibrated" if brier < 0.005 else "Moderate"
            }
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f"{name} (Brier: {brier:.4f})")

    plt.xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
    plt.ylabel('Fraction of Positives (Empirical Probability)', fontsize=11, fontweight='bold')
    plt.title('Model Probability Calibration Curves (Reliability Diagram)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "model_calibration_curves.png"), dpi=150)
    plt.close()

    # 3. Stratified 5-Fold Cross Validation
    print("   -> Running Stratified 5-Fold Cross-Validation...")
    # Using balanced training subset for snappy execution
    train_df = pd.concat([X_train_full, y_train_full], axis=1)
    fraud_tr = train_df[train_df['Class'] == 1]
    gen_tr = train_df[train_df['Class'] == 0].sample(n=len(fraud_tr) * 5, random_state=42)
    bal_cv = pd.concat([fraud_tr, gen_tr]).sample(frac=1, random_state=42)
    X_cv = bal_cv[feature_cols].values
    y_cv = bal_cv['Class'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in sup_models:
            model = sup_models[name]
            f1_scores_fold = []
            roc_scores_fold = []
            acc_scores_fold = []

            for tr_idx, val_idx in skf.split(X_cv, y_cv):
                fold_X_tr, fold_X_val = X_cv[tr_idx], X_cv[val_idx]
                fold_y_tr, fold_y_val = y_cv[tr_idx], y_cv[val_idx]
                
                # Clone/fit
                model.fit(fold_X_tr, fold_y_tr)
                val_pred = model.predict(fold_X_val)
                val_proba = model.predict_proba(fold_X_val)[:, 1] if hasattr(model, "predict_proba") else val_pred
                
                f1_scores_fold.append(f1_score(fold_y_val, val_pred, zero_division=0))
                roc_scores_fold.append(roc_auc_score(fold_y_val, val_proba))
                acc_scores_fold.append(accuracy_score(fold_y_val, val_pred))

            cv_summary[name] = {
                "f1_mean": round(float(np.mean(f1_scores_fold)) * 100, 2),
                "f1_std": round(float(np.std(f1_scores_fold)) * 100, 2),
                "roc_auc_mean": round(float(np.mean(roc_scores_fold)) * 100, 2),
                "roc_auc_std": round(float(np.std(roc_scores_fold)) * 100, 2),
                "acc_mean": round(float(np.mean(acc_scores_fold)) * 100, 2),
                "acc_std": round(float(np.std(acc_scores_fold)) * 100, 2)
            }

    # 4. Hyperparameter Search (RandomizedSearchCV on XGBoost & Random Forest)
    print("   -> Hyperparameter Search (RandomizedSearchCV on XGBoost)...")
    xgb_param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=6,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_cv, y_cv)
    best_xgb_params = random_search.best_params_
    best_xgb_f1 = round(float(random_search.best_score_) * 100, 2)
    print(f"   -> Best Hyperparameters: {best_xgb_params} (CV F1: {best_xgb_f1}%)")

    # Hyperparameter comparison plot
    cv_res = random_search.cv_results_
    ranks = np.argsort(cv_res['mean_test_score'])[::-1]
    param_names = [f"Trial {i+1}:\nDepth={cv_res['params'][i]['max_depth']}, LR={cv_res['params'][i]['learning_rate']}" for i in ranks[:5]]
    scores_trial = [round(cv_res['mean_test_score'][i] * 100, 2) for i in ranks[:5]]

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(param_names, scores_trial, color='#3b82f6', width=0.55)
    bars[0].set_color('#10b981')
    plt.title('Hyperparameter Search: Validation F1-Score across Search Trials', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Mean 3-Fold Cross-Val F1 (%)', fontsize=10, fontweight='bold')
    plt.ylim(min(scores_trial) - 5, 100)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "hyperparameter_tuning.png"), dpi=150)
    plt.close()

    tuning_summary = {
        "search_type": "Randomized Hyperparameter Search",
        "model": "XGBoost Classifier",
        "scoring_metric": "F1-Score",
        "best_parameters": best_xgb_params,
        "best_cv_f1": best_xgb_f1,
        "n_iterations": 6,
        "folds": 3
    }

    # ==========================================
    # SAVE PIPELINE & METRICS SUMMARY
    # ==========================================
    unsupervised_summary = {
        "clustering": {
            "kmeans": {
                "k_evaluated": k_range,
                "inertias": [round(x, 2) for x in inertias],
                "silhouette_scores": sil_scores,
                "best_k": best_k,
                "cluster_profiles": cluster_profiles
            },
            "hierarchical": {
                "linkage": "ward",
                "n_clusters": 3,
                "silhouette_score": agg_sil,
                "dendrogram_path": "hierarchical_dendrogram.png"
            },
            "dbscan": {
                "eps": 3.5,
                "min_samples": 15,
                "n_clusters": n_clusters_db,
                "noise_points": n_noise,
                "noise_frauds_detected": noise_frauds,
                "noise_fraud_rate": noise_fraud_rate
            }
        },
        "dim_reduction": {
            "pca": {
                "dimensions": 2,
                "explained_variance": pca_var,
                "total_variance_explained": round(sum(pca_var), 2)
            },
            "tsne": {
                "dimensions": 2,
                "perplexity": 35,
                "iterations": 800
            },
            "umap": {
                "dimensions": 2,
                "n_neighbors": 15,
                "min_dist": 0.1
            }
        },
        "anomaly_detection": anomaly_results,
        "evaluation_row2": {
            "pr_auc": pr_metrics,
            "calibration": calib_scores,
            "cross_validation": cv_summary,
            "hyperparameter_tuning": tuning_summary
        }
    }

    with open(os.path.join(MODELS_DIR, "unsupervised_summary.json"), "w") as f:
        json.dump(unsupervised_summary, f, indent=4)

    # Also update pipeline_summary.json to include PR-AUC in classification
    pipeline_summary_path = os.path.join(MODELS_DIR, "pipeline_summary.json")
    if os.path.exists(pipeline_summary_path):
        with open(pipeline_summary_path, "r") as f:
            full_summary = json.load(f)
        
        # Add PR-AUC to each model in classification if available
        for mname, mscore in pr_metrics.items():
            if mname in full_summary.get("classification", {}):
                full_summary["classification"][mname]["pr_auc"] = mscore
        
        full_summary["unsupervised"] = unsupervised_summary
        with open(pipeline_summary_path, "w") as f:
            json.dump(full_summary, f, indent=4)

    print("Unsupervised and Advanced Evaluation Pipeline completed successfully!")

if __name__ == "__main__":
    run_unsupervised_and_evaluation()
