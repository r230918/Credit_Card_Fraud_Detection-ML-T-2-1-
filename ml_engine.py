import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

MODELS_DIR = "Models"
STATIC_IMG_DIR = "Static/images"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

def run_pipeline():
    print("1. Loading dataset...")
    df = pd.read_csv("Data/creditcard.csv")
    df = df.drop_duplicates()
    
    # Handle any nulls
    numeric_cols = df.select_dtypes(include=np.number).columns
    for c in numeric_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
            
    print(f"Dataset shape: {df.shape}")

    # ==========================================
    # REGRESSION SECTION
    # ==========================================
    print("2. Running Simple and Multiple Linear Regression...")
    
    # Simple Linear Regression: Amount vs V2
    corr_with_amount = df.drop(columns=['Class']).corr()['Amount'].abs().sort_values(ascending=False)
    best_slr_feat = corr_with_amount.index[1] if len(corr_with_amount) > 1 else 'V2'
    
    # SLR
    X_slr = df[[best_slr_feat]].values
    y_slr = df['Amount'].values
    
    slr_model = LinearRegression()
    slr_model.fit(X_slr, y_slr)
    y_slr_pred = slr_model.predict(X_slr)
    
    slr_r2 = float(r2_score(y_slr, y_slr_pred))
    slr_mse = float(mean_squared_error(y_slr, y_slr_pred))
    slr_rmse = float(np.sqrt(slr_mse))
    slr_mae = float(mean_absolute_error(y_slr, y_slr_pred))
    slr_slope = float(slr_model.coef_[0])
    slr_intercept = float(slr_model.intercept_)
    pearson_r = float(df[[best_slr_feat, 'Amount']].corr().iloc[0, 1])

    # Plot SLR (Subsample for scatter plot clarity)
    sample_indices = np.random.RandomState(42).choice(len(df), size=min(1500, len(df)), replace=False)
    x_sub = df[best_slr_feat].iloc[sample_indices].values
    y_sub = df['Amount'].iloc[sample_indices].values
    
    x_line = np.linspace(x_sub.min(), x_sub.max(), 200).reshape(-1, 1)
    y_line = slr_model.predict(x_line)

    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_sub, y_sub, alpha=0.35, color="#3b82f6", edgecolors='none', s=25, label="Transaction Samples")
    plt.plot(x_line, y_line, color="#ef4444", linewidth=2.5, label=f"Fit: y = {slr_slope:.2f}x + {slr_intercept:.2f}")
    plt.title(f"Simple Linear Regression: Amount vs {best_slr_feat}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(f"Independent Variable ({best_slr_feat})", fontsize=11)
    plt.ylabel("Dependent Variable (Amount $)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "simple_linear_regression.png"), dpi=150)
    plt.close()

    # Multiple Linear Regression: Predicting Amount using multiple PCA components + Time
    mlr_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V20', 'V21']
    X_mlr = df[mlr_features].values
    y_mlr = df['Amount'].values
    
    mlr_model = LinearRegression()
    mlr_model.fit(X_mlr, y_mlr)
    y_mlr_pred = mlr_model.predict(X_mlr)
    
    mlr_r2 = float(r2_score(y_mlr, y_mlr_pred))
    n = len(df)
    p = len(mlr_features)
    mlr_adj_r2 = float(1 - (1 - mlr_r2) * (n - 1) / (n - p - 1))
    mlr_mse = float(mean_squared_error(y_mlr, y_mlr_pred))
    mlr_rmse = float(np.sqrt(mlr_mse))
    mlr_mae = float(mean_absolute_error(y_mlr, y_mlr_pred))
    
    mlr_coefs = {feat: float(coef) for feat, coef in zip(mlr_features, mlr_model.coef_)}
    mlr_intercept = float(mlr_model.intercept_)

    # Plot MLR Actual vs Predicted
    plt.figure(figsize=(9, 5.5))
    y_mlr_sub_actual = y_mlr[sample_indices]
    y_mlr_sub_pred = y_mlr_pred[sample_indices]
    plt.scatter(y_mlr_sub_actual, y_mlr_sub_pred, alpha=0.4, color="#8b5cf6", edgecolors='none', s=25, label="Predicted vs Actual")
    
    max_val = max(y_mlr_sub_actual.max(), y_mlr_sub_pred.max())
    plt.plot([0, max_val], [0, max_val], color="#10b981", linestyle="--", linewidth=2, label="Ideal 1:1 Identity Line")
    plt.title("Multiple Linear Regression: Actual vs Predicted Amount", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Actual Amount ($)", fontsize=11)
    plt.ylabel("Predicted Amount ($)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "multiple_linear_regression.png"), dpi=150)
    plt.close()

    # Save regression metrics
    regression_summary = {
        "slr": {
            "feature": best_slr_feat,
            "slope": round(slr_slope, 4),
            "intercept": round(slr_intercept, 4),
            "r2": round(slr_r2, 4),
            "mse": round(slr_mse, 4),
            "rmse": round(slr_rmse, 4),
            "mae": round(slr_mae, 4),
            "pearson_r": round(pearson_r, 4),
            "equation": f"Amount = {slr_slope:.4f} * {best_slr_feat} + ({slr_intercept:.4f})"
        },
        "mlr": {
            "features": mlr_features,
            "intercept": round(mlr_intercept, 4),
            "r2": round(mlr_r2, 4),
            "adjusted_r2": round(mlr_adj_r2, 4),
            "mse": round(mlr_mse, 4),
            "rmse": round(mlr_rmse, 4),
            "mae": round(mlr_mae, 4),
            "coefficients": {k: round(v, 4) for k, v in mlr_coefs.items()}
        }
    }

    # ==========================================
    # SUPERVISED CLASSIFICATION SECTION
    # ==========================================
    print("3. Preprocessing dataset for Classification...")
    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols].copy()
    y = df['Class'].copy()

    # Scaling Time and Amount with RobustScaler
    scaler = RobustScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "robust_scaler.pkl"))

    # Train Test Split (Stratified 80/20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balanced training subset
    train_df = pd.concat([X_train_full, y_train_full], axis=1)
    fraud_train = train_df[train_df['Class'] == 1]
    genuine_train = train_df[train_df['Class'] == 0].sample(n=len(fraud_train) * 5, random_state=42)
    balanced_train = pd.concat([fraud_train, genuine_train]).sample(frac=1, random_state=42)
    
    X_train = balanced_train[feature_cols]
    y_train = balanced_train['Class']

    print(f"Training shape: {X_train.shape} (Fraud: {(y_train==1).sum()}, Genuine: {(y_train==0).sum()})")
    print(f"Test shape: {X_test.shape} (Fraud: {(y_test==1).sum()}, Genuine: {(y_test==0).sum()})")

    # Define all supervised models up to XGBoost
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    model_metrics = {}
    saved_model_paths = {}
    roc_data = {}

    print("4. Training Supervised Classification Models...")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
        roc_data[name] = {
            "fpr": fpr[idx].tolist(),
            "tpr": tpr[idx].tolist(),
            "auc": round(auc, 4)
        }

        model_metrics[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc * 100, 2),
            "confusion_matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        }

        # Save model
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        model_filename = os.path.join(MODELS_DIR, f"{clean_name}.pkl")
        joblib.dump(model, model_filename)
        saved_model_paths[name] = model_filename

    # ==========================================
    # VISUALIZATIONS GENERATION
    # ==========================================
    print("5. Generating Supervised Learning Performance Visualizations...")

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    model_names = list(model_metrics.keys())
    x_pos = np.arange(len(model_names))
    width = 0.2

    precisions = [model_metrics[m]["precision"] for m in model_names]
    recalls = [model_metrics[m]["recall"] for m in model_names]
    f1s = [model_metrics[m]["f1_score"] for m in model_names]
    aucs = [model_metrics[m]["roc_auc"] for m in model_names]

    plt.bar(x_pos - 1.5 * width, precisions, width, label='Precision (%)', color='#3b82f6')
    plt.bar(x_pos - 0.5 * width, recalls, width, label='Recall (%)', color='#10b981')
    plt.bar(x_pos + 0.5 * width, f1s, width, label='F1-Score (%)', color='#f59e0b')
    plt.bar(x_pos + 1.5 * width, aucs, width, label='ROC-AUC (%)', color='#8b5cf6')

    plt.xlabel('Supervised Learning Models', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.title('Supervised Classification Models Benchmark (up to XGBoost)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x_pos, model_names, rotation=20, ha='right', fontsize=10)
    plt.ylim(0, 115)
    plt.legend(loc='upper right', frameon=True, ncol=4)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # 2. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, metrics) in enumerate(model_metrics.items()):
        cm = np.array(metrics["confusion_matrix"])
        ax = axes[i]
        cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{name}\nF1: {metrics['f1_score']}% | AUC: {metrics['roc_auc']}%", fontsize=11, fontweight='bold')
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Genuine', 'Fraud'], fontsize=9)
        ax.set_yticklabels(['Genuine', 'Fraud'], fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)

        thresh = cm.max() / 2.
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, f"{cm[r, c]:,}",
                        horizontalalignment="center",
                        color="white" if cm[r, c] > thresh else "black",
                        fontsize=10, fontweight='bold')

    plt.suptitle("Confusion Matrix Comparison Across Supervised Classifiers", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(os.path.join(STATIC_IMG_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    # 3. ROC Curves Comparison
    plt.figure(figsize=(9, 6.5))
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#ef4444']
    for (name, rinfo), color in zip(roc_data.items(), colors):
        plt.plot(rinfo["fpr"], rinfo["tpr"], label=f"{name} (AUC = {rinfo['auc']:.3f})", color=color, linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # 4. XGBoost Feature Importance
    xgb_model = models["XGBoost"]
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:12] # Top 12 features
    top_features = [feature_cols[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 5.5))
    bars = plt.barh(range(len(top_features)), top_importances[::-1], color='#3b82f6', edgecolor='none')
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=10)
    plt.xlabel('Relative Feature Importance (Gain / Weight)', fontsize=11, fontweight='bold')
    plt.title('Top 12 Most Influential Features (XGBoost Classifier)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "xgboost_feature_importance.png"), dpi=150)
    plt.close()

    # Save representative sample records for the live prediction demo
    fraud_sample = df[df['Class'] == 1].iloc[0][feature_cols].to_dict()
    genuine_sample = df[df['Class'] == 0].iloc[0][feature_cols].to_dict()

    summary_data = {
        "regression": regression_summary,
        "classification": model_metrics,
        "feature_cols": feature_cols,
        "samples": {
            "genuine": genuine_sample,
            "fraud": fraud_sample
        },
        "dataset_stats": {
            "total_transactions": len(df),
            "genuine_count": int((df['Class'] == 0).sum()),
            "fraud_count": int((df['Class'] == 1).sum()),
            "fraud_percentage": round(float((df['Class'] == 1).mean() * 100), 3),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
    }

    with open(os.path.join(MODELS_DIR, "pipeline_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    print("Pipeline execution and supervised model generation completed successfully!")

    # Execute Unsupervised & Advanced Evaluation Pipeline
    try:
        from run_unsupervised_pipeline import run_unsupervised_and_evaluation
        run_unsupervised_and_evaluation()
    except Exception as e:
        print(f"Warning: Unsupervised pipeline failed or skipped: {e}")

if __name__ == "__main__":
    run_pipeline()
