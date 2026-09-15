from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import joblib

app = Flask(__name__, template_folder="Templates", static_folder="Static")

# Load dataset for base stats and EDA
df = pd.read_csv("Data/creditcard.csv")
duplicates_before = int(df.duplicated().sum())
df = df.drop_duplicates()
duplicates_after = int(df.duplicated().sum())
missing_before = int(df.isnull().sum().sum())

numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

missing_after = int(df.isnull().sum().sum())
rows, cols = df.shape
columns = df.columns.tolist()
data_types = df.dtypes.astype(str).to_dict()

first_rows = df.head(10).to_html(classes="table", index=False)
last_rows = df.tail(5).to_html(classes="table", index=False)

os.makedirs("Static/images", exist_ok=True)
os.makedirs("Models", exist_ok=True)

# Generate / Ensure EDA Visualizations
class_counts = df["Class"].value_counts()

if not os.path.exists("Static/images/class_distribution.png"):
    plt.figure(figsize=(7, 5))
    plt.bar(["Genuine", "Fraud"], [class_counts.get(0, 0), class_counts.get(1, 0)], color=['#3b82f6', '#ef4444'])
    plt.title("Credit Card Transaction Distribution")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig("Static/images/class_distribution.png", dpi=150)
    plt.close()

if not os.path.exists("Static/images/amount_distribution.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(df["Amount"], bins=50, color='#3b82f6', edgecolor='white')
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("Static/images/amount_distribution.png", dpi=150)
    plt.close()

if not os.path.exists("Static/images/correlation_heatmap.png"):
    corr_cols = ["Time", "Amount", "Class", "V1", "V2", "V3", "V4", "V10", "V12", "V14", "V17"]
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title("Correlation Matrix of Key Features", fontsize=14, pad=15)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
    plt.yticks(range(len(corr_cols)), corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig("Static/images/correlation_heatmap.png", dpi=150)
    plt.close()

if not os.path.exists("Static/images/class_piechart.png"):
    plt.figure(figsize=(7, 5))
    plt.pie(
        [class_counts.get(0, 0), class_counts.get(1, 0)],
        labels=["Genuine", "Fraud"],
        autopct="%1.2f%%",
        colors=["#3b82f6", "#ef4444"],
        startangle=140,
        explode=[0, 0.2]
    )
    plt.title("Proportion of Genuine vs. Fraudulent Transactions")
    plt.tight_layout()
    plt.savefig("Static/images/class_piechart.png", dpi=150)
    plt.close()

# Load or run pipeline summary
SUMMARY_FILE = "Models/pipeline_summary.json"
if not os.path.exists(SUMMARY_FILE):
    import ml_engine
    ml_engine.run_pipeline()

with open(SUMMARY_FILE, "r") as f:
    pipeline_summary = json.load(f)

# Load trained models & scaler into memory for fast live inference
loaded_models = {}
for model_key in ["xgboost", "random_forest", "logistic_regression", "decision_tree", "k_nearest_neighbors", "gaussian_naive_bayes"]:
    path = f"Models/{model_key}.pkl"
    if os.path.exists(path):
        loaded_models[model_key] = joblib.load(path)

loaded_scaler = joblib.load("Models/robust_scaler.pkl") if os.path.exists("Models/robust_scaler.pkl") else None


@app.context_processor
def inject_global_vars():
    return dict(
        rows=rows,
        cols=cols,
        rows_str=f"{rows:,}"
    )


@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        stats=pipeline_summary.get("dataset_stats", {}),
        models_data=pipeline_summary.get("classification", {})
    )


@app.route("/data-loading")
def data_loading():
    return render_template(
        "data_loading.html",
        active_page="data_loading",
        columns=columns,
        data_types=data_types,
        first_rows=first_rows,
        last_rows=last_rows,
        duplicates_before=duplicates_before,
        duplicates_after=duplicates_after,
        missing_before=missing_before,
        missing_after=missing_after
    )


@app.route("/eda")
def eda():
    return render_template(
        "eda.html",
        active_page="eda"
    )


@app.route("/regression")
def regression():
    reg_data = pipeline_summary.get("regression", {})
    return render_template(
        "regression.html",
        active_page="regression",
        slr=reg_data.get("slr", {}),
        mlr=reg_data.get("mlr", {})
    )


@app.route("/preprocessing")
def preprocessing():
    stats = pipeline_summary.get("dataset_stats", {})
    return render_template(
        "preprocessing.html",
        active_page="preprocessing",
        stats=stats
    )


@app.route("/training")
def training():
    models_data = pipeline_summary.get("classification", {})
    stats = pipeline_summary.get("dataset_stats", {})
    return render_template(
        "training.html",
        active_page="training",
        models=models_data,
        stats=stats
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    samples = pipeline_summary.get("samples", {})
    feature_cols = pipeline_summary.get("feature_cols", [])
    
    prediction_result = None
    selected_model_name = "xgboost"
    input_values = samples.get("genuine", {})

    if request.method == "POST":
        selected_model_name = request.form.get("model_choice", "xgboost")
        
        # Read form values
        input_values = {}
        for feat in feature_cols:
            val_str = request.form.get(feat, "0")
            try:
                input_values[feat] = float(val_str)
            except ValueError:
                input_values[feat] = 0.0

        # Construct single-row DataFrame
        input_df = pd.DataFrame([input_values], columns=feature_cols)
        
        # Scale Time and Amount
        if loaded_scaler is not None:
            input_df[['Time', 'Amount']] = loaded_scaler.transform(input_df[['Time', 'Amount']])
        
        # Inference
        model = loaded_models.get(selected_model_name)
        if model is not None:
            pred_class = int(model.predict(input_df)[0])
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(input_df)[0][1])
            else:
                proba = 1.0 if pred_class == 1 else 0.0
            
            risk_level = "High Risk (Fraudulent)" if pred_class == 1 else "Low Risk (Genuine)"
            risk_color = "danger" if pred_class == 1 else "success"
            
            prediction_result = {
                "prediction": pred_class,
                "is_fraud": (pred_class == 1),
                "fraud_probability": round(proba * 100, 2),
                "genuine_probability": round((1 - proba) * 100, 2),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "model_used": selected_model_name.replace("_", " ").title()
            }

    return render_template(
        "predict.html",
        active_page="predict",
        feature_cols=feature_cols,
        samples=samples,
        input_values=input_values,
        selected_model=selected_model_name,
        result=prediction_result
    )


@app.route("/api/sample/<sample_type>")
def get_sample(sample_type):
    samples = pipeline_summary.get("samples", {})
    if sample_type in samples:
        return jsonify(samples[sample_type])
    return jsonify({}), 404


@app.route("/docs")
def docs():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    if per_page not in [10, 20, 50, 100]:
        per_page = 50
    if page < 1:
        page = 1
        
    start = (page - 1) * per_page
    end = start + per_page
    
    total_rows = len(df)
    total_pages = (total_rows + per_page - 1) // per_page
    
    if page > total_pages and total_pages > 0:
        page = total_pages
        start = (page - 1) * per_page
        end = start + per_page
        
    df_slice = df.iloc[start:end]
    table_html = df_slice.to_html(classes="table", index=False)
    
    return render_template(
        "docs.html",
        active_page="docs",
        table_html=table_html,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_rows=total_rows,
        start_row=min(start + 1, total_rows),
        end_row=min(end, total_rows)
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        active_page="about"
    )


@app.route("/contact")
def contact():
    return render_template(
        "contact.html",
        active_page="contact"
    )


if __name__ == "__main__":
    app.run(debug=True)