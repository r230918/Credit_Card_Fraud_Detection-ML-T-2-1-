from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

df = pd.read_csv("Data/creditcard.csv")

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

missing_before = df.isnull().sum().sum()

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

missing_after = df.isnull().sum().sum()

rows, cols = df.shape

columns = df.columns.tolist()

data_types = df.dtypes.astype(str).to_dict()

first_rows = df.head(10).to_html(
    classes="table",
    index=False
)

last_rows = df.tail(5).to_html(
    classes="table",
    index=False
)

os.makedirs("Static/images", exist_ok=True)

class_counts = df["Class"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Genuine", "Fraud"],
    [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
)

plt.title("Credit Card Transaction Distribution")

plt.xlabel("Transaction Type")

plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig("Static/images/class_distribution.png")

plt.close()

plt.figure(figsize=(8, 5))

plt.hist(
    df["Amount"],
    bins=50
)

plt.title("Transaction Amount Distribution")

plt.xlabel("Transaction Amount")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("Static/images/amount_distribution.png")

plt.close()

# 3. Correlation Heatmap of Key Features
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
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.savefig("Static/images/correlation_heatmap.png")
plt.close()

# 4. Proportions of Transactions (Pie Chart)
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
plt.savefig("Static/images/class_piechart.png")
plt.close()


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
        active_page="dashboard"
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
    table_html = df_slice.to_html(
        classes="table",
        index=False
    )
    
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