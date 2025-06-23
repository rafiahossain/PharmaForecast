# Imports
from flask import Flask, render_template, redirect, request
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# To upload files in data entry page
import os
from werkzeug.utils import secure_filename
# For preprocessing
import pandas as pd
import re
# For forecasting
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
# to measure execution time
import time

# create the app
app = Flask(__name__)
Scss(app)

# configure the SQLite database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pharmaDB.db"
app.config["SQLALCHEMY_TRACK_MODIFICATION"] = False
db = SQLAlchemy(app)

# ~ kind of a Data class ~ Row of data
class MyTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(100), nullable=False)
    complete = db.Column(db.Integer, default=0)
    created = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"Task {self.id}"

with app.app_context():
    db.create_all()

# Home Page index, and a route to it
@app.route("/",methods=["POST","GET"])
def index():
    # Add a task
    if request.method == "POST":
        current_task = request.form['content']
        new_task = MyTask(content=current_task)
        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect("/")
        except Exception as e:
            print(f"ERROR:{e}")
            return f"ERROR:{e}"
    # See all current tasks
    else:
        tasks = MyTask.query.order_by(MyTask.created).all()
        return render_template("index.html", tasks=tasks)

# DELETE an item
@app.route("/delete/<int:id>")
def delete(id:int):
    delete_task = MyTask.query.get_or_404(id)
    try:
        db.session.delete(delete_task)
        db.session.commit()
        return redirect("/") # redirects back to home page
    except Exception as e:
        return f"ERROR:{e}"
    # button only appears if we have the task
    # so we dont have to validate if task exists first
    # so 404 should not occur

# Edit an item
@app.route("/edit/<int:id>", methods=["GET","POST"])
def edit(id:int):
    task = MyTask.query.get_or_404(id)
    if request.method == "POST":
        task.content = request.form['content']
        try:
            db.session.commit()
            return redirect("/")
        except Exception as e:
            return f"ERROR:{e}"
    else: # create a new edit webpage
        return render_template('edit.html',task=task)

######################################

# Upload File in Data entry page

######################################
# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max size

# Global storage for modeling results and always-zero items
forecast_plot = None
model_eval_results = []
always_zero_items_list = []

@app.route("/upload-data", methods=["POST"])
def upload_data():
    global forecast_plot, model_eval_results, always_zero_items_list

    # File handling
    goods_file = request.files.get("goods_file")
    consumption_file = request.files.get("consumption_file")
    if not goods_file or not consumption_file:
        return "Both files are required!", 400

    goods_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(goods_file.filename))
    consumption_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(consumption_file.filename))
    goods_file.save(goods_path)
    consumption_file.save(consumption_path)

    # Data loading & preprocessing
    df = pd.read_excel(consumption_path, skiprows=10, header=0)
    df = df[df['Unit Name'].notna()]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # cols_to_drop = ['Login Store Stock', 'Dept C Stock', 'Stock Value', 'No.Of days Stock', 'Cost Per Item', 'Avg Con P.M.', 'Total cons.', 'Cons Per Day']
    # df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    def get_item_type_flags(name):
        name_lower = str(name).lower()
        injectable_pattern = r"\b(injection|inj\\.|inj|iv)\b"
        has_injectable_keyword = bool(re.search(injectable_pattern, name_lower))
        return pd.Series({
            "SUSPENSION": "suspension" in name_lower,
            "SYRUP": "syrup" in name_lower,
            "INJECTABLES": has_injectable_keyword,
            "TABLET": any(word in name_lower for word in ["tab", "tablet"]),
            "CAPSULE": "capsule" in name_lower,
            "DENTAL": any(word in name_lower for word in ["dental", "tooth", "teeth"])
        }).astype(int)

    df = pd.concat([df, df["ItemName"].apply(get_item_type_flags)], axis=1)
    df["IS_BOX"] = df["Unit Name"].apply(lambda x: "BOX" in str(x).upper()).astype(int)

    # Melt the date columns
    month_columns = [col for col in df.columns if re.match(r'^[A-Za-z]{3} \d{4}$', col)]
    df_melted = df.melt(
        id_vars=[col for col in df.columns if col not in month_columns],
        value_vars=month_columns,
        var_name='Month',
        value_name='Consumption'
    )

    df_melted['Month'] = pd.to_datetime(df_melted['Month'], format='%b %Y')
    df_melted = df_melted[df_melted['Consumption'] >= 0]
    df_melted['Month_Num'] = df_melted['Month'].dt.month
    df_melted['Year'] = df_melted['Month'].dt.year

    # Remove always-zero items before modeling
    item_grouped = df_melted.groupby('ItemName')['Consumption'].sum()
    always_zero_items_list = item_grouped[item_grouped == 0].index.tolist()
    df_melted = df_melted[~df_melted['ItemName'].isin(always_zero_items_list)]

    # Lag features
    df_melted['Lag_1'] = df_melted.groupby('ItemName')['Consumption'].shift(1)
    df_melted['Lag_2'] = df_melted.groupby('ItemName')['Consumption'].shift(2)
    df_melted.dropna(subset=['Lag_1', 'Lag_2'], inplace=True)

    # ========== Forecasting for Entire Dataset ==============
    feature_cols = ['Lag_1', 'Lag_2'] + ['SUSPENSION', 'SYRUP', 'INJECTABLES', 'TABLET', 'CAPSULE', 'DENTAL', 'IS_BOX']
    X = df_melted[feature_cols]
    y = df_melted['Consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_eval_results = evaluate_models(X_train, X_test, y_train, y_test, "Lag + Type Features")

    best_model = Ridge(alpha=1.0)
    best_model.fit(X_train, y_train)

    # Forecast June–Aug for each item
    latest_month = df_melted['Month'].max()
    latest_df = df_melted[df_melted['Month'] == latest_month].copy()

    forecast_months = pd.date_range(start="2025-06-01", periods=3, freq='MS')
    forecast_rows = []

    for forecast_month in forecast_months:
        latest_df['Month'] = forecast_month
        X_future = latest_df[feature_cols]
        latest_df['Predicted_Consumption'] = best_model.predict(X_future)
        forecast_rows.append(latest_df[['ItemName', 'Month', 'Predicted_Consumption']].copy())
        latest_df['Lag_2'] = latest_df['Lag_1']
        latest_df['Lag_1'] = latest_df['Predicted_Consumption']

    forecast_df = pd.concat(forecast_rows, ignore_index=True)
    forecast_df.rename(columns={'Predicted_Consumption': 'Value'}, inplace=True)
    forecast_df['Source'] = 'Forecast'

    history_df = df_melted[df_melted['Month'] >= (latest_month - pd.DateOffset(months=6))][
        ['ItemName', 'Month', 'Consumption']
    ].copy()
    history_df.rename(columns={'Consumption': 'Value'}, inplace=True)
    history_df['Source'] = 'Historical'

    plot_df = pd.concat([history_df, forecast_df], ignore_index=True)

    top_items = df_melted.groupby('ItemName')['Consumption'].sum().sort_values(ascending=False).head(5).index
    plot_df = plot_df[plot_df['ItemName'].isin(top_items)]

    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")
    sns.lineplot(
        data=plot_df,
        x='Month',
        y='Value',
        hue='ItemName',
        style='Source',
        markers=True,
        dashes=True
    )
    plt.title("Top 5 Items: Forecast vs Historical (June–Aug 2025)")
    plt.xlabel("Month")
    plt.ylabel("Consumption")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    forecast_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    return redirect("/forecasting")

######################################

# Web pages and routing

######################################
@app.route("/consumption")
def consumption():
    return render_template("dashboard_consumption.html")

@app.route("/suppliers")
def suppliers():
    return render_template("dashboard_suppliers.html")

@app.route("/forecasting")
def forecasting():
    return render_template("dashboard_forecasting.html",
                           plot=forecast_plot,
                           results=model_eval_results,
                           zero_items=always_zero_items_list)
    # return render_template("dashboard_forecasting.html")
    
def evaluate_models(X_train, X_test, y_train, y_test, feature_set_label):
    results = []
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42),
        'ANN (MLPRegressor)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True),
    }
    y_mean = y_test.mean()
    y_std = y_test.std()
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        
        # calculating evaluation metrics
        exec_time = round(end_time - start_time, 4)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        results.append({
            "Model": name,
            "Feature Set": feature_set_label,
            "Time (s)": exec_time,
            "R2": r2_score(y_test, y_pred),
            "MAE": mae,
            "RMSE": rmse,
            "MAE/Mean": mae / y_mean if y_mean != 0 else None,
            "MAE/Std": mae / y_std if y_std != 0 else None,
            "RMSE/Mean": rmse / y_mean if y_mean != 0 else None,
            "RMSE/Std": rmse / y_std if y_std != 0 else None
        })
        results = sorted(results, key=lambda x: x['Time (s)'])  # Fastest model first
    return results

@app.route("/data-entry")
def data_entry():
    return render_template("data_entry.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/performance")
def performance():
    return render_template("performance.html")

# RUNNER and DEBUGGER
# Keep Flask updating itself
if __name__ == "__main__":      
    app.run(debug=True)

