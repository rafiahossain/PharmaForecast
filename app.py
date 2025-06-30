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
import numpy as np
import re
# For forecasting
# For forecasting
import matplotlib
matplotlib.use('Agg')  # ✅ Add this line
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
# to download always zero items
from flask import send_file

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
    
# Global storage for modeling results and always-zero items
forecast_plot = None
model_eval_results = []
always_zero_items_list = []
forecast_months = pd.DatetimeIndex([])
overall_forecast_plot = None
# For consumption overview page
consumption_plot = None
stock_status_plot = None
avg_consumption_plot = None
box_vs_regular_plot = None




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
# app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB


@app.route("/upload-data", methods=["POST"])
def upload_data():
    # global forecast_plot, model_eval_results, always_zero_items_list, forecast_months
    global forecast_plot, model_eval_results, always_zero_items_list, forecast_months, overall_forecast_plot
    # for cons overview page
    global consumption_plot, stock_status_plot, avg_consumption_plot, box_vs_regular_plot

    # Clear previous state
    forecast_plot = None
    model_eval_results = []
    always_zero_items_list = []
    forecast_months = pd.DatetimeIndex([])
    # for cons overview page
    consumption_plot = None
    stock_status_plot = None
    avg_consumption_plot = None
    box_vs_regular_plot = None

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
    try:
        df = pd.read_excel(consumption_path, skiprows=10, header=0)
    except Exception as e:
        return f"Error reading consumption file: {str(e)}", 400

    
    df = pd.read_excel(consumption_path, skiprows=10, header=0)
    df = df[df['Unit Name'].notna()]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # cols_to_drop = ['Login Store Stock', 'Dept C Stock', 'Stock Value', 'No.Of days Stock', 'Cost Per Item', 'Avg Con P.M.', 'Total cons.', 'Cons Per Day']
    cols_to_drop = ['Stock Value', 'No.Of days Stock', 'Cost Per Item', 'Total cons.', 'Cons Per Day']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

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

    plot_df, forecast_months = generate_forecast(df_melted, best_model, feature_cols)
    overall_forecast_plot = generate_overall_forecast_plot(df_melted)


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
    # PLOTTING TITLE DYNAMICALLY
    plt.title(f"Top 5 Items: Forecast vs Historical ({forecast_months[0].strftime('%b %Y')} to {forecast_months[-1].strftime('%b %Y')})")
    plt.xlabel("Month")
    plt.ylabel("Consumption")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    forecast_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    # ------------------- Consumption by Product Type Chart -------------------
    categories = ['CAPSULE', 'DENTAL', 'INJECTABLES', 'SUSPENSION', 'SYRUP', 'TABLET']
    category_consumption = df_melted[categories].sum()

    plt.figure(figsize=(8, 6))
    category_consumption.plot(
        kind='bar', 
        color=sns.color_palette("pastel")
    )

    plt.ylabel('Total Consumption', color='white', fontsize=12, weight='bold')

    plt.xticks(
        rotation=45, 
        color='white', 
        fontsize=10, 
        weight='bold'
    )
    plt.yticks(
        color='white', 
        fontsize=10, 
        weight='bold'
    )

    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')

    plt.gca().tick_params(colors='white')

    plt.tight_layout()

    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    consumption_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    # ------------------- Stock Status Distribution Pie Chart -------------------
    df_melted['Months_Supply'] = df_melted['Current Stock'] / df_melted['Consumption'].replace(0, np.nan)
    df_melted['Stock_Status'] = np.where(
        df_melted['Months_Supply'] < 1, 'Low',
        np.where(df_melted['Months_Supply'] > 3, 'High', 'Normal')
    )

    plt.figure(figsize=(6, 6))
    colors = ['#ffffff', '#7677c4', '#4DC8F2']  # Adjust your colors here

    counts = df_melted['Stock_Status'].value_counts()
    labels = counts.index
    sizes = counts.values

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))
            return f'{labels[list(sizes).index(count)]}\n{pct:.1f}%' if count in sizes else ''
        return my_autopct

    # Plot with labels inside
    plt.pie(
        sizes,
        autopct=make_autopct(sizes),
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'},
        textprops={'color': 'black', 'weight': 'bold', 'fontsize': 10}  # ✅ Text inside
    )

    plt.tight_layout()
    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    stock_status_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    # ------------------- top 5 items by avg monthly consumption horizontal bar chart -------------------
    avg_consumption = (
        df_melted.groupby('ItemName')['Consumption']
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    plt.figure(figsize=(10, 3))
    bars = avg_consumption.plot(
        kind='barh',
        color='#4cc8f2'  # Accent color
    )

    plt.xlabel('Average Monthly Consumption', color='white')
    plt.ylabel('')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.gca().invert_yaxis()

    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    avg_consumption_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

     # ------------------- box vs non-box Box Plot -------------------
     
    plt.figure(figsize=(8, 5))
    # plt.figure(figsize=(7, 4))
    sns.boxplot(
        x='IS_BOX', 
        y='Consumption', 
        data=df_melted,
        boxprops=dict(color='white'),
        medianprops=dict(color='white'),
        whiskerprops=dict(color='white'),
        capprops=dict(color='white'),
        flierprops=dict(markerfacecolor='white', markeredgecolor='white')
    )

    plt.xticks([0, 1], ['Regular Units', 'Box'], color='white')
    plt.yticks(color='white')
    plt.ylabel("Consumption", color='white')
    plt.xlabel(None)

    plt.grid(True, color='gray')

    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    box_vs_regular_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    return redirect("/forecasting")

######################################

# Web pages and routing

######################################
@app.route("/consumption")
def consumption():
    return render_template(
        "dashboard_consumption.html",
        consumption_plot=consumption_plot,
        stock_status_plot=stock_status_plot,
        avg_consumption_plot=avg_consumption_plot,
        box_vs_regular_plot=box_vs_regular_plot
    )

@app.route("/suppliers")
def suppliers():
    return render_template("dashboard_suppliers.html")

@app.route("/forecasting")
def forecasting():
    date_range_str = "No forecast yet. Please upload data."
    
    if forecast_plot and model_eval_results:
        if not forecast_months.empty:
            date_range_str = f"{forecast_months[0].strftime('%b %Y')} – {forecast_months[-1].strftime('%b %Y')}"
        else:
            date_range_str = "Forecast date not available."

    return render_template(
        "dashboard_forecasting.html",
        overall_plot=overall_forecast_plot,
        plot=forecast_plot,
        results=model_eval_results,
        zero_items=always_zero_items_list,
        date_range=date_range_str
    )




# FUNCTION TO TRAIN AND TEST ALL TOP 5 MODELS FROM JNOTE    
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

# FORECASTING UPCOMING 3 MONTHS FOR TOP 5 ITEMS USING RIDGE REGRESSION
def generate_forecast(df_melted, model, feature_cols):
    # latest month from data
    latest_month = df_melted['Month'].max()

    # Generate next 3 months dynamically
    forecast_months = pd.date_range(start=latest_month + pd.DateOffset(months=1), periods=3, freq='MS')

    forecast_rows = []

    # the latest available data for each item
    latest_df = df_melted[df_melted['Month'] == latest_month].copy()

    # Loop per each forecast month
    for forecast_month in forecast_months:
        latest_df['Month'] = forecast_month

        # input features
        X_future = latest_df[feature_cols]
        # Make predictions
        latest_df['Predicted_Consumption'] = model.predict(X_future)
        # Save results for this month
        forecast_rows.append(latest_df[['ItemName', 'Month', 'Predicted_Consumption']].copy())

        # Updating lags for the next prediction month otherwise error
        latest_df['Lag_2'] = latest_df['Lag_1']
        latest_df['Lag_1'] = latest_df['Predicted_Consumption']

    # Combine forecast results into one df
    forecast_df = pd.concat(forecast_rows, ignore_index=True)
    forecast_df.rename(columns={'Predicted_Consumption': 'Value'}, inplace=True)
    forecast_df['Source'] = 'Forecast'

    # Historical data (past 6 months)
    history_df = df_melted[df_melted['Month'] >= (latest_month - pd.DateOffset(months=6))][
        ['ItemName', 'Month', 'Consumption']
    ].copy()
    history_df.rename(columns={'Consumption': 'Value'}, inplace=True)
    history_df['Source'] = 'Historical'

    # Combine history and forecast to plot
    plot_df = pd.concat([history_df, forecast_df], ignore_index=True).sort_values(by='Month')

    return plot_df, forecast_months

# FORECASTING OVERALL DEMAND
def generate_overall_forecast_plot(df_melted):
    global forecast_months

    # Aggregate total consumption per month
    overall_df = df_melted.groupby('Month')['Consumption'].sum().reset_index()

    # Create lag features
    overall_df['Lag_1'] = overall_df['Consumption'].shift(1)
    overall_df['Lag_2'] = overall_df['Consumption'].shift(2)
    overall_df = overall_df.dropna()

    # Train Ridge Regression on total consumption
    X = overall_df[['Lag_1', 'Lag_2']]
    y = overall_df['Consumption']

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Get latest lags
    latest_month = overall_df['Month'].max()
    lag_1 = overall_df.loc[overall_df['Month'] == latest_month, 'Lag_1'].values[0]
    lag_2 = overall_df.loc[overall_df['Month'] == latest_month, 'Lag_2'].values[0]

    forecast_months = pd.date_range(start=latest_month + pd.DateOffset(months=1), periods=3, freq='MS')

    forecast_results = []

    for forecast_month in forecast_months:
        X_future = pd.DataFrame({'Lag_1': [lag_1], 'Lag_2': [lag_2]})
        pred = model.predict(X_future)[0]
        forecast_results.append({'Month': forecast_month, 'Consumption': pred})

        # Update lags
        lag_2 = lag_1
        lag_1 = pred

    forecast_df = pd.DataFrame(forecast_results)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(overall_df['Month'], overall_df['Consumption'], marker='o', label='Historical')
    plt.plot(forecast_df['Month'], forecast_df['Consumption'], marker='o', linestyle='--', label='Forecast')

    plt.title(f'Overall Demand Forecast ({forecast_months[0].strftime("%b %Y")} – {forecast_months[-1].strftime("%b %Y")})')
    plt.xlabel('Month')
    plt.ylabel('Total Consumption')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    return plot_base64

from flask import send_file

@app.route("/download-zero-items")
def download_zero_items():
    if not always_zero_items_list:
        return "No data available to download.", 404

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "always_zero_items.txt")
    with open(file_path, 'w') as f:
        for item in always_zero_items_list:
            f.write(f"{item}\n")

    return send_file(file_path, as_attachment=True)


@app.route("/data-entry")
def data_entry():
    return render_template("data_entry.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

# @app.route("/performance")
# def performance():
#     return render_template("performance.html")

# RUNNER and DEBUGGER
# Keep Flask updating itself
if __name__ == "__main__":      
    app.run(debug=True)

