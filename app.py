# Imports
from flask import Flask, render_template, redirect, request
from flask_scss import Scss
# from flask_sqlalchemy import SQLAlchemy
# To upload files in data entry page
import os
from werkzeug.utils import secure_filename
# For preprocessing
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
# For forecasting
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
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pharmaDB.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATION"] = False
# db = SQLAlchemy(app)

# with app.app_context():
#     db.create_all()

######################################  
# Global variables for storing 
######################################

# For modeling results and always-zero items
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
# For GRN data
cl2 = None 
supplier_dependence_plot = None
delivery_delay_plot = None
expiry_risk_plot = None
itemvalue_vs_frequency_plot = None

######################################
# Home Page index, and a route to it
######################################
@app.route("/")
def index():
    return render_template("index.html")

######################################
# Upload File in Data entry page
######################################
# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB maximum size

@app.route("/upload-data", methods=["POST"])
def upload_data():
    # global forecasting variables
    global forecast_plot, model_eval_results, always_zero_items_list, forecast_months, overall_forecast_plot
    # for consumption overview page
    global consumption_plot, stock_status_plot, avg_consumption_plot, box_vs_regular_plot
    # for grn data
    global cl2

    # clear previous state
    forecast_plot = None
    model_eval_results = []
    always_zero_items_list = []
    forecast_months = pd.DatetimeIndex([])
    # for cons overview page
    consumption_plot = None
    stock_status_plot = None
    avg_consumption_plot = None
    box_vs_regular_plot = None

    # files
    goods_file = request.files.get("goods_file")
    consumption_file = request.files.get("consumption_file")
    if not goods_file or not consumption_file:
        return "Both files are required!", 400

    goods_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(goods_file.filename))
    consumption_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(consumption_file.filename))
    goods_file.save(goods_path)
    consumption_file.save(consumption_path)

    ######################################
    # preprocess consumption data
    ######################################
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

    # melt the date columns to long format
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

    # remove always-zero items to separate list
    item_grouped = df_melted.groupby('ItemName')['Consumption'].sum()
    always_zero_items_list = item_grouped[item_grouped == 0].index.tolist()
    df_melted = df_melted[~df_melted['ItemName'].isin(always_zero_items_list)]

    # create consumption lag features and drop NaNs produced for the first few months
    df_melted['Lag_1'] = df_melted.groupby('ItemName')['Consumption'].shift(1)
    df_melted['Lag_2'] = df_melted.groupby('ItemName')['Consumption'].shift(2)
    df_melted.dropna(subset=['Lag_1', 'Lag_2'], inplace=True)
    
    ######################################
    # preprocessing GRN data
    ######################################

    try:
        df2 = pd.read_excel(goods_path, skiprows=9)
    except Exception as e:
        return f"Error reading goods file: {str(e)}", 400

    # strip whitespaces from every column name
    df2.columns = df2.columns.str.strip()

    # drop summary rows based on nulls in important columns like Batch No (unique identifier)
    cl2 = df2.dropna(subset=[
        'Batch No', 'Item Name', 'Challan No./ Date :', 'PO No. / Date :', 'PO Unit Rate'
    ])

    # drop 'Unnamed' columns from xlsx formatting
    cl2 = cl2.loc[:, ~cl2.columns.str.contains('^Unnamed')]

    # drop redundant, less meaningful, one value and all empty columns
    cl2 = cl2.drop(columns=[
        "Rate", "Free Qty.", "Status", "Unit Name", "GRN type", "GIR No.", 
        "Disc. %", "Disc. Amt", "VAT%", "VAT"
    ], errors='ignore')

    # standardize injectables in 'Sub Category'
    cl2['Sub Category'] = cl2['Sub Category'].replace(
        ['INJ.', 'INJECTION'], 'INJECTABLES'
    )

    # split the 'No / Date' columns for Bill, PO and Challan
    cl2[['Bill No', 'Bill Date']] = cl2['Bill No. / Date'].str.split(r' / ', n=1, expand=True)
    cl2[['PO No', 'PO Date']] = cl2['PO No. / Date :'].str.split(r' / ', n=1, expand=True)
    cl2[['Challan No', 'Challan Date']] = cl2['Challan No./ Date :'].str.split(r'\*/', n=1, expand=True)

    # format date columns to datetime
    date_columns = ['Bill Date', 'PO Date', 'Challan Date', 'Grn Date', 'Expiry Date']
    for col in date_columns:
        cl2[col] = pd.to_datetime(cl2[col], dayfirst=True, errors='coerce')

    # drop original combo columns of No / Date
    cl2.drop(columns=[
        'Bill No. / Date', 'PO No. / Date :', 'Challan No./ Date :'
    ], inplace=True)

    ######################################
    # Forecasting
    ######################################
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
    
    ######################################
    # Consumption by Product Type Chart
    ######################################
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

    ######################################
    # Stock Status Distribution
    ######################################
    df_melted['Months_Supply'] = df_melted['Current Stock'] / df_melted['Consumption'].replace(0, np.nan)
    df_melted['Stock_Status'] = np.where(
        df_melted['Months_Supply'] < 1, 'Low',
        np.where(df_melted['Months_Supply'] > 3, 'High', 'Normal')
    )

    plt.figure(figsize=(6, 6))
    colors = ['#ffffff', '#7677c4', '#4DC8F2']

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
        textprops={'color': 'black', 'weight': 'bold', 'fontsize': 10}
    )

    plt.tight_layout()
    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    stock_status_plot = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    ######################################
    # top 5 items by avg monthly consumption 
    ######################################
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

    ######################################
    # box vs non-box Box Plot
    ######################################
     
    plt.figure(figsize=(8, 5))
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

######################################
# Supply Chain Risk Dashboard
######################################
@app.route("/suppliers")
def suppliers():
    global cl2
    global supplier_dependence_plot, delivery_delay_plot, expiry_risk_plot, itemvalue_vs_frequency_plot

    if cl2 is None:
        return render_template("dashboard_suppliers.html", kpis=None)
    try:
        ######################################
        # KPI Calculations
        ######################################
        total_grns = cl2['Grn No'].nunique()
        total_suppliers = cl2['Supplier Name'].nunique()
        total_items = cl2['Item Name'].nunique()

        total_item_value = cl2['Item Value'].sum()
        avg_item_value_per_grn = total_item_value / total_grns if total_grns else 0
        total_rec_qty = cl2['Rec. Qty.'].sum()

        cl2['Lead_Time_Days'] = (cl2['Grn Date'] - cl2['Challan Date']).dt.days
        avg_lead_time = cl2['Lead_Time_Days'].mean()

        cl2['Days_to_Expiry'] = (cl2['Expiry Date'] - pd.Timestamp.today()).dt.days
        near_expiry_batches = (cl2['Days_to_Expiry'] < 180).sum()

        price_variance_cases = (cl2['Unit Cost'].round(2) != cl2['PO Unit Rate'].round(2)).sum()

        grn_monthly = cl2.groupby(cl2['Grn Date'].dt.to_period('M')).size()
        avg_grns_per_month = grn_monthly.mean()

        kpis = {
            'Total GRNs Processed': total_grns,
            'Total Unique Suppliers': total_suppliers,
            'Total Item Value (Procured)': round(total_item_value, 2),
            'Average Lead Time (Days)': round(avg_lead_time, 2),
            'Near-Expiry Batches (<180 days)': int(near_expiry_batches),
            'Average GRNs Per Month': round(avg_grns_per_month, 2)
        }
        
        ######################################
        # Supplier Dependence
        ######################################
        delayed_suppliers = (
            cl2.groupby('Supplier Name')['Lead_Time_Days']
            .mean()
            .loc[lambda x: x > 0]
            .index.tolist()
        )

        supplier_value = cl2.groupby('Supplier Name')['Item Value'].sum().sort_values(ascending=False)
        supplier_share = supplier_value / supplier_value.sum() * 100
        top_suppliers = supplier_share.head(10)

        colors1 = ['#4DC8F2' if supplier in delayed_suppliers else 'yellow' for supplier in top_suppliers.index]

        plt.figure(figsize=(6, 4))
        top_suppliers.plot(kind='barh', color=colors1)

        plt.xlabel('% of Total Item Value', color='white')
        plt.ylabel('Supplier Name', color='white')

        plt.xticks(color='white')
        plt.yticks(color='white')

        plt.gca().invert_yaxis()

        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().spines['top'].set_color('white')
        plt.gca().spines['right'].set_color('white')

        plt.gca().tick_params(colors='white')

        # Transparent background
        plt.gca().set_facecolor('none')
        plt.gcf().set_facecolor('none')

        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', transparent=True)
        buf1.seek(0)
        supplier_dependence_plot = base64.b64encode(buf1.getvalue()).decode('utf8')
        plt.close()
        
        ######################################
        # Delivery Delay
        ######################################
        delivery_delay_supplier = (
            cl2.groupby('Supplier Name')['Lead_Time_Days']
            .mean()
            .loc[lambda x: x > 0]
            .sort_values(ascending=False)
        )

        colors2 = ['#4DC8F2' if supplier in top_suppliers.index else 'firebrick' for supplier in delivery_delay_supplier.index]

        plt.figure(figsize=(6, 4))
        delivery_delay_supplier.plot(kind='barh', color=colors2)

        plt.xlabel('Average Delivery Delay (Days)', color='white')
        plt.ylabel('Supplier Name', color='white')

        plt.xticks(color='white')
        plt.yticks(color='white')

        plt.gca().invert_yaxis()

        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().spines['top'].set_color('white')
        plt.gca().spines['right'].set_color('white')

        plt.gca().tick_params(colors='white')

        plt.gca().set_facecolor('none')
        plt.gcf().set_facecolor('none')

        plt.tight_layout()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', transparent=True)
        buf2.seek(0)
        delivery_delay_plot = base64.b64encode(buf2.getvalue()).decode('utf8')
        plt.close()
        
        ######################################
        # Expiry Risk
        ######################################
        cl2['Days_to_Expiry_Today'] = (cl2['Expiry Date'] - pd.Timestamp.today()).dt.days
        cl2['Expiry_Risk'] = pd.cut(
            cl2['Days_to_Expiry_Today'],
            bins=[-9999, 0, 90, 180, 365, 9999],
            labels=['Expired', '<3 Months', '3-6 Months', '6-12 Months', '>1 Year']
        )

        expiry_risk_subcat = cl2.groupby(['Sub Category', 'Expiry_Risk'])['Item Value'].sum().unstack().fillna(0)

        top6_subcats = expiry_risk_subcat.sum(axis=1).sort_values(ascending=False).head(6).index
        expiry_risk_top6 = expiry_risk_subcat.loc[top6_subcats]

        color_map = {
            'Expired': '#d62728',
            '<3 Months': '#4C72B0',
            '3-6 Months': '#4DC8F2',
            '6-12 Months': '#B6AED4',
            '>1 Year': '#ffffff'
        }

        ax = expiry_risk_top6.plot(
            kind='bar',
            stacked=True,
            figsize=(8, 6),
            color=[color_map.get(x, '#333333') for x in expiry_risk_top6.columns]
        )

        ax.set_ylabel('Total Item Value', color='white')
        ax.set_xlabel('Sub Category', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', transparent=True)
        buf3.seek(0)
        expiry_risk_plot = base64.b64encode(buf3.getvalue()).decode('utf8')
        plt.close()

        ######################################
        # Subcategory with ItemValue n Frequency
        ######################################
        top_subcat_item_value = cl2.groupby('Sub Category')['Item Value'].sum().sort_values(ascending=False).head(10)
        top_subcat_counts = cl2['Sub Category'].value_counts().reindex(top_subcat_item_value.index)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color1 = '#4DC8F2'
        color2 = '#B6AED4'

        ax1.bar(top_subcat_item_value.index, top_subcat_item_value.values, color=color1, width=0.4, align='center')
        ax1.set_xlabel('Sub Category', color='white')
        ax1.set_ylabel('Total Item Value', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x', colors='white')
        ax1.set_xticks(range(len(top_subcat_item_value.index)))
        ax1.set_xticklabels(top_subcat_item_value.index, rotation=45, ha='right', color='white')

        ax2 = ax1.twinx()
        ax2.bar([i + 0.4 for i in range(len(top_subcat_counts))], top_subcat_counts.values,
                color=color2, width=0.4, align='center')
        ax2.set_ylabel('Frequency', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.tight_layout()

        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png', transparent=True)
        buf4.seek(0)
        itemvalue_vs_frequency_plot = base64.b64encode(buf4.getvalue()).decode('utf8')
        plt.close()
        
        ######################################
        # Table for Items Expiring in <3 Months
        ######################################

        expiry_3_months = cl2[(cl2['Expiry_Risk'] == '<3 Months') & (cl2['Days_to_Expiry_Today'] >= 0)]

        expiry_display = expiry_3_months[[
            'Item Name', 'Sub Category', 'Supplier Name', 'Batch No',
            'Expiry Date', 'Days_to_Expiry_Today', 'Item Value'
        ]].sort_values('Days_to_Expiry_Today').reset_index(drop=True)

        expiry_display['Days_to_Expiry_Today'] = expiry_display['Days_to_Expiry_Today'].astype(int)
        expiry_display['Item Value'] = expiry_display['Item Value'].round(2)

        # Converting to HTML table
        expiry_table_html = expiry_display.to_html(
            classes='table table-striped table-bordered table-hover expiry-table',
            index=False,
            border=0,
            justify='center'
        )

    except Exception as e:
        print(f"KPI Calculation Error: {e}")
        kpis = None
        supplier_dependence_plot = None
        delivery_delay_plot = None
        expiry_risk_plot = None
        itemvalue_vs_frequency_plot = None
        
    return render_template(
        "dashboard_suppliers.html",
        kpis=kpis,
        supplier_dependence_plot=supplier_dependence_plot,
        delivery_delay_plot=delivery_delay_plot,
        expiry_risk_plot=expiry_risk_plot,
        itemvalue_vs_frequency_plot=itemvalue_vs_frequency_plot,
        expiry_table=expiry_table_html
    )


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

######################################
# FUNCTION TO TRAIN AND TEST ALL TOP 5 MODELS FROM JNOTE   
###################################### 
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
        results = sorted(results, key=lambda x: x['Time (s)'])  # fastest model on top
    return results

######################################
# FORECASTING UPCOMING 3 MONTHS FOR TOP 5 ITEMS USING RIDGE REGRESSION
######################################
def generate_forecast(df_melted, model, feature_cols):
    latest_month = df_melted['Month'].max() # latest month from data
    forecast_months = pd.date_range(start=latest_month + pd.DateOffset(months=1), periods=3, freq='MS') # dynamic next 3 months

    forecast_rows = []

    latest_df = df_melted[df_melted['Month'] == latest_month].copy() # the latest data per item

    # Loop per forecast month
    for forecast_month in forecast_months:
        latest_df['Month'] = forecast_month

        # features columns
        X_future = latest_df[feature_cols]
        # predictions
        latest_df['Predicted_Consumption'] = model.predict(X_future)
        # Saving results for this month
        forecast_rows.append(latest_df[['ItemName', 'Month', 'Predicted_Consumption']].copy())

        # update lags for the next prediction month otherwise error
        latest_df['Lag_2'] = latest_df['Lag_1']
        latest_df['Lag_1'] = latest_df['Predicted_Consumption']

    # combine forecast results into one df
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

######################################
# FORECASTING OVERALL DEMAND
######################################
def generate_overall_forecast_plot(df_melted):
    global forecast_months

    # total consumption per month
    overall_df = df_melted.groupby('Month')['Consumption'].sum().reset_index()
    # lag features
    overall_df['Lag_1'] = overall_df['Consumption'].shift(1)
    overall_df['Lag_2'] = overall_df['Consumption'].shift(2)
    overall_df = overall_df.dropna()
    # Train Ridge Regression on total consumption
    X = overall_df[['Lag_1', 'Lag_2']]
    y = overall_df['Consumption']

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # latest lags
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

######################################
# Download inactive items for inventory check
######################################
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

######################################
# RUNNER and DEBUGGER
# Keep Flask updating itself
######################################
if __name__ == "__main__":      
    app.run(debug=True)

