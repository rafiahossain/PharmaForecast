# Imports
from flask import Flask, render_template, redirect, request
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# To upload files in data entry page
import os
from werkzeug.utils import secure_filename

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

@app.route("/upload-data", methods=["POST"])
def upload_data():
    goods_file = request.files.get("goods_file")
    consumption_file = request.files.get("consumption_file")

    if not goods_file or not consumption_file:
        return "Both files are required!", 400

    # Secure and save the files
    goods_filename = secure_filename(goods_file.filename)
    consumption_filename = secure_filename(consumption_file.filename)

    goods_path = os.path.join(app.config['UPLOAD_FOLDER'], goods_filename)
    consumption_path = os.path.join(app.config['UPLOAD_FOLDER'], consumption_filename)

    goods_file.save(goods_path)
    consumption_file.save(consumption_path)

    # Optional: add flash messages or redirect to another page
    return f"Files uploaded successfully:<br>Goods File: {goods_filename}<br>Consumption File: {consumption_filename}"


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
    return render_template("dashboard_forecasting.html")

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

