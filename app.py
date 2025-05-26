# Imports
from flask import Flask, render_template, redirect, request
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

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

# RUNNER and DEBUGGER
# Keep Flask updating itself
if __name__ == "__main__":      
    app.run(debug=True)

