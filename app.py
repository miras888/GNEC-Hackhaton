import os
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, send_file
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import apology, login_required, lookup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

db = SQL("sqlite:///aquaclean.db")
 
@app.route("/login", methods = ["GET", "POST"])
def login():
     
    session.clear()

    
    if request.method == "POST":

        
        if not request.form.get("mail"):
            return apology("must provide mail", 403)

        
        elif not request.form.get("password"):
            return apology("must provide password", 403)
        

        
        rows = db.execute("SELECT * FROM users WHERE mail = ?", request.form.get("mail"))

        
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid mail and/or password", 403)

        
        session["user_id"] = rows[0]["id"]

        
        return redirect("/")

    
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""

    
    session.clear()

    
    return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    session.clear()

    if request.method == "POST":
        mail = request.form.get("mail")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        if not mail:
            return apology("Type mail")

        elif not password:
            return apology("Type Password")

        elif not confirmation:
            return apology("Type Confirmation")
     
        elif password != confirmation:
            return apology("password don't Match")

        rows = db.execute("SELECT * FROM users WHERE mail = ?", mail)

        if len(rows) > 0:
            return apology("mail already exists")
        db.execute("INSERT INTO users (mail, hash) VALUES (?, ?)", mail, generate_password_hash(password))

        rows1 = db.execute("SELECT * FROM users WHERE mail = ?", request.form.get("mail"))

        session["user_id"] = rows1[0]["id"]

        return redirect("/")

    else:
        return render_template("register.html")

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
        return render_template("index.html")

@app.route('/monitoring')
def monitoring():
    data = pd.read_csv('static/waterquality.csv')    
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # First Graph: pH Level
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['pH'], label='pH Level', color='blue', marker='o')
    plt.legend()
    plt.title("Water's pH Level Over Time")
    plt.xlabel("Date")
    plt.ylabel("pH")
    plt.grid(True)
    graph_path = 'static/phlevel.png'
    plt.savefig(graph_path)
    plt.close()

    # Second Graph: Water Temperature
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['WaterTemp (C)'], label='Water Temperature (°C)', color='red', marker='o')
    plt.legend()
    plt.title("Water's Temperature Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    graph_path1 = 'static/temprature.png'
    plt.savefig(graph_path1)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['DissolvedOxygen (mg/L)'], label='DissolvedOxygen (mg/L)', color='green', marker='o')
    plt.legend()
    plt.title("Dissolved Oxygen Over Time")
    plt.xlabel("Date")
    plt.ylabel("Dissolved Oxygen")
    plt.grid(True)
    graph_path2 = 'static/dissolvedoxygen.png'
    plt.savefig(graph_path2)
    plt.close()

    return render_template('monitoring.html', title="Monitoring", graph=graph_path, graph1=graph_path1, graph2=graph_path2)