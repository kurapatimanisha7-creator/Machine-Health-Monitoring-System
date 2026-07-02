import os
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "machine-health-secret")
app.config["DATABASE"] = os.path.join(app.root_path, "app.db")
MODEL_PATH = os.path.join(app.root_path, "models", "random_forest_model.joblib")


def get_db():
    if "db" not in g:
        conn = sqlite3.connect(app.config["DATABASE"])
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            phone TEXT,
            location TEXT,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            machine_type TEXT NOT NULL,
            location TEXT NOT NULL,
            manufacturer TEXT,
            install_date TEXT NOT NULL,
            status TEXT DEFAULT 'Active',
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id INTEGER NOT NULL,
            temperature REAL NOT NULL,
            vibration REAL NOT NULL,
            pressure REAL NOT NULL,
            humidity REAL NOT NULL,
            rpm REAL NOT NULL,
            status TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY(machine_id) REFERENCES machines(id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            machine_id INTEGER NOT NULL,
            temperature REAL NOT NULL,
            vibration REAL NOT NULL,
            pressure REAL NOT NULL,
            humidity REAL NOT NULL,
            rpm REAL NOT NULL,
            predicted_status TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(machine_id) REFERENCES machines(id)
        );

        CREATE TABLE IF NOT EXISTS maintenance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id INTEGER NOT NULL,
            task TEXT NOT NULL,
            description TEXT,
            priority TEXT NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY(machine_id) REFERENCES machines(id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            FOREIGN KEY(machine_id) REFERENCES machines(id)
        );
        """
    )
    db.commit()


def seed_demo_data():
    db = get_db()
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        db.execute(
            "INSERT INTO users (username, email, password, full_name, phone, location, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "admin",
                "admin@example.com",
                generate_password_hash("admin123"),
                "Ava Chen",
                "+1-555-0123",
                "Chicago",
                "admin",
                datetime.now().isoformat(),
            ),
        )
    if db.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0:
        user_id = db.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()[0]
        machines = [
            (user_id, "Compressor A1", "Compressor", "Production Floor 1", "Atlas Co.", "2020-03-12", "Active"),
            (user_id, "Pump B2", "Pump", "Production Floor 2", "Westline", "2019-08-19", "Active"),
            (user_id, "Generator C3", "Generator", "Power Bay", "Northstar", "2021-01-09", "Maintenance"),
        ]
        db.executemany(
            "INSERT INTO machines (user_id, name, machine_type, location, manufacturer, install_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(u, n, t, l, m, i, s, datetime.now().isoformat()) for u, n, t, l, m, i, s in machines],
        )
    if db.execute("SELECT COUNT(*) FROM maintenance").fetchone()[0] == 0:
        machine_ids = [row[0] for row in db.execute("SELECT id FROM machines ORDER BY id LIMIT 3")]
        db.executemany(
            "INSERT INTO maintenance (machine_id, task, description, priority, due_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (machine_ids[0], "Lubrication Check", "Inspect oil level and lubrication schedule.", "Medium", "2026-07-05", "Pending", datetime.now().isoformat()),
                (machine_ids[1], "Bearing Replacement", "Replace worn bearings to prevent overheating.", "High", "2026-07-03", "In Progress", datetime.now().isoformat()),
                (machine_ids[2], "Filter Cleaning", "Clean filters and verify airflow stability.", "Low", "2026-07-08", "Scheduled", datetime.now().isoformat()),
            ],
        )
    if db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0:
        machine_ids = [row[0] for row in db.execute("SELECT id FROM machines ORDER BY id LIMIT 3")]
        db.executemany(
            "INSERT INTO alerts (machine_id, title, message, severity, created_at, resolved) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (machine_ids[0], "Temperature Spike", "Compressor A1 exceeded safe operating temperature range.", "High", datetime.now().isoformat(), 0),
                (machine_ids[1], "Vibration anomaly", "Pump B2 shows inconsistent vibration patterns.", "Medium", datetime.now().isoformat(), 0),
            ],
        )
    if db.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0] == 0:
        machine_ids = [row[0] for row in db.execute("SELECT id FROM machines ORDER BY id LIMIT 3")]
        sample_rows = []
        for idx, machine_id in enumerate(machine_ids):
            for offset in range(4):
                sample_rows.append(
                    (
                        machine_id,
                        78 + idx * 4 + offset,
                        0.08 + offset * 0.01,
                        83 + idx,
                        60 + offset,
                        1500 + idx * 50,
                        "Healthy" if offset < 2 else "Warning",
                        datetime.now().isoformat(),
                    )
                )
        db.executemany(
            "INSERT INTO sensor_data (machine_id, temperature, vibration, pressure, humidity, rpm, status, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sample_rows,
        )
    db.commit()


def train_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    rng = np.random.default_rng(42)
    features = []
    labels = []
    for _ in range(1000):
        temperature = round(float(rng.uniform(60, 105)), 2)
        vibration = round(float(rng.uniform(0.03, 0.25)), 3)
        pressure = round(float(rng.uniform(70, 110)), 2)
        humidity = round(float(rng.uniform(40, 95)), 2)
        rpm = round(float(rng.uniform(800, 2200)), 2)
        if temperature > 95 or vibration > 0.2 or pressure > 100 or humidity > 85 or rpm > 2000:
            label = "Critical"
        elif temperature > 80 or vibration > 0.12 or pressure > 90 or humidity > 70 or rpm > 1600:
            label = "Warning"
        else:
            label = "Healthy"
        features.append([temperature, vibration, pressure, humidity, rpm])
        labels.append(label)

    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    model.fit(np.array(features), labels)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


@app.before_request
def load_user():
    g.user = get_current_user()


@app.before_request
def initialize_app():
    init_db()
    seed_demo_data()
    train_model_if_needed()


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not g.user:
            flash("Please sign in to access the dashboard.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user_id"] = user["id"]
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        full_name = request.form.get("full_name", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not username or not email or not full_name or not password:
            flash("Please fill in all required fields.", "warning")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        else:
            try:
                db = get_db()
                db.execute(
                    "INSERT INTO users (username, email, password, full_name, phone, location, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (username, email, generate_password_hash(password), full_name, request.form.get("phone", ""), request.form.get("location", ""), "user", datetime.now().isoformat()),
                )
                db.commit()
                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username or email already exists.", "danger")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    db = get_db()
    machine_count = db.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
    sensor_count = db.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
    maintenance_count = db.execute("SELECT COUNT(*) FROM maintenance").fetchone()[0]
    alert_count = db.execute("SELECT COUNT(*) FROM alerts WHERE resolved = 0").fetchone()[0]
    recent_predictions = db.execute(
        "SELECT p.*, m.name AS machine_name FROM predictions p JOIN machines m ON p.machine_id = m.id ORDER BY p.id DESC LIMIT 5"
    ).fetchall()
    recent_alerts = db.execute(
        "SELECT a.*, m.name AS machine_name FROM alerts a JOIN machines m ON a.machine_id = m.id WHERE a.resolved = 0 ORDER BY a.id DESC LIMIT 5"
    ).fetchall()
    recent_sensor_readings = db.execute(
        "SELECT s.*, m.name AS machine_name FROM sensor_data s JOIN machines m ON s.machine_id = m.id ORDER BY s.id DESC LIMIT 6"
    ).fetchall()
    return render_template(
        "dashboard.html",
        machine_count=machine_count,
        sensor_count=sensor_count,
        maintenance_count=maintenance_count,
        alert_count=alert_count,
        recent_predictions=recent_predictions,
        recent_alerts=recent_alerts,
        recent_sensor_readings=recent_sensor_readings,
    )


@app.route("/machines")
@login_required
def machines():
    db = get_db()
    machine_list = db.execute(
        "SELECT * FROM machines WHERE user_id = ? ORDER BY id DESC", (g.user["id"],)
    ).fetchall()
    return render_template("machines.html", machine_list=machine_list)


@app.route("/machines/add", methods=["GET", "POST"])
@login_required
def add_machine():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        machine_type = request.form.get("machine_type", "").strip()
        location = request.form.get("location", "").strip()
        manufacturer = request.form.get("manufacturer", "").strip()
        install_date = request.form.get("install_date", "")
        status = request.form.get("status", "Active")
        if not all([name, machine_type, location, install_date]):
            flash("Please fill in the required machine fields.", "warning")
        else:
            db = get_db()
            db.execute(
                "INSERT INTO machines (user_id, name, machine_type, location, manufacturer, install_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (g.user["id"], name, machine_type, location, manufacturer, install_date, status, datetime.now().isoformat()),
            )
            db.commit()
            flash("Machine added successfully.", "success")
            return redirect(url_for("machines"))
    return render_template("add_machine.html")


@app.route("/machines/<int:machine_id>/edit", methods=["GET", "POST"])
@login_required
def edit_machine(machine_id):
    db = get_db()
    machine = db.execute("SELECT * FROM machines WHERE id = ? AND user_id = ?", (machine_id, g.user["id"])).fetchone()
    if not machine:
        flash("Machine not found.", "danger")
        return redirect(url_for("machines"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        machine_type = request.form.get("machine_type", "").strip()
        location = request.form.get("location", "").strip()
        manufacturer = request.form.get("manufacturer", "").strip()
        install_date = request.form.get("install_date", "")
        status = request.form.get("status", "Active")
        db.execute(
            "UPDATE machines SET name = ?, machine_type = ?, location = ?, manufacturer = ?, install_date = ?, status = ? WHERE id = ?",
            (name, machine_type, location, manufacturer, install_date, status, machine_id),
        )
        db.commit()
        flash("Machine updated successfully.", "success")
        return redirect(url_for("machines"))
    return render_template("edit_machine.html", machine=machine)


@app.route("/machines/<int:machine_id>/delete")
@login_required
def delete_machine(machine_id):
    db = get_db()
    db.execute("DELETE FROM machines WHERE id = ? AND user_id = ?", (machine_id, g.user["id"]))
    db.commit()
    flash("Machine deleted.", "info")
    return redirect(url_for("machines"))


@app.route("/sensor-data", methods=["GET", "POST"])
@login_required
def sensor_data():
    db = get_db()
    machines = db.execute("SELECT * FROM machines WHERE user_id = ? ORDER BY id DESC", (g.user["id"],)).fetchall()
    if request.method == "POST":
        machine_id = request.form.get("machine_id")
        temperature = float(request.form.get("temperature", 0))
        vibration = float(request.form.get("vibration", 0))
        pressure = float(request.form.get("pressure", 0))
        humidity = float(request.form.get("humidity", 0))
        rpm = float(request.form.get("rpm", 0))
        model = train_model_if_needed()
        features = np.array([[temperature, vibration, pressure, humidity, rpm]])
        predicted_status = model.predict(features)[0]
        confidence = round(float(np.max(model.predict_proba(features))) * 100, 1)
        db.execute(
            "INSERT INTO sensor_data (machine_id, temperature, vibration, pressure, humidity, rpm, status, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (machine_id, temperature, vibration, pressure, humidity, rpm, predicted_status, datetime.now().isoformat()),
        )
        db.execute(
            "INSERT INTO predictions (user_id, machine_id, temperature, vibration, pressure, humidity, rpm, predicted_status, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (g.user["id"], machine_id, temperature, vibration, pressure, humidity, rpm, predicted_status, confidence, datetime.now().isoformat()),
        )
        if predicted_status != "Healthy":
            severity = "High" if predicted_status == "Critical" else "Medium"
            db.execute(
                "INSERT INTO alerts (machine_id, title, message, severity, created_at, resolved) VALUES (?, ?, ?, ?, ?, ?)",
                (machine_id, f"{predicted_status} health alert", f"AI detected {predicted_status.lower()} conditions from the latest sensor data.", severity, datetime.now().isoformat(), 0),
            )
        db.commit()
        flash(f"Sensor reading saved. Predicted health status: {predicted_status} ({confidence}% confidence)", "success")
        return redirect(url_for("sensor_data"))
    return render_template("sensor_data.html", machines=machines)


@app.route("/predictions")
@login_required
def predictions():
    db = get_db()
    history = db.execute(
        "SELECT p.*, m.name AS machine_name FROM predictions p JOIN machines m ON p.machine_id = m.id WHERE p.user_id = ? ORDER BY p.id DESC",
        (g.user["id"],),
    ).fetchall()
    return render_template("predictions.html", history=history)


@app.route("/reports")
@login_required
def reports():
    db = get_db()
    readings = db.execute(
        "SELECT s.*, m.name AS machine_name FROM sensor_data s JOIN machines m ON s.machine_id = m.id ORDER BY s.id DESC LIMIT 12"
    ).fetchall()
    counts = {"Healthy": 0, "Warning": 0, "Critical": 0}
    for row in readings:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    labels = [row["recorded_at"][:10] for row in readings]
    temperature_data = [row["temperature"] for row in readings]
    vibration_data = [row["vibration"] for row in readings]
    return render_template(
        "reports.html",
        readings=list(readings),
        labels=labels,
        temperature_data=temperature_data,
        vibration_data=vibration_data,
        counts=counts,
    )


@app.route("/maintenance", methods=["GET", "POST"])
@login_required
def maintenance():
    db = get_db()
    machines = db.execute("SELECT * FROM machines WHERE user_id = ? ORDER BY id DESC", (g.user["id"],)).fetchall()
    records = db.execute(
        "SELECT m.*, ma.name AS machine_name FROM maintenance m JOIN machines ma ON m.machine_id = ma.id WHERE ma.user_id = ? ORDER BY m.id DESC",
        (g.user["id"],),
    ).fetchall()
    if request.method == "POST":
        machine_id = request.form.get("machine_id")
        task = request.form.get("task", "").strip()
        description = request.form.get("description", "").strip()
        priority = request.form.get("priority", "Medium")
        due_date = request.form.get("due_date", "")
        status = request.form.get("status", "Pending")
        db.execute(
            "INSERT INTO maintenance (machine_id, task, description, priority, due_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (machine_id, task, description, priority, due_date, status, datetime.now().isoformat()),
        )
        db.commit()
        flash("Maintenance record saved.", "success")
        return redirect(url_for("maintenance"))
    return render_template("maintenance.html", machines=machines, records=records)


@app.route("/alerts")
@login_required
def alerts():
    db = get_db()
    alert_list = db.execute(
        "SELECT a.*, m.name AS machine_name FROM alerts a JOIN machines m ON a.machine_id = m.id WHERE m.user_id = ? ORDER BY a.id DESC",
        (g.user["id"],),
    ).fetchall()
    return render_template("alerts.html", alert_list=alert_list)


@app.route("/alerts/<int:alert_id>/resolve")
@login_required
def resolve_alert(alert_id):
    db = get_db()
    db.execute("UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,))
    db.commit()
    flash("Alert resolved.", "info")
    return redirect(url_for("alerts"))


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=g.user)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        location = request.form.get("location", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if password and password != confirm:
            flash("New passwords do not match.", "danger")
        else:
            db = get_db()
            if password:
                db.execute(
                    "UPDATE users SET full_name = ?, email = ?, phone = ?, location = ?, password = ? WHERE id = ?",
                    (full_name, email, phone, location, generate_password_hash(password), g.user["id"]),
                )
            else:
                db.execute(
                    "UPDATE users SET full_name = ?, email = ?, phone = ?, location = ? WHERE id = ?",
                    (full_name, email, phone, location, g.user["id"]),
                )
            db.commit()
            flash("Settings updated.", "success")
            return redirect(url_for("settings"))
    return render_template("settings.html", user=g.user)


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html")


@app.route("/faq")
@login_required
def faq():
    return render_template("faq.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
