"""
Author: Isaac Monroy
Title: Methane and Nitrogen Detection System Server
Description: 
    This algorithm serves as the back-end server for the Methane 
    and Nitrogen Detection System. The server is built using the Flask 
    framework, and it's designed to handle HTTP requests from the 
    client-side dashboard for displaying real-time gas readings and 
    statistics. It provides endpoints for user authentication, fetching
    the latest gas readings, and acquiring statistical data about the 
    readings. The data is presented in the form of JSON for easy 
    manipulation on the client-side. The server integrates with a 
    hypothetical gas sensor data source, fetching and processing the 
    data as required for testing and development purposes only, but
    it can display the data successfully.
"""

# Import necessary modules and libraries
from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from flask_bcrypt import Bcrypt 
import psycopg2
import psycopg2.extras
from apscheduler.schedulers.background import BackgroundScheduler
import json
from datetime import datetime, timedelta

# Initialize the background scheduler
sched = BackgroundScheduler(daemon=True)

# Initialize the Flask application
app = Flask(__name__)
# Load configurations from the flaskapp.cfg file
app.config.from_pyfile('flaskapp.cfg')

# Initialize Bcrypt hashing library
bcrypt = Bcrypt(app) 

# Set up Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Redirect non-authorized users to the login page
login_manager.login_view = "login"  

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Function to load the current user
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Dashboard route showing the most recent sensor reading and statistics
@app.route('/')
@login_required
def dashboard():
    """Fetch the latest sensor readings and statistics for the dashboard."""
    
    # Connect to the database
    conn = connect_to_db()

    # Fetch the most recent reading
    cur = conn.cursor()
    cur.execute("SELECT methane, nitrogen FROM sensor_readings ORDER BY timestamp DESC LIMIT 1")
    methane_latest, nitrogen_latest = cur.fetchone()
    cur.close()
    conn.close()

    # Load the stats from the statistics.json file
    with open('statistics.json', 'r') as f:
        statistics = json.load(f)
    methane_stats = statistics.get('methane')
    nitrogen_stats = statistics.get('nitrogen')

    # Calculate the current time and time five minutes ago
    now = datetime.now()
    five_minutes_ago = now - timedelta(minutes=5)

    # Render the dashboard template
    return render_template('dashboard.html', methane_stats=methane_stats, nitrogen_stats=nitrogen_stats, 
    methane_latest=methane_latest, nitrogen_latest=nitrogen_latest, 
    start_time=five_minutes_ago, end_time=now)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    
    # Check if it's a POST request
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Validate the entered username and password against the ones in the config file
        if username == app.config['USERNAME'] and bcrypt.check_password_hash(app.config['PASSWORD'], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials", 401
    else:
        return render_template('login.html')

# Logout route
@app.route("/logout")
@login_required
def logout():
    """Handle user logout."""
    
    logout_user()
    return redirect(url_for('login'))

# Route to get the statistics of the sensor data
@app.route('/stats', methods=['GET'])
@login_required
def stats():
    """Return the statistics of the sensor data in the last 24 hours."""
    
    # Load the stats from the statistics.json file
    with open('statistics.json', 'r') as f:
        statistics = json.load(f)

    # Connect to the database
    conn = connect_to_db()
    cur = conn.cursor()

    # Get the current time and time 24 hours ago
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)

    # Fetch the sensor readings from the last 24 hours
    cur.execute("SELECT timestamp, methane, nitrogen FROM sensor_readings WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp", 
                (start_time, end_time))
    readings = cur.fetchall()

    cur.close()
    conn.close()

    # Format the readings for Chart.js
    timestamps = [reading[0] for reading in readings]
    methane_data = [reading[1] for reading in readings]
    nitrogen_data = [reading[2] for reading in readings]

    # Select every 100th data point
    timestamps = timestamps[::100]
    methane_data = methane_data[::100]
    nitrogen_data = nitrogen_data[::100]

    # Add the sensor readings data to the stats data
    statistics['timestamps'] = timestamps
    statistics['methane_data'] = methane_data
    statistics['nitrogen_data'] = nitrogen_data

    return jsonify(statistics)

# Connect to the PostgreSQL database
def connect_to_db():
    """Create a connection to the PostgreSQL database."""
    
    conn = psycopg2.connect(
        dbname=app.config['DATABASE'],
        user=app.config['DB_USER'],
        password=app.config['DB_PASSWORD'],
        host=app.config['DB_HOST'],
        port=app.config['DB_PORT']
    )
    return conn

# Create the sensor_readings table if it does not exist
def create_table():
    """Create the sensor_readings table in the PostgreSQL database."""
    
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings(
            timestamp TIMESTAMP,
            methane REAL,
            nitrogen REAL
        )
    ''')
    conn.commit()

# Insert sensor data into the database
def insert_data(data):
    """Insert sensor data into the PostgreSQL database."""
    
    conn = connect_to_db()
    with conn.cursor() as cur:
        try:
            cur.execute("INSERT INTO sensor_readings VALUES (now(), %s, %s)",
                    (data['methane'], data['nitrogen']))
            conn.commit()
        except Exception as e:
            app.logger.error(e)
            return False
    return True

# Calculate the statistics of a particular gas's readings
def calculate_statistics(gas):
    """Calculate and return the statistics of a particular gas's readings."""
    
    # Connect to the PostgreSQL database
    conn = connect_to_db()

    cur = conn.cursor()

    # Get the current time and time 5 minutes ago
    now = datetime.now()
    five_minutes_ago = now - timedelta(minutes=5)

    # Calculate the max, min, average, and percentiles of the gas's readings in the last 5 minutes
    cur.execute(f"SELECT MAX({gas}), MIN({gas}), AVG({gas}), \
                  percentile_cont(0.25) WITHIN GROUP (ORDER BY {gas}), \
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY {gas}), \
                  percentile_cont(0.75) WITHIN GROUP (ORDER BY {gas}) \
                  FROM sensor_readings WHERE timestamp > %s;", (five_minutes_ago,))

    max_value, min_value, avg_value, perc_25, perc_50, perc_75 = cur.fetchone()

    cur.close()
    conn.close()

    return max_value, min_value, avg_value, perc_25, perc_50, perc_75

def update_statistics():
    """
    Calculate the statistics for methane and nitrogen, and update them in a JSON file.
    """
    # Calculate statistics for methane and nitrogen gases
    max_value_m, min_value_m, avg_value_m, perc_25_m, perc_50_m, perc_75_m = calculate_statistics('methane')
    max_value_n, min_value_n, avg_value_n, perc_25_n, perc_50_n, perc_75_n = calculate_statistics('nitrogen')

    # Store the statistics in a dictionary
    statistics = {
        'methane': {
            'max': max_value_m,
            'min': min_value_m,
            'mean': avg_value_m,
            '25_percentile': perc_25_m,
            'median': perc_50_m,
            '75_percentile': perc_75_m
        },
        'nitrogen': {
            'max': max_value_n,
            'min': min_value_n,
            'mean': avg_value_n,
            '25_percentile': perc_25_n,
            'median': perc_50_n,
            '75_percentile': perc_75_n
        }
    }

    # Save the statistics in a JSON file
    with open('statistics.json', 'w') as f:
        json.dump(statistics, f)

@app.route('/sensordata', methods=['POST'])
def sensor_data():
    """
    Receive and validate the sensor data for methane and nitrogen.
    """

    # Check if the request is JSON
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    # Extract the JSON data from the request
    data = request.get_json()

    # Check if both "methane" and "nitrogen" are in the data
    if not all(k in data for k in ("methane", "nitrogen")):
        return jsonify({"msg": "Missing methane or nitrogen value in JSON"}), 400

    # Validate the methane and nitrogen values: they should be numbers between 0 and 100
    for key in ('methane', 'nitrogen'):
        value = data.get(key)
        if value is None or not isinstance(value, (int, float)) or not (0 <= value <= 100):
            return jsonify({"msg": f'Invalid value for {key}'}), 400

    # Insert the data into the database, and if successful, send a response back to the client
    if insert_data(data):
        return 'Received!', 200
    else:
        return 'Internal Server Error', 500

@app.route('/latest', methods=['GET'])
def latest_reading():
    """
    Return the latest sensor reading for methane and nitrogen.
    """

    # Connect to the database
    conn = connect_to_db()

    cur = conn.cursor()

    # Execute a SQL query to get the latest sensor readings
    cur.execute("SELECT methane, nitrogen FROM sensor_readings ORDER BY timestamp DESC LIMIT 1")

    # Fetch the result of the query
    methane, nitrogen = cur.fetchone()

    # Close the database connection
    cur.close()
    conn.close()

    # Return the result as a JSON response
    return jsonify({'methane': methane, 'nitrogen': nitrogen})

# Schedule the statistics update function to run every 5 minutes
sched.start()
sched.add_job(update_statistics,'interval', minutes=5)

# Run the Flask application
if __name__ == '__main__':
    create_table()
    app.run()
