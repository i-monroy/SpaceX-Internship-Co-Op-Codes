# Methane and Nitrogen Detection System

## Author 
Isaac Monroy

## Overview
This comprehensive project serves as a detection system for Methane and Nitrogen gases. It consists of three main components: a back-end server for handling and displaying gas readings, an example data publisher for generating and sending random sensor readings, and a configuration file generator for a Flask application.

---

## Methane and Nitrogen Detection System Server

### Project Description
This algorithm serves as the back-end server for the Methane and Nitrogen Detection System. The server is built using the Flask framework, and it's designed to handle HTTP requests from the client-side dashboard for displaying real-time gas readings and statistics. It provides endpoints for user authentication, fetching the latest gas readings, and acquiring statistical data about the readings. The data is presented in the form of JSON for easy manipulation on the client-side. The server integrates with a hypothetical gas sensor data source, fetching and processing the data as required for testing and development purposes only, but it can display the data successfully.

### Libraries Used
- **Flask**: Used for setting up the web server and defining HTTP routes.
- **Flask-Login**: Provides user authentication functionality.
- **Flask-Bcrypt**: Handles password hashing.
- **psycopg2**: Communicates with PostgreSQL database.
- **apscheduler**: Schedules tasks to run at regular intervals.
- **json**: Handles JSON data.
- **datetime**: Manages date and time functionalities.

### How to Run
1. Set up a PostgreSQL database with the required configurations.
2. Define environment variables or create a configuration file with the necessary credentials.
3. Install the required libraries using pip.
4. Run the code using the command `python app.py`.
5. Access the server using a web browser at `http://localhost:5000`.

### Input and Output
- **Input**: The server accepts HTTP requests containing sensor data for methane and nitrogen.
- **Output**: It outputs JSON data containing the latest readings, statistics, and other information related to the gas sensors. The data is used by a client-side dashboard to display real-time information.

---

## Example Sensor Data Publisher 

### Project Description
This Python script emulates a publisher that generates random methane and nitrogen sensor readings. These readings are sent to a Flask server via a POST request. The server's response is then printed to the console. The script continues to generate and send data every second.  

### Libraries Used
- `random`: To generate random readings for methane and nitrogen.
- `time`: To add a delay of one second between sending readings.
- `requests`: To send the POST request to the Flask server.

### How to Run
1. Ensure the Flask server is running locally on port 8000 and has an endpoint /sensordata.
2. Run the script.

### Input and Output
- **Input:** None (Readings are generated within the script).
- **Output:** Response from the server printed to the console, displaying the server's reaction to the sensor data.

---

## Flask Configuration File Generator

### Project Description
This Python script generates a configuration file (`flaskapp.cfg`) for a Flask application. The configuration includes settings for Flask, a PostgreSQL database, and user credentials for an admin. A hashed version of the password is also generated and stored using Bcrypt for secure handling of sensitive information.  

### Libraries Used
- `flask_bcrypt`: To hash the password for secure handling.
- `configparser`: To write the configuration parameters to a file.

### How to Run
1. Update the script with your desired configuration (database, user credentials, etc.).
2. Run the script.
3. The configuration file `flaskapp.cfg` will be created in the working directory.

### Input and Output
- **Input:** Configuration parameters defined within the script (database, user credentials, etc.).
- **Output:** A `flaskapp.cfg` file containing the configuration for a Flask application.
