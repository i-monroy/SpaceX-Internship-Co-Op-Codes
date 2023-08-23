"""
Author: Isaac Monroy
Title: Sensor Data Publisher
Description: 
    This Python script emulates a publisher that generates
    random methane and nitrogen sensor readings. These readings
    are sent to a Flask server via a POST request. The server's
    response is then printed to the console. The script continues
    to generate and send data every second.
"""

# Import necessary libraries
import random
import time
import requests

while True:
    # Generate random readings for methane and nitrogen between 0 and 100
    methane_reading = random.uniform(0, 100)
    nitrogen_reading = random.uniform(0, 100)
    
    # Create a dictionary with the sensor data
    data = {
        'methane': methane_reading,
        'nitrogen': nitrogen_reading
    }
    
    # Send a POST request to the Flask server with the sensor data
    # The server is expected to be running locally on port 8000 and have an endpoint /sensordata
    response = requests.post('http://localhost:8000/sensordata', json=data)
    
    # Print the server's response to the console
    print(f'Response from server: {response.text}')
    
    # Wait for a second before generating and sending the next reading
    time.sleep(1)