"""
Author: Isaac Monroy
Title: Flask Configuration File Generator
Description: 
    This Python script generates a configuration file ('flaskapp.cfg')
    for a Flask application. The configuration includes settings for Flask, 
    a PostgreSQL database, and user credentials for an admin. A hashed version
    of the password is also generated and stored using Bcrypt for secure 
    handling of sensitive information.
"""

# Import necessary libraries
from flask_bcrypt import Bcrypt
from configparser import ConfigParser

# Custom ConfigParser class to preserve case sensitivity of keys
class MyConfigParser(ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

config = MyConfigParser()

bcrypt = Bcrypt()

# Flaskapp Config File
DEBUG = False  # Turns off debugging features in Flask
DATABASE = "'sensor_data'"
DB_USER = "'postgres'"
DB_PASSWORD = "'your_password'"
DB_HOST = "'localhost'"
DB_PORT = "'5432'"
SECRET_KEY = "'secret_key'"
USERNAME = "'admin'"
PASSWORD = "'12345'"
HASHED_PASSWORD = bcrypt.generate_password_hash(PASSWORD).decode('utf-8')

# Add the sections and options to the configparser
config['DEFAULT'] = { 
                    'DEBUG': DEBUG,
                    'DATABASE': DATABASE,
                    'DB_USER': DB_USER,
                    'DB_PASSWORD': DB_PASSWORD,
                    'DB_HOST': DB_HOST,
                    'DB_PORT': DB_PORT,
                    'SECRET_KEY': SECRET_KEY,
                    'USERNAME': USERNAME,
                    'PASSWORD': HASHED_PASSWORD}

# Write the config to a file
with open('flaskapp.cfg', 'w') as configfile:
    config.write(configfile)