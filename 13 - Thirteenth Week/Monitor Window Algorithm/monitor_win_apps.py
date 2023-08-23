"""
Author: Isaac Monroy
Title: Monitor Window Algorithm
Description:
    The Monitor Window Algorithm is a simple application for tracking user's application usage. 
    Its use is for capturing the current active window title, monitor time spent on each application, 
    and present the tracking status and statistics via a graphical user interface (GUI). The 
    application also uses a separate thread to perform the tracking operations, allowing the GUI
    to remain responsive during tracking.

    Logging is managed by Python's 'logging' module, with a rotating file handler ensuring logs 
    are kept in a series of files based on time intervals.

    The GUI provides a straightforward interface for the user to start and stop tracking, displays 
    the list of tracked applications, and shows summary statistics of time spent on each application.
    
    Data is stored in a JSON file, making it easy to parse and manipulate with Python's built-in 
    'json' module. Now, the purpose of this file is for the algorithm to avoid taking into account
    these apps. So, the creation and management of this JSON file should be administered by an 
    outside administrator before the application is used. This ensures the consistency and integrity 
    of the data being tracked and prevents any potential misconfigurations or data loss that could 
    occur from uncontrolled modifications.
"""
# Time-related libraries
import time  # Time access and conversions
import datetime  # Basic date and time types

# Logging libraries
import logging  # Event logging system for applications and libraries
from logging.handlers import TimedRotatingFileHandler  # Handler for logging to a set of files, which switches from one file to the next

# Provides functions to interact with native windows
import pygetwindow as gw

# Threading library
import threading

# GUI library
import tkinter as tk  # Python interface for graphical user interfaces
from tkinter import messagebox  # Access to Tkinter's message boxes

# JSON library
import json

# Load configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# The apps to ignore are now stored in a list
apps_to_ignore = config["apps_to_ignore"]

# Store the name of the currently active window
active_window_name = ""

# Timestamp to track when we started monitoring an app
start_time = datetime.datetime.now()

# Dictionary to store total time spent on each application
app_times = {}

# Control variable for the tracking loop
tracking_active = True

def get_base_app_name(window_title):
    """
    Extracts the base application name from the 
    window title.
    """
    # Split the window title at '-', take first part and strip whitespaces
    base_app_name = window_title.split('-')[0].strip()
    return base_app_name

def track_usage():
    """
    It tracks active window title changes and accumulates
    the time spent on each application. Next, it updates 
    the application usage summary and logs any errors.
    """
    # Global variables used in this function
    global tracking_active, active_window_name, start_time, app_times
    
    # Setting tracking active
    tracking_active = True
    
    # Continuously track until tracking_active is set to False
    while tracking_active:
        try:
            # Get active window's base app name
            new_window_name = get_base_app_name(gw.getActiveWindow().title)
            
            # If the active app is in the ignore list, skip this iteration
            if any(app in new_window_name for app in apps_to_ignore):
                continue
            
            # Check if the active window app has changed
            if active_window_name != new_window_name:
                # Calculate time spent on the previous app
                end_time = datetime.datetime.now()
                total_time = (end_time - start_time).total_seconds()
                
                # Update the time spent on the previous app
                if active_window_name not in app_times:
                    app_times[active_window_name] = total_time
                else:
                    app_times[active_window_name] += total_time
                
                # Update the current active window app and start time
                active_window_name = new_window_name
                start_time = datetime.datetime.now()
                
                # Add the new active window to the GUI listbox if it's not already there
                if new_window_name not in listbox.get(0, tk.END):
                    listbox.insert(tk.END, new_window_name)
                
                # Update the summary statistics
                try:
                    update_summary()
                except Exception as e:
                    logger.error(f"Error while updating summary: {e}")
                
        except Exception as e:
            logger.error(f"Error while tracking usage: {e}")
        
        # Pause for 1 second before next iteration
        time.sleep(1)
    
    # After tracking is stopped, log the total time spent on each app
    try:
        for app, total_time in app_times.items():
            logger.info(f"Total time spent on {app}: {total_time} seconds")
    except Exception as e:
        logger.error(f"Error while writing to log file: {e}")

def start_tracking():
    """
    Starts the application usage tracking by setting up a logger
    for the session, starts the tracking thread, and updates the 
    GUI status.
    """
    # Global variables used in this function
    global tracking_thread, logger, handler
    
    # Get the current time to create unique logger and file handler
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create a unique logger using timestamp
    logger = logging.getLogger(__name__ + current_time)
    logger.setLevel(logging.INFO)
    
    # Create a file handler with a unique filename using timestamp
    handler = logging.FileHandler(f'application_usage_log_{current_time}.txt')
    handler.setLevel(logging.INFO)
    
    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    
    # If logger already has handlers, clear them
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add the new handler to the logger
    logger.addHandler(handler)
    
    # Create and start the tracking thread
    tracking_thread = threading.Thread(target=track_usage)
    tracking_thread.start()
    
    # Update the GUI status to show that tracking is active
    status_label.config(text="Status: Tracking")


def stop_tracking():
    """
    Stops the application usage tracking. It sets 'tracking_active'
    to False, joins the tracking thread, updates the GUI status, 
    and shows the summary.
    """
    global tracking_active, tracking_thread
    tracking_active = False
    
    # Wait for tracking thread to finish
    tracking_thread.join()
    
    # Update GUI status
    status_label.config(text="Status: Not Tracking")
    
    # Show usage summary in a message box
    messagebox.showinfo("Summary", "\n".join([f"{app}: {total_time} seconds" for app, total_time in app_times.items()]))

def update_summary():
    """
    Updates the summary of application usage times in the GUI.
    """
    summary_text = "\n".join([f"* {app}: {total_time} seconds" for app, total_time in app_times.items()])
    summary_label.config(text=summary_text)

# Create a Tkinter window
root = tk.Tk()
root.geometry('900x500')
root.title("Monitor Window")
root.configure(background='#005288')

# Create a Frame in the window to hold other GUI elements and center it
frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor='c')

# Title of the application
title_label = tk.Label(frame, text="Monitor Window", font=("Helvetica", 22))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

# Display instructions
instructions = """
Instructions: 
    Click 'Start Tracking' to begin tracking your
    application usage. Click 'Stop Tracking' when
    you are done.
"""
instruction_label = tk.Label(frame, text=instructions, justify="left", font=('Helvetica', 10))
instruction_label.grid(row=1, column=0, columnspan=2, padx=20)

# Current status of the tracking
status_label = tk.Label(frame, text="Status: Not Tracking", font=('Helvetica', 10))
status_label.grid(row=2, column=0, columnspan=2) 

# Buttons to start and stop tracking
start_button = tk.Button(frame, text="Start Tracking", command=start_tracking, width=20)
start_button.grid(row=3, column=0, pady=10)

stop_button = tk.Button(frame, text="Stop Tracking", command=stop_tracking, width=20)
stop_button.grid(row=3, column=1)

# Label for apps being tracked
tracked_apps_label = tk.Label(frame, text="Tracked Apps", font=('Helvetica', 10))
tracked_apps_label.grid(row=4, column=0, columnspan=2, pady=10)

# Listbox to display the tracked apps
listbox = tk.Listbox(frame, width=60, height=10)
listbox.grid(row=5, column=0, columnspan=2)

# Label for the summary statistics
summary_name_label = tk.Label(frame, text="Summary Statistics:", justify='left', font=('Helvetica', 10), padx=20)
summary_name_label.grid(row=1, column=2)

# Display the summary statistics
summary_label = tk.Label(frame, text="", justify='left', font=('Helvetica', 10), padx=20, wraplength=400)
summary_label.grid(row=2, column=2, rowspan=4, sticky='n')

# Run the Tkinter event loop
root.mainloop()