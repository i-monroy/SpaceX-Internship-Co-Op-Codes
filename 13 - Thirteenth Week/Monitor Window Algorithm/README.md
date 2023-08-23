# Monitor Window Algorithm

## Author 
Isaac Monroy

## Project Description
The Monitor Window Algorithm is a robust application for tracking user's application usage. It captures the active window title, monitors time spent on each application, and presents the tracking status and statistics through a GUI. The app also uses threading to ensure responsiveness and leverages a JSON file to control ignored applications. Logging is efficiently managed, and a rotating file handler ensures the logs are kept in time-based intervals.

## Libraries Used
- **time**: For time access and conversions.
- **datetime**: For handling date and time types.
- **logging**: For event logging with rotating file handler.
- **pygetwindow**: For interacting with native windows.
- **threading**: For multithreading to allow responsive GUI.
- **tkinter**: For GUI creation and manipulation.
- **json**: For handling JSON files including loading app configuration.

## How to Run
1. Ensure the configuration file `config.json` contains the list of apps to ignore.
2. Run the code to start the GUI.
3. Click 'Start Tracking' to begin monitoring.
4. Click 'Stop Tracking' to stop and view the summary.

## Input and Output
- **Input**: Configuration in `config.json` to list apps to ignore.
- **Output**: GUI displaying tracking status and summary, and log files containing tracking details.
