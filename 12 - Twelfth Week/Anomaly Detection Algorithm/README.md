# Anomaly Detection Algorithm

## Author
Isaac Monroy

## Project Description
This algorithm classifies packets as 'normal' or 'anomaly' using packet header information such as length, protocol, source IP, and destination IP. It can detect potential Denial of Service attacks by tracking a source IP sending many packets in a short time span, or data breaches by identifying large amounts of data sent to the same destination IP. The input is a packet capture, and the code performs various statistical analyses to identify anomalies, visualizing the data and providing summaries.

## Libraries Used
- **Scapy**: Packet manipulation program and library used for packet sniffing/network monitoring.
- **Pandas**: Managing and manipulating datasets, and use of statistics for easier computations.
- **NumPy**: Supporting calculations for anomaly detection.
- **Matplotlib, Seaborn**: Libraries used to create plots for visualization of the data.

## How to Run
1. Install the required libraries (Scapy, Pandas, NumPy, Matplotlib, Seaborn).
2. Place the CSV file containing packet data in the correct path.
3. Run the code in a compatible environment (e.g., Anaconda and Jupyter Notebooks).
4. The code will process the packet data, detect anomalies, and visualize the results.

## Input and Output
- **Input**: A CSV file with packet data, including fields like Timestamp, Source IP, Destination IP, Protocol, Length.
- **Output**: Plots visualizing packet length distribution, protocol frequency, packet counts over time, and printed summaries of detected anomalies and potential data breaches.

Note: The code is designed to work on large datasets by processing data in chunks, and you may adjust the `chunk_size` as needed.
