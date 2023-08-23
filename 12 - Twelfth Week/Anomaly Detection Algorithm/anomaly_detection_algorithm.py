"""
Author: Isaac Monroy
Title: Anomaly Detection Algorithm
Description:
    The objective of this algorithm is for it to correctly classify 
    whether a packet is 'normal' or an 'anomaly' by utilizing a packet's 
    header information, more specifically its packet length, protocol 
    being used, source IP address, and destination IP address. In 
    addition, with the packet's timestamp the algorithm is able to 
    use this information to detect whether a source IP address is
    sending a lot of packets in a short time span (Denial of Service) 
    or whether a lot information is being extracted, packets going to
    the same destination IP address, in a short time span (data breach).
    And all of this information is obtained from the user feeding in 
    a packet capture so the algorithm can begin the process.
"""
# Packet manipulation program and library used for packet sniffing/network monitoring.
import scapy

# Managing and manipulating datasets, and use of statistics for easier computations.
import pandas as pd

# Used for supporting calculations for anomaly detection.
import numpy as np

# Libraries used to create plots for visualization of the data.
import matplotlib.pyplot as plt
import seaborn as sns

# Loading packet data from csv.
packet = pd.read_csv("path/to/packets.csv")

# Convert Unix timestamp to datetime and drop null values for the main fields.
packet['Timestamp'] = pd.to_datetime(packet['Timestamp'], unit='s')
packet = packet.dropna(subset=['Source IP', 'Destination IP', 'Protocol', 'Length'])

def establish_baseline(df):
    """
    Establish the baseline for normal 
    behavior in the network.
    """
    baseline = {}

    # Calculate statistics for Length.
    baseline['length_mean'] = df['Length'].mean()
    baseline['length_median'] = df['Length'].median()
    baseline['length_std'] = df['Length'].std()

    # Calculate frequency for Protocol.
    baseline['protocol_counts'] = df['Protocol'].value_counts()

    # Calculate frequency for Source IP and Destination IP.
    baseline['source_ip_counts'] = df['Source IP'].value_counts()
    baseline['destination_ip_counts'] = df['Destination IP'].value_counts()

    return baseline

def detect_anomalies(df, baseline):
    """
    Detect anomalies based on the baseline
    and pre-defined thresholds.
    """
    # Define thresholds.
    z_threshold = 1
    proto_freq_threshold = 0.01
    ip_freq_threshold = 0.03
    
    # Calculate Z-scores for packet length.
    length_z = np.abs((df['Length'] - baseline['length_mean']) / baseline['length_std'])
    
    # Identify protocols and IP addresses used less frequently.
    protocol_counts = baseline['protocol_counts'] / df['Protocol'].count()
    rare_protocols = protocol_counts[protocol_counts < proto_freq_threshold].index
    
    source_counts = baseline['source_ip_counts'] / df['Source IP'].count()
    rare_sources = source_counts[source_counts < ip_freq_threshold].index
    
    dest_counts = baseline['destination_ip_counts'] / df['Destination IP'].count()
    rare_dests = dest_counts[dest_counts < ip_freq_threshold].index
    
    # Flag anomalies.
    length_anomaly = (length_z > z_threshold)
    protocol_anomaly = df['Protocol'].isin(rare_protocols)
    source_anomaly = df['Source IP'].isin(rare_sources)
    dest_anomaly = df['Destination IP'].isin(rare_dests)
    
    # Calculate 'anomaly_score' for each record, by adding the binary flags of different anomaly checks.
    df['anomaly_score'] = (length_anomaly.astype(int) + protocol_anomaly.astype(int) + 
                           source_anomaly.astype(int) + dest_anomaly.astype(int))

    # Label record as 'anomaly' if 'anomaly_score' is 3 or more.
    df['is_anomaly'] = df['anomaly_score'] >= 3
    
    return df

# Set chunk size (how many rows to process at once).
chunk_size = 10000
chunks = []

# Initialize dictionaries to store counts, sums, and sum of squares for each IP.
ip_counts = {}
ip_sums = {}
ip_sumsq = {}

# Read and process data in chunks. This is done to handle large datasets.
for chunk in pd.read_csv("packets.csv", chunksize=chunk_size):
    
    # Convert unix timestamp to datetime and remove rows with missing values.
    chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], unit='s')
    chunk = chunk.dropna(subset=['Source IP', 'Destination IP', 'Protocol', 'Length'])

    # Establish the baseline using z-score anomaly detection.
    baseline = establish_baseline(chunk)
    
    # Detect anomalies within each chunk based on the baseline.
    chunk = detect_anomalies(chunk, baseline)

    # Add Timestamp_minute column to make time-based counts more feasible.
    chunk['Timestamp_minute'] = chunk['Timestamp'].dt.floor('min')

    # Count the number of packets per IP per minute.
    counts = chunk.groupby(['Source IP', 'Timestamp_minute']).size().reset_index(name='counts')

    # Calculate the mean and standard deviation of the counts for each IP.
    baseline_counts = counts.groupby('Source IP')['counts'].agg(['mean', 'std']).reset_index()

    # Merge counts and baseline_counts to be able to calculate Z-score.
    merged_counts = pd.merge(counts, baseline_counts, on='Source IP', how='left')

    # Calculate Z-scores for packet counts for each IP.
    merged_counts['counts_z'] = (merged_counts['counts'] - merged_counts['mean']) / merged_counts['std']

    # Replace NaN Z-scores with 0 (happens when standard deviation is 0).
    merged_counts['counts_z'].fillna(0, inplace=True)
    
    # Flag as a potential data breach any IP-minute with a significantly high packet count (z-score > 3).
    merged_counts['potential_breach'] = merged_counts['counts_z'] > 3

    # Merge potential_breach back into the main DataFrame.
    chunk = pd.merge(chunk, merged_counts[['Source IP', 'Timestamp_minute', 'potential_breach']], on=['Source IP', 'Timestamp_minute'], how='left')

    # Append the processed chunk to the list of all chunks.
    chunks.append(chunk)
    
    # Update dictionaries with count, sum, and sum of squares for each IP.
    ip_counts_chunk = counts.groupby('Source IP').size().to_dict()
    ip_sums_chunk = counts.groupby('Source IP')['counts'].sum().to_dict()
    ip_sumsq_chunk = (counts['counts']**2).groupby(counts['Source IP']).sum().to_dict()

    # Update cumulative dictionaries.
    for ip, count in ip_counts_chunk.items():
        ip_counts[ip] = ip_counts.get(ip, 0) + count
    for ip, sum_ in ip_sums_chunk.items():
        ip_sums[ip] = ip_sums.get(ip, 0) + sum_
    for ip, sumsq in ip_sumsq_chunk.items():
        ip_sumsq[ip] = ip_sumsq.get(ip, 0) + sumsq

# Concatenate all chunks into a single DataFrame.
packet = pd.concat(chunks)

# Packet length size plot.
plt.figure(figsize=(10, 6))
sns.histplot(packet['Length'], bins=30, kde=True)
plt.title('Packet Length Distribution')
plt.xlabel('Packet Length')
plt.ylabel('Frequency')
plt.show()

# Plot how frequent each type of detected protocol is used.
plt.figure(figsize=(10, 6))
packet['Protocol'].value_counts().plot(kind='bar')
plt.title('Protocol Frequency')
plt.xlabel('Protocol')
plt.ylabel('Frequency')
plt.show()

# Plot how many packets are sent throughout a time span.
packet_counts_over_time = packet.resample('T', on='Timestamp')['Length'].count()
plt.figure(figsize=(10, 6))
packet_counts_over_time.plot(kind='line')
plt.title('Packet Counts Over Time')
plt.xlabel('Time')
plt.ylabel('Packet Count')
plt.show()

# Number of detected anomalies.
num_anomalies = packet[packet['anomaly_score'] >= 3].shape[0]
print(f"Number of detected anomalies: {num_anomalies}")

# Count the number of potential data breaches.
potential_breach = packet[packet['potential_breach']].shape[0]
print(f"Potential data breaches: {potential_breach}")