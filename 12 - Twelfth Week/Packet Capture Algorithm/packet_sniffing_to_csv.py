#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Author: Isaac Monroy
Title: Packet Capture Algorithm
Description:
    The objective of this algorithm is to
    capture packets from the network. First,
    the columns are created into the CSV file,
    which are timestamp, source IP address,
    destination IP address, the protocol that
    was used, and the length of the packet. Next,
    the sniff function from scapy is used to 
    begin sniffing the network and capture the
    information of the packets and store it in 
    the CSV file.
"""
# Import necessary modules
from scapy.all import *
import csv
import time


# In[2]:


def packet_callback(packet):
    """
    Function to be applied to every
    packet that is sniffed.
    """
    # Open the output file in append mode.
    with open('packets.csv', 'a') as f:
        writer = csv.writer(f)
        
        # Extracting details from packet.
        # Get the current timestamp.
        timestamp = time.time()
        # Get source IP address if IP layer exists in packet, else 'N/A'.
        src_ip = packet[IP].src if IP in packet else 'N/A'
        # Get destination IP address if IP layer exists in packet, else 'N/A'.
        dst_ip = packet[IP].dst if IP in packet else 'N/A'
        # Get protocol if IP layer exists in packet, else 'N/A'.
        proto = packet[IP].proto if IP in packet else 'N/A'
        # Get the length of the packet.
        length = len(packet)
        
        # Writing extracted information to CSV.
        writer.writerow([timestamp, src_ip, dst_ip, proto, length])

# Attempt to create a new CSV file with headers.
# If the file already exists, this will raise a FileExistsError, which is ignored.
try:
    with open('packets.csv', 'x') as f:
        writer = csv.writer(f)
        # Write column headers into the CSV file.
        writer.writerow(["Timestamp", "Source IP", "Destination IP", "Protocol", "Length"])
# If file already exists, pass and do nothing.
except FileExistsError:
    pass

# Start sniffing packets and apply packet_callback function to each packet.
sniff(prn=packet_callback)


# In[ ]:




