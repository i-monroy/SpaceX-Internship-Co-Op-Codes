# Packet Capture Algorithm

## Author
Isaac Monroy

## Project Description
This algorithm captures packets from the network and stores vital information about them in a CSV file. The captured details include timestamp, source IP address, destination IP address, protocol used, and packet length. The Scapy library is utilized to sniff the network and extract this information.

## Libraries Used
- **Scapy**: For sniffing the network and capturing packet details.
- **csv**: Used to create and write information to a CSV file.
- **time**: Provides the current timestamp for each captured packet.

## How to Run
1. Ensure the Scapy library is installed in your environment.
2. Run the code to start capturing packets from the network.
3. The packets' details will be stored in a CSV file named `packets.csv`.

## Input and Output
- **Input**: None (The code continuously sniffs the network and captures packet information).
- **Output**: A CSV file (`packets.csv`) containing the following details for each packet:
  - Timestamp
  - Source IP Address
  - Destination IP Address
  - Protocol
  - Length
