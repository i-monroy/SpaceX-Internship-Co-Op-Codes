# Open MPI Pipeline Communication

## Author
Isaac Monroy

## Project Description
This project implements a peer-to-peer communication system using Open MPI to handle data processing across both local and remote machines. By employing efficient pipeline communication and passwordless SSH connections, the project facilitates the distribution of data and computational tasks for improved performance.

## Libraries Used
- **Open MPI**: Utilized for parallel programming and communication between different nodes in a distributed system.
- **OpenSSH**: Used to set up passwordless communication between local and remote machines, enabling seamless data transfer.

## How to Run
- The `README_Local.txt` has a thorough step-by-step guide on how to set up passwordless communication, install Open MPI, and run the files on the Local machine.
- The `README_Remote.txt` has a thorough step-by-step guide on how to install Open MPI and run the files on the Remote machine.

## Input and Output
**Input**: The code requires the number of processes, local and remote IPv4 addresses, and the path to the executable.
**Output**: The program prints out the information about data processing across different processes, including the time taken for execution and the names of the servers involved.

## Additional Notes
Make sure to follow the instructions in both `README_Local.txt` and `README_Remote.txt` to set up and run the project correctly. Any debugging and changes made to the C file must be updated on both the local and remote machines, as they need to have the same file and be compiled before executing the "mpirun" command.

## Installation
Assuming that both local and remote machines are running on Linux, follow the instructions in the `install_openmpi.sh` script to install Open MPI and enable peer-to-peer communication. Make sure the C file contents are the same on both machines, as the same task is performed, with the workload split between both machines.
