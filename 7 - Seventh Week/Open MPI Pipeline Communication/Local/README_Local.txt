README_Local

Assuming that both local and remote machines are running on Linux, execute the following commands to install passwordless communication and Open MPI for peer-to-peer communication:

- Make the scripts in this folder executable.

$ chmod +x install_openmpi.sh setup_passwordless_ssh.sh


Password Less Communication For Local Machine
-----------------------------------------------

- Execute the following command on the local machine. (<username>@<hostname>)

$ ./setup_passwordless_ssh.sh (remote username) (remote IPv4)

- While executing, the terminal will ask for the password of the remote machine. Provide it to complete the process.

- When done, attempt to ssh into the remote machine by entering the command in the terminal:

$ ssh (IPv4)

- Summary: The shell script installed openssh-server (if it wasn't already installed), generated a public key, and shared it with the remote machine. Then, an ssh config file was edited to include the remote user and path for the key.


Install Open MPI on Local Machine
----------------------------------

- Execute the following command to install Open MPI on the local machine:

$ ./install_openmpi.sh


Running Test File on Local Machine
----------------------------------

- Once Open MPI has been installed on both machines (local and remote), the following steps must be done simultaneously.

- Execute the following command to compile the C code:

$ mpicc -O3 -o pipeline_test pipeline_test.c -lm

- Run the following command now to execute the test code:

$ mpirun --oversubscribe -np 2 --host (Local IPv4) -x PATH /path/on/localmachine/to/pipeline_test : -np 2 --host (Remote IPv4) -x PATH /path/on/remotemachine/to/pipeline_test

- If no errors appear, the output should look like the following:

Process 0 on server (Local Hostname) generates random data array
Process 1 on server (Local Hostname) received data array from process 0
Process 2 on server (Remote Hostname) received data array from process 0
Process 3 on server (Remote Hostname) received data array from process 0
Total elapsed time: ... seconds


Running Main File on Local Machine
-----------------------------------

- The test file's purpose was for demonstration. The main file that shall be edited and used as needed is pipeline.c. The following command can be used again, but now in the following manner when ready:

$ mpicc -O3 -o pipeline pipeline.c -lm

- Within the "mpirun" command, there is an argument that goes as follows "-np" and it is followed by a number. This stands for the number of processes. If need be, the number can be increased the number of processes needed for both hosts.

** Inside the pipeline.c file ** 
- The perform_operation function is left blank on purpose because this would be the place where the main operation that the processes shall be executing with the given data.

- Also, this statement is left blank too because this is where the data that shall be executed would be placed.

if (world_rank == 0) {
        // Initialize data array here
}

- At last, the following variables can be used or disregarded as needed.

ARRAY_SIZE
ITERATIONS


*IMPORTANT: When debugging and making changes to the C file, make sure to also update it on the remote machine, as they both need to have the same file and also be compiled before executing the "mpirun" command.
