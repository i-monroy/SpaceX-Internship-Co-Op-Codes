README_Remote

Assuming that both local and remote machines are running on Linux, execute the following commands to install Open MPI for peer-to-peer communication:

- Make the script in this folder executable.

$ chmod +x install_openmpi.sh


Install Open MPI on Remote Machine
-----------------------------------

- Execute the following command to install Open MPI on the remote machine:

$ ./install_openmpi.sh


Running Open MPI along with Local Machine
------------------------------------------

- Once Open MPI has been installed on both machines (local and remote), the following steps must be done simultaneously.

- Execute the following command to compile the C code:

$ mpicc -O3 -o pipeline_test pipeline_test.c -lm


*IMPORTANT: When debugging and making changes to the C file, make sure to also update it on the remote machine, as they both need to have the same file and also be compiled before executing the "mpirun" command.
