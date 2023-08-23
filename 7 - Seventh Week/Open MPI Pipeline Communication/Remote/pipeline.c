#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ARRAY_SIZE 2000000
#define ITERATIONS 1

// Function to perform a computationally intensive operation on each element of the array
void perform_operation(int *array, int size) {
    // Enter operation to be performed by processes.
}

// MPI pipeline communication function that performs a given operation on a data array
// and measures the total elapsed time for the operation across all processes
void mpi_pipeline_communication(int world_rank, int world_size, int *data, int data_size, void (*operation)(int*, int)) {
    int chunk_size = data_size / world_size;
    double start_time, end_time, elapsed_time, max_elapsed_time;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);

    if (world_rank == 0) {
        printf("Process 0 on server %s has data array\n", processor_name);

        for (int i = 1; i < world_size; i++) {
            MPI_Send(&data[i * chunk_size], chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        start_time = MPI_Wtime();
        operation(data, chunk_size);
        end_time = MPI_Wtime();
    }
    else {
        MPI_Recv(data, chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d on server %s received data array from process 0\n", world_rank, processor_name);
        start_time = MPI_Wtime();
        operation(data, chunk_size);
        end_time = MPI_Wtime();
    }

    elapsed_time = end_time - start_time;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Total elapsed time: %.10f seconds\n", max_elapsed_time);
    }
}

int main(int argc, char *argv[]) {
    int world_size, world_rank;

    // Initialize MPI environment and get process information
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Check if there are at least two processes
    if (world_size < 2) {
        printf("Please run with at least two processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory for the data array
    int data[ARRAY_SIZE];
    
    // Initialize the data array with user's data (e.g., read from a file or generate data as required)
    if (world_rank == 0) {
        // Initialize data array here
    }

    // Call the mpi_pipeline_communication function with the specific operation to perform
    mpi_pipeline_communication(world_rank, world_size, data, ARRAY_SIZE, perform_operation);

    // Finalize MPI environment and exit
    MPI_Finalize();
    return 0;
}