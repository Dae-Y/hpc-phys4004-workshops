#include <mpi.h>
#include <stdio.h>
#include <unistd.h>   // sleep(1)

/*
 * hello_ordered.c
 *
 * Goal:
 *   Print output in strict rank order: 0, 1, 2, 3, ...
 *   Use MPI_Barrier for synchronization.
 *   Pause 1 second between each printed line using sleep(1).
 *
 * How it works:
 *   We loop over r = 0..(size-1).
 *   In each iteration:
 *     - Everyone reaches a barrier.
 *     - Only rank==r prints, flushes output, then sleeps 1 second.
 *     - Everyone reaches another barrier, ensuring next rank prints after.
 */

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int r = 0; r < size; r++) {
        /* Ensure all ranks start this "turn" together */
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == r) {
            printf("Hello from rank %d of %d\n", rank, size);
            fflush(stdout);   // reduce output buffering issues
            sleep(1);         // required pause between printed lines
        }

        /* Ensure rank r finishes printing before moving to r+1 */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
