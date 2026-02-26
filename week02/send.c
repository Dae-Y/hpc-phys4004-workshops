#include <mpi.h>
#include <stdio.h>
#include <unistd.h>   // sleep(1)

/*
 * send.c
 *
 * This single file supports two tasks:
 *
 * (A) Two-process bidirectional exchange (run with -n 2)
 *     - rank 0 sends 10 to rank 1
 *     - rank 1 sends 5  to rank 0
 *     Deadlock-safe order:
 *       rank 0: send -> recv
 *       rank 1: recv -> send
 *
 * (B) Ring communication for arbitrary number of processes N>=2 (run with -n 4, -n 6, etc.)
 *     - An integer (10) is sent around the ring:
 *         0 -> 1 -> 2 -> ... -> N-1 -> 0
 *     - The program terminates after the value returns to rank 0.
 *     Order rule (deadlock-safe):
 *       rank 0: send -> recv
 *       others: recv -> send
 *     A short sleep(1) after each receive helps the output appear in the expected order.
 */

int main(int argc, char *argv[]) {
    int rank, size;
    const int tag = 88;
    int num;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ------------------------------
     * Task (A): Bidirectional (N=2)
     * ------------------------------ */
    if (size == 2) {
        if (rank == 0) {
            num = 10;
            MPI_Send(&num, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
            printf("Process %d sent the number %d to process %d\n", rank, num, 1);

            MPI_Recv(&num, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &status);
            printf("Process %d received the number %d from process %d\n", rank, num, 1);
        } else { // rank == 1
            MPI_Recv(&num, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
            printf("Process %d received the number %d from process %d\n", rank, num, 0);

            num = 5;
            MPI_Send(&num, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
            printf("Process %d sent the number %d to process %d\n", rank, num, 0);
        }

        MPI_Finalize();
        return 0;
    }

    /* -----------------------------------------
     * Task (B): Ring for arbitrary N (N >= 2)
     * ----------------------------------------- */
    if (size < 2) {
        MPI_Finalize();
        return 0;
    }

    // Compute ring neighbours
    // source: the rank that sends to me
    // dest  : the rank I send to
    int source = (rank == 0) ? (size - 1) : (rank - 1);
    int dest   = (rank == size - 1) ? 0 : (rank + 1);

    if (rank == 0) {
        // Start the ring with value 10
        num = 10;
        MPI_Send(&num, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        printf("Process %d sent the number %d to process %d\n", rank, num, dest);

        // Receive from the last rank to complete the ring
        MPI_Recv(&num, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        sleep(1);
        printf("Process %d received the number %d from process %d\n", rank, num, source);
    } else {
        // Other ranks receive first, then forward to dest
        MPI_Recv(&num, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        sleep(1);
        printf("Process %d received the number %d from process %d\n", rank, num, source);

        MPI_Send(&num, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        printf("Process %d sent the number %d to process %d\n", rank, num, dest);
    }

    MPI_Finalize();
    return 0;
}
