#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NMAX 100000

int main(int argc, char *argv[]) {
    int         rank, ncpu, num0, num1, i;
    float       data0[NMAX], data1[NMAX];
    MPI_Status  status[2];          /* We will wait for 2 requests */
    MPI_Request request[2];         /* 0: Irecv, 1: Isend (or vice versa) */

    /* Basic Initialisation */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    /* Check there are enough processes */
    if (ncpu == 1) {
        printf("Need at least two processes\n");
        MPI_Finalize();
        return 0;
    }

    /* Parse the command-line */
    num0 = 100;
    num1 = 100;

    if (argc > 1) {
        num0 = atoi(argv[1]);
        if (num0 < 1)    num0 = 1;
        if (num0 > NMAX) num0 = NMAX;
        num1 = num0;
    }

    if (argc > 2) {
        num1 = atoi(argv[2]);
        if (num1 < 1)    num1 = 1;
        if (num1 > NMAX) num1 = NMAX;
    }

    /* ------------------------------------------------------------
       Non-blocking communication to avoid deadlock:
         - Post Irecv first (so the matching receive is available)
         - Post Isend
         - Waitall for both to complete
       ------------------------------------------------------------ */

    /* Instructions for Process 0 */
    if (rank == 0) {
        /* Fill send buffer (data0) */
        for (i = 0; i < num0; i++) {
            data0[i] = (float)(i * i);
        }

        /* 1) Post non-blocking receive FIRST (from rank 1, tag 200) */
        MPI_Irecv(data1, num1, MPI_FLOAT, 1, 200, MPI_COMM_WORLD, &request[0]);

        /* 2) Post non-blocking send (to rank 1, tag 100) */
        MPI_Isend(data0, num0, MPI_FLOAT, 1, 100, MPI_COMM_WORLD, &request[1]);

        /* 3) Wait for both operations to complete */
        MPI_Waitall(2, request, status);

        /* Print after completion (to match original idea) */
        printf("Process %d sent %d floats\n", rank, num0);
        printf("Process %d received %d floats\n", rank, num1);
    }

    /* Instructions for Process 1 */
    if (rank == 1) {
        /* Fill send buffer (data1) */
        for (i = 0; i < num1; i++) {
            data1[i] = sqrt((float)i);
        }

        /* 1) Post non-blocking receive FIRST (from rank 0, tag 100) */
        MPI_Irecv(data0, num0, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &request[0]);

        /* 2) Post non-blocking send (to rank 0, tag 200) */
        MPI_Isend(data1, num1, MPI_FLOAT, 0, 200, MPI_COMM_WORLD, &request[1]);

        /* 3) Wait for both operations to complete */
        MPI_Waitall(2, request, status);

        /* Print after completion */
        printf("Process %d sent %d floats\n", rank, num1);
        printf("Process %d received %d floats\n", rank, num0);
    }

    MPI_Finalize();
    return 0;
}
