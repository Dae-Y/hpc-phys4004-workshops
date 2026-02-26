#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NMAX 100000

int main(int argc, char *argv[]) {
    int rank, ncpu;
    int num0, num1;
    int i;

    /* We use double precision arrays (heap-allocated to avoid stack limits). */
    double *data0 = NULL;
    double *data1 = NULL;

    MPI_Status status;

    /* ---------------------------
       Basic MPI initialization
       --------------------------- */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    /* This lab needs at least two processes (rank 0 and rank 1). */
    if (ncpu < 2) {
        /* Only rank 0 prints the message to avoid duplicated output. */
        if (rank == 0) {
            printf("Need at least two processes\n");
        }
        MPI_Finalize();
        return 0;
    }

    /* ---------------------------
       Parse command-line arguments
       --------------------------- */
    num0 = 100;
    num1 = 100;

    /* If one argument is given, use it for both directions. */
    if (argc > 1) {
        num0 = atoi(argv[1]);
        if (num0 < 1)    num0 = 1;
        if (num0 > NMAX) num0 = NMAX;
        num1 = num0;
    }

    /* If two arguments are given, set sizes independently. */
    if (argc > 2) {
        num1 = atoi(argv[2]);
        if (num1 < 1)    num1 = 1;
        if (num1 > NMAX) num1 = NMAX;
    }

    /* Allocate arrays on the heap. */
    data0 = (double *)malloc((size_t)NMAX * sizeof(double));
    data1 = (double *)malloc((size_t)NMAX * sizeof(double));
    if (data0 == NULL || data1 == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        free(data0);
        free(data1);
        MPI_Finalize();
        return 1;
    }

    /* ------------------------------------------------------------
       IMPORTANT: This program is intentionally written to create
       deadlock for large messages when using blocking MPI_Send.

       Both rank 0 and rank 1 do:
         1) MPI_Send(...)
         2) MPI_Recv(...)

       For small messages, MPI may buffer the send, so it "works".
       For larger messages, MPI may require a matching receive to
       progress (rendezvous protocol), so both processes can block
       in MPI_Send -> deadlock.
       ------------------------------------------------------------ */

    /* ---------------------------
       Instructions for rank 0
       --------------------------- */
    if (rank == 0) {
        /* Fill data0 with a simple dataset: i^2 (as double). */
        for (i = 0; i < num0; i++) {
            data0[i] = (double)i * (double)i;
        }

        /* Send num0 doubles to rank 1 using tag 100. */
        MPI_Send(data0, num0, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD);
        printf("Process %d sent %d doubles\n", rank, num0);

        /* Receive num1 doubles from rank 1 using tag 200. */
        MPI_Recv(data1, num1, MPI_DOUBLE, 1, 200, MPI_COMM_WORLD, &status);
        printf("Process %d received %d doubles\n", rank, num1);
    }

    /* ---------------------------
       Instructions for rank 1
       --------------------------- */
    if (rank == 1) {
        /* Fill data1 with a different dataset: sqrt(i) (as double). */
        for (i = 0; i < num1; i++) {
            data1[i] = sqrt((double)i);
        }

        /* Send num1 doubles to rank 0 using tag 200. */
        MPI_Send(data1, num1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
        printf("Process %d sent %d doubles\n", rank, num1);

        /* Receive num0 doubles from rank 0 using tag 100. */
        MPI_Recv(data0, num0, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &status);
        printf("Process %d received %d doubles\n", rank, num0);
    }

    /* Other ranks (>=2) do nothing for this lab; they just finalize. */

    free(data0);
    free(data1);

    MPI_Finalize();
    return 0;
}
