#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* This exercise is designed for exactly 2 processes. */
    if (size != 2) {
        MPI_Finalize();
        return 0;
    }

    /* Default: 3 million floats. Allow override: ./main 100000 */
    int N = 3000000;
    if (argc > 1) {
        long tmp = atol(argv[1]);
        if (tmp > 0) N = (int)tmp;
    }

    const int tag = 123;

    if (rank == 0) {
        /* Allocate and fill send buffer */
        float *buf = (float *)malloc((size_t)N * sizeof(float));
        if (!buf) {
            fprintf(stderr, "Rank 0: malloc failed for N=%d\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; i++) {
            buf[i] = (float)i;  /* simple pattern; last value is N-1 */
        }

        /* Post non-blocking send */
        MPI_Request req;
        MPI_Isend(buf, N, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &req);

        /* Busy loop doing "computation" while we poll for completion */
        long long counter = 0;
        int done = 0;
        while (!done) {
            counter++;  /* the "computation" */
            MPI_Test(&req, &done, MPI_STATUS_IGNORE);
        }

        printf("Rank 0: Isend completed after counter = %lld iterations (N=%d floats)\n",
               counter, N);

        /* Safe to free only after send has completed */
        free(buf);
    }

    if (rank == 1) {
        /* Blocking receive */
        float *buf = (float *)malloc((size_t)N * sizeof(float));
        if (!buf) {
            fprintf(stderr, "Rank 1: malloc failed for N=%d\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Recv(buf, N, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Print the last received value */
        printf("Rank 1: received last value = %.1f (expected %.1f)\n",
               buf[N - 1], (float)(N - 1));

        free(buf);
    }

    MPI_Finalize();
    return 0;
}
