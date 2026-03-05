#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * dot_mpi.c
 *
 * MPI dot product using Scatter + Reduce.
 * Simpler version
 *
 * Steps:
 *  1) Choose a fixed chunk size per rank: count
 *  2) Total vector length N = count * nprocs
 *  3) Rank 0 allocates full x,y, fills them, and computes serial dot
 *  4) Scatter x and y into local chunks xpart, ypart on every rank
 *  5) Each rank computes partial dot over its chunk
 *  6) Reduce partial dots (sum) into dot on rank 0
 *  7) Rank 0 compares serial vs parallel results
 */

int main(int argc, char *argv[]) {
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Hard-code how many elements each rank will own */
    const int count = 100000;                 // adjust if needed
    const long long N = (long long)count * nprocs;

    double *x = NULL;
    double *y = NULL;

    /* Local chunks on every rank */
    double *xpart = (double*)malloc((size_t)count * sizeof(double));
    double *ypart = (double*)malloc((size_t)count * sizeof(double));
    if (!xpart || !ypart) {
        fprintf(stderr, "Rank %d: failed to allocate local arrays\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double dot_serial = 0.0;

    /* Rank 0 creates the full arrays and computes a serial reference */
    if (rank == 0) {
        x = (double*)malloc((size_t)N * sizeof(double));
        y = (double*)malloc((size_t)N * sizeof(double));
        if (!x || !y) {
            fprintf(stderr, "Rank 0: failed to allocate full arrays (N=%lld)\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Fill with simple deterministic values */
        for (long long i = 0; i < N; i++) {
            x[i] = (double)i * 0.001;     // e.g., 0, 0.001, 0.002, ...
            y[i] = 1.0 / (1.0 + (double)i); // decreasing positive values
        }

        /* Serial dot product on rank 0 for verification */
        for (long long i = 0; i < N; i++) {
            dot_serial += x[i] * y[i];
        }
    }

    /* Scatter chunks of x and y from rank 0 to all ranks */
    MPI_Scatter(x, count, MPI_DOUBLE, xpart, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, count, MPI_DOUBLE, ypart, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Each rank computes its partial dot product */
    double dotpart = 0.0;
    for (int i = 0; i < count; i++) {
        dotpart += xpart[i] * ypart[i];
    }

    /* Reduce all dotpart values into dot on rank 0 */
    double dot = 0.0;
    MPI_Reduce(&dotpart, &dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Rank 0 verifies and prints results */
    if (rank == 0) {
        double diff = dot - dot_serial;
        if (diff < 0) diff = -diff;

        printf("N = %lld (count=%d per rank), nprocs = %d\n", N, count, nprocs);
        printf("dot_serial   = %.15e\n", dot_serial);
        printf("dot_parallel = %.15e\n", dot);
        printf("abs diff     = %.15e\n", diff);

        free(x);
        free(y);
    }

    free(xpart);
    free(ypart);

    MPI_Finalize();
    return 0;
}
