#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * dot_scatter_reduce.c
 *
 * Goal:
 *   Compute dot = sum_{i=0..N-1} x[i]*y[i] using many MPI processes.
 *
 * Required steps:
 *   1) Allocate big arrays x and y (same size).
 *   2) On rank 0 only:
 *        - populate x and y
 *        - compute serial dot product (reference)
 *   3) Scatter x and y across all ranks:
 *        - each rank receives count elements (xpart, ypart)
 *        - N is set to count * nprocs (hard-coded count, arbitrary nprocs)
 *   4) Each rank computes dotpart over its local chunk.
 *   5) Use MPI_Reduce (MPI_SUM) to sum dotpart into dot on rank 0.
 *   6) Verify reduced dot matches serial dot on rank 0.
 *
 * Notes:
 *   - We use double precision.
 *   - MPI_Scatter requires equal chunk sizes for each rank.
 *   - Using a deterministic fill ensures reproducibility.
 */

static double serial_dot(const double *x, const double *y, long long N) {
    double s = 0.0;
    for (long long i = 0; i < N; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Hard-code chunk size per rank (adjust if you want larger/smaller) */
    const int count = 100000;                 // elements per rank
    const long long N = (long long)count * (long long)size;  // total length

    double *x = NULL;
    double *y = NULL;

    /* Only rank 0 allocates and initializes the full arrays */
    double dot_serial = 0.0;
    if (rank == 0) {
        x = (double *)malloc((size_t)N * sizeof(double));
        y = (double *)malloc((size_t)N * sizeof(double));
        if (!x || !y) {
            fprintf(stderr, "Rank 0: allocation failed for N=%lld\n", N);
            free(x);
            free(y);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Fill arrays with deterministic values */
        for (long long i = 0; i < N; i++) {
            x[i] = 0.001 * (double)i;            // simple increasing pattern
            y[i] = sin(0.001 * (double)i);       // smooth function
        }

        /* Serial reference dot product on rank 0 */
        dot_serial = serial_dot(x, y, N);
    }

    /* Each rank allocates local chunks */
    double *xpart = (double *)malloc((size_t)count * sizeof(double));
    double *ypart = (double *)malloc((size_t)count * sizeof(double));
    if (!xpart || !ypart) {
        fprintf(stderr, "Rank %d: allocation failed for local chunks\n", rank);
        free(xpart);
        free(ypart);
        if (rank == 0) { free(x); free(y); }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Scatter x and y from rank 0 to all ranks */
    MPI_Scatter(x, count, MPI_DOUBLE, xpart, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, count, MPI_DOUBLE, ypart, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Local dot product on each rank */
    double dotpart = 0.0;
    for (int i = 0; i < count; i++) {
        dotpart += xpart[i] * ypart[i];
    }

    /* Reduce local partial dots into final dot on rank 0 */
    double dot_parallel = 0.0;
    MPI_Reduce(&dotpart, &dot_parallel, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Verify results on rank 0 */
    if (rank == 0) {
        double abs_diff = fabs(dot_parallel - dot_serial);
        double rel_diff = abs_diff / (fabs(dot_serial) + 1e-300);

        printf("N = %lld (count=%d per rank), nprocs = %d\n", N, count, size);
        printf("dot_serial   = %.15e\n", dot_serial);
        printf("dot_parallel = %.15e\n", dot_parallel);
        printf("abs diff     = %.15e\n", abs_diff);
        printf("rel diff     = %.15e\n", rel_diff);

        free(x);
        free(y);
    }

    free(xpart);
    free(ypart);

    MPI_Finalize();
    return 0;
}
