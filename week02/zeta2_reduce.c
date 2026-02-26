#include <mpi.h>
#include <stdio.h>
#include <math.h>

/*
 * zeta2_reduce.c
 *
 * Goal:
 *   Compute S = sum_{k=1..N} 1/k^2 in parallel using an arbitrary number of MPI processes.
 *   Each rank computes a partial sum over a subset of k.
 *   Use MPI_Reduce with MPI_SUM to sum partial results on rank 0.
 *   Use double precision.
 *   Print:
 *     - the final estimate
 *     - the exact value pi^2/6
 *     - percentage error
 *
 * Work distribution:
 *   We distribute the integer range [1..N] across ranks using a "block with remainder" method:
 *     base = N / size
 *     rem  = N % size
 *   Ranks 0..(rem-1) get (base+1) elements, others get base elements.
 *
 * This makes the result independent of number of processes (correctness check).
 */

static void get_range_1_to_N(int N, int rank, int size, int *start, int *end) {
    int base = N / size;
    int rem  = N % size;

    /* Number of items for this rank */
    int my_count = base + (rank < rem ? 1 : 0);

    /* Starting index in 0-based counting */
    int offset = rank * base + (rank < rem ? rank : rem);

    /* Convert to k in [1..N] */
    *start = offset + 1;
    *end   = offset + my_count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    const int N = 100000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine this rank's k-range */
    int k_start, k_end;
    get_range_1_to_N(N, rank, size, &k_start, &k_end);

    /* Compute partial sum in double precision */
    double partial = 0.0;
    for (int k = k_start; k <= k_end; k++) {
        double kk = (double)k;
        partial += 1.0 / (kk * kk);   // faster & simpler than pow(kk, 2)
    }

    /* Reduce all partial sums into total on rank 0 */
    double total = 0.0;
    MPI_Reduce(&partial, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Compute exact value pi^2/6 safely (no reliance on M_PI) */
        double pi = acos(-1.0);
        double exact = (pi * pi) / 6.0;

        double pct_error = fabs((total - exact) / exact) * 100.0;

        printf("N = %d, nprocs = %d\n", N, size);
        printf("approx = %.15f\n", total);
        printf("exact  = %.15f\n", exact);
        printf("percent error = %.12e %%\n", pct_error);
    }

    MPI_Finalize();
    return 0;
}
