#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    double total = 0.0;
    double pi = 3.141592653589793;

    // Starts MPI, rank = which process you are, size = total number of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // If processes are not exactly 4, exit
    // For this exercise, we only use 4 processes, 4 different summations p = 2, 3, 4, 5
    if (size != 4) {
        MPI_Finalize();
        return 0;
    }

    // Determine the power 'p' dynamically based on rank (0->2, 1->3, 2->4, 3->5)
    // Assign each process a different exponent
    int p = rank + 2;

    // Use an array to store exact values mapped to rank indices (No 'if' needed)
    // zeta(3) and zeta(5) use their known mathematical constants
    double exact_values[4] = {
        (pi * pi) / 6.0,             // rank 0 (p=2) ζ(2)
        1.202056903159594,           // rank 1 (p=3) Apéry’s constant
        pow(pi, 4) / 90.0,           // rank 2 (p=4)
        1.036927755143370            // rank 3 (p=5) ζ(5)
    };

    // Calculate the partial sum
    // Each process computes its own series, no communication needed
    for (int i = 1; i <= 100000; i++) {
        total += 1.0 / pow((double)i, p);
    }

    // Print the result for each rank
    // Each process prints it's rank, exponent p, its computed sum, the exact value
    printf("rank %d (p=%d) : total=%.15f   exact=%.15f\n", rank, p, total, exact_values[rank]);

    MPI_Finalize();
    return 0;
}
