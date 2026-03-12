
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#define SIZE 10

int main(int argc, char *argv[]) {

   int numtasks, rank, tag=1, i;
   int start, num, recv;
   int *a,*b;
   MPI_Status stat;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

   start=atoi(argv[1]);
   num=atoi(argv[2]);
   recv=atoi(argv[3]);

   if (rank == 0) {
        a= (int *) malloc(SIZE * sizeof(int));

        printf("Array 'a' on Process 0\n");
        for (i=0; i<SIZE; i++) {
            a[i]=i+1;
            printf("%3d ",a[i]);
        }
        printf("\n");

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(a+start, num, MPI_INT, 1, tag, MPI_COMM_WORLD);
    }

    if (rank == 1) {
	b= (int *) malloc(SIZE * sizeof(int));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("\nArray 'b' on Process 1 (before receive)\n");
        for (i=0; i<SIZE; i++) {
	    b[i]=0;
	    printf("%3d ",b[i]);
	}
        printf("\n");

        MPI_Recv(b+recv, num, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat);
        printf("\nArray 'b' on Process 1 (after receive)\n");
        for (i=0; i<SIZE; i++) {
	    printf("%3d ",b[i]);
	}
	printf("\n");
   }
  
   MPI_Finalize();
}

