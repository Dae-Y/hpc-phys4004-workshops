#include <stdio.h>
#include <mpi.h>
 
int main(int argc, char *argv[]) {
   int          rank, num, tag = 88;
   MPI_Status   status;
   MPI_Request  request;  // 1. Added Request handle for non-blocking tracking
 
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
   if (rank == 0) {
      num = 10;
      // 2. Changed MPI_Send to MPI_Isend (requires &request at the end)
      MPI_Isend(&num, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &request);
      
      // 3. Wait for the send to finish before the program ends or modifies 'num'
      MPI_Wait(&request, &status);
      printf("Process %d sent the number %d\n", rank, num);
   }
 
   if (rank == 1) {
      // 2. Changed MPI_Recv to MPI_Irecv (requires &request at the end)
      MPI_Irecv(&num, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &request);
      
      // 3. Wait to ensure the data has actually arrived before we print 'num'
      MPI_Wait(&request, &status);
      printf("Process %d received the number %d\n", rank, num);
   }
   
   MPI_Finalize();
   return 0;
}
