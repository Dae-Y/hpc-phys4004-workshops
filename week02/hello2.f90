program hello
  implicit none
  include 'mpif.h'
  integer :: rank, ierr

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

  print*, 'Hello World. I am process ',rank

  call MPI_Finalize(ierr)
end program hello