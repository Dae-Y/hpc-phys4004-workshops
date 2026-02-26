program hello
  implicit none
  include 'mpif.h'
  integer :: ierr

  call MPI_Init(ierr)
  print*, 'Hello World!'

  call MPI_Finalize(ierr)
end program hello
