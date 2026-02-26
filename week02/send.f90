program send
  implicit none
  include 'mpif.h'
  integer :: rank, ierr
  integer :: num, tag, stat(MPI_STATUS_SIZE)

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

  tag=88

  if (rank == 0) then
    num=10
    call MPI_Send(num,1,MPI_INTEGER,1,tag,MPI_COMM_WORLD,ierr)
    print*, 'Process',rank,'sent the number',num
  end if

  if (rank == 1) then
    call MPI_Recv(num,1,MPI_INTEGER,0,tag,MPI_COMM_WORLD,stat,ierr)
    print*, 'Process',rank,'received the number',num
  end if

  call MPI_Finalize(ierr)
end program send