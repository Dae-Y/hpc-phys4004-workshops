program array
  implicit none 
  include 'mpif.h'
 
  integer :: rank, nranks, ierr
  integer :: num, num0, num1, i, tag, stat(MPI_STATUS_SIZE)
  integer, parameter :: NMAX=100000
  character*80 :: value
     
  real data0(NMAX),data1(NMAX)
 
  !Standard MPI initialisation 
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nranks,ierr)
 
  !Check there are enough processes 
  if (nranks.eq.1) then
    print*, 'Need at least two processes'
    call MPI_Finalize(ierr)
    stop
  end if
 
  !Parse the command-line
  num0=100
  num1=100
 
  if (iargc() >= 1) then
    call getarg(1,value)
    read(value,*) num0
 
    if (num0 < 1)    num0=1000
    if (num0 > NMAX) num0=NMAX
    num1=num0
  end if
 
  if (iargc() >= 2) then
    call getarg(2,value)
    read(value,*) num1
 
    if (num1 < 1)    num1=1000
    if (num1 > NMAX) num1=NMAX
  end if
 
  !Instructions for Process 0
  if (rank == 0) then
    do i=1,num0
        data0(i)=i*i
    end do
 
    call MPI_Send(data0,num0,MPI_REAL,1,100,MPI_COMM_WORLD,ierr)
    print*, 'Process',rank,'sent',num0,'reals'
 
    call MPI_Recv(data1,num1,MPI_REAL,1,200,MPI_COMM_WORLD,stat,ierr)
    print*, 'Process',rank,'received',num1,'reals'
  end if
 
  !Instructions for Process 1
  if (rank == 1) then
    do i=1,num1
        data1(i)=sqrt(real(i))
    end do
 
    call MPI_Send(data1,num1,MPI_REAL,0,200,MPI_COMM_WORLD,ierr)
    print*, 'Process',rank,'sent ',num1,'reals'
 
    call MPI_Recv(data0,num0,MPI_REAL,0,100,MPI_COMM_WORLD,stat,ierr)
    print*, 'Process',rank,'received',num0,'reals'
  end if
 
  call MPI_Finalize(ierr)
end program array
