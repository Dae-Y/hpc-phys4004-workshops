program memory

include 'mpif.h'

parameter (N=10)
integer   err, rank, ncpu, stat(MPI_STATUS_SIZE)
integer   start, num, recv, i
integer   a(N), b(N)
character*80 value

call MPI_INIT(err)
call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, err)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, err)

call getarg(1,value)
read(value,*) start

call getarg(2,value)
read(value,*) num

call getarg(3,value)
read(value,*) recv

if (rank.eq.0) then
    write(6,*) "Array 'a' on Process 0"
    do i=1,N
       a(i)=i
    end do
    write(6,'(10I4,/)') (a(i),i=1,N)

    call MPI_BARRIER(MPI_COMM_WORLD,err)
    call MPI_SEND(a(start),num,MPI_INTEGER,1,0,MPI_COMM_WORLD,err)
endif

if (rank.eq.1) then
    call MPI_BARRIER(MPI_COMM_WORLD,err)

    write(6,*) "Array 'b' on Process 1 (before receive)"
    do i=1,N
       b(i)=0
    end do
    write(6,'(10I4,/)') (b(i),i=1,N)

    call MPI_RECV(b(recv),num,MPI_INTEGER,0,0,MPI_COMM_WORLD,stat,err)

    write(6,*) "Array 'b' on Process 1 (after receive)"
    write(6,'(10I4)') (b(i),i=1,N)
endif

call MPI_FINALIZE(err)
end
