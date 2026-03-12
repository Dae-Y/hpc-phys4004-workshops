program mandelbrot

complex z,kappa
integer green,blue,i,j,k

integer   N,maxiter
parameter (N=2000,maxiter=1000)
real      x(0:N-1,0:N-1)
    
! ----------------------------------------------------------------

do i=0,N-1
    do j=0,N-1
        kappa = cmplx((4.0*(i-N/2)/N),(4.0*(j-N/2)/N))

        k=1
        z=kappa
        do while (abs(z).le.2 .and. k.lt.maxiter) 
            k=k+1
            z=z*z + kappa
        end do

        x(i,j)=log(real(k))/log(real(maxiter))
    end do
 
    if (mod(i+1,N/10).eq.0) write(6,*) 100*(i+1)/N,'% complete.'
end do

! ----------------------------------------------------------------

write(6,*) 'Writing mandelbrot.ppm'

open(7,file='mandelbrot.ppm',status='unknown')
write(7,100) 'P6', N, N, 255

do j=0,N-1
    do i=0,N-1
        if (x(i,j).lt.0.5) then
            green=2.0*x(i,j)*255
            write(7,110) achar(255-green),achar(green),achar(0)
        else 
            blue= 2.0*x(i,j)*255 - 255
            write(7,110) achar(0),achar(255-blue),achar(blue)
        end if
    end do
end do

100  format(A2,/,I4,I5,/,I3)
110  format(3A1,$)

close(7)
end
