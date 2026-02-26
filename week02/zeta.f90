program zeta
  implicit none
  integer :: i
  real(kind=8) :: total, pi, exact

  pi=3.1415926536d0
  exact=pi**2 / 6.0d0

  total=0.0d0
  do i=1, 100000
    total = total + 1.0d0/dble(i)**2
  end do
  
  print*, 'total=',total,'   exact=',exact

end program zeta