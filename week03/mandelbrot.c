
#define MAXITER 1000
#define N	2000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

/* ----------------------------------------------------------------*/

int main() {
    int	   i,j,k,tenth;
    short  green,blue;
    float  **x;
    FILE   *fp;

    float complex   z, kappa;

    // Allocate x as a 2-D array - requires use of a double pointer
    x=(float**)malloc(N*sizeof(float*));
    for (i=0;i<N;i++) {
      x[i]=(float*)malloc(N*sizeof(float));
    }
  
    // Iterate over 2-D grid: i covers N cols and j covers N rows
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
	    
            //kappa is the coordinate in the complex plane of the (i,j)th pixel
            //both the real and imaginary axis range from -2 to 2 (4 units)
            kappa = (4.0*(i-N/2))/N + (4.0*(j-N/2))/N * I;

            //z_0 = kappa, initial value of z before iteration begins
            z = kappa;
	
            //now we iterate until either the max number of iterations is reached 
            //OR z leaves the circle of radius 2 centred at the origin
	    k=1; 
	    while ((cabs(z)<=2) && (k++<MAXITER)) {
	        z = z*z + kappa;
            }
	  
            // Store the number of iterations required for this point to leave the mandelbrot set.
            // If this point IS IN the Mandelbrot set, then k=MAXITER.
            // We actually store log(k)/log(MAXITER) to see more colour variation in the points 
            // outside the Mandelbrot set.
            // Points that are in the set have k=MAXITER and therefore log(k)/log(MAXITER)=1.0.
            // When the mandelbrot.ppm file is created below, x=1.0 translates to 100% blue.
	    x[i][j]=log((float)k) / log((float)MAXITER);
        }

        tenth=(int) N/10;
        if ((i+1)%tenth==0) 
            printf("%d%% complete.\n",100*(i+1)/N);
    }

/* ----------------------------------------------------------------*/
  
    printf("Writing mandelbrot.ppm\n");
    fp = fopen ("mandelbrot.ppm", "w");
    fprintf (fp, "P6\n%4d %4d\n255\n", N, N);
    
    for (j=0; j<N; j++)
        for (i=0; i<N; i++) 
	    if (x[i][j]<0.5) {
	        green= (int) (2*x[i][j]*255);
                fprintf (fp, "%c%c%c", 255-green,green,0);
	    } else {
	        blue= (int) (2*x[i][j]*255-255);
                fprintf (fp, "%c%c%c", 0,255-blue,blue);
	    }
    
    fclose(fp);

/* ----------------------------------------------------------------*/

    for (i=0; i<N; i++)
        free(x[i]);
    free(x);
}
