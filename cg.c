#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#define MAX 1024
//#define MAX 384
#define EPS 1e-10
void main(int argc, char **argv) {
    int i,j,k;
    int pid, nproc;
    int ss;
    float a[MAX][MAX], b[MAX], x[MAX], p[MAX], y[MAX], r[MAX], psum[MAX], alpha,alpha0,alpha1,beta,beta0;
    double ts1,ts2;
    double t;
    double remain;
    int add;
    int fin;
    double start, end;

// Initialize
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    for (i=0; i< MAX; i++) b[i] = 50.0+i;
    for (i=0; i< MAX; i++) x[i] = 0;
    for (j=0; j< MAX; j++) {
        for(i=0; i<MAX; i++) {
            if(i== j) a[i][j] = 8.0;
            else a[i][j] = 5.0;
        }
    }
    for (j=0; j< MAX; j++) {
        y[j]=0;
        for (i=0; i< MAX; i++)
            y[j] += a[i][j]*x[i];
    }
    for(i=0; i<MAX; i++) {
        p[i]= b[i]- y[i];
        r[i] = p[i];
    }
// Loop
    if (pid == 0)
        start = MPI_Wtime();
    for(k=0; k<MAX; k++) {
    // A x p
        if (pid == 0) {
            //Send
            for (i = 1; i < nproc; i++) {
                if (i != nproc - 1) {
                    MPI_Send(&a[i*MAX/nproc][0], MAX*(MAX/nproc), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&p[i*MAX/nproc], MAX/nproc, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
                else {
                    add = MAX % nproc;
                    MPI_Send(&a[i*MAX/nproc][0], MAX*(MAX/nproc + add), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&p[i*MAX/nproc], MAX/nproc + add, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
            //Compute
            for (j=0; j<MAX; j++) {
                y[j] = 0.0;
                for(i=0; i<MAX/nproc; i++)
                    y[j] += a[i][j]*p[i];
            }
            //Receive
            for (i = 1; i < nproc; i++) {
                MPI_Recv(&psum[0], MAX, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                for (j = 0; j < MAX; j++)
                    y[j] += psum[j];
            }

            alpha0 = 0; alpha1=0;
            for(i=0; i<MAX; i++) {
                alpha0 += p[i]*r[i];
                alpha1 += p[i]*y[i];
            }
            alpha = alpha0/alpha1;
            for(i=0;i<MAX;i++)
                x[i] = x[i]+alpha*p[i];
            for(i=0;i<MAX;i++)
                r[i] = r[i]- alpha*y[i];
            beta0 = 0.0;
            for(i=0; i<MAX;i++)
                beta0 += r[i]*y[i];
            beta = -beta0/alpha1;

            for(i=0; i<MAX;i++)
                p[i] = r[i] + beta*p[i];

            remain=0.0;
            for(i=0;i<MAX;i++)
                remain+=r[i]*r[i];
            printf("%d: %lf\n",k, remain);
            if (remain<EPS) {
                fin = 1;
                for (i = 1; i < nproc; i++)
                    MPI_Send(&fin, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                end = MPI_Wtime();
                break;
            }
            else {
                fin = 0;
                for (i = 1; i < nproc; i++)
                    MPI_Send(&fin, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        else {
            if (pid != nproc - 1) {
                i = 0;
                MPI_Recv(&a[i][i], MAX*(MAX/nproc), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&p[i], MAX/nproc, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                for (i = 0; i < MAX; i++) {
                    psum[i] = 0;
                    for (j = 0; j < MAX/nproc; j++)
                        psum[i] += a[j][i] * p[j];
                }
            }
            else {
                i = 0;
                add = MAX % nproc;
                MPI_Recv(&a[i][i], MAX*(MAX/nproc + add), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&p[i], MAX/nproc + add, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                for (i = 0; i < MAX; i++) {
                    psum[i] = 0;
                    for (j = 0; j < MAX/nproc + add; j++)
                        psum[i] += a[j][i] * p[j];
                }
            }
            MPI_Send(&psum[0], MAX, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&fin, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (fin)
                break;
        }
    }

    for(i=0;i<MAX;i++)
        printf("%d: %f %f\n",i,x[i],r[i]);

    MPI_Finalize();

    if (pid == 0)
        printf("実行時間: %lf\n", end - start);
}