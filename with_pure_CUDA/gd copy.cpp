#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <cmath>
#include <chrono>
#include <unistd.h>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define rnd(x) ((x) * rand() / RAND_MAX)
#define rrnd() (rnd(6.f) - (3.f))

void generate_conditions(int N, int d, float *X, float *y);

int main() {
    srand(960501);

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *X, *y, *theta;
    float *d_X, *d_y, *d_theta;
    float *d_theta1, *d_tmpy;

    float h = 1e-2;
    float F0, F1;
    int iter;
    double threshold = 5e-6;   // stop condition

    float alpha = 1., beta = -1.;
    float alpha1 = - 2.f * h, beta1 = 1.f;

    chrono::system_clock::time_point t0, t1;
    chrono::duration<double> dt;

    for (int N = 1e3; N < 1e6; N *= 10) for (int d = 1e2; d < 1e6; d *= 10) {
        X = new float[N * d];
        y = new float[N];
        theta = new float[d];
        generate_conditions(N, d, X, y);
        for (int j = 0; j < d; j++) theta[j] = rrnd();
        printf("Completed Initialization!\t");

        t0 = chrono::system_clock::now();
        cudaStat = cudaMalloc((void**)&d_X, N * d * sizeof(*X));
        cudaStat = cudaMalloc((void**)&d_y, N * sizeof(*y));
        cudaStat = cudaMalloc((void**)&d_tmpy, N * sizeof(*y));
        cudaStat = cudaMalloc((void**)&d_theta, d * sizeof(*theta));
        cudaStat = cudaMalloc((void**)&d_theta1, d * sizeof(*theta));
        
        stat = cublasCreate(&handle);
        // why N and N??????????
        stat = cublasSetMatrix(N, d, sizeof(*X), X, N, d_X, N);  // could be done asynchronously
        stat = cublasSetVector(N, sizeof(*y), y, 1, d_y, 1);
        stat = cublasSetVector(d, sizeof(*theta), theta, 1, d_theta, 1);

        stat = cublasScopy(handle, N, d_y, 1, d_tmpy, 1);
        // stat = F(handle, d_X, d_theta, d_tmpy, N, d, F0);
        stat = cublasSgemv(
            handle, CUBLAS_OP_N,
            N, d, &alpha, d_X, N,
            d_theta, 1, &beta, d_tmpy, 1
        );
        stat = cublasSnrm2(handle, N, d_tmpy, 1, &F0);

        iter = 0;
        while (true) {
            stat = cublasScopy(handle, N, d_y, 1, d_tmpy, 1);
            stat = cublasScopy(handle, d, d_theta, 1, d_theta1, 1);
            // stat = newtheta(handle, d_X, d_theta1, d_tmpy, N, d, h);
            alpha1 = - 2.f * h;
            stat = cublasSgemv(
                handle, CUBLAS_OP_N,
                N, d, &alpha, d_X, N,
                d_theta1, 1, &beta, d_tmpy, 1
            );
            // stat = cublasGetVector(d, sizeof(float), d_theta, 1, theta, 1);
            // for (int i = 0; i < 10; i++) cout << theta[i] << " "; cout << endl;
            stat = cublasSgemv(
                handle, CUBLAS_OP_T,
                N, d, &alpha1, d_X, N,
                d_tmpy, 1, &beta1, d_theta1, 1
            );

            // cout << endl;
            // stat = cublasGetVector(d, sizeof(float), d_theta, 1, theta, 1);
            // for (int i = 0; i < 10; i++) cout << theta[i] << " ";
            // stat = cublasGetVector(d, sizeof(float), d_theta1, 1, theta, 1); cout << endl;
            // for (int i = 0; i < 10; i++) cout << theta[i] << " "; cout << endl;
            // sleep(1);

            stat = cublasScopy(handle, N, d_y, 1, d_tmpy, 1);
            // stat = F(handle, d_X, d_theta1, d_tmpy, N, d, F1);
            stat = cublasSgemv(
                handle, CUBLAS_OP_N,
                N, d, &alpha, d_X, N,
                d_theta1, 1, &beta, d_tmpy, 1
            );
            stat = cublasSnrm2(handle, N, d_tmpy, 1, &F1);
            
            // cout << F0 << " " << F1 << endl;
            // sleep(1);
            if (F0 / F1 - 1. < threshold && F1 <= F0) break;
            if (F1 > F0) h /= 2.f;
            else {
                h *= 1.2f;
                stat = cublasScopy(handle, d, d_theta1, 1, d_theta, 1);
                F0 = F1;
            }
            iter++;
        }
        stat = cublasGetVector(d, sizeof(float), d_theta, 1, theta, 1);
        t1 = chrono::system_clock::now();
        dt = t1 - t0;
        printf(
            "Iterated: %4d, N: %6d, d: %6d, t: %10.5fs, err: %9.5f\n",
            iter, N, d, dt.count(), F0
        );

        cudaFree(d_X);
        cudaFree(d_y);
        cudaFree(d_theta);
        cudaFree(d_theta1);
        cudaFree(d_tmpy);
        delete[] X, y, theta;
    }
    cublasDestroy(handle);
}

void generate_conditions(int N, int d, float *X, float *y) {
    for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
        X[IDX2C(i, j, N)] = rrnd();
    for (int i = 0; i < N; i++) y[i] = rrnd();
}