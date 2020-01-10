#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define rnd(x) ((x) * rand() / RAND_MAX)
#define rrnd() (rnd(6.f) - (3.f))

void generate_conditions(float *A, float *y, int N, int d);
void F(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float &result);
void newtheta(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float h);

int main() {

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *A, *x, *y, *x1;
    float *dA, *dx, *dy, *dx1;
    float h, F0, F1;
    int updated, realiter;

    chrono::system_clock::time_point t0, t1;
    chrono::duration<double> dt;

    for (int N = 1e3; N < 1e6; N *= 10) for (int d = 1e2; d < 1e6; d *= 10) {
        h = 1e-2;
        A = new float[N * d];
        x = new float[d];
        y = new float[N];
        x1 = new float[d];
        generate_conditions(A, y, N, d);
        for (int j = 0; j < d; j++) x[j] = rrnd();
        printf("Completed Initialization!\t");
        t0 = chrono::system_clock::now();

        cudaMalloc((void**)&dA, N * d * sizeof(*A));
        cudaMalloc((void**)&dx, d * sizeof(*x));
        cudaMalloc((void**)&dy, N * sizeof(*y));
        cudaMalloc((void**)&dx1, d * sizeof(*x1));
        cublasSetMatrix(N, d, sizeof(*A), A, N, dA, N);
        cublasSetVector(d, sizeof(*x), x, 1, dx, 1);
        cublasSetVector(N, sizeof(*y), y, 1, dy, 1);

        F(handle, dA, dx, dy, N, d, F0); realiter = 0;
        for (int iter = 0; iter < 10; iter+=updated, realiter++) {
            cublasScopy(handle, d, dx, 1, dx1, 1);
            newtheta(handle, dA, dx1, dy, N, d, h);
            F(handle, dA, dx1, dy, N, d, F1);
            if (F1 > F0) {
                updated = 0;
                h /= 2.f;
            } else {
                updated = 1;
                cublasScopy(handle, d, dx1, 1, dx, 1);
                h *= 1.2f;
                F0 = F1;
            }
        }

        cudaFree(dA);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dx1);

        t1 = chrono::system_clock::now();
        dt = t1 - t0;

        printf(
            "Iterated: %4d, N: %6d, d: %6d, rmse: %10.5f, t: %9.3f ms, time/iter: %9.4f ms\n",
            realiter, N, d, F0, dt.count() * 1000, dt.count() / realiter * 1000
        );

        delete[] A, x, x1, y;
    }
    cublasDestroy(handle); 
}

void generate_conditions(float *A, float *y, int N, int d) {
    float *env = new float[d];
    float sum;
    for (int j = 0; j < d; j++) env[j] = rrnd();
    for (int i = 0; i < N; i++) {
        sum = 0;
        for (int j = 0; j < d; j++) A[IDX2C(i, j, N)] = rrnd() * 3.;
        for (int j = 0; j < d; j++) sum += A[IDX2C(i, j, N)] * env[j];
        y[i] = sum + rrnd() / 3.;
    }
    delete[] env;
}

void F(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float &result) {
    float *tmpdy; cudaMalloc((void**)&tmpdy, N * sizeof(*dy));
    cublasScopy(handle, N, dy, 1, tmpdy, 1);
    
    float alpha = 1.f, beta = -1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, dA, N,
        dx, 1, &beta, tmpdy, 1
    );
    cublasSnrm2(handle, N, tmpdy, 1, &result);
    result = sqrt(result * result / N); // RMSE
    cudaFree(tmpdy);
}

void newtheta(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float h) {
    float *tmpdy; cudaMalloc((void**)&tmpdy, N * sizeof(*dy));
    cublasScopy(handle, N, dy, 1, tmpdy, 1);

    float alpha = 1.f, beta = -1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, dA, N,
        dx, 1, &beta, tmpdy, 1
    );

    alpha = - 2.f * h, beta = 1.f;
    cublasSgemv(
        handle, CUBLAS_OP_T,
        N, d, &alpha, dA, N,
        tmpdy, 1, &beta, dx, 1
    );

    cudaFree(tmpdy);
}
