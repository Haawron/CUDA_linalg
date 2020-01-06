#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
using namespace std;

#define N 1024

int main() {
    cudaError_t cudaStat;   // cudaMalloc status
    cublasStatus_t stat;    // CUBLAS functions status
    cublasHandle_t handle;  // CUBLAS context
    
    float *x, *y;
    x = new float[N];
    y = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = 2 * i;
        y[i] = 3 * i;
    }

    for (int i = 0; i < 10; i++) cout << x[i] << " " << y[i] << endl;

    float *d_x, *d_y;
    cudaStat = cudaMalloc((void**)&d_x, N * sizeof(*x));
    cudaStat = cudaMalloc((void**)&d_y, N * sizeof(*y));
    
    stat = cublasCreate(&handle);   // initialize CUBLAS context
    stat = cublasSetVector(N, sizeof(*x), x, 1, d_x, 1);
    stat = cublasSetVector(N, sizeof(*y), y, 1, d_y, 1);

    float alpha = 2.;
    stat = cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);      // y <- alpha * x + y
    
    stat = cublasGetVector(N, sizeof(float), d_y, 1, y, 1);

    for (int i = 0; i < 10; i++) cout << y[i] << endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    delete[] x, y;
}