#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))    // ld : leading dimension

#define M 6     // row
#define N 5     // col

int main() {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *A, *x, *y;
    A = new float[M * N];
    x = new float[N];
    y = new float[M];

    // [a00, a10, a20, ..., a50], [a01, a11, ...], ..., [..., a44, a54]
    int ind = 11;
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        A[IDX2C(i, j, M)] = ind;
        cout << ind++ << (j == N-1 ? "\n" : " ");
    }
    for (int i = 0; i < N; i++) x[i] = 1.f;
    for (int j = 0; j < M; j++) y[j] = 0.f;

    float *d_A, *d_x, *d_y;
    cudaStat = cudaMalloc((void**)&d_A, M * N * sizeof(*A));
    cudaStat = cudaMalloc((void**)&d_x, N * sizeof(*x));
    cudaStat = cudaMalloc((void**)&d_y, M * sizeof(*y));
    
    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(M, N, sizeof(*A), A, M, d_A, M);
    stat = cublasSetVector(N, sizeof(*x), x, 1, d_x, 1);
    stat = cublasSetVector(M, sizeof(*y), y, 1, d_y, 1);
    float alpha = 1.f;
    float beta  = 0.f;

    // y = alpha * Ax + beta * y
    stat = cublasSgemv(
        handle, CUBLAS_OP_N,
        M, N,
        &alpha,
        d_A, M,
        d_x, 1,
        &beta,
        d_y, 1
    );
    stat = cublasGetVector(M, sizeof(*y), d_y, 1, y, 1);
    for (int j = 0; j < M; j++) cout << y[j] << endl;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    delete[] A, x, y;
}