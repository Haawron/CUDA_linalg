#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <random>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))


cublasStatus_t
eval(
    cublasHandle_t handle,
    float *d_X, float *d_theta, float *copied_d_y,
    int N, int d) {
    /*
        y <- alpha * Ax + beta * y
        => y <- Ax - y
    */
    cublasStatus_t stat;
    float alpha = 1., beta = -1.;
    float *X = new float[N * d];

    // stat = cublasGetMatrix(N, d, sizeof(float), d_X, N, X, d);
    // for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
    //     cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");
    cout << endl;

    stat = cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, d_X, N,
        d_theta, 1, &beta, copied_d_y, 1
    );

    // stat = cublasGetMatrix(N, d, sizeof(float), d_X, N, X, d);
    // for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
    //     cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");

    delete[] X;
    return stat;
}

cublasStatus_t
F(
    cublasHandle_t handle,
    float *&d_X, float *d_theta, float *copied_d_y,
    int N, int d, float &result) {
    /*
        This function works inplace.
        d_y must be copied in advance.
    */
    cublasStatus_t stat;

    float *X = new float[N * d];
    stat = cublasGetMatrix(N, d, sizeof(float), d_X, N, X, d);
    for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
        cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");
    cout << endl;

    stat = eval(handle, d_X, d_theta, copied_d_y, N, d);

    // stat = cublasGetMatrix(N, d, sizeof(float), d_X, N, X, d);
    // for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
    //     cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");

    stat = cublasSnrm2(handle, N, copied_d_y, 1, &result);
    return stat;
}

int main() {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *X = new float[10 * 5];
    float *theta = new float[5];
    float *y = new float[10];
    float *d_X, *d_theta, *d_y;

    const int N = 10, d = 5;

    int idx = 1;
    for (int j = 0; j < d; j++) for (int i = 0; i < N; i++) X[IDX2C(i, j, N)] = idx++;
    for (int j = 0; j < d; j++) theta[j] = 1;
    for (int i = 0; i < N; i++) y[i] = -1;

    // for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
    //     cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");
    // for (int j = 0; j < d; j++) cout << theta[j] << endl;
    
    float f0;
    cudaMalloc((void**)&d_X, N * d * sizeof(float));
    cudaMalloc((void**)&d_theta, d * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(N, d, sizeof(float), X, N, d_X, d);
    stat = cublasSetVector(N, sizeof(float), y, 1, d_y, 1);
    stat = cublasSetVector(d, sizeof(float), theta, 1, d_theta, 1);

    stat = cublasGetMatrix(N, d, sizeof(float), d_X, N, X, d);
    for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
        cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");

    cout << endl;
    F(handle, d_X, d_theta, d_y, N, d, f0);
    // float alpha = 1., beta = -1.;
    // stat = cublasSgemv(
    //     handle, CUBLAS_OP_N,
    //     N, d, &alpha, d_X, N,
    //     d_theta, 1, &beta, d_y, 1
    // );

    // stat = cublasGetMatrix(N, d, sizeof(float), d_X, N, X, d);
    // for (int i = 0; i < N; i++) for (int j = 0; j < d; j++)
    //     cout << X[IDX2C(i, j, N)] << (j == d - 1 ? "\n" : " ");

    // stat = cublasGetVector(N, sizeof(float), d_y, 1, y, 1);
    // for (int i = 0; i < N; i++) cout << y[i] << " ";
    cout << endl;

    cout << f0 << endl;

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    delete[] X, theta, y;
}