#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <random>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define rnd(x) ((x) * rand() / RAND_MAX)
#define rrnd() (rnd(6.f) - (3.f))


void generate_conditions(float *A, float *y, int N, int d) {
    float *env = new float[d];
    float sum;
    for (int i = 0; i < N; i++) {
        sum = 0;
        for (int j = 0; j < d; j++) {
            A[IDX2C(i, j, N)] = rrnd() * 3.;
            env[j] = rrnd();
        }
        for (int j = 0; j < d; j++)
            sum += A[IDX2C(i, j, N)] * env[j];
        y[i] = sum + rrnd() / 3.;
    }
    delete[] env;
}

void func(cublasHandle_t handle, float *da, float *db, int N) {
    float *a = new float[N];
    float *b = new float[N];
    // cudaMemcpy(a, da, N * sizeof(float), cudaMemcpyDeviceToHost);
    cublasGetVector(N, sizeof(*da), da, 1, a, 1);
    cublasGetVector(N, sizeof(*db), db, 1, b, 1);
    for (int i = 0; i < N; i++) cout << a[i] << " "; cout << endl;
    for (int i = 0; i < N; i++) cout << b[i] << " "; cout << endl;

    float result;
    cublasSdot(handle, N, da, 1, db, 1, &result);

    cublasGetVector(N, sizeof(*da), da, 1, a, 1);
    cublasGetVector(N, sizeof(*db), db, 1, b, 1);
    for (int i = 0; i < N; i++) cout << a[i] << " "; cout << endl;
    for (int i = 0; i < N; i++) cout << b[i] << " "; cout << endl;

    cout << result << endl;
    delete[] a, b;
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

int main() {

    
    cublasHandle_t handle;
    cublasCreate(&handle);

    // const int N = 10;
    // float result;
    // float *a = new float[N], *da;
    // float *b = new float[N], *db;
    // for (int i = 0; i < N; i++) a[i] = i * i;
    // for (int i = 0; i < N; i++) b[i] = 3 * i;
    // cudaMalloc((void**)&da, N * sizeof(float));
    // cudaMalloc((void**)&db, N * sizeof(float));
    // cublasSetVector(N, sizeof(*a), a, 1, da, 1);
    // cublasSetVector(N, sizeof(*b), b, 1, db, 1);
    // func(handle, da, db, N);
    // cudaMemcpy(a, da, N * sizeof(*a), cudaMemcpyDeviceToHost);
    // cout << endl;
    // cudaFree(da);
    // cudaFree(db);
    // delete[] a, b;

    const int N = 10, d = 6;
    float *A = new float[N * d], *dA;
    float *x = new float[d], *dx;
    float *y = new float[N], *dy;
    // for (int i = 0; i < N; i++) for (int j = 0; j < d; j++) A[IDX2C(i, j, N)] = 1 + i + j * N;
    for (int j = 0; j < d; j++) x[j] = d - j;
    // for (int i = 0; i < N; i++) y[i] = 450;
    generate_conditions(A, y, N, d);

    cudaMalloc((void**)&dA, N * d * sizeof(*A));
    cudaMalloc((void**)&dx, d * sizeof(*x));
    cudaMalloc((void**)&dy, N * sizeof(*y));

    cublasSetMatrix(N, d, sizeof(*A), A, N, dA, N);
    cublasSetVector(d, sizeof(*x), x, 1, dx, 1);
    cublasSetVector(N, sizeof(*y), y, 1, dy, 1);

    cublasGetMatrix(N, d, sizeof(*A), dA, N, A, N);
    cublasGetVector(d, sizeof(*x), dx, 1, x, 1);
    cublasGetVector(N, sizeof(*y), dy, 1, y, 1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++)
            printf("%8.5f ", A[IDX2C(i, j, N)]);
        if (i < d) cout << setw(2) << x[i];
        printf("   \t%10.5f\n", y[i]);
    }
    cout << endl << endl;

    float result;
    F(handle, dA, dx, dy, N, d, result);

    cublasGetMatrix(N, d, sizeof(*A), dA, N, A, N);
    cublasGetVector(d, sizeof(*x), dx, 1, x, 1);
    cublasGetVector(N, sizeof(*y), dy, 1, y, 1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++)
            printf("%8.5f ", A[IDX2C(i, j, N)]);
        if (i < d) cout << setw(2) << x[i];
        printf("   \t%10.5f\n", y[i]);
    }
    cout << endl << result << endl << endl;
    
    float h = 1e-2, F0, F1;
    int updated = 0;
    float *x1 = new float[d], *dx1;
    cudaMalloc((void**)&dx1, d * sizeof(*x1));
    F(handle, dA, dx, dy, N, d, F0);
    for (int iter = 0; iter < 10; iter+=updated) {
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
        // cublasGetVector(d, sizeof(*x), dx, 1, x, 1);
        // for (int j = 0; j < d; j++) cout << x[j] << endl;
        // cout << endl << F0 << " " << F1 << endl << endl;
        cout << F0 << " " << F1 << endl;
    }
    
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dx1);
    delete[] A, x, x1, y;
    cublasDestroy(handle); 
}
