#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <curand.h>

#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define now() chrono::system_clock::now()
#define SEED 960501

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void GPU_fill_rand(float *dx, int size, float mean, float std);
chrono::duration<double> generate_conditions(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d);
void F(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float &result);
void newtheta(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float h);

int main() {

    cudaProfilerStart();

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *dA, *dx, *dy, *dx1;
    float h, F0, F1;
    int updated, realiter, niter = 30;

    chrono::system_clock::time_point t0, t1;
    chrono::duration<double> dt, dt1, dtrans;

    for (int64_t N = 1e3; N < 1e6; N *= 10) for (int64_t d = 1e2; d < 1e6; d *= 10) {
        h = 1e-2;
        t1 = now();
        gpuErrchk(cudaMalloc((void**)&dA, N * d * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&dx, d * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&dy, N * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&dx1, d * sizeof(float)));
        t0 = now();
        dtrans = generate_conditions(handle, dA, dx, dy, N, d);
        dt1 = now() - t1 - dtrans;

        F(handle, dA, dx, dy, N, d, F0); realiter = 0;
        for (int iter = 0; iter < niter; iter+=updated, realiter++) {
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

        dt = now() - t0;

        printf(
            "N: %6lld, d: %6lld, t_init: %7.3f ms, t_trans: %8.5f s, iterated: %4d, rmse: %10.5f, t: %8.5f s, time/iter: %9.4f ms\n",
            N, d, dt1.count() * 1000, dtrans.count(), realiter, F0, dt.count(), dt.count() / realiter * 1000
        );
    }
    cublasDestroy(handle);
    cudaDeviceReset();
    cudaProfilerStop();
}

void GPU_fill_rand(curandGenerator_t prng, float *dx, int size, float mean, float std) {
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, SEED);
    curandGenerateNormal(prng, dx, size, mean, std);
}

chrono::duration<double> generate_conditions(
    cublasHandle_t handle,
    float *dA, float *dx, float *dy, int N, int d) {

    curandGenerator_t prng;
    GPU_fill_rand(prng, dA, N * d, 0., 9.);
    GPU_fill_rand(prng, dx, d,     0., 3.);   // environment
    GPU_fill_rand(prng, dy, N,     0., 1.);   // noise

    // intentionally added this useless D2H transfer for the impartial time measurement.
    float *A = new float[N * d], *y = new float[N];
    chrono::system_clock::time_point t0 = now();
    cublasGetMatrix(N, d, sizeof(float), dA, N, A, N);
    cublasGetVector(N, sizeof(float), dy, 1, y, 1);
    chrono::duration<double> dt = now() - t0;
    delete[] A, y;

    float alpha = 1.f, beta = 1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, dA, N,
        dx, 1, &beta, dy, 1
    );
    GPU_fill_rand(prng, dx, d, 0., 3.);       // initial theta
    return dt; 
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
