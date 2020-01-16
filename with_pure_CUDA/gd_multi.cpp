#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <nccl.h>

#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <functional>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define now() chrono::system_clock::now()
#define SEED 960501

#define CUDACHECK(cmd) do {                             \
    cudaError_t e = cmd;                                \
    if( e != cudaSuccess ) {                            \
        printf("Failed: Cuda error %s:%d '%s'\n",       \
            __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)


#define NCCLCHECK(cmd) do {                             \
    ncclResult_t r = cmd;                               \
    if (r!= ncclSuccess) {                              \
        printf("Failed, NCCL error %s:%d '%s'\n",       \
            __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)


// global variable init
const int nGPU = 4;
const uint64_t N = 12, d = 4;
const uint64_t Nd = N * d;
const uint64_t NdGPU = Nd / nGPU;
float *h_X, *h_y, *h_theta;
float *d_X[nGPU];
float *d_y[nGPU];
float *d_theta[nGPU];
float F0[nGPU], F1[nGPU];
ncclComm_t *comms = new ncclComm_t[nGPU];
cudaStream_t *nccl_streams = new cudaStream_t[nGPU];
cudaStream_t *blas_streams = new cudaStream_t[nGPU];


void F(cublasHandle_t handle, float *dA, float *dx, float *dy, int N, int d, float &result) {
    float *tmpdy; cudaMalloc((void**)&tmpdy, N * sizeof(*dy));
    cublasScopy(handle, N, dy, 1, tmpdy, 1);
    
    float alpha = 1.f, beta = -1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N / 4, d, &alpha, dA, N / 4,
        dx, 1, &beta, tmpdy, 1
    );
    cublasSnrm2(handle, N, tmpdy, 1, &result);
    cudaFree(tmpdy);
}

void worker(int rank) {
    
    cudaSetDevice(rank);

    cublasHandle_t handle;
    auto& blas_stream = blas_streams[rank];
    cublasCreate(&handle);
    cublasSetStream(handle, blas_stream);

    auto& nccl_stream = nccl_streams[rank];
    
    F(handle, d_X[rank], d_theta[rank], d_y[rank], N, d, F0[rank]);

    // sum F0's
    /////////////////////////////////////////////////////////////////////////////////////
    // how????????????????????????????????????? to do inter-thread sum??
    // -> use MPI...
}

int main(int argc, char* argv[]) {
    
    // nccl init
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    ncclGroupStart();
    for (int rank = 0; rank < nGPU; rank++) {
        cudaSetDevice(rank);
        ncclCommInitRank(&comms[rank], nGPU, nccl_id, rank);  // New communicators
        cudaStreamCreate(&nccl_streams[rank]);
        cudaStreamCreate(&blas_streams[rank]);
    }
    ncclGroupEnd();

    // data init
    for (int rank = 0; rank < nGPU; rank++) {
        // host alloc
        h_X = new float[NdGPU];
        h_y = new float[N];
        h_theta = new float[d];
        // host init
        for (int i = 0; i < NdGPU; i++) h_X[i] = 1;
        for (int i = 0; i < N; i++) h_y[i] = 0;
        for (int j = 0; j < d; j++) h_theta[j] = 1;
        // cuda alloc
        cudaSetDevice(rank);
        cudaMalloc((void**)&d_X[rank], NdGPU * sizeof(float));
        cudaMalloc((void**)&d_y[rank], N * sizeof(float));
        cudaMalloc((void**)&d_theta[rank], d * sizeof(float));
        // cublas init
        cublasSetVector(NdGPU, sizeof(float), h_X, 1, d_X[rank], 1);
        cublasSetVector(N, sizeof(float), h_y, 1, d_y[rank], 1);
        cublasSetVector(d, sizeof(float), h_theta, 1, d_theta[rank], 1);
        // host free
        delete[] h_X, h_y, h_theta;
    }

    vector<thread> threads;
    for (int rank = 0; rank < nGPU; rank++) {
        thread t(bind(&worker, rank));
        threads.push_back(move(t));
    }
    for (auto& t : threads) t.join();

    float sum = 0;
    for (int rank = 0; rank < nGPU; rank++) sum += F0[rank] * F0[rank];
    cout << sum << endl;

    delete[] comms, nccl_streams, blas_streams;
}
