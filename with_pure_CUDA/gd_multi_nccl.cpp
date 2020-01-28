#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <curand.h>

#include <mpi.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <unistd.h>
#include <stdint.h>
using namespace std;


#define now() chrono::system_clock::now()
#define SEED 960501


#define MPICHECK(cmd) do {                              \
    int e = cmd;                                        \
    if( e != MPI_SUCCESS ) {                            \
        printf("Failed: MPI error %s:%d '%d'\n",        \
            __FILE__,__LINE__, e);                      \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)


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

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}
#define CUBLASCHECK(cmd) do {                           \
    cublasStatus_t s = cmd;                             \
    if (cmd != CUBLAS_STATUS_SUCCESS) {                 \
        printf("Failed, CUBLAS error %s:%d '%s'\n",     \
            __FILE__,__LINE__, _cudaGetErrorEnum(s));   \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)


static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

float cublasNorm(cublasHandle_t handle, int N, float *d_y) {
    float alpha = 1.f;
    float *result, *hresult = new float; cudaMalloc((void**)&result, sizeof(float));
    cublasSgemv(
        handle, CUBLAS_OP_T,
        N, 1, &alpha, d_y, N,
        d_y, 1, &alpha, result, 1
    );
    cublasGetVector(1, sizeof(float), result, 1, hresult, 1);
    return *hresult;
}

float F(cublasHandle_t handle, int N, int d, float *d_X, float *d_theta, float *d_y) {
    float *tmpdy; cudaMalloc((void**)&tmpdy, N * sizeof(float));
    cublasScopy(handle, N, d_y, 1, tmpdy, 1);
    float alpha = 1.f, beta = -1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, d_X, N,
        d_theta, 1, &beta, tmpdy, 1
    );
    // cublasSnrm2 generates an unknown bug. Probably because of the internal data transfer. 
    // CUBLASCHECK(cublasSnrm2(handle, N_gpu, tmpdy, 1, &result));
    float result = cublasNorm(handle, N, tmpdy);
    cudaFree(tmpdy);
    return result;
}

void newtheta(cublasHandle_t handle, float *d_X, float *d_theta, float *d_y, int N, int d, float h, int nRanks) {
    float *tmpdy; cudaMalloc((void**)&tmpdy, N * sizeof(float));
    cublasScopy(handle, N, d_y, 1, tmpdy, 1);

    float alpha = 1.f, beta = -1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, d_X, N,
        d_theta, 1, &beta, tmpdy, 1
    );

    alpha = - 2.f * h, beta = 1.f / nRanks;
    cublasSgemv(
        handle, CUBLAS_OP_T,
        N, d, &alpha, d_X, N,
        tmpdy, 1, &beta, d_theta, 1
    );

    cudaFree(tmpdy);
}

chrono::duration<double> generate_conditions(
    cublasHandle_t handle, int myRank,
    float *d_X, float *d_theta, float *d_y, int N, int d) {

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, SEED);
    curandGenerateNormal(prng, d_theta, d,     0., 3.);   // environment (shared)
    curandSetPseudoRandomGeneratorSeed(prng, SEED + myRank);
    curandGenerateNormal(prng, d_X,     N * d, 0., 9.);   // observation (scattered)
    curandGenerateNormal(prng, d_y,     N,     0., 1.);   // noise (scattered)

    // intentionally added this useless D2H transfer for the impartial time measurement.
    // float *X = new float[N * d], *y = new float[N];
    chrono::system_clock::time_point t0 = now();
    // cublasGetMatrix(N, d, sizeof(float), d_X, N, X, N);
    // cublasGetVector(N, sizeof(float), d_y, 1, y, 1);
    chrono::duration<double> dt = now() - t0;
    // delete[] X, y;

    float alpha = 1.f, beta = 1.f;
    cublasSgemv(
        handle, CUBLAS_OP_N,
        N, d, &alpha, d_X, N,
        d_theta, 1, &beta, d_y, 1
    );

    curandSetPseudoRandomGeneratorSeed(prng, SEED + 2339);
    curandGenerateNormal(prng, d_theta, d, 0., 3.);       // initial theta
    return dt;
}

int main(int argc, char* argv[]) {
    /*
    **  https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/examples.html
    **  Example 2: One Device per Process or Thread
    */

    ////////////////////////////////////////////////////////////
    ///////////////////////// MPI Init /////////////////////////
    int myRank, nRanks, localRank = 0;
    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    // rank of this device in each host
    // localRank == nRanks if there's only 1 host (PC).
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }
    //get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ///////////////////////// MPI Init /////////////////////////
    ////////////////////////////////////////////////////////////

    ncclComm_t comm;
    cudaStream_t nccl_stream;

    //picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaStreamCreate(&nccl_stream));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    cublasHandle_t handle;
    cublasCreate(&handle);

    const size_t N = 92000, d = 92000;
    const size_t N_gpu = N / nRanks;
    const size_t Nd = N * d;
    const size_t Nd_gpu = Nd / nRanks;  // N * d / nGPUs
    float *d_X, *d_y, *d_theta, *d_theta1, done;
    double h, F0, F1, crit = 0., eps = 1e-6;
    int updated, realiter, iter;
    const int niter = 1e4;
    string s;

    chrono::system_clock::time_point    t0, t1;
    chrono::duration<double>            dt, dt1, dtrans;
    t0 = now();

    CUDACHECK(cudaMalloc(&d_X, Nd_gpu * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_y, N_gpu * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_theta, d * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_theta1, d * sizeof(float)));
    // size_t tmpf, tmpt; cudaMemGetInfo(&tmpf, &tmpt);
    // cout << myRank << " Free/Total: " << (tmpf >> 20)  << " / " << (tmpt >> 20) << " MiB" << endl;
    t1 = now();
    
    dtrans = generate_conditions(handle, myRank, d_X, d_theta, d_y, N_gpu, d);
    dt1 = now() - t0 - dtrans;

    h = 1e-2; realiter = 0;
    F0 = F(handle, N_gpu, d, d_X, d_theta, d_y);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &F0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    // cout << myRank << " " << F0 << endl;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (iter = 0; iter < niter; iter+=updated, realiter++) {
        cublasScopy(handle, d, d_theta, 1, d_theta1, 1);
        newtheta(handle, d_X, d_theta1, d_y, N_gpu, d, h, nRanks);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        ncclAllReduce(d_theta1, d_theta1, d, ncclFloat, ncclSum, comm, nccl_stream);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        F1 = F(handle, N_gpu, d, d_X, d_theta1, d_y);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        MPI_Allreduce(MPI_IN_PLACE, &F1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (F1 > F0) {
            updated = 0;
            h /= 2.f;
        } else {

            // logging
            if (iter % 5 == 4 && myRank == 0) {
                done = (iter + 1) / (float)niter;
                s.clear();
                for (int i = 0; i < 30 * done; i++) s += "="; s += ">";
                for (int i = 0; i < 30 * (1. - done); i++) s += " ";
                printf("\r[ Step %5d / %5d | ", iter + 1, niter);
                cout << s << " ]";
                printf(" F0: %10.6f, F1: %10.6f", sqrt(F0 / N), sqrt(F1 / N));
            }
            // MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

            // early stopping
            crit = 1. - F1 / F0;
            if (crit > 0 && crit < eps) break;

            // update
            updated = 1;
            cublasScopy(handle, d, d_theta1, 1, d_theta, 1);
            h *= 1.2f;
            F0 = F1;

        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // free device buffers
    CUDACHECK(cudaFree(d_X));
    CUDACHECK(cudaFree(d_y));
    CUDACHECK(cudaFree(d_theta));
    CUDACHECK(cudaFree(d_theta1));

    dt = now() - t1;
    if (myRank == 0) 
        printf(
            "\n\nN: %6lu, d: %6lu, iterated: %5d, updated: %5d, rmse: %10.5Lf\n",
            N, d, realiter, iter, sqrtl(F0 / N)
        );
    else sleep(1);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    printf(
        "Rank: %d, t_init: %7.3f ms, t_trans: %8.5f s, t: %8.5f s, time/iter: %9.4f ms\n",
        myRank, dt1.count() * 1000, dtrans.count(), dt.count(), dt.count() / realiter * 1000
    );

    cublasDestroy(handle);
    ncclCommDestroy(comm);
    cudaDeviceReset();
    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}
