#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <mpi.h>

#include <iostream>
#include <unistd.h>
using namespace std;

#define CUDACHECK(cmd) do {                             \
    cudaError_t e = cmd;                                \
    if( e != cudaSuccess ) {                            \
        printf("Failed: Cuda error %s:%d '%s'\n",       \
            __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

int main() {
  float *t;
  CUDACHECK(cudaMalloc((void**)&t, ((93000l * 93000l / 4) + ((1. / 4 + 2.) * 93000l)) * sizeof(float)));
  cudaFree(t);
}