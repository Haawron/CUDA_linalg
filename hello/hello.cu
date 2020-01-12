#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
using namespace std;

__global__
void Sadds(float *dx, const float scalar, int N) {
    // Single (precision) add scalar
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(; tidx < N; tidx += stride) dx[tidx] += scalar;
}

int main() {
    
    const int N = 10;
    float *x = new float[N], *dx;
    cudaMalloc((void**)&dx, N * sizeof(float));

    Sadds<<<(N + 15)/16, 16>>>(dx, 1, N);
    cublasGetVector(N, sizeof(float), dx, 1, x, 1);
    for (int i = 0; i < N; i++) cout << x[i] << " ";
    cout << endl;

    cudaFree(dx);
    delete[] x;

}