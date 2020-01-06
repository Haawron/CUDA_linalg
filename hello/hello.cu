#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
using namespace std;


__global__ void func(int *a, int *b, int *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 100;
    cout << "hi" << endl;
    int *a = new int[N], *b = new int[N], *c = new int[N];  // alloc c?
    int *dev_a, *dev_b, *dev_c;

    for (int i = 0; i < N; i++) {
        a[i] = 2 * i;
        b[i] = -3 * i + 43;
    }
    cudaMalloc((void**)&dev_a, sizeof(int) * N);
    cudaMalloc((void**)&dev_b, sizeof(int) * N);
    cudaMalloc((void**)&dev_c, sizeof(int) * N);
    cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
    func<<<(N + 15)/16, 16>>>(dev_a, dev_b, dev_c, N);
    cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        cout << setw(3) << c[i];
        if (i % 10 == 9) cout << endl;
    }
    cout << endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    delete[] a, b, c;
}