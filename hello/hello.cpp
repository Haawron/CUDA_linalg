#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <random>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))


void func(float *da, int N) {
    
}

int main() {
    const int N = 10;
    float *a = new float[N], *da = new float[N];
    for (int i = 0; i < N; i++) a[i] = i * i;
    cudaMalloc((void**)&da, N * sizeof(float));
    cudaMemcpy(da, a, N * sizeof(*a), cudaMemcpyHostToDevice);
    func(da, N);
    cudaMemcpy(a, da, N * sizeof(*a), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) cout << a[i] << " ";
    cout << endl;
    cudaFree(da);
    delete[] a;
}
