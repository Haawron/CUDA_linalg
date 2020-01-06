#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;


__global__ void func(int *a, int *b, int *c, int N) {
    int i = blockDim.x * gridDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    cout << "hi" << endl;
    
}