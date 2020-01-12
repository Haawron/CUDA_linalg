#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <bitset>
#include <string>

#include <thread>
using namespace std;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define rnd(x) ((x) * rand() / RAND_MAX)
#define rrnd() (rnd(6.f) - (3.f))
#define now() chrono::system_clock::now()

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void func(int id) {
    for (int i = 0; i < 1e8; i++) int x = 1;
    cout << "Hi!! from ";
    printf("%5d\n", id);
}

int main() {
    const int N = 100;
    thread t[N];
    for (int i = 0; i < N; i++) t[i] = thread(func, i);
    for (int i = 0; i < N; i++) t[i].join();
}
