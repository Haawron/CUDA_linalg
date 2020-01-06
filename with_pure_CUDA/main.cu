#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define DIM 960

struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;
    
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

Sphere *s;

int main() {
    cudaEvent_t start, stop;  // Capture the start time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    using uchar = unsigned char;
    uchar *bitmap = new uchar[DIM * DIM];
    // header, color
    uchar *dev_bitmap;

    cudaMalloc((void**)&dev_bitmap, sizeof(uchar) * DIM * DIM);
    cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);
    
    Sphere *temp_s = new Sphere[SPHERES];
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.f);
        temp_s[i].g = rnd(1.f);
        temp_s[i].b = rnd(1.f);
        temp_s[i].x = rnd(1000.f);
        temp_s[i].y = rnd(1000.f);
        temp_s[i].z = rnd(1000.f);
        temp_s[i].radius = rnd(100.f) + 20;
    }
    
    cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERE, cudaMemcpyHostToDevice);
    delete[] temp_s;

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);
    cudaMemcpy(bitmap, dev_bitmap, sizeof(uchar) * DIM * DIM, cudaMemcpyDeviceToHost);
    
    // draw

    cudaFree(dev_bitmap);
    cudaFree(s);
}