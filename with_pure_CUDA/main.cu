#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
using namespace std;

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define DIM 1024

#define uchar unsigned char
#define uint32_t unsigned int
#define uint16_t unsigned short
#define int32_t int

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t file_type;          // File type always BM which is 0x4D42
    uint32_t file_size;               // Size of the file (in bytes)
    uint16_t reserved1;               // Reserved, always 0
    uint16_t reserved2;               // Reserved, always 0
    uint32_t offset_data;            // Start position of pixel data (bytes from the beginning of the file)
    BMPFileHeader() {
        file_type = 0x4D42;
        file_size = 0;
        reserved1 = 0;
        reserved2 = 0;
        offset_data = 54;
    };
};

struct BMPInfoHeader {
     uint32_t size;                      // Size of this header (in bytes)
     int32_t width;                      // width of bitmap in pixels
     int32_t height;                     // width of bitmap in pixels
                                              //       (if positive, bottom-up, with origin in lower left corner)
                                              //       (if negative, top-down, with origin in upper left corner)
     uint16_t planes;                    // No. of planes for the target device, this is always 1
     uint16_t bit_count;                 // No. of bits per pixel
     uint32_t compression;               // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
     uint32_t size_image;                // 0 - for uncompressed images
     int32_t x_pixels_per_meter;
     int32_t y_pixels_per_meter;
     uint32_t colors_used;               // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
     uint32_t colors_important;          // No. of colors used for displaying the bitmap. If 0 all colors are required
     BMPInfoHeader() {
         size = 40;
         width = DIM;
         height = DIM;
         planes = 1;
         bit_count = 24;
         compression = 0;
         size_image = 0;
         x_pixels_per_meter = 0;
         y_pixels_per_meter = 0;
         colors_used = 0;
         colors_important = 0;
     }
 };

struct Header {
    BMPFileHeader fileheader;
    BMPInfoHeader infoheader;
    Header() {
        fileheader.file_size = 54 + 3 * DIM * DIM;
    }
};
#pragma pack(pop)

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

__global__ void kernel(Sphere *, uchar *);

int main() {
    cudaEvent_t start, stop;  // Capture the start time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    uchar *bitmap = new uchar[3 * DIM * DIM];
    uchar *dev_bitmap;
    Sphere *s;

    cudaMalloc((void**)&dev_bitmap, sizeof(uchar) * 3 * DIM * DIM);
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
    
    cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
    delete[] temp_s;

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(s, dev_bitmap);
    cudaMemcpy(bitmap, dev_bitmap, sizeof(uchar) * 3 * DIM * DIM, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time to generate: " << elapsedTime << " s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // draw
    Header header;
    ofstream out;
    out.open("with_pure_CUDA/nonconst.bmp", ios::binary);
    out.write((char*)&header, sizeof(Header));
    out.write((char*)bitmap, 3 * DIM * DIM);

    cudaFree(dev_bitmap);
    cudaFree(s);
    delete[] bitmap;
}

__global__ void kernel(Sphere *s, uchar *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = x - DIM / 2;
    float oy = y - DIM / 2;
    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }
    
    ptr[offset * 3 + 0] = (int)(b * 255);
    ptr[offset * 3 + 1] = (int)(g * 255);
    ptr[offset * 3 + 2] = (int)(r * 255);
}