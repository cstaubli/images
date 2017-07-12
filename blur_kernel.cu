#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include "blur_kernel.h"

__global__
void cuda_blur (unsigned char* in_image, unsigned char* out_image, int width, int height, int radius) {
    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset-x) / width;
    int fsize = radius; 
    
    if (offset < width*height) {
        float output_r = 0;
        float output_g = 0;
        float output_b = 0;
        float output_a = 0;
        int hits = 0;
        for (int ox = -fsize; ox < fsize + 1; ++ox) {
            for (int oy = -fsize; oy < fsize + 1; ++oy) {
                if ((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int currentoffset = (offset + ox + oy * width) * 4;
                    output_r += in_image[currentoffset + 0]; 
                    output_g += in_image[currentoffset + 1];
                    output_b += in_image[currentoffset + 2];
                    output_a += in_image[currentoffset + 3];
                    hits++;
                }
            }
        }
        out_image[offset * 4 + 0] = output_r / hits;
        out_image[offset * 4 + 1] = output_g / hits;
        out_image[offset * 4 + 2] = output_b / hits;
        out_image[offset * 4 + 3] = output_a / hits;
    }
}

void print_device_info() {
    int ndevices;
    char star[] = "***************************************\n";
    cudaGetDeviceCount(&ndevices);
    fprintf(stdout, "number of devices: %d\n", ndevices);
    fprintf(stdout, "%s", star);
    for (int i = 0; i < ndevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        fprintf(stdout, "device number: %d\n", i);
        fprintf(stdout, "device name:   %s\n", prop.name);
        fprintf(stdout, "clock rate:    %d Mhz\n", prop.clockRate / 1000);
        fprintf(stdout, "global memory: %lu MB\n", prop.totalGlobalMem / 1024 / 1024);
        fprintf(stdout, "cpu count:     %d\n", prop.multiProcessorCount);
        fprintf(stdout, "bus width:     %d bit\n", prop.memoryBusWidth);
        fprintf(stdout, "max t block:   %d\n", prop.maxThreadsPerBlock);
        fprintf(stdout, "dim x max:     %d\n", prop.maxThreadsDim[0]);
        fprintf(stdout, "dim y max:     %d\n", prop.maxThreadsDim[1]);
        fprintf(stdout, "dim z max:     %d\n", prop.maxThreadsDim[2]);
        fprintf(stdout, "compute cap.:  %d.%d\n", prop.major, prop.minor);
        fprintf(stdout, "asyncEngines:  %d\n", prop.asyncEngineCount);
        fprintf(stdout, "warpSize:      %d\n", prop.warpSize);
        fprintf(stdout, "is unified:    %d\n", prop.unifiedAddressing);
        fprintf(stdout, "%s", star);
    }
}

void cuda_blur_prepare (unsigned char* in_image, unsigned char* out_image, int width, int height, int radius, int testmode) {
    unsigned char* dev_input; // device char array input image
    unsigned char* dev_output; // device char array output image
    int blockSize;
    int minGridSize;
    int gridSize;
    const int N = 1000000;
    cudaEvent_t start;
    cudaEvent_t stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int malsize = width * height * 4 * sizeof(unsigned char);

    fprintf(stdout, "Doing cuda blur...\n");

    cudaError_t mallocstatus = cudaMalloc( (void**) &dev_input, malsize);
    if (mallocstatus != cudaSuccess) {
        fprintf(stderr, "Malloc went wrong: %s\n", cudaGetErrorString(mallocstatus));
    }

    cudaError_t memcpystatus = cudaMemcpy( dev_input, in_image, malsize, cudaMemcpyHostToDevice );
    if (memcpystatus != cudaSuccess) {
        fprintf(stderr, "Memcpy went wrong: %s\n", cudaGetErrorString(memcpystatus));
    }

    cudaError_t mallocoutputstatus = cudaMalloc( (void**) &dev_output, malsize);
    if (mallocoutputstatus != cudaSuccess) {
        fprintf(stderr, "Malloc went wrong: %s\n", cudaGetErrorString(mallocoutputstatus));
    }

    if (testmode > 0) {
        fprintf(stdout, "Test mode detected...\n");
        print_device_info();
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_blur, 0, N);
        gridSize = (N + blockSize - 1) / blockSize;
    } else {
        blockSize = 256;
        gridSize = (unsigned int) ceil( (double)(width * height * 4 / blockSize));
    }
    
    fprintf(stdout, "gridSize: %i\n", gridSize);
    fprintf(stdout, "blockSize: %i\n", blockSize);

    cudaEventRecord(start, 0); 
    cuda_blur<<<gridSize, blockSize>>> (dev_input, dev_output, width, height, radius);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    fprintf(stdout, "Kernel elapsed time: %3.3f ms\n", time);

    cudaError_t copybackstatus = cudaMemcpy(out_image, dev_output, malsize, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);

    cudaDeviceReset();
}

