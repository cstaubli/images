#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include "paint_kernel.h"

#define C_MAX_INTENSITIES 256

__global__
void cuda_paint (unsigned char* in_image, unsigned char* out_image, int width, int height, int radius, int nBins) {
    // http://supercomputingblog.com/cuda/advanced-image-processing-with-cuda/2/
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // Test to see if we're testing a valid pixel
    if (i >= height || j >= width) return;  // Don't bother doing the calculation. We're not in a valid pixel location
    int intensityCount[C_MAX_INTENSITIES];
    int avgR[C_MAX_INTENSITIES];
    int avgG[C_MAX_INTENSITIES];
    int avgB[C_MAX_INTENSITIES];
    for (int k=0; k <= nBins; k++) {
        intensityCount[k] = 0;
        avgR[k] = 0;
        avgG[k] = 0;
        avgB[k] = 0;
    }
    // we have a radius r
    int maxIntensityCount = 0;
    int maxIntensityCountIndex = 0;
    for (int k=i-radius; k <= i+radius;k++) {
        if (k < 0 || k >= height) continue;
        for (int l=j-radius; l <= j+radius; l++) {
            if (l < 0 || l >= width) continue;
            //int curPixel = in_image[k*stride/4 + l];
            const int currentoffset = (j + k + l * width) * 4;
            int curPixelr = in_image[currentoffset + 0];
            int curPixelg = in_image[currentoffset + 1];
            int curPixelb = in_image[currentoffset + 2];
            int r = ((curPixelr & 0x00ff0000) >> 16);
            int g = ((curPixelg & 0x0000ff00) >> 8);
            int b = ((curPixelb & 0x000000ff) >> 0);
            int curIntensity = (int)((float)((r+g+b)/3*nBins)/255.0f);
            intensityCount[curIntensity]++;
            if (intensityCount[curIntensity] > maxIntensityCount) {
                maxIntensityCount = intensityCount[curIntensity];
                maxIntensityCountIndex = curIntensity;
            }
            avgR[curIntensity] += r;
            avgG[curIntensity] += g;
            avgB[curIntensity] += b;
        }
    }
    int finalR = avgR[maxIntensityCountIndex] / maxIntensityCount;
    int finalG = avgG[maxIntensityCountIndex] / maxIntensityCount;
    int finalB = avgB[maxIntensityCountIndex] / maxIntensityCount;
    out_image[j * 4 + 0] = finalR;
    out_image[j * 4 + 1] = finalG;
    out_image[j * 4 + 2] = finalB;
    out_image[j * 4 + 3] = 255;
}

void cuda_paint_prepare (unsigned char* in_image, unsigned char* out_image, int width, int height, int radius, int nBins) {
    unsigned char* dev_input; // device char array input image
    unsigned char* dev_output; // device char array output image

    int blockSize;
    int gridSize;

    int malsize = width * height * 4 * sizeof(unsigned char);

    fprintf(stdout, "Doing cuda paint...\n");

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

    blockSize = 256;
    gridSize = (unsigned int) ceil( (double)(width * height * 4 / blockSize));

    fprintf(stdout, "gridSize: %i\n", gridSize);
    fprintf(stdout, "blockSize: %i\n", blockSize);

    cuda_paint<<<gridSize, blockSize>>> (dev_input, dev_output, width, height, radius, nBins);

    cudaError_t copybackstatus = cudaMemcpy(out_image, dev_output, malsize, cudaMemcpyDeviceToHost);
    if (copybackstatus != cudaSuccess) {
        fprintf(stderr, "Copy back went wrong: %s\n", cudaGetErrorString(copybackstatus));
    }

    cudaFree(dev_input);
    cudaFree(dev_output);

    cudaDeviceReset();
}

