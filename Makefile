# Making images
#
CC = gcc
NVCC = nvcc
INCLUDES = -I/home/cstaubli/workspace/lib/lodepng
DFLAGS = -g -Wall

default: images

images: lodepng.o images.o blur_kernel.o box_blur.o paint_kernel.o
	$(NVCC) -o images lodepng.o images.o blur_kernel.o box_blur.o paint_kernel.o

lodepng.o: lodepng.h lodepng.cu
	$(NVCC) -c lodepng.cu

images.o: lodepng.h images.h blur_kernel.h box_blur.h lodepng.cu images.cu blur_kernel.cu box_blur.cu
	$(NVCC) -c images.cu 

blur_kernel.o: blur_kernel.h blur_kernel.cu
	$(NVCC) -c blur_kernel.cu 

paint_kernel.o: paint_kernel.h paint_kernel.cu
	$(NVCC) -c paint_kernel.cu

box_blur.o: box_blur.h box_blur.cu
	$(NVCC) -c box_blur.cu

clean:
	rm -f *.o images_debug images

rebuild: clean images

