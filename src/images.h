#pragma once

void print_args(int argc, char** argv);

void print_help();

void copy_image(unsigned char* in_image, unsigned char* out_image, int width, int height);

void box_blur(unsigned char* in_image, unsigned char* out_image, int width, int height, int radius);

