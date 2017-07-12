#include "lodepng.h"
#include "images.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include "blur_kernel.h"
#include "paint_kernel.h"
#include "box_blur.h"

void print_args (int argc, char** argv) {
    int i;
    for (i = 0; i<argc; ++i) {
        fprintf(stdout, "Argument %i has value %s\n", i, argv[i]);
    }
}

void print_help() {
    fprintf(stdout, "Usage:\n");
    fprintf(stdout, "-i input png file\n");
    fprintf(stdout, "-o output png file\n");
    fprintf(stdout, "-r blur radius in pixels\n");
    fprintf(stdout, "-t switches to testmode (Optional)\n");
    fprintf(stdout, "Example: images -i test.png -o output.png -r 10\n");
}

int main (int argc, char** argv) {
   
    char* ivalue = NULL;
    char* ovalue = NULL;
    char* rvalue = NULL;
    int testmode = 0;
    int radius = 0;
    int c;

    if (argc < 7) {
        print_help();
        return EXIT_FAILURE;
    }

    while ((c = getopt (argc, argv, "i:o:r:t")) != -1)
        switch(c)
        {
            case 'i':
                ivalue = optarg;
                break;
            case 'o':
                ovalue = optarg;
                break;
            case 'r':
                rvalue = optarg;
                break;
            case 't':
                testmode = 1;
                break;
            case '?':
                if (optopt == 'i' || optopt == 'o' || optopt == 'r')
                    fprintf(stderr, "Option -%c requires an argument\n", optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE; 
        }


    // print_args(argc, argv);

    const char* input_file = ivalue; 
    const char* output_file = ovalue; 
    radius = atoi(rvalue);

    unsigned int error;
    unsigned char* in_image;
    unsigned int width, height;

    error = lodepng_decode32_file(&in_image, &width, &height, input_file);

    if (error) {
        fprintf(stderr, "Error Code %i: %s\n", error, lodepng_error_text(error));
        return EXIT_FAILURE;
    }

    fprintf(stdout, "Image %s has width: %i and height: %i\n", input_file, width, height);

    unsigned char* out_image = (unsigned char*) malloc(width * height * 4);

    // box_blur(in_image, out_image, width, height, radius);
   
    // copy_image(in_image, out_image, width, height);

    // cuda_blur_prepare(in_image, out_image, width, height, radius, testmode);

    cuda_paint_prepare(in_image, out_image, width, height, radius, 4);

    unsigned int out_error;
    out_error = lodepng_encode32_file(output_file, out_image, width, height);
    if (out_error) {
        fprintf(stderr, "Error Code %i: %s\n", error, lodepng_error_text(out_error));
        return EXIT_FAILURE;
    }

    free(in_image);
    return EXIT_SUCCESS;
}

