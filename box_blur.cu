#include <stdio.h>
#include <stdlib.h>
#include <cmath>

void box_blur(unsigned char* in_image, unsigned char* out_image, int width, int height, int radius) {
    const int fsize = radius; 
    int ox, oy, x, y;

    fprintf(stdout, "Doing box blur\n");

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            int index_r = 4 * width * y + 4 * x + 0;
            int index_g = 4 * width * y + 4 * x + 1;
            int index_b = 4 * width * y + 4 * x + 2;
            int index_a = 4 * width * y + 4 * x + 3;

            float output_red = 0;
            float output_green = 0;
            float output_blue = 0;
            float output_alpha = 0;
            int hits = 0;

            for (ox = -fsize; ox < fsize+1; ++ox) {
                for (oy = -fsize; oy  < fsize+1; ++oy) {
                    if (x+ox > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                        //int currentoffset = (x + y + ox + oy)*4;
                        int currentoffset = 4 * width * (y+oy) + 4 * (x+ox);
                        output_red    += in_image[currentoffset + 0];
                        output_green  += in_image[currentoffset + 1];
                        output_blue   += in_image[currentoffset + 2];
                        output_alpha  += in_image[currentoffset + 3];
                        ++hits;
                    }
                }
            }
            out_image[index_r] = output_red / hits;
            out_image[index_g] = output_green / hits;
            out_image[index_b] = output_blue / hits;
            out_image[index_a] = output_alpha / hits;
        }
    }
}

