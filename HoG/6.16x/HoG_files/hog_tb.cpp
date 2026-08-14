// This is the HLS testbench that validates the hog_compute kernel by reading a test image from input.dat,
// running the computation, and comparing the output against a golden reference from out.gold.dat.

#include "hog.h"

#include <cmath>
#include <cstdio>

int main()
{
    unsigned char image[IMG_H][IMG_W];
    float golden[FEATURE_LEN];
    float features[FEATURE_LEN];

    FILE *fin = fopen("input.dat", "r");
    if (!fin)
    {
        printf("ERROR: could not open input.dat\n");
        return 1;
    }
    for (int h = 0; h < IMG_H; h++)
    {
        for (int w = 0; w < IMG_W; w++)
        {
            int v;
            if (fscanf(fin, "%d", &v) != 1)
            {
                printf("ERROR: malformed input.dat\n");
                fclose(fin);
                return 1;
            }
            image[h][w] = (unsigned char)v;
        }
    }
    fclose(fin);

    FILE *fgold = fopen("out.gold.dat", "r");
    if (!fgold)
    {
        printf("ERROR: could not open out.gold.dat\n");
        return 1;
    }
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        if (fscanf(fgold, "%f", &golden[i]) != 1)
        {
            printf("ERROR: malformed out.gold.dat\n");
            fclose(fgold);
            return 1;
        }
    }
    fclose(fgold);

    hog_compute(image, features);

    const float TOL = 1e-4f;
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        float diff = std::fabs(features[i] - golden[i]);
        if (diff > max_diff)
            max_diff = diff;
        if (diff > TOL)
            errors++;
    }

    printf("Max abs diff: %.8f\n", max_diff);
    if (errors == 0)
    {
        printf("Test passed!\n");
        return 0;
    }

    printf("Test failed! %d of %d values exceeded tolerance %.6f\n", errors, FEATURE_LEN, TOL);
    return 1;
}
