// This is the testbench for the HoG hardware accelerator. It reads a test image and golden
// reference features from files, then validates that the hardware implementation produces
// correct results matching the golden data.

#include "hog.h"

#include <cmath>
#include <cstdio>

#define TB_BATCH 2

static int check_against_golden(const float *features, const float *golden,
                                const char *label)
{
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
    printf("[%s] Max abs diff: %.8f\n", label, max_diff);
    if (errors != 0)
    {
        printf("[%s] FAILED! %d of %d values exceeded tolerance %.6f\n",
               label, errors, FEATURE_LEN, TOL);
    }
    return errors;
}

int main()
{
    unsigned char image[IMG_H][IMG_W];
    float golden[FEATURE_LEN];

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

    int total_errors = 0;

    {
        static float features[FEATURE_LEN];
        hog_compute(&image[0][0], features, 1);
        total_errors += check_against_golden(features, golden, "single");
    }

    {
        static unsigned char batch_images[TB_BATCH * IMG_PIX];
        static float batch_features[TB_BATCH * FEATURE_LEN];
        for (int n = 0; n < TB_BATCH; n++)
        {
            for (int h = 0; h < IMG_H; h++)
                for (int w = 0; w < IMG_W; w++)
                    batch_images[n * IMG_PIX + h * IMG_W + w] = image[h][w];
        }

        hog_compute(batch_images, batch_features, TB_BATCH);

        total_errors += check_against_golden(&batch_features[0], golden,
                                             "batch img 0");
        total_errors += check_against_golden(&batch_features[FEATURE_LEN],
                                             golden, "batch img 1");
    }

    if (total_errors == 0)
    {
        printf("Test passed!\n");
        return 0;
    }
    printf("Test failed!\n");
    return 1;
}