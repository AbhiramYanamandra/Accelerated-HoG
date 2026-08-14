// This file holds the main HoG image-processing pipeline, where a 32x32 grayscale image is turned into a histogram-based feature descriptor.
#include "hog.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Here we clean and normalize the image so the later gradient work is stable and the feature extraction behaves consistently.
static void normalize_image(const unsigned char image[IMG_H][IMG_W], float norm[IMG_H][IMG_W])
{
#pragma HLS pipeline off
    float sum = 0.0f;
    for (int h = 0; h < IMG_H; h++)
    {
#pragma HLS pipeline off
        for (int w = 0; w < IMG_W; w++)
        {
#pragma HLS pipeline off
            float x = (float)image[h][w];
            float v = x * x * std::sqrt(x);
            norm[h][w] = v;
            sum += v;
        }
    }
    float mean = sum / 1024;

    float var_sum = 0.0f;
    for (int h = 0; h < IMG_H; h++)
    {
#pragma HLS pipeline off
        for (int w = 0; w < IMG_W; w++)
        {
#pragma HLS pipeline off
            float d = norm[h][w] - mean;
            var_sum += d * d;
        }
    }
    float stddev = std::sqrt(var_sum / (IMG_H * IMG_W));

    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
            norm[h][w] = (norm[h][w] - mean) / stddev;
}

// This function here estimates the local gradient directions and magnitudes, which are the key ingredients for the HoG descriptor.
static void compute_gradients(const float norm[IMG_H][IMG_W],
                              float grad[IMG_H][IMG_W],
                              float mag[IMG_H][IMG_W])
{
    for (int h = 0; h < IMG_H; h++)
    {
        for (int w = 0; w < IMG_W; w++)
        {
            grad[h][w] = 0.0f;
            mag[h][w] = 0.0f;
        }
    }

    for (int h = 1; h < IMG_H - 1; h++)
    {
        for (int w = 1; w < IMG_W - 1; w++)
        {
            float dy = norm[h + 1][w] - norm[h - 1][w];
            float dx = norm[h][w + 1] - norm[h][w - 1] + 0.0001f;

            float g = std::atan(dy / dx) * (180.0f / (float)M_PI);
            if (g < 0.0f)
                g += 180.0f;

            grad[h][w] = g;
            mag[h][w] = std::sqrt(dx * dx + dy * dy);
        }
    }
}

// Here we build the histogram for one image cell, grouping nearby gradient angles into the eight bins used by HoG.
static void cell_histogram(const float grad[IMG_H][IMG_W], const float mag[IMG_H][IMG_W],
                           int h0, int w0, float hist[NUM_BINS])
{
    const float bin_width = 180.0f / NUM_BINS;

    for (int b = 0; b < NUM_BINS; b++)
        hist[b] = 0.0f;

    for (int h = 0; h < CELL_SIZE; h++)
    {
        for (int w = 0; w < CELL_SIZE; w++)
        {
#pragma HLS pipeline off
            float angle = grad[h0 + h][w0 + w];
            float weight = mag[h0 + h][w0 + w];

            int bin = (int)(angle / bin_width);
            if (bin >= NUM_BINS)
                bin = NUM_BINS - 1;
            if (bin < 0)
                bin = 0;

            hist[bin] += weight;
        }
    }
}

// This function here brings the whole pipeline together, from normalization to block normalization, so the final output is a usable HoG feature vector.
void hog_compute(const unsigned char image[IMG_H][IMG_W], float features[FEATURE_LEN])
{
    static float norm[IMG_H][IMG_W];
    static float grad[IMG_H][IMG_W];
    static float mag[IMG_H][IMG_W];
    static float cells[CELLS_H][CELLS_W][NUM_BINS];

    normalize_image(image, norm);
    compute_gradients(norm, grad, mag);

    for (int ci = 0; ci < CELLS_H; ci++)
#pragma HLS pipeline off
        for (int cj = 0; cj < CELLS_W; cj++)
            cell_histogram(grad, mag, ci * CELL_SIZE, cj * CELL_SIZE, cells[ci][cj]);

    const int BLOCK_LEN = BLOCK_CELLS * BLOCK_CELLS * NUM_BINS;

    int out_idx = 0;
    for (int bi = 0; bi < BLOCKS_H; bi++)
    {
#pragma HLS pipeline off
        for (int bj = 0; bj < BLOCKS_W; bj++)
        {
            float block[BLOCK_CELLS * BLOCK_CELLS * NUM_BINS];
            int k = 0;
            for (int ci = 0; ci < BLOCK_CELLS; ci++)
                for (int cj = 0; cj < BLOCK_CELLS; cj++)
                    for (int b = 0; b < NUM_BINS; b++)
                        block[k++] = cells[bi + ci][bj + cj][b];

            float norm_sq = 0.0f;
            for (int k2 = 0; k2 < BLOCK_LEN; k2++)
                norm_sq += block[k2] * block[k2];
            float block_norm = std::sqrt(norm_sq);

            for (int k2 = 0; k2 < BLOCK_LEN; k2++)
                features[out_idx++] = block[k2] / block_norm;
        }
    }
}
