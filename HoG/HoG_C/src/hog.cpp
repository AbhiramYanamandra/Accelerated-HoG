#include "hog.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gamma correction (x^2.5) followed by local contrast normalization
// ((x - mean) / stddev), matching the reference Python implementation.
static void normalize_image(const unsigned char image[IMG_H][IMG_W], double norm[IMG_H][IMG_W]) {
    double sum = 0.0;
    for (int h = 0; h < IMG_H; h++) {
        for (int w = 0; w < IMG_W; w++) {
            double v = std::pow((double)image[h][w], 2.5);
            norm[h][w] = v;
            sum += v;
        }
    }
    double mean = sum / (IMG_H * IMG_W);

    double var_sum = 0.0;
    for (int h = 0; h < IMG_H; h++) {
        for (int w = 0; w < IMG_W; w++) {
            double d = norm[h][w] - mean;
            var_sum += d * d;
        }
    }
    double stddev = std::sqrt(var_sum / (IMG_H * IMG_W));

    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
            norm[h][w] = (norm[h][w] - mean) / stddev;
}

// [-1 0 1] gradient kernel. Border pixels (row/col 0 and IMG_H-1/IMG_W-1)
// are left at 0, matching the reference implementation's boundary handling.
static void compute_gradients(const double norm[IMG_H][IMG_W],
                               double grad[IMG_H][IMG_W],
                               double mag[IMG_H][IMG_W]) {
    for (int h = 0; h < IMG_H; h++) {
        for (int w = 0; w < IMG_W; w++) {
            grad[h][w] = 0.0;
            mag[h][w] = 0.0;
        }
    }

    for (int h = 1; h < IMG_H - 1; h++) {
        for (int w = 1; w < IMG_W - 1; w++) {
            double dy = norm[h + 1][w] - norm[h - 1][w];
            double dx = norm[h][w + 1] - norm[h][w - 1] + 0.0001;

            double g = std::atan(dy / dx) * (180.0 / M_PI);
            if (g < 0.0) g += 180.0;

            grad[h][w] = g;
            mag[h][w] = std::sqrt(dx * dx + dy * dy);
        }
    }
}

// 8-bin histogram of grad[h0:h0+CELL_SIZE][w0:w0+CELL_SIZE] over [0,180),
// weighted by the corresponding magnitude.
static void cell_histogram(const double grad[IMG_H][IMG_W], const double mag[IMG_H][IMG_W],
                            int h0, int w0, double hist[NUM_BINS]) {
    const double bin_width = 180.0 / NUM_BINS;

    for (int b = 0; b < NUM_BINS; b++) hist[b] = 0.0;

    for (int h = 0; h < CELL_SIZE; h++) {
        for (int w = 0; w < CELL_SIZE; w++) {
            double angle = grad[h0 + h][w0 + w];
            double weight = mag[h0 + h][w0 + w];

            int bin = (int)(angle / bin_width);
            if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            if (bin < 0) bin = 0;

            hist[bin] += weight;
        }
    }
}

void hog_compute(const unsigned char image[IMG_H][IMG_W], float features[FEATURE_LEN]) {
    static double norm[IMG_H][IMG_W];
    static double grad[IMG_H][IMG_W];
    static double mag[IMG_H][IMG_W];
    static double cells[CELLS_H][CELLS_W][NUM_BINS];

    normalize_image(image, norm);
    compute_gradients(norm, grad, mag);

    for (int ci = 0; ci < CELLS_H; ci++)
        for (int cj = 0; cj < CELLS_W; cj++)
            cell_histogram(grad, mag, ci * CELL_SIZE, cj * CELL_SIZE, cells[ci][cj]);

    const int BLOCK_LEN = BLOCK_CELLS * BLOCK_CELLS * NUM_BINS; // 32

    int out_idx = 0;
    for (int bi = 0; bi < BLOCKS_H; bi++) {
        for (int bj = 0; bj < BLOCKS_W; bj++) {
            double block[BLOCK_CELLS * BLOCK_CELLS * NUM_BINS];
            int k = 0;
            for (int ci = 0; ci < BLOCK_CELLS; ci++)
                for (int cj = 0; cj < BLOCK_CELLS; cj++)
                    for (int b = 0; b < NUM_BINS; b++)
                        block[k++] = cells[bi + ci][bj + cj][b];

            double norm_sq = 0.0;
            for (int k2 = 0; k2 < BLOCK_LEN; k2++)
                norm_sq += block[k2] * block[k2];
            double block_norm = std::sqrt(norm_sq);

            for (int k2 = 0; k2 < BLOCK_LEN; k2++)
                features[out_idx++] = (float)(block[k2] / block_norm);
        }
    }
}
