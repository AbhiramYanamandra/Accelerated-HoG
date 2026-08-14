// This header file defines the core data structures and constants for our HOG (Histogram of Oriented Gradients) implementation.
// It also declares the main hog_compute function that we'll be optimizing with fixed-point arithmetic and HLS pragmas.

#include <ap_fixed.h>
#include <ap_int.h>

#ifndef HOG_H
#define HOG_H

#define IMG_H 32
#define IMG_W 32

#define CELL_SIZE 8
#define BLOCK_CELLS 2
#define NUM_BINS 8

#define CELLS_H (IMG_H / CELL_SIZE)
#define CELLS_W (IMG_W / CELL_SIZE)
#define BLOCKS_H (CELLS_H - BLOCK_CELLS + 1)
#define BLOCKS_W (CELLS_W - BLOCK_CELLS + 1)

typedef ap_fixed<64, 32> VAR_TYPE;
typedef ap_fixed<96, 64> ACC_TYPE;
typedef ap_fixed<72, 40> VAR_MEAN_TYPE;
typedef ap_fixed<48, 24> PIX_TYPE;
typedef ap_fixed<32, 16> HIST_T;
typedef ap_uint<3> BIN_T;

template <int W, int I>
ap_fixed<W, I> fixed_sqrt(ap_fixed<W, I> x);

#define FEATURE_LEN (BLOCKS_H * BLOCKS_W * BLOCK_CELLS * BLOCK_CELLS * NUM_BINS)

void hog_compute(const unsigned char image[IMG_H][IMG_W], float features[FEATURE_LEN]);

#endif