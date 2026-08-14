// This file defines the shared HoG constants, image sizing, and function interface used across the project.
#include <ap_fixed.h>

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

#define FEATURE_LEN (BLOCKS_H * BLOCKS_W * BLOCK_CELLS * BLOCK_CELLS * NUM_BINS)

void hog_compute(const unsigned char image[IMG_H][IMG_W], float features[FEATURE_LEN]);

#endif
