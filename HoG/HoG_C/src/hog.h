#ifndef HOG_H
#define HOG_H

#define IMG_H 32
#define IMG_W 32

#define CELL_SIZE   8
#define BLOCK_CELLS 2
#define NUM_BINS    8

#define CELLS_H (IMG_H / CELL_SIZE)             // 4
#define CELLS_W (IMG_W / CELL_SIZE)              // 4
#define BLOCKS_H (CELLS_H - BLOCK_CELLS + 1)     // 3
#define BLOCKS_W (CELLS_W - BLOCK_CELLS + 1)     // 3

#define FEATURE_LEN (BLOCKS_H * BLOCKS_W * BLOCK_CELLS * BLOCK_CELLS * NUM_BINS) // 288

// Top-level HoG kernel: 32x32 8-bit grayscale image in, 288-length feature
// vector out. Written HLS-synthesizable: fixed-size arrays only, no dynamic
// allocation, no STL containers.
void hog_compute(const unsigned char image[IMG_H][IMG_W], float features[FEATURE_LEN]);

#endif // HOG_H
