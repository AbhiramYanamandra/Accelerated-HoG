// This source file is where the HOG pipeline actually runs end-to-end,
// from per-pixel prep all the way to the final feature vector output.
#include "hog.h"
#include <cstdio>
#include <hls_math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Here we turn each pixel into its gamma-scaled value and also estimate one
// shared normalization factor, so later stages can stay fast and consistent.
static void normalize_image(const unsigned char image[IMG_H][IMG_W],
                            float gamma_img[IMG_H][IMG_W],
                            float &inv_stddev_out)
{
  VAR_TYPE sum = 0;
  ACC_TYPE sum_sq = 0;
  for (int h = 0; h < IMG_H; h++)
  {
    for (int w = 0; w < IMG_W; w++)
    {
#pragma HLS pipeline II = 1
      PIX_TYPE x = (PIX_TYPE)image[h][w];
      PIX_TYPE v = x * x * fixed_sqrt<48, 24>(x);
      gamma_img[h][w] = (float)v;
      sum += (VAR_TYPE)v;
      sum_sq += (ACC_TYPE)v * (ACC_TYPE)v;
    }
  }
  VAR_TYPE mean_fx = sum / 1024;

  ACC_TYPE var = sum_sq / 1024 - (ACC_TYPE)mean_fx * (ACC_TYPE)mean_fx;
  if (var < 0)
    var = 0;
  float stddev = (float)fixed_sqrt<72, 40>((VAR_MEAN_TYPE)var) + 1e-6f;
  inv_stddev_out = 1.0f / stddev;
#ifndef __SYNTHESIS__
  printf("DEBUG mean=%f stddev=%f\n", (float)mean_fx, stddev);
#endif
}

// This function here computes gradient direction and strength per interior
// pixel, then drops each direction into one of the 8 orientation bins.
static void compute_gradients(const float gamma_img[IMG_H][IMG_W],
                              float inv_stddev,
                              BIN_T bin_map[IMG_H][IMG_W],
                              float mag[IMG_H][IMG_W])
{
  const float T1 = 0.41421356237309503f;
  const float T3 = 2.41421356237309515f;

BORDER_COL:
  for (int w = 0; w < IMG_W; w++)
  {
#pragma HLS pipeline II = 1
    bin_map[0][w] = 0;
    mag[0][w] = 0.0f;
    bin_map[IMG_H - 1][w] = 0;
    mag[IMG_H - 1][w] = 0.0f;
  }
BORDER_ROW:
  for (int h = 1; h < IMG_H - 1; h++)
  {
#pragma HLS pipeline II = 1
    bin_map[h][0] = 0;
    mag[h][0] = 0.0f;
    bin_map[h][IMG_W - 1] = 0;
    mag[h][IMG_W - 1] = 0.0f;
  }

GRAD_ROW:
  for (int h = 1; h < IMG_H - 1; h++)
  {
  GRAD_COL:
    for (int w = 1; w < IMG_W - 1; w++)
    {
#pragma HLS pipeline II = 1
      float dy = (gamma_img[h + 1][w] - gamma_img[h - 1][w]) * inv_stddev;
      float dx = (gamma_img[h][w + 1] - gamma_img[h][w - 1]) * inv_stddev + 0.0001f;

      bool dx_neg = (dx < 0.0f);
      float ax = dx_neg ? -dx : dx;
      float ay = dx_neg ? -dy : dy;

      float e1 = T1 * ax;
      float e2 = ax;
      float e3 = T3 * ax;

      bool c[7];
#pragma HLS array_partition variable = c complete
      c[0] = (ay >= -e3);
      c[1] = (ay >= -e2);
      c[2] = (ay >= -e1);
      c[3] = (ay >= 0.0f);
      c[4] = (ay >= e1);
      c[5] = (ay >= e2);
      c[6] = (ay >= e3);

      ap_uint<3> k = 0;
    COUNT:
      for (int i = 0; i < 7; i++)
      {
#pragma HLS unroll
        k += (ap_uint<3>)(c[i] ? 1 : 0);
      }
      BIN_T bin = (BIN_T)((k + 4) & 7);

      if (ax == 0.0f && ay > 0.0f)
        bin = 4;

      bin_map[h][w] = bin;
      mag[h][w] = hls::sqrt(dx * dx + dy * dy);
    }
  }
}

// Here we collect a small 8x8 cell and accumulate its weighted orientation
// votes, so each cell becomes a compact 8-bin histogram summary.
static void cell_histogram(const BIN_T bin_map[IMG_H][IMG_W],
                           const float mag[IMG_H][IMG_W], int h0, int w0,
                           float hist[NUM_BINS])
{
#pragma HLS inline
  HIST_T local_hist[NUM_BINS];
#pragma HLS array_partition variable = local_hist complete
INIT:
  for (int b = 0; b < NUM_BINS; b++)
  {
#pragma HLS unroll
    local_hist[b] = 0;
  }

ROW:
  for (int h = 0; h < CELL_SIZE; h++)
  {
  COL:
    for (int w = 0; w < CELL_SIZE; w++)
    {
#pragma HLS pipeline II = 1
      BIN_T bin = bin_map[h0 + h][w0 + w];
      HIST_T weight = (HIST_T)mag[h0 + h][w0 + w];

      for (int b = 0; b < NUM_BINS; b++)
      {
#pragma HLS unroll
        if (bin == b)
          local_hist[b] += weight;
      }
    }
  }

OUT:
  for (int b = 0; b < NUM_BINS; b++)
  {
#pragma HLS unroll
    hist[b] = (float)local_hist[b];
  }
}

// Here we have done the per-image orchestration: gradients, cell histograms,
// and block normalization to build one complete HOG descriptor.
static void hog_single(const unsigned char image[IMG_H][IMG_W],
                       float features[FEATURE_LEN])
{
#pragma HLS inline off
#ifndef __SYNTHESIS__
  printf("DEBUG BUILD MARKER: v5\n");
#endif

  static float gamma_img[IMG_H][IMG_W];
  static BIN_T bin_map[IMG_H][IMG_W];
  static float mag[IMG_H][IMG_W];
  static float cells[CELLS_H][CELLS_W][NUM_BINS];
#pragma HLS array_partition variable = bin_map block factor = 4 dim = 2
#pragma HLS array_partition variable = mag block factor = 4 dim = 2
#pragma HLS array_partition variable = cells complete dim = 2

  float inv_stddev;
  normalize_image(image, gamma_img, inv_stddev);
  compute_gradients(gamma_img, inv_stddev, bin_map, mag);

  for (int ci = 0; ci < CELLS_H; ci++)
  {
#pragma HLS pipeline off
    for (int cj = 0; cj < CELLS_W; cj++)
    {
#pragma HLS unroll
      cell_histogram(bin_map, mag, ci * CELL_SIZE, cj * CELL_SIZE, cells[ci][cj]);
    }
  }

  int out_idx = 0;
  for (int bi = 0; bi < BLOCKS_H; bi++)
  {
#pragma HLS pipeline off
    for (int bj = 0; bj < BLOCKS_W; bj++)
    {
      VAR_TYPE norm_sq = 0;
      for (int ci = 0; ci < BLOCK_CELLS; ci++)
        for (int cj = 0; cj < BLOCK_CELLS; cj++)
          for (int b = 0; b < NUM_BINS; b++)
          {
#pragma HLS pipeline II = 1
            VAR_TYPE bv = (VAR_TYPE)cells[bi + ci][bj + cj][b];
            norm_sq += bv * bv;
          }

      float block_norm = (float)fixed_sqrt<64, 32>(norm_sq) + 1e-6f;
#ifndef __SYNTHESIS__
      printf("DEBUG bi=%d bj=%d norm_sq=%f block_norm=%f\n", bi, bj,
             (float)norm_sq, block_norm);
#endif
      float inv_norm = 1.0f / block_norm;
      for (int ci = 0; ci < BLOCK_CELLS; ci++)
        for (int cj = 0; cj < BLOCK_CELLS; cj++)
          for (int b = 0; b < NUM_BINS; b++)
          {
#pragma HLS pipeline II = 1
            features[out_idx++] = cells[bi + ci][bj + cj][b] * inv_norm;
          }
    }
  }
}

// This function here is the batched entry point, so one call can process
// many images and reduce launch overhead while reusing the same pipeline.
void hog_compute(const unsigned char *images, float *features, int num_images)
{
#pragma HLS interface m_axi port = images offset = slave bundle = gmem0 depth = 2048
#pragma HLS interface m_axi port = features offset = slave bundle = gmem1 depth = 576
#pragma HLS interface s_axilite port = num_images
#pragma HLS interface s_axilite port = return

  unsigned char img_local[IMG_H][IMG_W];
#pragma HLS array_partition variable = img_local cyclic factor = 16 dim = 2
  float feat_local[FEATURE_LEN];
#pragma HLS array_partition variable = feat_local cyclic factor = 16 dim = 1

BATCH:
  for (int n = 0; n < num_images; n++)
  {
#pragma HLS loop_tripcount min = 1 max = 256 avg = 256
  LOAD_IMG:
    for (int i = 0; i < IMG_PIX; i += 16)
    {
#pragma HLS pipeline II = 1
      for (int j = 0; j < 16; j++)
      {
#pragma HLS unroll
        int idx = i + j;
        img_local[idx / IMG_W][idx % IMG_W] = images[n * IMG_PIX + idx];
      }
    }

    hog_single(img_local, feat_local);

  STORE_FEAT:
    for (int i = 0; i < FEATURE_LEN; i += 16)
    {
#pragma HLS pipeline II = 1
      for (int j = 0; j < 16; j++)
      {
#pragma HLS unroll
        features[n * FEATURE_LEN + i + j] = feat_local[i + j];
      }
    }
  }
}
