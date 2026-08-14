// This is the main HoG accelerator kernel that processes batches of images. It implements the complete
// histogram of oriented gradients pipeline, optimized for hardware execution with dataflow parallelism,
// pipelining, and custom fixed-point arithmetic to maximize throughput on FPGAs

#include "hog.h"
#include <cstdio>
#include <hls_math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Here we normalize the image with gamma correction and compute the inverse standard deviation
// This optimized version uses dual-lane accumulators to avoid recurrence and compute the stats in a single pass.
static void normalize_image(const unsigned char image[IMG_H][IMG_W],
                            float gamma_img[IMG_H][IMG_W],
                            float &inv_stddev_out)
{
  VAR_TYPE sum_lane[2];
  ACC_TYPE sum_sq_lane[2];
  #pragma HLS array_partition variable = sum_lane complete
  #pragma HLS array_partition variable = sum_sq_lane complete
  sum_lane[0] = 0;
  sum_lane[1] = 0;
  sum_sq_lane[0] = 0;
  sum_sq_lane[1] = 0;

  for (int h = 0; h < IMG_H; h++)
  {
    for (int w = 0; w < IMG_W; w += 2)
    {
      #pragma HLS pipeline II = 1
      for (int j = 0; j < 2; j++)
      {
        #pragma HLS unroll
        PIX_TYPE x = (PIX_TYPE)image[h][w + j];
        PIX_TYPE v = x * x * fixed_sqrt<48, 24>(x);
        gamma_img[h][w + j] = (float)v;
        sum_lane[j] += (VAR_TYPE)v;
        sum_sq_lane[j] += (ACC_TYPE)v * (ACC_TYPE)v;
      }
    }
  }
  VAR_TYPE sum = sum_lane[0] + sum_lane[1];
  ACC_TYPE sum_sq = sum_sq_lane[0] + sum_sq_lane[1];
  VAR_TYPE mean_fx = sum / 1024;

  ACC_TYPE var = sum_sq / 1024 - (ACC_TYPE)mean_fx * (ACC_TYPE)mean_fx;
  if (var < 0)
    var = 0;
  float stddev = (float)fixed_sqrt<72, 40>((VAR_MEAN_TYPE)var) + 1e-6f;
  inv_stddev_out = 1.0f / stddev;
}

// This function computes gradients and classifies them into orientation bins using efficient comparison trees
// We extract both magnitude and bin index for each pixel, which are then accumulated into histograms
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
    for (int w = 1; w < IMG_W - 1; w += 2)
    {
      #pragma HLS pipeline II = 1
      for (int j = 0; j < 2; j++)
      {
        #pragma HLS unroll
        int wc = w + j;
        float dy = (gamma_img[h + 1][wc] - gamma_img[h - 1][wc]) * inv_stddev;
        float dx =
            (gamma_img[h][wc + 1] - gamma_img[h][wc - 1]) * inv_stddev + 0.0001f;

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

        bin_map[h][wc] = bin;
        mag[h][wc] = hls::sqrt(dx * dx + dy * dy);
      }
    }
  }
}

// Here we compute a single cell histogram by accumulating weighted magnitudes into 8 orientation bins.
// This function is inlined and unrolled to enable massive parallelism across all cells in the image.
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

// This stage loads an image from external memory into local RAM using burst-aligned transfers.
// Its the first step in our dataflow pipeline and feeds data to the normalize stage.
static void load_image_stage(const unsigned char *images, int n,
                             unsigned char img[IMG_H][IMG_W])
{
LOAD_IMG:
  for (int i = 0; i < IMG_PIX; i += 16)
  {
    #pragma HLS pipeline II = 1
    for (int j = 0; j < 16; j++)
    {
      #pragma HLS unroll
      int idx = i + j;
      img[idx / IMG_W][idx % IMG_W] = images[n * IMG_PIX + idx];
    }
  }
}

// Here we compute all 16 cell histograms by calling the cell_histogram function in parallel.
// The loop is completely unrolled and operates on the pre-computed gradients and magnitudes.
static void compute_cells(const BIN_T bin_map[IMG_H][IMG_W],
                          const float mag[IMG_H][IMG_W],
                          float cells[CELLS_H][CELLS_W][NUM_BINS])
{
CELL_ROW:
  for (int ci = 0; ci < CELLS_H; ci++)
  {
    #pragma HLS pipeline off
  CELL_COL:
    for (int cj = 0; cj < CELLS_W; cj++)
    {
      #pragma HLS unroll
      cell_histogram(bin_map, mag, ci * CELL_SIZE, cj * CELL_SIZE, cells[ci][cj]);
    }
  }
}

// This function performs L2 normalization on 2x2 cell blocks using precomputed per-cell energy values.
// We optimize this by reusing energy across overlapping blocks instead of recomputing norms repeatedly.
static void block_normalize(const float cells[CELLS_H][CELLS_W][NUM_BINS],
                            float features[FEATURE_LEN])
{
  VAR_TYPE cell_energy[CELLS_H][CELLS_W];
  #pragma HLS array_partition variable = cell_energy complete dim = 0

ENERGY_ROW:
  for (int ci = 0; ci < CELLS_H; ci++)
  {
  ENERGY_COL:
    for (int cj = 0; cj < CELLS_W; cj++)
    {
      #pragma HLS pipeline II = 1
      VAR_TYPE e = 0;
      for (int b = 0; b < NUM_BINS; b++)
      {
        #pragma HLS unroll
        VAR_TYPE bv = (VAR_TYPE)cells[ci][cj][b];
        e += bv * bv;
      }
      cell_energy[ci][cj] = e;
    }
  }

  float inv_norm[BLOCKS_H * BLOCKS_W];
  #pragma HLS array_partition variable = inv_norm complete

NORM_SQRT:
  for (int k = 0; k < BLOCKS_H * BLOCKS_W; k++)
  {
    #pragma HLS pipeline II = 1
    int bi = k / BLOCKS_W;
    int bj = k % BLOCKS_W;
    VAR_TYPE norm_sq = cell_energy[bi][bj] + cell_energy[bi][bj + 1] +
                       cell_energy[bi + 1][bj] + cell_energy[bi + 1][bj + 1];
    float block_norm = hls::sqrt((float)norm_sq) + 1e-6f;
    inv_norm[k] = 1.0f / block_norm;
  }

  const int BLK_LEN = BLOCK_CELLS * BLOCK_CELLS * NUM_BINS;

OUTPUT:
  for (int out_idx = 0; out_idx < FEATURE_LEN; out_idx++)
  {
    #pragma HLS pipeline II = 1
    int blk = out_idx / BLK_LEN;
    int r = out_idx % BLK_LEN;
    int bi = blk / BLOCKS_W;
    int bj = blk % BLOCKS_W;
    int ci = r / (BLOCK_CELLS * NUM_BINS);
    int cj = (r / NUM_BINS) % BLOCK_CELLS;
    int b = r % NUM_BINS;
    features[out_idx] = cells[bi + ci][bj + cj][b] * inv_norm[blk];
  }
}

// This stage writes the computed 288-element feature vector back to external memory using burst transfers.
// It's the final step in our dataflow pipeline and outputs the HoG descriptor.
static void store_feat_stage(float *features, int n,
                             const float feat[FEATURE_LEN])
{
STORE_FEAT:
  for (int i = 0; i < FEATURE_LEN; i += 16)
  {
    #pragma HLS pipeline II = 1
    for (int j = 0; j < 16; j++)
    {
      #pragma HLS unroll
      features[n * FEATURE_LEN + i + j] = feat[i + j];
    }
  }
}

// This is the main top-level batched kernel that orchestrates the entire HoG computation.
// It processes multiple images in a dataflow region where consecutive images overlap, achieving pipelining.
void hog_compute(const unsigned char *images, float *features, int num_images)
{
  #pragma HLS interface m_axi port = images offset = slave bundle = gmem0 depth = 2048
  #pragma HLS interface m_axi port = features offset = slave bundle = gmem1 depth = 576
  #pragma HLS interface s_axilite port = num_images
  #pragma HLS interface s_axilite port = return

BATCH:
  for (int n = 0; n < num_images; n++)
  {
    #pragma HLS loop_tripcount min = 1 max = 256 avg = 256
    #pragma HLS dataflow
    unsigned char img_local[IMG_H][IMG_W];
    float gamma_img[IMG_H][IMG_W];
    BIN_T bin_map[IMG_H][IMG_W];
    float mag[IMG_H][IMG_W];
    float cells[CELLS_H][CELLS_W][NUM_BINS];
    float feat_local[FEATURE_LEN];
    float inv_stddev;
    
    #pragma HLS array_partition variable = img_local cyclic factor = 16 dim = 2
    #pragma HLS array_partition variable = gamma_img cyclic factor = 3 dim = 1
    #pragma HLS array_partition variable = gamma_img cyclic factor = 2 dim = 2
    #pragma HLS array_partition variable = bin_map block factor = 4 dim = 2
    #pragma HLS array_partition variable = mag block factor = 4 dim = 2
    #pragma HLS array_partition variable = cells complete dim = 2
    #pragma HLS array_partition variable = feat_local cyclic factor = 16 dim = 1

    load_image_stage(images, n, img_local);
    normalize_image(img_local, gamma_img, inv_stddev);
    compute_gradients(gamma_img, inv_stddev, bin_map, mag);
    compute_cells(bin_map, mag, cells);
    block_normalize(cells, feat_local);
    store_feat_stage(features, n, feat_local);
  }
}