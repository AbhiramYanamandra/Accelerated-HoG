// This file acts as the host-side driver, loading the image, running the kernel on the device, and comparing hardware results with the software reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <string>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#include "hog.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define NUM_RUNS 100

#define TOL 1e-2f

void hog_sw(const unsigned char image[IMG_H][IMG_W], float features[FEATURE_LEN])
{
    static float norm[IMG_H][IMG_W];
    static float grad[IMG_H][IMG_W];
    static float mag[IMG_H][IMG_W];
    static float cells[CELLS_H][CELLS_W][NUM_BINS];

    float sum = 0.0f;
    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
        {
            float x = (float)image[h][w];
            float v = x * x * std::sqrt(x);
            norm[h][w] = v;
            sum += v;
        }
    float mean = sum / (IMG_H * IMG_W);

    float var_sum = 0.0f;
    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
        {
            float d = norm[h][w] - mean;
            var_sum += d * d;
        }
    float stddev = std::sqrt(var_sum / (IMG_H * IMG_W));

    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
            norm[h][w] = (norm[h][w] - mean) / stddev;

    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
        {
            grad[h][w] = 0.0f;
            mag[h][w] = 0.0f;
        }

    for (int h = 1; h < IMG_H - 1; h++)
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

    const float bin_width = 180.0f / NUM_BINS;
    for (int ci = 0; ci < CELLS_H; ci++)
        for (int cj = 0; cj < CELLS_W; cj++)
        {
            int h0 = ci * CELL_SIZE, w0 = cj * CELL_SIZE;
            for (int b = 0; b < NUM_BINS; b++)
                cells[ci][cj][b] = 0.0f;
            for (int h = 0; h < CELL_SIZE; h++)
                for (int w = 0; w < CELL_SIZE; w++)
                {
                    float angle = grad[h0 + h][w0 + w];
                    float weight = mag[h0 + h][w0 + w];
                    int bin = (int)(angle / bin_width);
                    if (bin >= NUM_BINS)
                        bin = NUM_BINS - 1;
                    if (bin < 0)
                        bin = 0;
                    cells[ci][cj][bin] += weight;
                }
        }

    const int BLOCK_LEN = BLOCK_CELLS * BLOCK_CELLS * NUM_BINS;
    int out_idx = 0;
    for (int bi = 0; bi < BLOCKS_H; bi++)
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

static bool load_image(const std::string &path, unsigned char img[IMG_H][IMG_W])
{
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp)
        return false;
    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
        {
            int v;
            if (fscanf(fp, "%d", &v) != 1)
            {
                fclose(fp);
                return false;
            }
            if (v < 0)
                v = 0;
            if (v > 255)
                v = 255;
            img[h][w] = (unsigned char)v;
        }
    fclose(fp);
    return true;
}

static void gen_image(unsigned char img[IMG_H][IMG_W])
{
    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
            img[h][w] = (unsigned char)((h * 13 + w * 7 + h * w) & 0xFF);
}

int main(int argc, char **argv)
{
    std::string xclbin_file;
    std::string image_file;

    for (int i = 1; i < argc; i++)
    {
        std::string a(argv[i]);
        if (a == "-x" && i + 1 < argc)
            xclbin_file = argv[++i];
        else if (a == "-i" && i + 1 < argc)
            image_file = argv[++i];
    }
    if (xclbin_file.empty())
    {
        std::cout << "Usage: " << argv[0] << " -x <xclbin> [-i <image.dat>]\n";
        return 1;
    }

    unsigned char img[IMG_H][IMG_W];
    if (!image_file.empty())
    {
        if (!load_image(image_file, img))
        {
            std::cout << "Cannot read image file " << image_file << "\n";
            return 1;
        }
        std::cout << "Loaded image from " << image_file << "\n";
    }
    else
    {
        gen_image(img);
        std::cout << "Using generated synthetic image\n";
    }

    std::cout << "Opening device 0\n";
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_file);
    auto krnl = xrt::kernel(device, uuid, "hog_compute");

    auto bo_image = xrt::bo(device, IMG_H * IMG_W * sizeof(unsigned char), krnl.group_id(0));
    auto bo_feat = xrt::bo(device, FEATURE_LEN * sizeof(float), krnl.group_id(1));
    auto image_map = bo_image.map<unsigned char *>();
    auto feat_map = bo_feat.map<float *>();

    float sw_features[FEATURE_LEN];
    hog_sw(img, sw_features);

    double sw_total_us = 0.0;
    for (int run = 0; run < NUM_RUNS; run++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        hog_sw(img, sw_features);
        auto t1 = std::chrono::high_resolution_clock::now();
        sw_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    double sw_us = sw_total_us / NUM_RUNS;

    for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++)
            image_map[h * IMG_W + w] = img[h][w];

    bo_image.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    krnl(bo_image, bo_feat).wait();
    bo_feat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    double write_total_us = 0.0, kernel_total_us = 0.0, read_total_us = 0.0;

    for (int run = 0; run < NUM_RUNS; run++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        bo_image.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto krnl_run = krnl(bo_image, bo_feat);
        krnl_run.wait();
        auto t2 = std::chrono::high_resolution_clock::now();

        bo_feat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto t3 = std::chrono::high_resolution_clock::now();

        write_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        kernel_total_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        read_total_us += std::chrono::duration<double, std::micro>(t3 - t2).count();
    }

    double write_us = write_total_us / NUM_RUNS;
    double kernel_us = kernel_total_us / NUM_RUNS;
    double read_us = read_total_us / NUM_RUNS;
    double hw_us = write_us + kernel_us + read_us;

    float sum_sq = 0.0f, max_diff = 0.0f;
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        float d = feat_map[i] - sw_features[i];
        sum_sq += d * d;
        float ad = std::fabs(d);
        if (ad > max_diff)
            max_diff = ad;
    }
    float rmse = std::sqrt(sum_sq / FEATURE_LEN);

    // Print statements are AI-Generated
    printf("----------------------------------------------\n");
    printf("Feature length        : %d\n", FEATURE_LEN);
    printf("RMSE (HW vs SW)       : %0.15f\n", rmse);
    printf("Max abs diff          : %0.15f\n", max_diff);
    printf("----------------------------------------------\n");
    printf("Averaged over %d runs (plus 1 warm-up)\n", NUM_RUNS);
    printf("----------------------------------------------\n");
    printf("Software (ARM)        : %10.2f us\n", sw_us);
    printf("----------------------------------------------\n");
    printf("HW write (H2D)        : %10.2f us\n", write_us);
    printf("HW kernel only        : %10.2f us\n", kernel_us);
    printf("HW read  (D2H)        : %10.2f us\n", read_us);
    printf("HW end-to-end         : %10.2f us\n", hw_us);
    printf("----------------------------------------------\n");
    printf("Speedup (kernel only) : %10.2f x\n", sw_us / kernel_us);
    printf("Speedup (end-to-end)  : %10.2f x\n", sw_us / hw_us);
    printf("Transfer overhead     : %9.1f %% of end-to-end\n",
           100.0 * (write_us + read_us) / hw_us);
    printf("----------------------------------------------\n");

    if (max_diff > TOL)
    {
        printf("FAIL: HW output does not match SW within %.0e\n", TOL);
        return 1;
    }
    printf("PASS: HW output matches SW reference!\n");
    return 0;
}
