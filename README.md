# COMP4601 Project: Accelerated Histogram of Oriented Gradients (HoG)

## Team Name: HoG Riders

## Project Overview

This project focuses on accelerating the computation of **Histogram of Oriented Gradients (HoG)** descriptors using FPGA-based High-Level Synthesis (HLS). HoG descriptors are a powerful feature extraction technique used in traditional (non-deep-learning) machine learning pipelines. When paired with a Support Vector Machine (SVM), they are highly effective for image classification tasks such as digit recognition.

For this project, we restrict our implementation to **8-bin HoG** descriptors and evaluate performance on a batch of 10,000 images from the **CIFAR-BW dataset** (provided in bitmap format). Speedup is measured by comparing the latency and/or throughput of the hardware-accelerated HLS design against a software-only implementation running on the Processing System (PS).

### HoG Algorithm Summary

The HoG algorithm works by:

1. Computing the gradient magnitude and direction at each pixel of an input image.
2. Dividing the image into small spatial cells.
3. Building a histogram of gradient orientations (8 bins) within each cell.
4. Normalising histograms across overlapping blocks of cells to improve robustness to lighting variation.
5. Concatenating all block descriptors into a final feature vector for use with a classifier (e.g., SVM).

### Approach

1. Study the HoG algorithm and identify opportunities for parallelism.
2. Develop and thoroughly test a software (C/C++) implementation on the PS.
3. Profile the software implementation to identify performance bottlenecks.
4. Port the implementation to an HLS design, applying directives (e.g., loop unrolling, pipelining, array partitioning) to exploit parallelism on the FPGA fabric.
5. Benchmark the HLS design against the PS baseline using latency and throughput metrics.

### Project Documentation

- [Initial Investigation and Plan](Initial_Investigation_And_Plan.pdf) — the project's initial research, scope, implementation plan, and proposed FPGA acceleration strategy.
- [Project Updates](project_updates.md) — a dated record of changes to the project direction and the reasons behind them.
- [Final Project Presentation](COMP4601_Project_Presentation_Final.pdf) — the final presentation covering the implementation, optimisations, and results.
- [Demonstration Plan](demonstration_plan.pdf) — the planned walkthrough of the accelerator versions, performance comparison, and demonstration fallback.
- [Final Reprt](COMP4601_ProjectReport.pdf) — final report, which covers content which we could fit in within the presentation and demonstration.

### Sample HoG Input and Output

- [`9999.bmp`](9999.bmp) is a 32 × 32, 8-bit bitmap that serves as a single-image input example. The Python HoG implementation reads it as a grayscale image and runs the complete feature-extraction pipeline on it.
- [`9999.txt`](9999.txt) is the output generated from `9999.bmp`. It contains the resulting 288-value normalised HoG feature vector (9 blocks × 4 cells × 8 orientation bins), with one value per line. This provides an easy way to inspect and validate how an input image is converted into features for comparison or classification.

### Resources

- [HoG Background Video](https://www.youtube.com/watch?v=4ESLTAd3IOM)
- [Object Recognition for Dummies, Part 1 (Lilian Weng)](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html)
- [Histogram of Oriented Gradients (LearnOpenCV)](https://learnopencv.com/histogram-of-oriented-gradients/)
- [HoG + SVM Digit Recognition Example](https://medium.com/@basu369victor/handwritten-digits-recognition-d3d383431845)

---

## Team Members

| Name | zID | Contact |
|------|-----|---------|
| Srikaran Gomadam | z5478391 | z5478391@ad.unsw.edu.au |
| Rhythm Nath | z5417295 | z5417295@ad.unsw.edu.au |
| Samyak Diwan | z5611048 | z5611048@ad.unsw.edu.au |
| Abhiram Yanamandra | z5492119 | z5492119@ad.unsw.edu.au |

---

## Notes

- Added in the HoG files, but removed the actual data files since there's so many

---

## Contact

For further information, please contact the team representative at: **z5492119@ad.unsw.edu.au**
