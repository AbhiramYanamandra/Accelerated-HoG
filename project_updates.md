# Project Updates

This page records unplanned changes in project direction, dated and briefly explained.

---

| Date | Change | Reason |
|------|--------|--------|
| 16 June 2026 | Selected **HOG (Histogram of Oriented Gradients) feature extraction** as the project topic and created a shared GitHub repository. | HOG's cell- and block-based pipeline maps well to FPGA dataflow and pipelining, while also being a relatively familiar algorithm for the team. |
| 1 July 2026 | Scoped the implementation to **8-bin HOG descriptors on 32 × 32 images** and defined the expected software-versus-hardware comparison. | The reduced image size and descriptor complexity kept the FPGA design tractable while still allowing meaningful performance comparisons. |
| 8 July 2026 | Began a **baseline C implementation** and divided the algorithm into normalisation, gradient computation, cell histogram generation, and block normalisation stages. | A known-correct software reference was required to validate the FPGA implementation and made it easier to optimise each pipeline stage independently. |
| 8 July 2026 | Identified initial optimisation targets, including **pipelining per-pixel loops** and **unrolling cell histogram computation**. | These optimisations were expected to improve throughput by increasing parallelism in the most frequently executed parts of the algorithm. |
| 15 July 2026 | Refined the hardware optimisation strategy to consider **array partitioning** and methods for removing **loop-carried dependencies** in histogram accumulation. | The baseline implementation was passing against reference feature vectors, allowing attention to shift from correctness to improving hardware performance. |
| 29 July 2026 | Confirmed a **top-level dataflow architecture** so consecutive images could overlap across pipeline stages. | Overlapping stages enables higher throughput by allowing different images to be processed simultaneously in different parts of the HOG pipeline. |
| 29 July 2026 | Began integrating the individual HOG stages into a **single kernel**. | Integration was needed ahead of the project presentation and to evaluate end-to-end hardware performance. |
| 5 August 2026 | Chose to focus the presentation results on the **software-versus-hardware timing comparison** and **per-stage performance breakdown**. | These results most clearly demonstrated the effect of the FPGA optimisations and the performance characteristics of each stage. |
| 12 August 2026 | Structured the final written report to avoid repeating material already presented or demonstrated. | This allowed the report to focus on new analysis, implementation detail, limitations, and evaluation rather than duplicating earlier presentation content. |
| 12 August 2026 | Added discussion of **potential design changes and future work** to the report scope. | Further optimisation and architectural improvements remained possible beyond the current implementation and were relevant to the project evaluation. |

---

*New entries should be added to the table above in chronological order as they occur.*
