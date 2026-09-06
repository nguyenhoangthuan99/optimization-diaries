#ifndef KERNELS_H
#define KERNELS_H

typedef struct {
  int threads; /* used by omp kernels */
  int TI, TJ, TK; /* tile sizes; 0 = kernel default */
} KernOpts;

typedef void (*kern_fn)(const float *A, const float *B, float *C,
                        int M, int N, int K, const KernOpts *o);

typedef struct {
  const char *name;
  kern_fn fn;
  const char *desc;
} KernelEntry;

extern const KernelEntry KERNELS[];
extern const int NUM_KERNELS;

#endif
