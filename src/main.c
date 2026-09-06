/* main.c - benchmark harness for the matmul ladder.
 *
 * Usage: matmul [-k kernel] [-t threads] [-n iters] [-w warmups] [-T TI,TJ,TK]
 *               [-v] [-l] M N K
 * Prints one machine-parseable RESULT line per run.
 */
#define _GNU_SOURCE
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kernels.h"

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* deterministic fill so every kernel sees identical data */
static void fill(float *x, long n, unsigned seed) {
  unsigned s = seed;
  for (long i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;
    x[i] = (float)(s >> 8) / (float)(1 << 24) - 0.5f;
  }
}

/* reference in double precision, ikj order (fast enough for -v sizes) */
static void ref_gemm(const float *A, const float *B, double *C,
                     int M, int N, int K) {
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
      double a = A[i * (long)K + k];
      for (int j = 0; j < N; j++)
        C[i * (long)N + j] += a * (double)B[k * (long)N + j];
    }
}

static int cmp_dbl(const void *a, const void *b) {
  double d = *(const double *)a - *(const double *)b;
  return d < 0 ? -1 : d > 0;
}

int main(int argc, char **argv) {
  const char *kname = "ijk";
  KernOpts o = {1, 0, 0, 0};
  int iters = 5, warmups = 1, validate = 0;
  int c;
  while ((c = getopt(argc, argv, "k:t:n:w:T:vlh")) != -1) {
    switch (c) {
    case 'k': kname = optarg; break;
    case 't': o.threads = atoi(optarg); break;
    case 'n': iters = atoi(optarg); break;
    case 'w': warmups = atoi(optarg); break;
    case 'T': sscanf(optarg, "%d,%d,%d", &o.TI, &o.TJ, &o.TK); break;
    case 'v': validate = 1; break;
    case 'l':
      for (int i = 0; i < NUM_KERNELS; i++)
        printf("%-10s %s\n", KERNELS[i].name, KERNELS[i].desc);
      return 0;
    default:
      fprintf(stderr,
              "usage: %s [-k kernel] [-t threads] [-n iters] [-w warmups] "
              "[-T TI,TJ,TK] [-v] [-l] M N K\n", argv[0]);
      return c == 'h' ? 0 : 1;
    }
  }
  if (argc - optind < 3) { fprintf(stderr, "need M N K\n"); return 1; }
  int M = atoi(argv[optind]), N = atoi(argv[optind + 1]),
      K = atoi(argv[optind + 2]);

  kern_fn fn = NULL;
  for (int i = 0; i < NUM_KERNELS; i++)
    if (!strcmp(KERNELS[i].name, kname)) fn = KERNELS[i].fn;
  if (!fn) { fprintf(stderr, "unknown kernel '%s' (-l to list)\n", kname); return 1; }

  float *A = aligned_alloc(64, (size_t)M * K * sizeof(float));
  float *B = aligned_alloc(64, (size_t)K * N * sizeof(float));
  float *C = aligned_alloc(64, (size_t)M * N * sizeof(float));
  if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
  fill(A, (long)M * K, 1);
  fill(B, (long)K * N, 2);

  double flops = 2.0 * M * N * K;

  if (validate) {
    double *R = calloc((size_t)M * N, sizeof(double));
    ref_gemm(A, B, R, M, N, K);
    memset(C, 0, (size_t)M * N * sizeof(float));
    fn(A, B, C, M, N, K, &o);
    /* normalize by RMS of the reference: elementwise relative error is
     * meaningless on near-zero entries of a random-input GEMM */
    double rms = 0, max_abs = 0;
    for (long i = 0; i < (long)M * N; i++) {
      rms += R[i] * R[i];
      double d = fabs(C[i] - R[i]);
      if (d > max_abs) max_abs = d;
    }
    rms = sqrt(rms / ((long)M * N));
    double err = max_abs / rms;
    printf("VALIDATE kernel=%s M=%d N=%d K=%d max_err_vs_rms=%.3e %s\n",
           kname, M, N, K, err, err < 1e-4 ? "OK" : "FAIL");
    free(R);
    if (err >= 1e-4) return 2;
  }

  for (int w = 0; w < warmups; w++) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    fn(A, B, C, M, N, K, &o);
  }

  double *ts = malloc(iters * sizeof(double));
  for (int it = 0; it < iters; it++) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    double t0 = now_s();
    fn(A, B, C, M, N, K, &o);
    ts[it] = now_s() - t0;
    fprintf(stderr, "  iter %d: %.4f s  %.2f GFLOPS\n", it, ts[it],
            flops / ts[it] / 1e9);
  }
  qsort(ts, iters, sizeof(double), cmp_dbl);
  double med = ts[iters / 2];
  double mean = 0, var = 0;
  for (int i = 0; i < iters; i++) mean += ts[i];
  mean /= iters;
  for (int i = 0; i < iters; i++) var += (ts[i] - mean) * (ts[i] - mean);
  double sd_pct = iters > 1 ? 100.0 * sqrt(var / (iters - 1)) / mean : 0;

  printf("RESULT kernel=%s M=%d N=%d K=%d threads=%d TI=%d TJ=%d TK=%d "
         "iters=%d median_s=%.6f gflops=%.3f sd_pct=%.2f\n",
         kname, M, N, K, o.threads, o.TI, o.TJ, o.TK, iters, med,
         flops / med / 1e9, sd_pct);
  free(A); free(B); free(C); free(ts);
  return 0;
}
