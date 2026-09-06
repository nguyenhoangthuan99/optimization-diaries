/* kernels.c — the optimization ladder, one kernel per rung.
 *
 * Conventions: A is MxK, B is KxN, C is MxN, all row-major, C zeroed by the
 * harness; every kernel computes C += A*B. Same flops everywhere — only the
 * order the bytes move in changes.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kernels.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* ---------------- rung 0/1: the six loop orders ---------------- */

static void k_ijk(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  (void)o;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < K; k++)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

static void k_ikj(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  (void)o;
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
      float a = A[i * K + k];
      for (int j = 0; j < N; j++)
        C[i * N + j] += a * B[k * N + j];
    }
}

static void k_jik(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  (void)o;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

static void k_jki(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  (void)o;
  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++) {
      float b = B[k * N + j];
      for (int i = 0; i < M; i++)
        C[i * N + j] += A[i * K + k] * b;
    }
}

static void k_kij(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  (void)o;
  for (int k = 0; k < K; k++)
    for (int i = 0; i < M; i++) {
      float a = A[i * K + k];
      for (int j = 0; j < N; j++)
        C[i * N + j] += a * B[k * N + j];
    }
}

static void k_kji(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  (void)o;
  for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++) {
      float b = B[k * N + j];
      for (int i = 0; i < M; i++)
        C[i * N + j] += A[i * K + k] * b;
    }
}

/* ---------------- rung 2: cache blocking on top of ikj ---------------- */

static void k_tiled(const float *A, const float *B, float *C,
                    int M, int N, int K, const KernOpts *o) {
  int TI = o->TI ? o->TI : 64;
  int TJ = o->TJ ? o->TJ : 512;
  int TK = o->TK ? o->TK : 64;
  for (int ii = 0; ii < M; ii += TI)
    for (int kk = 0; kk < K; kk += TK)
      for (int jj = 0; jj < N; jj += TJ) {
        int li = MIN(ii + TI, M), lk = MIN(kk + TK, K), lj = MIN(jj + TJ, N);
        for (int i = ii; i < li; i++)
          for (int k = kk; k < lk; k++) {
            float a = A[i * K + k];
            for (int j = jj; j < lj; j++)
              C[i * N + j] += a * B[k * N + j];
          }
      }
}

/* ---------------- rung 3: register blocking (4x16 tile in locals) --------
 * Same tiling as rung 2, but the innermost work keeps a 4-row x 16-col C tile
 * in local variables across the whole K-panel: each B load now feeds 4 rows
 * (vs 1 in rung 2) and C is touched once per panel instead of once per k.
 * Written as plain C with the j-loop innermost so the compiler can still
 * auto-vectorize it — no intrinsics yet.
 */
static void k_regblock(const float *A, const float *B, float *C,
                       int M, int N, int K, const KernOpts *o) {
  int TJ = o->TJ ? o->TJ : 512;
  int TK = o->TK ? o->TK : 256;
  for (int kk = 0; kk < K; kk += TK)
    for (int jj = 0; jj < N; jj += TJ) {
      int lk = MIN(kk + TK, K), lj = MIN(jj + TJ, N);
      int i = 0;
      for (; i + 3 < M; i += 4) {
        int j = jj;
        for (; j + 15 < lj; j += 16) {
          float c0[16] = {0}, c1[16] = {0}, c2[16] = {0}, c3[16] = {0};
          for (int k = kk; k < lk; k++) {
            const float *b = &B[k * (long)N + j];
            float a0 = A[(i + 0) * (long)K + k], a1 = A[(i + 1) * (long)K + k];
            float a2 = A[(i + 2) * (long)K + k], a3 = A[(i + 3) * (long)K + k];
            for (int v = 0; v < 16; v++) {
              c0[v] += a0 * b[v];
              c1[v] += a1 * b[v];
              c2[v] += a2 * b[v];
              c3[v] += a3 * b[v];
            }
          }
          for (int v = 0; v < 16; v++) {
            C[(i + 0) * (long)N + j + v] += c0[v];
            C[(i + 1) * (long)N + j + v] += c1[v];
            C[(i + 2) * (long)N + j + v] += c2[v];
            C[(i + 3) * (long)N + j + v] += c3[v];
          }
        }
        for (; j < lj; j++) /* N remainder */
          for (int k = kk; k < lk; k++) {
            C[(i + 0) * N + j] += A[(i + 0) * K + k] * B[k * N + j];
            C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j];
            C[(i + 2) * N + j] += A[(i + 2) * K + k] * B[k * N + j];
            C[(i + 3) * N + j] += A[(i + 3) * K + k] * B[k * N + j];
          }
      }
      for (; i < M; i++) /* M remainder */
        for (int k = kk; k < lk; k++) {
          float a = A[i * K + k];
          for (int j = jj; j < lj; j++)
            C[i * N + j] += a * B[k * N + j];
        }
    }
}

/* ---------------- rung 4: SIMD microkernels ---------------- */

/* 6x16 AVX2 microkernel: C tile lives in 12 ymm registers across the whole
 * K-panel; per k-step, 2 B loads + 6 A broadcasts feed 12 FMAs. */
static inline void micro_6x16_avx2(const float *A, const float *B, float *C,
                                   int K, int N, int kc) {
  __m256 c[6][2];
  for (int r = 0; r < 6; r++) {
    c[r][0] = _mm256_loadu_ps(C + r * N);
    c[r][1] = _mm256_loadu_ps(C + r * N + 8);
  }
  for (int k = 0; k < kc; k++) {
    __m256 b0 = _mm256_loadu_ps(B + k * N);
    __m256 b1 = _mm256_loadu_ps(B + k * N + 8);
    for (int r = 0; r < 6; r++) {
      __m256 a = _mm256_set1_ps(A[r * K + k]);
      c[r][0] = _mm256_fmadd_ps(a, b0, c[r][0]);
      c[r][1] = _mm256_fmadd_ps(a, b1, c[r][1]);
    }
  }
  for (int r = 0; r < 6; r++) {
    _mm256_storeu_ps(C + r * N, c[r][0]);
    _mm256_storeu_ps(C + r * N + 8, c[r][1]);
  }
}

/* scalar edge handler shared by the SIMD kernels */
static void edge_ikj(const float *A, const float *B, float *C, int K, int N,
                     int i0, int i1, int j0, int j1, int k0, int k1) {
  for (int i = i0; i < i1; i++)
    for (int k = k0; k < k1; k++) {
      float a = A[i * K + k];
      for (int j = j0; j < j1; j++)
        C[i * N + j] += a * B[k * N + j];
    }
}

static void block_avx2(const float *A, const float *B, float *C,
                       int M, int N, int K, int i0, int i1,
                       int TJ, int TK) {
  for (int kk = 0; kk < K; kk += TK) {
    int lk = MIN(kk + TK, K);
    for (int jj = 0; jj < N; jj += TJ) {
      int lj = MIN(jj + TJ, N);
      int i = i0;
      for (; i + 5 < i1; i += 6) {
        int j = jj;
        for (; j + 15 < lj; j += 16)
          micro_6x16_avx2(A + i * K + kk, B + kk * N + j, C + i * N + j,
                          K, N, lk - kk);
        if (j < lj) edge_ikj(A, B, C, K, N, i, i + 6, j, lj, kk, lk);
      }
      if (i < i1) edge_ikj(A, B, C, K, N, i, i1, jj, lj, kk, lk);
    }
  }
}

static void k_avx2(const float *A, const float *B, float *C,
                   int M, int N, int K, const KernOpts *o) {
  int TJ = o->TJ ? o->TJ : 512;
  int TK = o->TK ? o->TK : 256;
  block_avx2(A, B, C, M, N, K, 0, M, TJ, TK);
}

#ifdef __AVX512F__
/* 6x32 AVX-512 microkernel: same shape, zmm registers. */
static inline void micro_6x32_avx512(const float *A, const float *B, float *C,
                                     int K, int N, int kc) {
  __m512 c[6][2];
  for (int r = 0; r < 6; r++) {
    c[r][0] = _mm512_loadu_ps(C + r * N);
    c[r][1] = _mm512_loadu_ps(C + r * N + 16);
  }
  for (int k = 0; k < kc; k++) {
    __m512 b0 = _mm512_loadu_ps(B + k * N);
    __m512 b1 = _mm512_loadu_ps(B + k * N + 16);
    for (int r = 0; r < 6; r++) {
      __m512 a = _mm512_set1_ps(A[r * K + k]);
      c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
      c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
    }
  }
  for (int r = 0; r < 6; r++) {
    _mm512_storeu_ps(C + r * N, c[r][0]);
    _mm512_storeu_ps(C + r * N + 16, c[r][1]);
  }
}

static void block_avx512(const float *A, const float *B, float *C,
                         int M, int N, int K, int i0, int i1,
                         int TJ, int TK) {
  for (int kk = 0; kk < K; kk += TK) {
    int lk = MIN(kk + TK, K);
    for (int jj = 0; jj < N; jj += TJ) {
      int lj = MIN(jj + TJ, N);
      int i = i0;
      for (; i + 5 < i1; i += 6) {
        int j = jj;
        for (; j + 31 < lj; j += 32)
          micro_6x32_avx512(A + i * K + kk, B + kk * N + j, C + i * N + j,
                            K, N, lk - kk);
        if (j < lj) edge_ikj(A, B, C, K, N, i, i + 6, j, lj, kk, lk);
      }
      if (i < i1) edge_ikj(A, B, C, K, N, i, i1, jj, lj, kk, lk);
    }
  }
}

static void k_avx512(const float *A, const float *B, float *C,
                     int M, int N, int K, const KernOpts *o) {
  int TJ = o->TJ ? o->TJ : 512;
  int TK = o->TK ? o->TK : 256;
  block_avx512(A, B, C, M, N, K, 0, M, TJ, TK);
}
#endif

/* ---------------- rung 5: threads over row-blocks ---------------- */

/* One contiguous row-panel per thread (multiple of 6 so no thread splits a
 * microtile). Panel height M/nt keeps the B-tile reuse of the single-core
 * kernel intact — parallelizing over 6-row blocks instead cuts B reuse to 6
 * rows and turns the whole thing memory-bound (measured: 19 vs 93 GFLOP/s at
 * one thread). */
static void k_omp(const float *A, const float *B, float *C,
                  int M, int N, int K, const KernOpts *o) {
  int TJ = o->TJ ? o->TJ : 512;
  int TK = o->TK ? o->TK : 256;
  int nt = o->threads ? o->threads : 1;
  int rows6 = (M + 5) / 6;
  int per = (rows6 + nt - 1) / nt * 6; /* panel height, multiple of 6 */
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
  {
    int t = omp_get_thread_num();
#else
  for (int t = 0; t < nt; t++) {
#endif
    int i0 = t * per, i1 = MIN(i0 + per, M);
    if (i0 < M) {
#ifdef __AVX512F__
      block_avx512(A, B, C, M, N, K, i0, i1, TJ, TK);
#else
      block_avx2(A, B, C, M, N, K, i0, i1, TJ, TK);
#endif
    }
  }
}

/* ---------------- ceiling: OpenBLAS ---------------- */

#ifdef USE_BLAS
#include <cblas.h>
static void k_blas(const float *A, const float *B, float *C,
                   int M, int N, int K, const KernOpts *o) {
  (void)o; /* thread count set via OPENBLAS_NUM_THREADS */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
              1.0f, A, K, B, N, 1.0f, C, N);
}
#endif

const KernelEntry KERNELS[] = {
  {"ijk",      k_ijk,      "rung 0: naive triple loop"},
  {"ikj",      k_ikj,      "rung 1: best loop order"},
  {"jik",      k_jik,      "loop order jik"},
  {"jki",      k_jki,      "loop order jki"},
  {"kij",      k_kij,      "loop order kij"},
  {"kji",      k_kji,      "loop order kji"},
  {"tiled",    k_tiled,    "rung 2: cache blocking (ikj inside tiles)"},
  {"regblock", k_regblock, "rung 3: 4x4 register blocking, scalar"},
  {"avx2",     k_avx2,     "rung 4a: 6x16 AVX2 FMA microkernel"},
#ifdef __AVX512F__
  {"avx512",   k_avx512,   "rung 4b: 6x32 AVX-512 microkernel"},
#endif
  {"omp",      k_omp,      "rung 5: threads over row-blocks (best SIMD)"},
#ifdef USE_BLAS
  {"blas",     k_blas,     "ceiling: OpenBLAS sgemm"},
#endif
};
const int NUM_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));
