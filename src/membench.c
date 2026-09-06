/* membench.c — measure the memory hierarchy the post talks about.
 *
 * 1) Latency: dependent pointer chase over a random cycle, one hop per cache
 *    line, working set swept 4 KB -> 1 GB. Each load's address depends on the
 *    previous load, so the CPU can't overlap them: pure latency.
 * 2) Bandwidth: streaming read-sum of a large array, 1 thread and all threads.
 *
 * Output: CSV lines "LAT,<bytes>,<ns_per_load>" and "BW,<threads>,<GB_s>".
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { uint64_t next; char pad[56]; } Line; /* one cache line */

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t rng_s = 88172645463325252ull;
static uint64_t rng(void) { /* xorshift64 */
  rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
  return rng_s;
}

/* Sattolo's algorithm: a single random cycle visiting every line once */
static void build_cycle(Line *a, uint64_t n) {
  uint64_t *perm = malloc(n * sizeof(uint64_t));
  for (uint64_t i = 0; i < n; i++) perm[i] = i;
  for (uint64_t i = n - 1; i > 0; i--) {
    uint64_t j = rng() % i;
    uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (uint64_t i = 0; i < n; i++)
    a[perm[i]].next = perm[(i + 1) % n];
  free(perm);
}

static double chase(Line *a, uint64_t n, uint64_t hops) {
  volatile uint64_t p = 0;
  uint64_t q = 0;
  double t0 = now_s();
  for (uint64_t h = 0; h < hops; h++) q = a[q].next;
  double dt = now_s() - t0;
  p = q; (void)p;
  return dt / hops * 1e9;
}

int main(void) {
  /* latency sweep */
  uint64_t max_bytes = 1ull << 30; /* 1 GB */
  Line *a = aligned_alloc(64, max_bytes);
  for (uint64_t ws = 4096; ws <= max_bytes; ws *= 2) {
    uint64_t n = ws / sizeof(Line);
    build_cycle(a, n);
    chase(a, n, n < 1000000 ? n * 4 : n); /* touch everything: warm */
    uint64_t hops = 20000000; /* ~x ms per point */
    if (ws >= (64ull << 20)) hops = 5000000;
    double ns = chase(a, n, hops);
    printf("LAT,%llu,%.3f\n", (unsigned long long)ws, ns);
    fflush(stdout);
  }
  free(a);

  /* bandwidth: streaming read-sum over 1 GB of floats */
  uint64_t nf = (1ull << 30) / sizeof(float);
  float *x = aligned_alloc(64, nf * sizeof(float));
  memset(x, 1, nf * sizeof(float));
  int max_t = 1;
#ifdef _OPENMP
  max_t = omp_get_max_threads();
#endif
  int tlist[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 144};
  uint64_t nchunk = nf / 4096; /* 16 KB chunks, each read with 8 ymm accs */
  for (unsigned ti = 0; ti < sizeof(tlist) / sizeof(int); ti++) {
    int t = tlist[ti];
    if (t > max_t) break;
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
      double t0 = now_s();
      float sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) num_threads(t) schedule(static)
#endif
      for (uint64_t c = 0; c < nchunk; c++) {
        const float *p = x + c * 4096;
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        __m256 s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        for (int i = 0; i < 4096; i += 64) {
          s0 = _mm256_add_ps(s0, _mm256_loadu_ps(p + i));
          s1 = _mm256_add_ps(s1, _mm256_loadu_ps(p + i + 8));
          s2 = _mm256_add_ps(s2, _mm256_loadu_ps(p + i + 16));
          s3 = _mm256_add_ps(s3, _mm256_loadu_ps(p + i + 24));
          s4 = _mm256_add_ps(s4, _mm256_loadu_ps(p + i + 32));
          s5 = _mm256_add_ps(s5, _mm256_loadu_ps(p + i + 40));
          s6 = _mm256_add_ps(s6, _mm256_loadu_ps(p + i + 48));
          s7 = _mm256_add_ps(s7, _mm256_loadu_ps(p + i + 56));
        }
        s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        s4 = _mm256_add_ps(_mm256_add_ps(s4, s5), _mm256_add_ps(s6, s7));
        s0 = _mm256_add_ps(s0, s4);
        float tmp[8];
        _mm256_storeu_ps(tmp, s0);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      }
      double dt = now_s() - t0;
      if (sum < 0) printf("#"); /* keep the reduction alive */
      double gbs = nchunk * 4096 * sizeof(float) / dt / 1e9;
      if (gbs > best) best = gbs;
    }
    printf("BW,%d,%.2f\n", t, best);
    fflush(stdout);
  }
  free(x);
  return 0;
}
