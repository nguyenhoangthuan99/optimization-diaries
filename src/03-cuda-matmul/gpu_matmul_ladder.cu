#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t _cuda_err = (x); if (_cuda_err != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_cuda_err)); std::exit(2); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t _cublas_status = (x); if (_cublas_status != CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)_cublas_status); std::exit(2); } } while (0)

template<int BM>
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BM + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// Uncoalesced baseline, for contrast with the naive kernel above.
// Swap the roles of threadIdx.x/threadIdx.y: now threadIdx.x walks the ROW
// dimension. Within a warp (consecutive threadIdx.x, fixed threadIdx.y) the
// threads read A rows a 4*K-byte stride apart and write C rows a 4*N-byte
// stride apart, so a warp touches 32 separate cache lines instead of one.
// B is broadcast (all lanes read the same element). This is the access bug
// that the "global memory coalescing" step fixes.
template<int BM>
__global__ void matmul_naive_bad(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * BM + threadIdx.x;   // x -> row  (strided A and C)
    int col = blockIdx.x * BM + threadIdx.y;   // y -> col  (B broadcast)
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    constexpr int T = 32;
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int kb = 0; kb < K; kb += T) {
        As[ty][tx] = (row < M && kb + tx < K) ? A[row * K + kb + tx] : 0.0f;
        Bs[ty][tx] = (kb + ty < K && col < N) ? B[(kb + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// 2x2 register tile per thread. A 16x16 thread block covers a 32x32 C tile.
__global__ void matmul_reg2x2(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    constexpr int BM = 32, BN = 32, BK = 16;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * BM + ty * 2;
    int col0 = blockIdx.x * BN + tx * 2;
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    for (int kb = 0; kb < K; kb += BK) {
        int ar = row0, ac = kb + tx;
        As[ty * 2 + 0][tx] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
        ar = row0 + 1;
        As[ty * 2 + 1][tx] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
        int br = kb + ty, bc = col0;
        Bs[ty][tx * 2 + 0] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
        Bs[ty][tx * 2 + 1] = (br < K && bc + 1 < N) ? B[br * N + bc + 1] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a0 = As[ty * 2 + 0][k], a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2 + 0], b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0; c01 += a0 * b1;
            c10 += a1 * b0; c11 += a1 * b1;
        }
        __syncthreads();
    }
    if (row0 < M && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < M && col0 + 1 < N) C[row0 * N + col0 + 1] = c01;
    if (row0 + 1 < M && col0 < N) C[(row0 + 1) * N + col0] = c10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = c11;
}

// 64x64 block tile, 16x16 threads, 4x4 outputs per thread.
// This increases data reuse while keeping the block at 256 threads.
__global__ void matmul_block64_reg4x4(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 16;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * 16 + threadIdx.x;
    int row0 = blockIdx.y * BM + threadIdx.y * 4;
    int col0 = blockIdx.x * BN + threadIdx.x * 4;
    float acc[4][4] = {};

    for (int kb = 0; kb < K; kb += BK) {
        // Cooperative, contiguous loads into the shared-memory block tile.
        for (int i = tid; i < BM * BK; i += 256) {
            int r = i / BK, c = i % BK;
            As[r][c] = (blockIdx.y * BM + r < M && kb + c < K)
                ? A[(blockIdx.y * BM + r) * K + kb + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += 256) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = (kb + r < K && blockIdx.x * BN + c < N)
                ? B[(kb + r) * N + blockIdx.x * BN + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[threadIdx.y * 4 + i][k];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][threadIdx.x * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// Explicit warp tiling: 4 warps arranged as a 2x2 grid. Each warp owns a
// 32x32 output tile; each lane accumulates an 4x8 register tile.
__global__ void matmul_warp4x4(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 16;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int warp_y = warp / 2, warp_x = warp % 2;
    int lane_y = lane / 4, lane_x = lane % 4;
    int row0 = blockIdx.y * BM + warp_y * 32 + lane_y * 4;
    int col0 = blockIdx.x * BN + warp_x * 32 + lane_x * 8;
    float acc[4][8] = {};

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = tid; i < BM * BK; i += 128) {
            int r = i / BK, c = i % BK;
            As[r][c] = (blockIdx.y * BM + r < M && kb + c < K)
                ? A[(blockIdx.y * BM + r) * K + kb + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += 128) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = (kb + r < K && blockIdx.x * BN + c < N)
                ? B[(kb + r) * N + blockIdx.x * BN + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[4], b[8];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[warp_y * 32 + lane_y * 4 + i][k];
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[k][warp_x * 32 + lane_x * 8 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// Larger register tile: 8x8 values per lane. Four warps cover a 64x128
// block tile; each warp owns a 32x64 tile. This raises arithmetic intensity
// while keeping the block at 128 threads.
__global__ void matmul_warp64x128_reg8x8(const float* __restrict__ A, const float* __restrict__ B,
                                         float* __restrict__ C, int M, int N, int K) {
    constexpr int BM = 64, BN = 128, BK = 16;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int warp_y = warp / 2, warp_x = warp % 2;
    int lane_y = lane / 8, lane_x = lane % 8;
    int row0 = blockIdx.y * BM + warp_y * 32 + lane_y * 8;
    int col0 = blockIdx.x * BN + warp_x * 64 + lane_x * 8;
    float acc[8][8] = {};

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = tid; i < BM * BK; i += 128) {
            int r = i / BK, c = i % BK;
            As[r][c] = (blockIdx.y * BM + r < M && kb + c < K)
                ? A[(blockIdx.y * BM + r) * K + kb + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += 128) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = (kb + r < K && blockIdx.x * BN + c < N)
                ? B[(kb + r) * N + blockIdx.x * BN + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[warp_y * 32 + lane_y * 8 + i][k];
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[k][warp_x * 64 + lane_x * 8 + j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// Vectorized float4 global loads into the same 64x128 / 8x8 / BK=32 kernel.
// Each thread moves four adjacent FP32 values per transaction where aligned;
// scalar tail loads preserve correctness for arbitrary dimensions.
__global__ void matmul_warp64x128_reg8x8_bk32_vec4(const float* __restrict__ A, const float* __restrict__ B,
                                                  float* __restrict__ C, int M, int N, int K) {
    constexpr int BM = 64, BN = 128, BK = 32;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int warp_y = warp / 2, warp_x = warp % 2;
    int lane_y = lane / 8, lane_x = lane % 8;
    int row0 = blockIdx.y * BM + warp_y * 32 + lane_y * 8;
    int col0 = blockIdx.x * BN + warp_x * 64 + lane_x * 8;
    float acc[8][8] = {};

    for (int kb = 0; kb < K; kb += BK) {
        // A tile: one float4 per work item, contiguous along K.
        constexpr int AVECS = BM * (BK / 4);
        for (int v = tid; v < AVECS; v += 128) {
            int r = v / (BK / 4), c = (v % (BK / 4)) * 4;
            int gr = blockIdx.y * BM + r, gc = kb + c;
            if (gr < M && gc + 3 < K) {
                float4 x = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
                As[r][c + 0] = x.x; As[r][c + 1] = x.y;
                As[r][c + 2] = x.z; As[r][c + 3] = x.w;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    As[r][c + j] = (gr < M && gc + j < K) ? A[gr * K + gc + j] : 0.0f;
            }
        }
        // B tile: one float4 per work item, contiguous along N.
        constexpr int BVECS = BK * (BN / 4);
        for (int v = tid; v < BVECS; v += 128) {
            int r = v / (BN / 4), c = (v % (BN / 4)) * 4;
            int gr = kb + r, gc = blockIdx.x * BN + c;
            if (gr < K && gc + 3 < N) {
                float4 x = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
                Bs[r][c + 0] = x.x; Bs[r][c + 1] = x.y;
                Bs[r][c + 2] = x.z; Bs[r][c + 3] = x.w;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    Bs[r][c + j] = (gr < K && gc + j < N) ? B[gr * N + gc + j] : 0.0f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[warp_y * 32 + lane_y * 8 + i][k];
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[k][warp_x * 64 + lane_x * 8 + j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// Same larger register tile, but with a 32-wide K tile. This amortizes
// synchronization and loop overhead at the cost of more shared memory.
__global__ void matmul_warp64x128_reg8x8_bk32(const float* __restrict__ A, const float* __restrict__ B,
                                              float* __restrict__ C, int M, int N, int K) {
    constexpr int BM = 64, BN = 128, BK = 32;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int warp_y = warp / 2, warp_x = warp % 2;
    int lane_y = lane / 8, lane_x = lane % 8;
    int row0 = blockIdx.y * BM + warp_y * 32 + lane_y * 8;
    int col0 = blockIdx.x * BN + warp_x * 64 + lane_x * 8;
    float acc[8][8] = {};

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = tid; i < BM * BK; i += 128) {
            int r = i / BK, c = i % BK;
            As[r][c] = (blockIdx.y * BM + r < M && kb + c < K)
                ? A[(blockIdx.y * BM + r) * K + kb + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += 128) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = (kb + r < K && blockIdx.x * BN + c < N)
                ? B[(kb + r) * N + blockIdx.x * BN + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[warp_y * 32 + lane_y * 8 + i][k];
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[k][warp_x * 64 + lane_x * 8 + j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

struct Result { std::string name; double ms; double tflops; double max_abs; double rms; };

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static Result bench_kernel(const std::string& name, void (*launch)(const float*, const float*, float*, int, int, int),
                           const float* A, const float* B, float* C, const float* ref,
                           int M, int N, int K, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times; times.reserve(iters);
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch(A, B, C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); times.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<float> host((size_t)M * N); CUDA_CHECK(cudaMemcpy(host.data(), C, host.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0.0, sum_sq = 0.0;
    if (ref) for (size_t i = 0; i < host.size(); ++i) {
        double a = (double)host[i] - ref[i];
        max_abs = std::max(max_abs, std::abs(a)); sum_sq += a * a;
    }
    double rms = ref ? std::sqrt(sum_sq / host.size()) : 0.0;
    double ms = median(times), tflops = (2.0 * M * N * K) / (ms * 1e9);
    return {name, ms, tflops, max_abs, rms};
}

static void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 32), grid((N+31)/32, (M+31)/32);
    matmul_naive<32><<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_naive_bad(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 32), grid((N+31)/32, (M+31)/32);
    matmul_naive_bad<32><<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 32), grid((N+31)/32, (M+31)/32);
    matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_reg2x2(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16), grid((N+31)/32, (M+31)/32);
    matmul_reg2x2<<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_block64(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16), grid((N+63)/64, (M+63)/64);
    matmul_block64_reg4x4<<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_warp64(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 4), grid((N+63)/64, (M+63)/64);
    matmul_warp4x4<<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_warp128_8x8(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 4), grid((N+127)/128, (M+63)/64);
    matmul_warp64x128_reg8x8<<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_warp128_8x8_bk32(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 4), grid((N+127)/128, (M+63)/64);
    matmul_warp64x128_reg8x8_bk32<<<grid, block>>>(A, B, C, M, N, K);
}
static void launch_warp128_8x8_bk32_vec4(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 4), grid((N+127)/128, (M+63)/64);
    matmul_warp64x128_reg8x8_bk32_vec4<<<grid, block>>>(A, B, C, M, N, K);
}

// Tunable native-FP32 kernel. The compile-time tile and thread shape make
// block/grid/thread experiments explicit instead of hiding them in launch code.
template<int BM, int BN, int BK, int THREADS_Y, int THREADS_X, int RY, int RX>
__global__ void matmul_tuned_fp32(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    static_assert(THREADS_X * THREADS_Y <= 1024, "invalid block size");
    static_assert(BM == THREADS_Y * RY && BN == THREADS_X * RX, "tile/thread mismatch");
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * THREADS_X + threadIdx.x;
    int row0 = blockIdx.y * BM + threadIdx.y * RY;
    int col0 = blockIdx.x * BN + threadIdx.x * RX;
    float acc[RY][RX] = {};

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = tid; i < BM * BK; i += THREADS_X * THREADS_Y) {
            int r = i / BK, k = i % BK;
            int gr = blockIdx.y * BM + r, gk = kb + k;
            As[r][k] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += THREADS_X * THREADS_Y) {
            int k = i / BN, c = i % BN;
            int gk = kb + k, gc = blockIdx.x * BN + c;
            Bs[k][c] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[RY], b[RX];
            #pragma unroll
            for (int i = 0; i < RY; ++i) a[i] = As[threadIdx.y * RY + i][k];
            #pragma unroll
            for (int j = 0; j < RX; ++j) b[j] = Bs[k][threadIdx.x * RX + j];
            #pragma unroll
            for (int i = 0; i < RY; ++i)
                #pragma unroll
                for (int j = 0; j < RX; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RY; ++i)
        #pragma unroll
        for (int j = 0; j < RX; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

static void launch_tuned_64x128_bk16_16x8(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 8), grid((N+127)/128, (M+63)/64);
    matmul_tuned_fp32<64,128,16,8,16,8,8><<<grid, block>>>(A,B,C,M,N,K);
}
static void launch_tuned_64x128_bk32_16x8(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 8), grid((N+127)/128, (M+63)/64);
    matmul_tuned_fp32<64,128,32,8,16,8,8><<<grid, block>>>(A,B,C,M,N,K);
}
static void launch_tuned_64x128_bk32_32x4(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 4), grid((N+127)/128, (M+63)/64);
    matmul_tuned_fp32<64,128,32,4,32,16,4><<<grid, block>>>(A,B,C,M,N,K);
}
static void launch_tuned_128x128_bk32_16x8(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 8), grid((N+127)/128, (M+127)/128);
    matmul_tuned_fp32<128,128,32,8,16,16,8><<<grid, block>>>(A,B,C,M,N,K);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 2048;
    int N = argc > 2 ? std::atoi(argv[2]) : M;
    int K = argc > 3 ? std::atoi(argv[3]) : M;
    int iters = argc > 4 ? std::atoi(argv[4]) : 30;
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s | SM %d.%d | %.1f GiB\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);
    std::printf("Shape: M=%d N=%d K=%d | iterations=%d\n", M, N, K, iters);
    size_t asz=(size_t)M*K, bsz=(size_t)K*N, csz=(size_t)M*N;
    std::vector<float> hA(asz), hB(bsz), hRef(csz);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng); for (auto& x : hB) x = dist(rng);
    float *A, *B, *C; CUDA_CHECK(cudaMalloc(&A, asz*sizeof(float))); CUDA_CHECK(cudaMalloc(&B, bsz*sizeof(float))); CUDA_CHECK(cudaMalloc(&C, csz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), asz*sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(B, hB.data(), bsz*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha=1.0f, beta=0.0f;
    // Row-major C=A*B is column-major C^T=B^T*A^T, represented by swapping A and B.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(hRef.data(), C, csz*sizeof(float), cudaMemcpyDeviceToHost));
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    for(int i=0;i<10;++i) CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N));
    CUDA_CHECK(cudaEventRecord(s)); CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N)); CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float cublas_ms; CUDA_CHECK(cudaEventElapsedTime(&cublas_ms,s,e));
    std::printf("%-14s %9.3f ms  %8.2f TFLOP/s  (reference)\n", "cuBLAS", cublas_ms, (2.0*M*N*K)/(cublas_ms*1e9));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    std::printf("%-14s %9s     %8s       max_abs          rms\n", "kernel", "time", "TFLOP/s");
    Result r0=bench_kernel("naive_bad", launch_naive_bad,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r1=bench_kernel("naive", launch_naive,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r2=bench_kernel("shared_32", launch_tiled,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r3=bench_kernel("reg_2x2", launch_reg2x2,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r4=bench_kernel("block64_4x4", launch_block64,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r5=bench_kernel("warp64_4x8", launch_warp64,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r6=bench_kernel("warp128_8x8", launch_warp128_8x8,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r7=bench_kernel("warp128_8x8_bk32", launch_warp128_8x8_bk32,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r8=bench_kernel("warp128_8x8_vec4", launch_warp128_8x8_bk32_vec4,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r9=bench_kernel("tuned64x128_bk16_16x8", launch_tuned_64x128_bk16_16x8,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r10=bench_kernel("tuned64x128_bk32_16x8", launch_tuned_64x128_bk32_16x8,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r11=bench_kernel("tuned64x128_bk32_32x4", launch_tuned_64x128_bk32_32x4,A,B,C,hRef.data(),M,N,K,5,iters);
    Result r12=bench_kernel("tuned128x128_bk32_16x8", launch_tuned_128x128_bk32_16x8,A,B,C,hRef.data(),M,N,K,5,iters);
    for (const auto& r : {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12}) std::printf("%-24s %9.3f ms  %8.2f       %.3e   %.3e\n",r.name.c_str(),r.ms,r.tflops,r.max_abs,r.rms);
    CUBLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(B)); CUDA_CHECK(cudaFree(C));
}
