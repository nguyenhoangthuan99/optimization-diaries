#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); std::exit(2);} } while(0)

// ---------------- step 0/1: naive (coalesced vs uncoalesced) ----------------
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
template<int BM>
__global__ void matmul_naive_bad(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * BM + threadIdx.x;
    int col = blockIdx.x * BM + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------- step 2: shared-memory tiling ----------------
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

// ---------------- step 3: 2x2 register tile ----------------
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

// ---------------- step 4: tuned 64x128 / BK=32 / 16x8 ----------------
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

// ---------------- step 5/6: cp.async double-buffered pipeline ----------------
template<int BM, int BN, int BK, int TX, int TY, int RY, int RX>
__global__ void matmul_pipe_dbl(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
    static_assert(TX*TY<=1024 && BM==TY*RY && BN==TX*RX, "shape");
    extern __shared__ float smem[];
    float (*As)[BM][BK] = reinterpret_cast<float(*)[BM][BK]>(smem);
    float (*Bs)[BK][BN] = reinterpret_cast<float(*)[BK][BN]>(smem + 2u*BM*BK);
    const int nthread = TX*TY;
    int tid = threadIdx.y*TX + threadIdx.x;
    int row0 = blockIdx.y*BM + threadIdx.y*RY;
    int col0 = blockIdx.x*BN + threadIdx.x*RX;
    float acc[RY][RX] = {};
    auto load_stage = [&](int st, int kb) {
        constexpr int A4 = BM*BK/4;
        for (int v=tid; v<A4; v+=nthread) {
            int r = v/(BK/4), c=(v%(BK/4))*4;
            int gr = blockIdx.y*BM + r;
            int gc = kb+c;
            if (gr<M && gc+3<K && ((gr*K+gc)&3)==0)
                __pipeline_memcpy_async(&As[st][r][c], &A[gr*K+gc], 16);
            else
                #pragma unroll
                for (int j=0;j<4;++j) As[st][r][c+j]=(gr<M && gc+j<K)?A[gr*K+gc+j]:0.0f;
        }
        constexpr int B4 = BK*BN/4;
        for (int v=tid; v<B4; v+=nthread) {
            int r = v/(BN/4), c=(v%(BN/4))*4;
            int gr = kb+r, gc = blockIdx.x*BN+c;
            if (gr<K && gc+3<N && ((gr*N+gc)&3)==0)
                __pipeline_memcpy_async(&Bs[st][r][c], &B[gr*N+gc], 16);
            else
                #pragma unroll
                for (int j=0;j<4;++j) Bs[st][r][c+j]=(gr<K && gc+j<N)?B[gr*N+gc+j]:0.0f;
        }
    };
    load_stage(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    int st = 0;
    for (int kb=0; kb<K; kb+=BK) {
        int cur = st, nxt = st^1;
        if (kb+BK < K) { load_stage(nxt, kb+BK); __pipeline_commit(); }
        #pragma unroll
        for (int k=0;k<BK;++k){
            float a[RY], b[RX];
            #pragma unroll
            for (int i=0;i<RY;++i) a[i]=As[cur][threadIdx.y*RY+i][k];
            #pragma unroll
            for (int j=0;j<RX;++j) b[j]=Bs[cur][k][threadIdx.x*RX+j];
            #pragma unroll
            for (int i=0;i<RY;++i)
                #pragma unroll
                for (int j=0;j<RX;++j) acc[i][j]+=a[i]*b[j];
        }
        __pipeline_wait_prior(0);
        __syncthreads();
        st = nxt;
    }
    #pragma unroll
    for (int i=0;i<RY;++i)
        #pragma unroll
        for (int j=0;j<RX;++j){
            int r=row0+i, c=col0+j;
            if (r<M && c<N) C[r*N+c]=acc[i][j];
        }
}

typedef void (*launch_fn)(const float*,const float*,float*,int,int,int);

static void l_naive_bad(const float*A,const float*B,float*C,int M,int N,int K){ dim3 b(32,32),g((N+31)/32,(M+31)/32); matmul_naive_bad<32><<<g,b>>>(A,B,C,M,N,K); }
static void l_naive(const float*A,const float*B,float*C,int M,int N,int K){ dim3 b(32,32),g((N+31)/32,(M+31)/32); matmul_naive<32><<<g,b>>>(A,B,C,M,N,K); }
static void l_tiled(const float*A,const float*B,float*C,int M,int N,int K){ dim3 b(32,32),g((N+31)/32,(M+31)/32); matmul_tiled<<<g,b>>>(A,B,C,M,N,K); }
static void l_reg2x2(const float*A,const float*B,float*C,int M,int N,int K){ dim3 b(16,16),g((N+31)/32,(M+31)/32); matmul_reg2x2<<<g,b>>>(A,B,C,M,N,K); }
static void l_tuned(const float*A,const float*B,float*C,int M,int N,int K){ dim3 b(16,8),g((N+127)/128,(M+63)/64); matmul_tuned_fp32<64,128,32,8,16,8,8><<<g,b>>>(A,B,C,M,N,K); }

static void l_pipe_8x8(const float*A,const float*B,float*C,int M,int N,int K){
    // Step 5 win: 64x128, BK=16, 16x8 threads, 8x8 register tile.
    constexpr int bm=64,bn=128,bk=16,tx=16,ty=8,ry=8,rx=8;
    dim3 b(tx,ty),g((N+bn-1)/bn,(M+bm-1)/bm);
    size_t sh=2u*bm*bk*sizeof(float)+2u*bk*bn*sizeof(float);
    static bool init=false;
    if(!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh)); init=true; }
    matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx><<<g,b,sh>>>(A,B,C,M,N,K);
}
static void l_pipe_8x4(const float*A,const float*B,float*C,int M,int N,int K){
    constexpr int bm=64,bn=128,bk=32,tx=32,ty=8,ry=8,rx=4;
    dim3 b(tx,ty),g((N+bn-1)/bn,(M+bm-1)/bm);
    size_t sh=2u*bm*bk*sizeof(float)+2u*bk*bn*sizeof(float);
    static bool init=false;
    if(!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh)); init=true; }
    matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx><<<g,b,sh>>>(A,B,C,M,N,K);
}

int main(int argc,char** argv){
    if (argc < 5) { std::printf("usage: profile <M> <N> <K> <kernel> [launches]\n"); return 1; }
    int M=std::atoi(argv[1]), N=std::atoi(argv[2]), K=std::atoi(argv[3]);
    std::string kern = argv[4];
    int launches = argc>5?std::atoi(argv[5]):3;
    launch_fn fn=nullptr;
    if (kern=="naive_bad") fn=l_naive_bad;
    else if (kern=="naive") fn=l_naive;
    else if (kern=="tiled") fn=l_tiled;
    else if (kern=="reg2x2") fn=l_reg2x2;
    else if (kern=="tuned") fn=l_tuned;
    else if (kern=="pipe_8x8") fn=l_pipe_8x8;
    else if (kern=="pipe_8x4") fn=l_pipe_8x4;
    else { std::printf("unknown kernel %s\n",kern.c_str()); return 1; }

    size_t asz=(size_t)M*K,bsz=(size_t)K*N,csz=(size_t)M*N;
    std::vector<float> hA(asz),hB(bsz);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for (auto& x:hA) x=dist(rng); for (auto& x:hB) x=dist(rng);
    float *A,*B,*C;
    CUDA_CHECK(cudaMalloc(&A,asz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B,bsz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C,csz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A,hA.data(),asz*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B,hB.data(),bsz*sizeof(float),cudaMemcpyHostToDevice));
    for (int i=0;i<launches;++i){ fn(A,B,C,M,N,K); CUDA_CHECK(cudaGetLastError()); }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("profiled kernel=%s M=%d N=%d K=%d launches=%d\n",kern.c_str(),M,N,K,launches);
    CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(B)); CUDA_CHECK(cudaFree(C));
    return 0;
}
