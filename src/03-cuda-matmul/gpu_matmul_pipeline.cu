#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); std::exit(2);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuBLAS err %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); std::exit(2);} } while(0)

// ---------------------------------------------------------------------------
// Baseline: same tuned 64x128 / BK=32 / 16x8 native-FP32 kernel as the ladder.
// ---------------------------------------------------------------------------
template<int BM, int BN, int BK, int TX, int TY, int RY, int RX>
__global__ void matmul_tuned(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    static_assert(TX*TY <= 1024 && BM==TY*RY && BN==TX*RX, "shape");
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y*TX + threadIdx.x;
    int row0 = blockIdx.y*BM + threadIdx.y*RY;
    int col0 = blockIdx.x*BN + threadIdx.x*RX;
    float acc[RY][RX] = {};
    for (int kb=0; kb<K; kb+=BK) {
        for (int i=tid; i<BM*BK; i+=TX*TY) {
            int r=i/BK, k=i%BK, gr=blockIdx.y*BM+r, gk=kb+k;
            As[r][k] = (gr<M && gk<K)? A[gr*K+gk] : 0.0f;
        }
        for (int i=tid; i<BK*BN; i+=TX*TY) {
            int k=i/BN, c=i%BN, gk=kb+k, gc=blockIdx.x*BN+c;
            Bs[k][c] = (gk<K && gc<N)? B[gk*N+gc] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k=0;k<BK;++k){
            float a[RY], b[RX];
            #pragma unroll
            for (int i=0;i<RY;++i) a[i]=As[threadIdx.y*RY+i][k];
            #pragma unroll
            for (int j=0;j<RX;++j) b[j]=Bs[k][threadIdx.x*RX+j];
            #pragma unroll
            for (int i=0;i<RY;++i)
                #pragma unroll
                for (int j=0;j<RX;++j) acc[i][j]+=a[i]*b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0;i<RY;++i)
        #pragma unroll
        for (int j=0;j<RX;++j){
            int r=row0+i, c=col0+j;
            if (r<M && c<N) C[r*N+c]=acc[i][j];
        }
}

// ---------------------------------------------------------------------------
// Pipelined variant: double-buffered cp.async global->shared prefetch.
// One buffer holds the tile being computed; the next K-tile is fetched
// asynchronously into the other buffer while compute runs.
// ---------------------------------------------------------------------------
template<int BM, int BN, int BK, int TX, int TY, int RY, int RX, bool SWAP=false>
__global__ void matmul_pipe_dbl(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
    static_assert(TX*TY<=1024 && BM==TY*RY && BN==TX*RX, "shape");
    extern __shared__ float smem[];
    // Two 3D buffers laid out as [stage][BM][BK] and [stage][BK][BN].
    float (*As)[BM][BK] = reinterpret_cast<float(*)[BM][BK]>(smem);
    float (*Bs)[BK][BN] = reinterpret_cast<float(*)[BK][BN]>(smem + 2u*BM*BK);
    const int nthread = TX*TY;
    int tid = threadIdx.y*TX + threadIdx.x;
    // SWAP reverses the block-id mapping so consecutive blocks share a B tile
    // (different A rows) instead of an A tile. Changes the L2 reuse pattern.
    int by = SWAP ? blockIdx.x : blockIdx.y;
    int bx = SWAP ? blockIdx.y : blockIdx.x;
    int row0 = by*BM + threadIdx.y*RY;
    int col0 = bx*BN + threadIdx.x*RX;
    float acc[RY][RX] = {};

    auto load_stage = [&](int st, int kb) {
        constexpr int A4 = BM*BK/4;
        for (int v=tid; v<A4; v+=nthread) {
            int r = v/(BK/4), c=(v%(BK/4))*4;
            int gr = by*BM+r, gc = kb+c;
            if (gr<M && gc+3<K && ((gr*K+gc)&3)==0)
                __pipeline_memcpy_async(&As[st][r][c], &A[gr*K+gc], 16);
            else
                #pragma unroll
                for (int j=0;j<4;++j) As[st][r][c+j]=(gr<M && gc+j<K)?A[gr*K+gc+j]:0.0f;
        }
        constexpr int B4 = BK*BN/4;
        for (int v=tid; v<B4; v+=nthread) {
            int r = v/(BN/4), c=(v%(BN/4))*4;
            int gr = kb+r, gc = bx*BN+c;
            if (gr<K && gc+3<N && ((gr*N+gc)&3)==0)
                __pipeline_memcpy_async(&Bs[st][r][c], &B[gr*N+gc], 16);
            else
                #pragma unroll
                for (int j=0;j<4;++j) Bs[st][r][c+j]=(gr<K && gc+j<N)?B[gr*N+gc+j]:0.0f;
        }
    };

    // Prologue: fetch tile 0 into buffer 0.
    load_stage(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int st = 0;
    for (int kb=0; kb<K; kb+=BK) {
        int cur = st, nxt = st^1;
        if (kb+BK < K) {            // prefetch next tile into the other buffer
            load_stage(nxt, kb+BK);
            __pipeline_commit();
        }
        // Compute on the current buffer while the next tile is in flight.
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
        __pipeline_wait_prior(0);   // wait for the prefetched tile
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

// ---------------------------------------------------------------------------
// Pipelined + swizzled B layout. The B tile is stored transposed so warps read
// contiguous columns, and a XOR swizzle on the row index removes conflicts.
// ---------------------------------------------------------------------------
// (Implemented and measured separately; included if it wins.)

struct Result { std::string name; double ms; double tflops; double max_abs; double rms; };
static double median(std::vector<float> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

static Result bench(const std::string& name,
                    void (*launch)(const float*,const float*,float*,int,int,int),
                    const float* A,const float* B,float* C,const float* ref,
                    int M,int N,int K,int warmup,int iters){
    for (int i=0;i<warmup;++i) launch(A,B,C,M,N,K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times; times.reserve(iters);
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    for (int i=0;i<iters;++i){
        CUDA_CHECK(cudaEventRecord(s)); launch(A,B,C,M,N,K);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e)); times.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    std::vector<float> host((size_t)M*N);
    CUDA_CHECK(cudaMemcpy(host.data(),C,host.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double max_abs=0.0,sum_sq=0.0;
    if (ref) for (size_t i=0;i<host.size();++i){
        double d=(double)host[i]-ref[i]; max_abs=std::max(max_abs,std::abs(d)); sum_sq+=d*d;
    }
    double rms=ref?std::sqrt(sum_sq/host.size()):0.0;
    double ms=median(times),tf=(2.0*M*N*K)/(ms*1e9);
    return {name,ms,tf,max_abs,rms};
}

static const int BM=64, BN=128, BK=32, TX=16, TY=8, RY=8, RX=8;

static void launch_tuned(const float* A,const float* B,float* C,int M,int N,int K){
    dim3 block(TX,TY), grid((N+BN-1)/BN,(M+BM-1)/BM);
    matmul_tuned<BM,BN,BK,TX,TY,RY,RX><<<grid,block>>>(A,B,C,M,N,K);
}
static void launch_pipe(const float* A,const float* B,float* C,int M,int N,int K){
    dim3 block(TX,TY), grid((N+BN-1)/BN,(M+BM-1)/BM);
    size_t shmem = 2u*BM*BK*sizeof(float) + 2u*BK*BN*sizeof(float);
    static bool init=false;
    if (!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)); init=true; }
    matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX><<<grid,block,shmem>>>(A,B,C,M,N,K);
}
static void launch_pipe_swap(const float* A,const float* B,float* C,int M,int N,int K){
    dim3 block(TX,TY), grid((M+BM-1)/BM,(N+BN-1)/BN);  // swapped: x maps to tile-row
    size_t shmem = 2u*BM*BK*sizeof(float) + 2u*BK*BN*sizeof(float);
    static bool init=false;
    if (!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX,true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)); init=true; }
    matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX,true><<<grid,block,shmem>>>(A,B,C,M,N,K);
}

static void launch_pipe_bk16(const float* A,const float* B,float* C,int M,int N,int K){
    constexpr int bm=64,bn=128,bk=16,tx=16,ty=8,ry=8,rx=8;
    dim3 block(tx,ty), grid((N+bn-1)/bn,(M+bm-1)/bm);
    size_t shmem = 2u*bm*bk*sizeof(float) + 2u*bk*bn*sizeof(float);
    static bool init=false;
    if (!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)); init=true; }
    matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx><<<grid,block,shmem>>>(A,B,C,M,N,K);
}

static void launch_pipe_bk8(const float* A,const float* B,float* C,int M,int N,int K){
    constexpr int bm=64,bn=128,bk=8,tx=16,ty=8,ry=8,rx=8;
    dim3 block(tx,ty), grid((N+bn-1)/bn,(M+bm-1)/bm);
    size_t shmem = 2u*bm*bk*sizeof(float) + 2u*bk*bn*sizeof(float);
    static bool init=false;
    if (!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)); init=true; }
    matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx><<<grid,block,shmem>>>(A,B,C,M,N,K);
}

// Triple-buffered cp.async pipeline: prefetches two K-tiles ahead, splitting
// the load latency across two KJUN-stage frontier instead of one.
template<int BM, int BN, int BK, int TX, int TY, int RY, int RX>
__global__ void matmul_pipe_tri(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
    static_assert(TX*TY<=1024 && BM==TY*RY && BN==TX*RX, "shape");
    extern __shared__ float smem[];
    float (*As)[BM][BK] = reinterpret_cast<float(*)[BM][BK]>(smem);
    float (*Bs)[BK][BN] = reinterpret_cast<float(*)[BK][BN]>(smem + 3u*BM*BK);
    const int nthread = TX*TY;
    int tid = threadIdx.y*TX + threadIdx.x;
    int row0 = blockIdx.y*BM + threadIdx.y*RY;
    int col0 = blockIdx.x*BN + threadIdx.x*RX;
    float acc[RY][RX] = {};

    auto load_stage = [&](int st, int kb) {
        constexpr int A4 = BM*BK/4;
        for (int v=tid; v<A4; v+=nthread) {
            int r = v/(BK/4), c=(v%(BK/4))*4;
            int gr = blockIdx.y*BM+r, gc = kb+c;
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

    // Prologue: fetch tiles 0 and BK into buffers 0 and 1.
    load_stage(0, 0); __pipeline_commit();
    load_stage(1, BK); __pipeline_commit();
    __pipeline_wait_prior(0); __syncthreads();

    int st = 0;
    for (int kb=0; kb<K; kb+=BK) {
        if (kb+2*BK < K) { load_stage((st+2)%3, kb+2*BK); __pipeline_commit(); }
        #pragma unroll
        for (int k=0;k<BK;++k){
            float a[RY], b[RX];
            #pragma unroll
            for (int i=0;i<RY;++i) a[i]=As[st][threadIdx.y*RY+i][k];
            #pragma unroll
            for (int j=0;j<RX;++j) b[j]=Bs[st][k][threadIdx.x*RX+j];
            #pragma unroll
            for (int i=0;i<RY;++i)
                #pragma unroll
                for (int j=0;j<RX;++j) acc[i][j]+=a[i]*b[j];
        }
        __pipeline_wait_prior(1);  // leave the just-issued prefetch in flight
        __syncthreads();
        st = (st+1)%3;
    }
    #pragma unroll
    for (int i=0;i<RY;++i)
        #pragma unroll
        for (int j=0;j<RX;++j){
            int r=row0+i, c=col0+j;
            if (r<M && c<N) C[r*N+c]=acc[i][j];
        }
}

static void launch_pipe_tri_bk16(const float* A,const float* B,float* C,int M,int N,int K){
    constexpr int bm=64,bn=128,bk=16,tx=16,ty=8,ry=8,rx=8;
    dim3 block(tx,ty), grid((N+bn-1)/bn,(M+bm-1)/bm);
    size_t shmem = 3u*bm*bk*sizeof(float) + 3u*bk*bn*sizeof(float);
    static bool init=false;
    if (!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_tri<bm,bn,bk,tx,ty,ry,rx>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)); init=true; }
    matmul_pipe_tri<bm,bn,bk,tx,ty,ry,rx><<<grid,block,shmem>>>(A,B,C,M,N,K);
}

// Winner: 64x128, BK=32, 8x4 register tile, 256 threads.
static void launch_pipe_8x4_bk32(const float* A,const float* B,float* C,int M,int N,int K){
    constexpr int bm=64,bn=128,bk=32,tx=32,ty=8,ry=8,rx=4;
    dim3 block(tx,ty), grid((N+bn-1)/bn,(M+bm-1)/bm);
    size_t shmem = 2u*bm*bk*sizeof(float) + 2u*bk*bn*sizeof(float);
    static bool init=false;
    if (!init){ CUDA_CHECK(cudaFuncSetAttribute(matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)); init=true; }
    matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx><<<grid,block,shmem>>>(A,B,C,M,N,K);
}

static void print_occupancy(){
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,(const void*)matmul_tuned<BM,BN,BK,TX,TY,RY,RX>));
    int blocks; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,(const void*)matmul_tuned<BM,BN,BK,TX,TY,RY,RX>,TX*TY,0));
    std::printf("tuned:   regs=%d shared=%zu blocks/SM=%d warps/SM=%d\n",attr.numRegs,attr.sharedSizeBytes,blocks,blocks*(TX*TY/32));
    CUDA_CHECK(cudaFuncGetAttributes(&attr,(const void*)matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>));
    size_t shmem = 2u*BM*BK*sizeof(float) + 2u*BK*BN*sizeof(float);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,(const void*)matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>,TX*TY,shmem));
    std::printf("pipe:    regs=%d shared=%zu blocks/SM=%d warps/SM=%d\n",attr.numRegs,shmem,blocks,blocks*(TX*TY/32));
}

int main(int argc,char** argv){
    int M=argc>1?std::atoi(argv[1]):2048;
    int N=argc>2?std::atoi(argv[2]):M;
    int K=argc>3?std::atoi(argv[3]):M;
    int iters=argc>4?std::atoi(argv[4]):50;
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::printf("GPU: %s | SM %d.%d | %.1f GiB\n",prop.name,prop.major,prop.minor,prop.totalGlobalMem/1e9);
    std::printf("Shape: M=%d N=%d K=%d | iterations=%d\n",M,N,K,iters);

    size_t asz=(size_t)M*K,bsz=(size_t)K*N,csz=(size_t)M*N;
    std::vector<float> hA(asz),hB(bsz),hRef(csz);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for (auto& x:hA) x=dist(rng); for (auto& x:hB) x=dist(rng);

    float *A,*B,*C;
    CUDA_CHECK(cudaMalloc(&A,asz*sizeof(float))); CUDA_CHECK(cudaMalloc(&B,bsz*sizeof(float))); CUDA_CHECK(cudaMalloc(&C,csz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A,hA.data(),asz*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B,hB.data(),bsz*sizeof(float),cudaMemcpyHostToDevice));

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)); // guaranteed true FP32 ref
    const float alpha=1.0f,beta=0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(),C,csz*sizeof(float),cudaMemcpyDeviceToHost));

    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    for(int i=0;i<20;++i) CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N));
    CUDA_CHECK(cudaEventRecord(s));
    for(int i=0;i<iters;++i) CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N));
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float cbl_ms; CUDA_CHECK(cudaEventElapsedTime(&cbl_ms,s,e)); cbl_ms/=iters;
    std::printf("cuBLAS(fp32 pedantic) %8.3f ms  %8.2f TFLOP/s\n",cbl_ms,(2.0*M*N*K)/(cbl_ms*1e9));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));

    std::printf("%s\n","--- occupancy ---");
    print_occupancy();
    std::printf("%-28s %9s %9s %10s  %10s\n","kernel","ms","TFLOP/s","max_abs","rms");
    for (const auto& r : {
        bench("tuned_64x128_bk32_16x8", launch_tuned,A,B,C,hRef.data(),M,N,K,10,iters),
        bench("pipe_dbl_64x128_bk32_16x8", launch_pipe,A,B,C,hRef.data(),M,N,K,10,iters),
        bench("pipe_dbl_64x128_bk32_16x8_L2swap", launch_pipe_swap,A,B,C,hRef.data(),M,N,K,10,iters),
        bench("pipe_dbl_64x128_bk16_16x8", launch_pipe_bk16,A,B,C,hRef.data(),M,N,K,10,iters),
        bench("pipe_dbl_64x128_bk32_8x4", launch_pipe_8x4_bk32,A,B,C,hRef.data(),M,N,K,10,iters),
        bench("pipe_dbl_64x128_bk8_16x8", launch_pipe_bk8,A,B,C,hRef.data(),M,N,K,10,iters),
        bench("pipe_tri_64x128_bk16_16x8", launch_pipe_tri_bk16,A,B,C,hRef.data(),M,N,K,10,iters)
    }) std::printf("%-28s %9.3f %9.2f %10.3e  %10.3e\n",r.name.c_str(),r.ms,r.tflops,r.max_abs,r.rms);

    CUBLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(B)); CUDA_CHECK(cudaFree(C));
    return 0;
}
