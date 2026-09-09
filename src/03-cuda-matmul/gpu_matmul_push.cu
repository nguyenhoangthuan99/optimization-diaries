#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){std::fprintf(stderr,"err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));std::exit(2);} } while(0)

// ---------------------------------------------------------------------------
// Baseline: exact copy of the best pipelined kernel (64x128, BK=16, 16x8, 8x8).
// ---------------------------------------------------------------------------
template<int BM,int BN,int BK,int TX,int TY,int RY,int RX>
__global__ void matmul_pipe_dbl(const float* __restrict__ A,const float* __restrict__ B,
                                float* __restrict__ C,int M,int N,int K){
    static_assert(TX*TY<=1024 && BM==TY*RY && BN==TX*RX,"shape");
    extern __shared__ float smem[];
    float (*As)[BM][BK]=reinterpret_cast<float(*)[BM][BK]>(smem);
    float (*Bs)[BK][BN]=reinterpret_cast<float(*)[BK][BN]>(smem+2u*BM*BK);
    const int nthread=TX*TY; int tid=threadIdx.y*TX+threadIdx.x;
    int row0=blockIdx.y*BM+threadIdx.y*RY, col0=blockIdx.x*BN+threadIdx.x*RX;
    float acc[RY][RX]={};
    auto load_stage=[&](int st,int kb){
        constexpr int A4=BM*BK/4;
        for(int v=tid;v<A4;v+=nthread){int r=v/(BK/4),c=(v%(BK/4))*4;int gr=blockIdx.y*BM+r,gc=kb+c;
            if(gr<M&&gc+3<K&&((gr*K+gc)&3)==0) __pipeline_memcpy_async(&As[st][r][c],&A[gr*K+gc],16);
            else for(int j=0;j<4;++j) As[st][r][c+j]=(gr<M&&gc+j<K)?A[gr*K+gc+j]:0.0f;}
        constexpr int B4=BK*BN/4;
        for(int v=tid;v<B4;v+=nthread){int r=v/(BN/4),c=(v%(BN/4))*4;int gr=kb+r,gc=blockIdx.x*BN+c;
            if(gr<K&&gc+3<N&&((gr*N+gc)&3)==0) __pipeline_memcpy_async(&Bs[st][r][c],&B[gr*N+gc],16);
            else for(int j=0;j<4;++j) Bs[st][r][c+j]=(gr<K&&gc+j<N)?B[gr*N+gc+j]:0.0f;}
    };
    load_stage(0,0); __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();
    int st=0;
    for(int kb=0;kb<K;kb+=BK){int cur=st,nxt=st^1;
        if(kb+BK<K){load_stage(nxt,kb+BK); __pipeline_commit();}
        #pragma unroll
        for(int k=0;k<BK;++k){float a[RY],b[RX];
            #pragma unroll
            for(int i=0;i<RY;++i) a[i]=As[cur][threadIdx.y*RY+i][k];
            #pragma unroll
            for(int j=0;j<RX;++j) b[j]=Bs[cur][k][threadIdx.x*RX+j];
            #pragma unroll
            for(int i=0;i<RY;++i)
                #pragma unroll
                for(int j=0;j<RX;++j) acc[i][j]+=a[i]*b[j];}
        __pipeline_wait_prior(0); __syncthreads(); st=nxt;}
    for(int i=0;i<RY;++i) for(int j=0;j<RX;++j){int r=row0+i,c=col0+j; if(r<M&&c<N) C[r*N+c]=acc[i][j];}
}

// ---------------------------------------------------------------------------
// Variant harness.
//   PAD_A   = pad the A row stride by one (fixes the 2-way As bank conflict)
//   SWIZB   = XOR-swizzle the B column index (fixes the 4-way Bs conflict)
//   MINBLKS = __launch_bounds__ min blocks/SM (0 = no constraint)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int swiz(int c){ return c ^ ((c>>5)&3); }

template<int BM,int BN,int BK,int TX,int TY,int RY,int RX,int PAD_A,int SWIZB,int MINBLKS>
__global__ void __launch_bounds__(TX*TY, MINBLKS)
matmul_push(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,
            int M,int N,int K){
    static_assert(TX*TY<=1024 && BM==TY*RY && BN==TX*RX,"shape");
    constexpr int AP = PAD_A ? BK+1 : BK;
    extern __shared__ float smem[];
    float (*As)[BM][AP]=reinterpret_cast<float(*)[BM][AP]>(smem);
    float (*Bs)[BK][BN]=reinterpret_cast<float(*)[BK][BN]>(smem+2u*BM*AP);
    const int nthread=TX*TY; int tid=threadIdx.y*TX+threadIdx.x;
    int row0=blockIdx.y*BM+threadIdx.y*RY, col0=blockIdx.x*BN+threadIdx.x*RX;
    float acc[RY][RX]={};
    auto load_stage=[&](int st,int kb){
        if (PAD_A) {
            // scalar (4-byte) async copies so the padded row stride (odd) needs no 16B alignment
            constexpr int Asz=BM*BK;
            #pragma unroll
            for(int v=tid;v<Asz;v+=nthread){int r=v/BK,c=v%BK;int gr=blockIdx.y*BM+r,gc=kb+c;
                if(gr<M&&gc<K) __pipeline_memcpy_async(&As[st][r][c],&A[gr*K+gc],4); else As[st][r][c]=0.0f;}
        } else {
        constexpr int A4=BM*BK/4;
        #pragma unroll
        for(int v=tid;v<A4;v+=nthread){int r=v/(BK/4),c=(v%(BK/4))*4;int gr=blockIdx.y*BM+r,gc=kb+c;
            if(gr<M&&gc+3<K&&((gr*K+gc)&3)==0) __pipeline_memcpy_async(&As[st][r][c],&A[gr*K+gc],16);
            else for(int j=0;j<4;++j) As[st][r][c+j]=(gr<M&&gc+j<K)?A[gr*K+gc+j]:0.0f;}
        }
        if (SWIZB) {
            constexpr int Bsz=BK*BN;
            #pragma unroll
            for(int v=tid;v<Bsz;v+=nthread){int r=v/BN,c=v%BN;int gr=kb+r,gc=blockIdx.x*BN+c;
                // scalar (4-byte) async copy into the swizzled column
                if(gr<K&&gc<N) __pipeline_memcpy_async(&Bs[st][r][swiz(c)],&B[gr*N+gc],4);
                else Bs[st][r][swiz(c)]=0.0f;}
        } else {
            constexpr int B4=BK*BN/4;
            #pragma unroll
            for(int v=tid;v<B4;v+=nthread){int r=v/(BN/4),c=(v%(BN/4))*4;int gr=kb+r,gc=blockIdx.x*BN+c;
                if(gr<K&&gc+3<N&&((gr*N+gc)&3)==0) __pipeline_memcpy_async(&Bs[st][r][c],&B[gr*N+gc],16);
                else for(int j=0;j<4;++j) Bs[st][r][c+j]=(gr<K&&gc+j<N)?B[gr*N+gc+j]:0.0f;}
        }
    };
    load_stage(0,0); __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();
    int st=0;
    for(int kb=0;kb<K;kb+=BK){int cur=st,nxt=st^1;
        if(kb+BK<K){load_stage(nxt,kb+BK); __pipeline_commit();}
        #pragma unroll
        for(int k=0;k<BK;++k){float a[RY],b[RX];
            #pragma unroll
            for(int i=0;i<RY;++i) a[i]=As[cur][threadIdx.y*RY+i][k];
            #pragma unroll
            for(int j=0;j<RX;++j) b[j]=Bs[cur][k][SWIZB?swiz(threadIdx.x*RX+j):(threadIdx.x*RX+j)];
            #pragma unroll
            for(int i=0;i<RY;++i)
                #pragma unroll
                for(int j=0;j<RX;++j) acc[i][j]+=a[i]*b[j];}
        __pipeline_wait_prior(0); __syncthreads(); st=nxt;}
    for(int i=0;i<RY;++i) for(int j=0;j<RX;++j){int r=row0+i,c=col0+j; if(r<M&&c<N) C[r*N+c]=acc[i][j];}
}

// ---------------------------------------------------------------------------
struct Result{std::string name;double ms,tflops,max_abs,rms;};
static double median(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

static void bench_into(std::vector<Result>& out,const std::string& name,
                       void(*launch)(const float*,const float*,float*,int,int,int),
                       const float*A,const float*B,float*C,const float*ref,int M,int N,int K,int it){
    for(int i=0;i<10;++i) launch(A,B,C,M,N,K); CK(cudaDeviceSynchronize());
    std::vector<float> t; t.reserve(it); cudaEvent_t s,e; CK(cudaEventCreate(&s));CK(cudaEventCreate(&e));
    for(int i=0;i<it;++i){CK(cudaEventRecord(s));launch(A,B,C,M,N,K);CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,s,e));t.push_back(ms);}
    CK(cudaEventDestroy(s));CK(cudaEventDestroy(e));
    std::vector<float> host((size_t)M*N); CK(cudaMemcpy(host.data(),C,host.size()*4,cudaMemcpyDeviceToHost));
    double ma=0,ss=0; for(size_t i=0;i<host.size();++i){double d=(double)host[i]-ref[i];ma=std::max(ma,std::abs(d));ss+=d*d;}
    double ms=median(t); out.push_back({name,ms,(2.0*M*N*K)/(ms*1e9),ma,std::sqrt(ss/host.size())});
}

struct Cfg{const char* name;int pad;int swiz;int minb;};
static void run(const Cfg& c,const float*A,const float*B,float*C,const float*ref,int M,int N,int K,int it,const char*label){
    std::vector<Result> out;
    #define INST(PAD,SWIZ,MINB, NAMELBL) do{ \
        constexpr int BM=64,BN=128,BK=16,TX=16,TY=8,RY=8,RX=8; \
        auto launch=[](const float*A_,const float*B_,float*C_,int M_,int N_,int K_){ \
            dim3 blk(TX,TY),g((N_+BN-1)/BN,(M_+BM-1)/BM); \
            size_t sh=2u*BM*(PAD?BK+1:BK)*4+2u*BK*BN*4; \
            static bool init=false; if(!init){CK(cudaFuncSetAttribute(matmul_push<BM,BN,BK,TX,TY,RY,RX,PAD,SWIZ,MINB>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));init=true;} \
            matmul_push<BM,BN,BK,TX,TY,RY,RX,PAD,SWIZ,MINB><<<g,blk,sh>>>(A_,B_,C_,M_,N_,K_);}; \
        bench_into(out, label, launch, A,B,C,ref,M,N,K,it); \
    }while(0)
    if(c.pad==0&&c.swiz==0&&c.minb==0){
        // clean baseline (129 regs, natural occupancy)
        constexpr int BM=64,BN=128,BK=16,TX=16,TY=8,RY=8,RX=8;
        auto launch=[](const float*A_,const float*B_,float*C_,int M_,int N_,int K_){
            dim3 blk(TX,TY),g((N_+BN-1)/BN,(M_+BM-1)/BM);size_t sh=2u*BM*BK*4+2u*BK*BN*4;
            static bool init=false;if(!init){CK(cudaFuncSetAttribute(matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));init=true;}
            matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX><<<g,blk,sh>>>(A_,B_,C_,M_,N_,K_);};
        bench_into(out,"base",launch,A,B,C,ref,M,N,K,it);
    }
    else if(c.pad==1&&c.swiz==0&&c.minb==3) INST(1,0,3,"padA");
    else if(c.pad==0&&c.swiz==1&&c.minb==3) INST(0,1,3,"swizB");
    else if(c.pad==1&&c.swiz==1&&c.minb==3) INST(1,1,3,"padA+swizB");
    else if(c.pad==0&&c.swiz==0&&c.minb==4) INST(0,0,4,"bmin4");
    else if(c.pad==0&&c.swiz==0&&c.minb==5) INST(0,0,5,"bmin5");
    else if(c.pad==0&&c.swiz==0&&c.minb==6) INST(0,0,6,"bmin6");
    else if(c.pad==1&&c.swiz==0&&c.minb==4) INST(1,0,4,"padA+bmin4");
    else if(c.pad==1&&c.swiz==1&&c.minb==4) INST(1,1,4,"padA+swizB+bmin4");
    else INST(0,0,3,"padA+swizB+base");
    for(auto&r:out) std::printf("%-22s %9.3f %9.2f %10.3e %10.3e\n",r.name.c_str(),r.ms,r.tflops,r.max_abs,r.rms);
    #undef INST
}

int main(int argc,char**argv){
    int M=argc>1?std::atoi(argv[1]):2048,N=argc>2?std::atoi(argv[2]):M,K=argc>3?std::atoi(argv[3]):M,it=argc>4?std::atoi(argv[4]):20;
    cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,0));
    std::printf("GPU %s SM %d.%d | %dx%dx%d iters=%d\n",p.name,p.major,p.minor,M,N,K,it);
    size_t asz=(size_t)M*K,bsz=(size_t)K*N,csz=(size_t)M*N;
    std::vector<float>hA(asz),hB(bsz),hRef(csz);std::mt19937 rng(42);std::uniform_real_distribution<float> d(-1,1);
    for(auto&x:hA)x=d(rng);for(auto&x:hB)x=d(rng);
    float*A,*B,*C;CK(cudaMalloc(&A,asz*4));CK(cudaMalloc(&B,bsz*4));CK(cudaMalloc(&C,csz*4));
    CK(cudaMemcpy(A,hA.data(),asz*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(B,hB.data(),bsz*4,cudaMemcpyHostToDevice));
    // reference (CPU equivalent via a straightforward device matmul is overkill; use tensor-core-free ref via cublas pedantic)
    // We rely on the device matmul_pipe_dbl (already validated) producing the reference, then compare variants.
    {constexpr int BM=64,BN=128,BK=16,TX=16,TY=8,RY=8,RX=8;
     const float alpha=1.0f; /* not using cublas; use host-generate via random * random won't validate */ }
    // Build reference on device with the baseline kernel.
    {constexpr int BM=64,BN=128,BK=16,TX=16,TY=8,RY=8,RX=8;
     dim3 blk(TX,TY),g((N+BN-1)/BN,(M+BM-1)/BM);size_t sh=2u*BM*BK*4+2u*BK*BN*4;
     static bool init=false;if(!init){CK(cudaFuncSetAttribute(matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));init=true;}
     for(int i=0;i<3;++i) matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX><<<g,blk,sh>>>(A,B,C,M,N,K);CK(cudaDeviceSynchronize());}
    CK(cudaMemcpy(hRef.data(),C,csz*4,cudaMemcpyDeviceToHost));

    const Cfg cfgs[]={
        {"base",0,0,0},{"padA",1,0,3},{"swizB",0,1,3},{"padA+swizB",1,1,3},
        {"bmin4",0,0,4},{"bmin5",0,0,5},{"bmin6",0,0,6},
        {"padA+bmin4",1,0,4},{"padA+swizB+bmin4",1,1,4}
    };
    for(auto&c:cfgs){std::printf("--- %s ---\n",c.name);run(c,A,B,C,hRef.data(),M,N,K,it,c.name);}
    return 0;
}
