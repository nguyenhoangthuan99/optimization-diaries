#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){std::fprintf(stderr,"err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));std::exit(2);} } while(0)

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
        #pragma unroll
        for(int v=tid;v<A4;v+=nthread){int r=v/(BK/4),c=(v%(BK/4))*4;int gr=blockIdx.y*BM+r,gc=kb+c;
            if(gr<M&&gc+3<K&&((gr*K+gc)&3)==0) __pipeline_memcpy_async(&As[st][r][c],&A[gr*K+gc],16);
            else for(int j=0;j<4;++j) As[st][r][c+j]=(gr<M&&gc+j<K)?A[gr*K+gc+j]:0.0f;}
        constexpr int B4=BK*BN/4;
        #pragma unroll
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

// Vectorized-B variant: when RX%4==0, load the B fragment as a single float4.
template<int BM,int BN,int BK,int TX,int TY,int RY,int RX>
__global__ void matmul_pipe_vec(const float* __restrict__ A,const float* __restrict__ B,
                                float* __restrict__ C,int M,int N,int K){
    static_assert(TX*TY<=1024 && BM==TY*RY && BN==TX*RX && RX%4==0,"shape");
    extern __shared__ float smem[];
    float (*As)[BM][BK]=reinterpret_cast<float(*)[BM][BK]>(smem);
    float (*Bs)[BK][BN]=reinterpret_cast<float(*)[BK][BN]>(smem+2u*BM*BK);
    const int nthread=TX*TY; int tid=threadIdx.y*TX+threadIdx.x;
    int row0=blockIdx.y*BM+threadIdx.y*RY, col0=blockIdx.x*BN+threadIdx.x*RX;
    float acc[RY][RX]={};
    auto load_stage=[&](int st,int kb){
        constexpr int A4=BM*BK/4;
        #pragma unroll
        for(int v=tid;v<A4;v+=nthread){int r=v/(BK/4),c=(v%(BK/4))*4;int gr=blockIdx.y*BM+r,gc=kb+c;
            if(gr<M&&gc+3<K&&((gr*K+gc)&3)==0) __pipeline_memcpy_async(&As[st][r][c],&A[gr*K+gc],16);
            else for(int j=0;j<4;++j) As[st][r][c+j]=(gr<M&&gc+j<K)?A[gr*K+gc+j]:0.0f;}
        constexpr int B4=BK*BN/4;
        #pragma unroll
        for(int v=tid;v<B4;v+=nthread){int r=v/(BN/4),c=(v%(BN/4))*4;int gr=kb+r,gc=blockIdx.x*BN+c;
            if(gr<K&&gc+3<N&&((gr*N+gc)&3)==0) __pipeline_memcpy_async(&Bs[st][r][c],&B[gr*N+gc],16);
            else for(int j=0;j<4;++j) Bs[st][r][c+j]=(gr<K&&gc+j<N)?B[gr*N+gc+j]:0.0f;}
    };
    load_stage(0,0); __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();
    int st=0;
    for(int kb=0;kb<K;kb+=BK){int cur=st,nxt=st^1;
        if(kb+BK<K){load_stage(nxt,kb+BK); __pipeline_commit();}
        #pragma unroll
        for(int k=0;k<BK;++k){float a[RY];
            #pragma unroll
            for(int i=0;i<RY;++i) a[i]=As[cur][threadIdx.y*RY+i][k];
            const float4 bv=*reinterpret_cast<const float4*>(&Bs[cur][k][threadIdx.x*RX]);
            #pragma unroll
            for(int i=0;i<RY;++i){acc[i][0]+=a[i]*bv.x;acc[i][1]+=a[i]*bv.y;acc[i][2]+=a[i]*bv.z;acc[i][3]+=a[i]*bv.w;}
        }
        __pipeline_wait_prior(0); __syncthreads(); st=nxt;}
    for(int i=0;i<RY;++i) for(int j=0;j<RX;++j){int r=row0+i,c=col0+j; if(r<M&&c<N) C[r*N+c]=acc[i][j];}
}

template<int BM,int BN,int BK,int TX,int TY,int RY,int RX>
void instvec(const char*name,const float*A,const float*B,float*C,const float*ref,int M,int N,int K,int it){
    auto l=[](const float*A,const float*B,float*C,int M,int N,int K){ \
        dim3 blk(TX,TY),g((N+BN-1)/BN,(M+BM-1)/BM);size_t sh=2u*BM*BK*4+2u*BK*BN*4; \
        static bool ini=false;if(!ini){CK(cudaFuncSetAttribute(matmul_pipe_vec<BM,BN,BK,TX,TY,RY,RX>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));ini=true;} \
        matmul_pipe_vec<BM,BN,BK,TX,TY,RY,RX><<<g,blk,sh>>>(A,B,C,M,N,K);}; \
    run_one(name,l,A,B,C,ref,M,N,K,it);
}

template<int BM,int BN,int BK,int TX,int TY,int RY,int RX>
void inst(const char*name,const float*A,const float*B,float*C,const float*ref,int M,int N,int K,int it){
    auto l=[](const float*A,const float*B,float*C,int M,int N,int K){ \
        dim3 blk(TX,TY),g((N+BN-1)/BN,(M+BM-1)/BM);size_t sh=2u*BM*BK*4+2u*BK*BN*4; \
        static bool ini=false;if(!ini){CK(cudaFuncSetAttribute(matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));ini=true;} \
        matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX><<<g,blk,sh>>>(A,B,C,M,N,K);}; \
    run_one(name,l,A,B,C,ref,M,N,K,it);
    // occupancy
    cudaFuncAttributes at;CK(cudaFuncGetAttributes(&at,(const void*)matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>));
    int blocks; size_t sh=2u*BM*BK*4+2u*BK*BN*4;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,(const void*)matmul_pipe_dbl<BM,BN,BK,TX,TY,RY,RX>,TX*TY,sh));
    std::printf("   [occ] regs=%d shared=%zu blocks/SM=%d warps/SM=%d\n",at.numRegs,sh,blocks,blocks*(TX*TY/32));
}

static void run_one(const char*name,void(*l)(const float*,const float*,float*,int,int,int),
                    const float*A,const float*B,float*C,const float*ref,int M,int N,int K,int it){
    for(int i=0;i<10;++i)l(A,B,C,M,N,K);CK(cudaDeviceSynchronize());
    std::vector<float>t;cudaEvent_t s,e;CK(cudaEventCreate(&s));CK(cudaEventCreate(&e));
    for(int i=0;i<it;++i){CK(cudaEventRecord(s));l(A,B,C,M,N,K);CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,s,e));t.push_back(ms);}
    std::sort(t.begin(),t.end());double ms=t[t.size()/2];
    std::vector<float> h((size_t)M*N);CK(cudaMemcpy(h.data(),C,h.size()*4,cudaMemcpyDeviceToHost));
    double ma=0;for(size_t i=0;i<h.size();++i){double d=(double)h[i]-ref[i];ma=std::max(ma,std::abs(d));}
    std::printf("%-16s %9.3f %8.2f max_abs %10.3e\n",name,ms,(2.0*M*N*K)/(ms*1e9),ma);
}

int main(int argc,char**argv){
    int M=argc>1?std::atoi(argv[1]):2048,N=argc>2?std::atoi(argv[2]):M,K=argc>3?std::atoi(argv[3]):M,it=argc>4?std::atoi(argv[4]):20;
    cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,0));std::printf("GPU %s SM %d.%d | %dx%dx%d\n",p.name,p.major,p.minor,M,N,K);
    size_t asz=(size_t)M*K,bsz=(size_t)K*N,csz=(size_t)M*N;
    std::vector<float>hA(asz),hB(bsz),hRef(csz);std::mt19937 rng(42);std::uniform_real_distribution<float> d(-1,1);
    for(auto&x:hA)x=d(rng);for(auto&x:hB)x=d(rng);
    float*A,*B,*C;CK(cudaMalloc(&A,asz*4));CK(cudaMalloc(&B,bsz*4));CK(cudaMalloc(&C,csz*4));
    CK(cudaMemcpy(A,hA.data(),asz*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(B,hB.data(),bsz*4,cudaMemcpyHostToDevice));
    // reference from base config's output
    {constexpr int bm=64,bn=128,bk=16,tx=16,ty=8,ry=8,rx=8;dim3 blk(tx,ty),g((N+bn-1)/bn,(M+bm-1)/bm);size_t sh=2u*bm*bk*4+2u*bk*bn*4;
     static bool ini=false;if(!ini){CK(cudaFuncSetAttribute(matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx>,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));ini=true;}
     for(int i=0;i<3;++i)matmul_pipe_dbl<bm,bn,bk,tx,ty,ry,rx><<<g,blk,sh>>>(A,B,C,M,N,K);CK(cudaDeviceSynchronize());}
    CK(cudaMemcpy(hRef.data(),C,csz*4,cudaMemcpyDeviceToHost));

    inst<64,128,16,16,8,8,8>("64x128_b16_8x8",A,B,C,hRef.data(),M,N,K,it);
    inst<64,128,16,16,8,8,8>("64x128_b16_8x8",A,B,C,hRef.data(),M,N,K,it);
    inst<64,128,16,32,8,8,4>("64x128_b16_8x4",A,B,C,hRef.data(),M,N,K,it);
    inst<64,128,8,32,8,8,4>("64x128_b8_8x4",A,B,C,hRef.data(),M,N,K,it);
    inst<64,128,32,32,8,8,4>("b32_8x4",A,B,C,hRef.data(),M,N,K,it);
    // ---- cuBLAS FP32 (pedantic) reference + final apples-to-apples ----
    cublasHandle_t h;cublasCreate(&h);cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH);
    const float alpha=1.0f,beta=0.0f;
    auto cublas_ms=[&](){cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
        for(int i=0;i<10;++i)cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N);
        cudaDeviceSynchronize();cudaEventRecord(s);
        for(int i=0;i<it;++i)cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N);
        cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);
        cudaEventDestroy(s);cudaEventDestroy(e);return ms/it;};
    double cbl=cublas_ms();
    std::printf("cuBLAS_fp32_pedantic   %9.3f %8.2f TFLOP/s\n",cbl,(2.0*M*N*K)/(cbl*1e9));
    return 0;
}
