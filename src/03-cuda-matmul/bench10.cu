
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels/10_kernel_warptiling.cuh"
#define CK(x) do{cudaError_t _e=(x);if(_e!=cudaSuccess){printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(2);}}while(0)
int M=4096,N=4096,K=4096;
float *hA,*hB,*hRef,*hOut; float *dA,*dB,*dC;
template<int BM,int BN,int BK,int WM,int WN,int WNITER,int TM,int TN,int NT>
void bench(const char*name){
    dim3 grid(CEIL_DIV(N,BN),CEIL_DIV(M,BM)); dim3 blk(NT);
    for(int i=0;i<3;++i) sgemmWarptiling<BM,BN,BK,WM,WN,WNITER,TM,TN,NT><<<grid,blk>>>(M,N,K,1.f,dA,dB,0.f,dC);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hOut,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    double ma=0; for(int i=0;i<M*N;++i){double d=std::abs((double)hOut[i]-hRef[i]); if(d>ma)ma=d;}
    if(ma>0.05){ printf("CFG %-34s FAIL ma=%.2e\n",name,ma); return; }
    std::vector<double> t; cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e);
    for(int it=0;it<20;++it){ cudaEventRecord(s); sgemmWarptiling<BM,BN,BK,WM,WN,WNITER,TM,TN,NT><<<grid,blk>>>(M,N,K,1.f,dA,dB,0.f,dC); cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e); t.push_back(ms);}
    std::sort(t.begin(),t.end()); double ms=t[t.size()/2];
    double tf=2.0*(double)M*N*K/(ms*1e-3)/1e12;
    printf("CFG %-34s ma=%.1e %7.3f ms  %7.2f TFLOP/s\n",name,ma,ms,tf);
}
int main(int argc,char**argv){
    if(argc>1) M=N=K=atoi(argv[1]);
    hA=(float*)malloc((size_t)M*K*4); hB=(float*)malloc((size_t)K*N*4); hRef=(float*)malloc((size_t)M*N*4); hOut=(float*)malloc((size_t)M*N*4);
    srand(42); for(int i=0;i<M*K;++i) hA[i]=((rand()%5)+0.01f*(rand()%5)); for(int i=0;i<K*N;++i) hB[i]=((rand()%5)+0.01f*(rand()%5));
    CK(cudaMalloc(&dA,(size_t)M*K*4)); CK(cudaMalloc(&dB,(size_t)K*N*4)); CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemcpy(dA,hA,(size_t)M*K*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,hB,(size_t)K*N*4,cudaMemcpyHostToDevice));
    cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH);
    float alpha=1.f,beta=0.f;
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC,N);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hRef,dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    printf("size=%d ref=cuBLAS pedantic FP32\n",M);
    cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e); std::vector<double> t;
    for(int it=0;it<20;++it){ cudaEventRecord(s); cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC,N); cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e); t.push_back(ms);}
    std::sort(t.begin(),t.end()); double cms=t[t.size()/2];
    printf("CUBLAS pedantic FP32  %7.3f ms  %7.2f TFLOP/s\n",cms,2.0*(double)M*N*K/(cms*1e-3)/1e12);

    bench<128,128,16,32,64,1,4,4,256>("BM128_BN128_BK16_WM32_WN64_WI1_TM4_TN4_NT256");
    bench<128,128,16,32,64,1,4,8,256>("BM128_BN128_BK16_WM32_WN64_WI1_TM4_TN8_NT256");
    bench<128,128,16,32,64,1,8,4,256>("BM128_BN128_BK16_WM32_WN64_WI1_TM8_TN4_NT256");
    bench<128,128,16,32,64,1,8,8,256>("BM128_BN128_BK16_WM32_WN64_WI1_TM8_TN8_NT256");
    bench<128,128,16,32,64,2,4,4,256>("BM128_BN128_BK16_WM32_WN64_WI2_TM4_TN4_NT256");
    bench<128,128,16,32,64,2,4,8,256>("BM128_BN128_BK16_WM32_WN64_WI2_TM4_TN8_NT256");
    bench<128,128,16,32,64,2,8,4,256>("BM128_BN128_BK16_WM32_WN64_WI2_TM8_TN4_NT256");
    bench<128,128,16,32,64,4,4,4,256>("BM128_BN128_BK16_WM32_WN64_WI4_TM4_TN4_NT256");
    bench<128,128,16,64,32,1,4,4,256>("BM128_BN128_BK16_WM64_WN32_WI1_TM4_TN4_NT256");
    bench<128,128,16,64,32,1,4,8,256>("BM128_BN128_BK16_WM64_WN32_WI1_TM4_TN8_NT256");
    bench<128,128,16,64,32,1,8,4,256>("BM128_BN128_BK16_WM64_WN32_WI1_TM8_TN4_NT256");
    bench<128,128,16,64,32,1,8,8,256>("BM128_BN128_BK16_WM64_WN32_WI1_TM8_TN8_NT256");
    bench<128,128,16,64,32,2,4,4,256>("BM128_BN128_BK16_WM64_WN32_WI2_TM4_TN4_NT256");
    bench<128,128,16,64,32,2,4,8,256>("BM128_BN128_BK16_WM64_WN32_WI2_TM4_TN8_NT256");
    bench<128,128,16,64,32,2,8,4,256>("BM128_BN128_BK16_WM64_WN32_WI2_TM8_TN4_NT256");
    bench<128,128,16,64,32,4,4,4,256>("BM128_BN128_BK16_WM64_WN32_WI4_TM4_TN4_NT256");
    bench<128,128,32,32,64,1,4,4,256>("BM128_BN128_BK32_WM32_WN64_WI1_TM4_TN4_NT256");
    bench<128,128,32,32,64,1,4,8,256>("BM128_BN128_BK32_WM32_WN64_WI1_TM4_TN8_NT256");
    bench<128,128,32,32,64,1,8,4,256>("BM128_BN128_BK32_WM32_WN64_WI1_TM8_TN4_NT256");
    bench<128,128,32,32,64,1,8,8,256>("BM128_BN128_BK32_WM32_WN64_WI1_TM8_TN8_NT256");
    bench<128,128,32,32,64,2,4,4,256>("BM128_BN128_BK32_WM32_WN64_WI2_TM4_TN4_NT256");
    bench<128,128,32,32,64,2,4,8,256>("BM128_BN128_BK32_WM32_WN64_WI2_TM4_TN8_NT256");
    bench<128,128,32,32,64,2,8,4,256>("BM128_BN128_BK32_WM32_WN64_WI2_TM8_TN4_NT256");
    bench<128,128,32,32,64,4,4,4,256>("BM128_BN128_BK32_WM32_WN64_WI4_TM4_TN4_NT256");
    bench<128,128,32,64,32,1,4,4,256>("BM128_BN128_BK32_WM64_WN32_WI1_TM4_TN4_NT256");
    bench<128,128,32,64,32,1,4,8,256>("BM128_BN128_BK32_WM64_WN32_WI1_TM4_TN8_NT256");
    bench<128,128,32,64,32,1,8,4,256>("BM128_BN128_BK32_WM64_WN32_WI1_TM8_TN4_NT256");
    bench<128,128,32,64,32,1,8,8,256>("BM128_BN128_BK32_WM64_WN32_WI1_TM8_TN8_NT256");
    bench<128,128,32,64,32,2,4,4,256>("BM128_BN128_BK32_WM64_WN32_WI2_TM4_TN4_NT256");
    bench<128,128,32,64,32,2,4,8,256>("BM128_BN128_BK32_WM64_WN32_WI2_TM4_TN8_NT256");
    bench<128,128,32,64,32,2,8,4,256>("BM128_BN128_BK32_WM64_WN32_WI2_TM8_TN4_NT256");
    bench<128,128,32,64,32,4,4,4,256>("BM128_BN128_BK32_WM64_WN32_WI4_TM4_TN4_NT256");
    bench<128,128,16,32,128,1,4,4,128>("BM128_BN128_BK16_WM32_WN128_WI1_TM4_TN4_NT128");
    bench<128,128,16,32,128,1,4,8,128>("BM128_BN128_BK16_WM32_WN128_WI1_TM4_TN8_NT128");
    bench<128,128,16,32,128,1,8,4,128>("BM128_BN128_BK16_WM32_WN128_WI1_TM8_TN4_NT128");
    bench<128,128,16,32,128,1,8,8,128>("BM128_BN128_BK16_WM32_WN128_WI1_TM8_TN8_NT128");
    bench<128,128,16,32,128,2,4,4,128>("BM128_BN128_BK16_WM32_WN128_WI2_TM4_TN4_NT128");
    bench<128,128,16,32,128,2,4,8,128>("BM128_BN128_BK16_WM32_WN128_WI2_TM4_TN8_NT128");
    bench<128,128,16,32,128,2,8,4,128>("BM128_BN128_BK16_WM32_WN128_WI2_TM8_TN4_NT128");
    bench<128,128,16,32,128,2,8,8,128>("BM128_BN128_BK16_WM32_WN128_WI2_TM8_TN8_NT128");
    bench<128,128,16,32,128,4,4,4,128>("BM128_BN128_BK16_WM32_WN128_WI4_TM4_TN4_NT128");
    bench<128,128,16,32,128,4,4,8,128>("BM128_BN128_BK16_WM32_WN128_WI4_TM4_TN8_NT128");
    bench<128,128,16,32,128,4,8,4,128>("BM128_BN128_BK16_WM32_WN128_WI4_TM8_TN4_NT128");
    bench<128,128,16,64,64,1,4,4,128>("BM128_BN128_BK16_WM64_WN64_WI1_TM4_TN4_NT128");
    bench<128,128,16,64,64,1,4,8,128>("BM128_BN128_BK16_WM64_WN64_WI1_TM4_TN8_NT128");
    bench<128,128,16,64,64,1,8,4,128>("BM128_BN128_BK16_WM64_WN64_WI1_TM8_TN4_NT128");
    bench<128,128,16,64,64,1,8,8,128>("BM128_BN128_BK16_WM64_WN64_WI1_TM8_TN8_NT128");
    bench<128,128,16,64,64,2,4,4,128>("BM128_BN128_BK16_WM64_WN64_WI2_TM4_TN4_NT128");
    bench<128,128,16,64,64,2,4,8,128>("BM128_BN128_BK16_WM64_WN64_WI2_TM4_TN8_NT128");
    bench<128,128,16,64,64,2,8,4,128>("BM128_BN128_BK16_WM64_WN64_WI2_TM8_TN4_NT128");
    bench<128,128,16,64,64,2,8,8,128>("BM128_BN128_BK16_WM64_WN64_WI2_TM8_TN8_NT128");
    bench<128,128,16,64,64,4,4,4,128>("BM128_BN128_BK16_WM64_WN64_WI4_TM4_TN4_NT128");
    bench<128,128,16,64,64,4,4,8,128>("BM128_BN128_BK16_WM64_WN64_WI4_TM4_TN8_NT128");
    bench<128,128,16,64,64,4,8,4,128>("BM128_BN128_BK16_WM64_WN64_WI4_TM8_TN4_NT128");
    bench<128,128,16,128,32,1,4,4,128>("BM128_BN128_BK16_WM128_WN32_WI1_TM4_TN4_NT128");
    bench<128,128,16,128,32,1,4,8,128>("BM128_BN128_BK16_WM128_WN32_WI1_TM4_TN8_NT128");
    return 0;
}