def valid(c):
    BM,BN,BK,WM,WN,WNI,TM,TN,NT=c
    if (BN*BK + BM*BK)*4 > 48*1024: return False
    if BM%WM or BN%WN: return False
    if (BN//WN)*(BM//WM) != NT//32: return False
    if WNI<1 or WNI>WN: return False
    if (WM*WN)%(32*TM*TN*WNI): return False
    WMITER=(WM*WN)//(32*TM*TN*WNI)
    if (WM%WMITER) or (WN%WNI): return False
    if (NT*4)%BK: return False
    if (NT*4)%BN: return False
    if BN%(16*TN): return False
    if BM%(16*TM): return False
    if (BM*BK)%(4*NT): return False
    if (BN*BK)%(4*NT): return False
    return True

cfg=set()
for NT in (128,256):
  for BK in (16,32):
    for BM in (128,256):
      for BN in (128,256):
        for TM in (4,8):
          for TN in (4,8):
            for WM in (32,64,128,256):
              for WN in (32,64,128,256):
                for WNI in (1,2,4):
                  c=(BM,BN,BK,WM,WN,WNI,TM,TN,NT)
                  if valid(c):
                    WMITER=(WM*WN)//(32*TM*TN*WNI)
                    acc=WMITER*TM*WNI*TN
                    if acc<=160:
                        cfg.add((acc,c))
cfg=sorted(cfg)[:56]
print("sweep configs:", len(cfg))

hdr = r'''
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
'''
lines=[hdr]
for _,c in cfg:
    BM,BN,BK,WM,WN,WNI,TM,TN,NT=c
    name="BM%d_BN%d_BK%d_WM%d_WN%d_WI%d_TM%d_TN%d_NT%d"%(BM,BN,BK,WM,WN,WNI,TM,TN,NT)
    lines.append("    bench<%d,%d,%d,%d,%d,%d,%d,%d,%d>(\"%s\");"%(BM,BN,BK,WM,WN,WNI,TM,TN,NT,name))
lines.append("    return 0;\n}")
open("bench10.cu","w").write("\n".join(lines))
print("wrote bench10.cu, configs:", len(cfg))
