// Independent FP32 Xgesvdp check, no Python/torch or legacy binding involved.
// Input: column-major float32 binary, m, n. Output: JSON diagnostics.
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define SOLVER(call) do { auto e=(call); if(e!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER status "+std::to_string(e)); } while(0)

int main(int argc, char** argv) {
  try {
    if(argc!=4) throw std::runtime_error("usage: polar_sanity matrix.bin m n");
    int64_t m=std::stoll(argv[2]), n=std::stoll(argv[3]), r=std::min(m,n);
    if(m<1 || n<1 || std::max(m,n)>4096) throw std::runtime_error("sanity check sizes must be 1..4096");
    std::vector<float> a(m*n), u(m*r), v(n*r), s(r);
    std::ifstream f(argv[1],std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(float))) throw std::runtime_error("input read failed");
    float *da,*du,*dv,*ds; int *di, info;
    CUDA(cudaMalloc(&da,a.size()*sizeof(float))); CUDA(cudaMalloc(&du,u.size()*sizeof(float)));
    CUDA(cudaMalloc(&dv,v.size()*sizeof(float))); CUDA(cudaMalloc(&ds,s.size()*sizeof(float)));
    CUDA(cudaMalloc(&di,sizeof(int)));
    CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
    cusolverDnHandle_t handle; cusolverDnParams_t params;
    SOLVER(cusolverDnCreate(&handle)); SOLVER(cusolverDnCreateParams(&params));
    size_t wd=0,wh=0;
    SOLVER(cusolverDnXgesvdp_bufferSize(handle,params,CUSOLVER_EIG_MODE_VECTOR,1,m,n,
      CUDA_R_32F,da,m,CUDA_R_32F,ds,CUDA_R_32F,du,m,CUDA_R_32F,dv,n,
      CUDA_R_32F,&wd,&wh));
    void *work; CUDA(cudaMalloc(&work,std::max(wd,size_t(1))));
    std::vector<unsigned char> hostwork(std::max(wh,size_t(1)));
    double perturbation=0;
    SOLVER(cusolverDnXgesvdp(handle,params,CUSOLVER_EIG_MODE_VECTOR,1,m,n,
      CUDA_R_32F,da,m,CUDA_R_32F,ds,CUDA_R_32F,du,m,CUDA_R_32F,dv,n,
      CUDA_R_32F,work,wd,hostwork.data(),wh,di,&perturbation));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(u.data(),du,u.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(v.data(),dv,v.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(s.data(),ds,s.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(&info,di,sizeof(int),cudaMemcpyDeviceToHost));
    double err=0, norm=0;
    for(int64_t j=0;j<n;++j) for(int64_t i=0;i<m;++i) {
      double approx=0;
      for(int64_t k=0;k<r;++k) approx+=double(u[i+k*m])*s[k]*v[j+k*n];
      double delta=double(a[i+j*m])-approx;
      err+=delta*delta; norm+=double(a[i+j*m])*a[i+j*m];
    }
    int major,minor,patch;
    SOLVER(cusolverGetProperty(MAJOR_VERSION,&major));
    SOLVER(cusolverGetProperty(MINOR_VERSION,&minor));
    SOLVER(cusolverGetProperty(PATCH_LEVEL,&patch));
    std::cout.precision(12);
    std::cout << "{\"cusolver_version\":[" << major << ',' << minor << ',' << patch
      << "],\"m\":" << m << ",\"n\":" << n << ",\"info\":" << info
      << ",\"h_err_sigma\":" << perturbation << ",\"reconstruction_relative_fro\":" << std::sqrt(err/(norm>0?norm:1))
      << ",\"workspace_device_bytes\":" << wd << ",\"workspace_host_bytes\":" << wh << "}\n";
    cudaFree(work); cudaFree(di); cudaFree(ds); cudaFree(dv); cudaFree(du); cudaFree(da);
    cusolverDnDestroyParams(params); cusolverDnDestroy(handle);
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
