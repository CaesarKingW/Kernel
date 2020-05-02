// includes, system
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <ctime>
#include <assert.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <gputimer.h>

using namespace std;

const int ARRAY_SIZE = 5120000;
typedef int32_t int32;

template <typename T>
__global__ void SetToValue(const int count, T* __restrict__ ptr, T value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i+=blockDim.x*gridDim.x) {
    ptr[i] = value;
  }
}

clock_t totalStart, totalEnd;
clock_t kernelStart, kernelEnd;
int main(int argc, char* argv[]){
    srand((unsigned)time(NULL));
    // 计时开始
    totalStart = clock();
    int nBytes = ARRAY_SIZE * sizeof(float);
    float *ptr;
    // 申请托管内存
    //cudaMallocManaged((void**)&ptr, nBytes);
    // 申请host内存
    ptr = (float *)malloc(nBytes);
    // 申请device内存
    float *d_x;
    cudaMalloc((void**)&d_x, nBytes);
    // 初始化数据
    for(int i=0; i<ARRAY_SIZE; i++){
        ptr[i] = rand()%1000;
    }
    float value = 100.0;
    // 将host数据拷贝到device
    cudaMemcpy((void*)d_x, (void*)ptr, nBytes, cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((ARRAY_SIZE + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    GpuTimer timer;
    timer.Start();
    // kernelStart = clock();
    SetToValue<<<gridSize, blockSize>>>(ARRAY_SIZE, d_x, value);
    // 同步device, 保证结果能正确访问
    cudaDeviceSynchronize();
    timer.Stop();
    double kernelTime = timer.Elapsed();
    cout<<"Kernel time:"<<kernelTime<<"ms"<<endl;	
    // kernelEnd = clock();
    // double kernelTime=(double)(kernelEnd-kernelStart)/CLOCKS_PER_SEC;
    // cout<<"Kernel time:"<<kernelTime*1000<<"ms"<<endl;	//ms为单位
    cudaMemcpy((void*)ptr, (void*)d_x, nBytes, cudaMemcpyDeviceToHost);
    // 释放内存
    cudaFree(d_x);
    free(ptr);
    // 计时结束
    totalEnd = clock();
    double totalTime=(double)(totalEnd-totalStart)/CLOCKS_PER_SEC;
	  cout<<"Total time:"<<totalTime*1000<<"ms"<<endl;	//ms为单位
    return 0;
}