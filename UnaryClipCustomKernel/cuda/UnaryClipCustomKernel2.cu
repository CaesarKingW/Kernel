// includes, system
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <ctime>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <gputimer.h>

using namespace std;

const int ARRAY_SIZE = 5120000;
typedef int32_t int32;

template <typename T>
__global__ void UnaryClipCustomKernel(const int32 size_in,
                                      const T *__restrict__ in0,
                                      const T *__restrict__ in1,
                                      const T *__restrict__ in2,
                                      T *__restrict__ out) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;
  int32 offset = thread_id;
  while(offset < size_in){
      T value = in2[0] < in0[offset] ? in2[0] : in0[offset];
      out[offset] = value < in1[0] ? in1[0] : value;
      offset += total_thread_count;
  }
}

clock_t totalStart, totalEnd;
clock_t kernelStart, kernelEnd;
void executeKernel();
double sum = 0;

int main(int argc, char* argv[]){
    
    // 计时开始
    totalStart = clock();
    for(int i=0; i<1;i++)
        executeKernel();
    // 计时结束
    totalEnd = clock();
    double totalTime=(double)(totalEnd-totalStart)/CLOCKS_PER_SEC;
    cout<<"Total time:"<<totalTime*1000<<"ms"<<endl;	//ms为单位
    cout<<"Total Kernel Time"<< sum <<"ms"<<endl;
    return 0;
}

void executeKernel(){
    // 设置随机数种子
    srand((unsigned)time(NULL));
    int nBytes = ARRAY_SIZE * sizeof(float);
    float *in0, *in1, *in2, *out;
    // 申请托管内存
    /*
    cudaMallocManaged((void**)&in0, nBytes);
    cudaMallocManaged((void**)&out, nBytes);
    cudaMallocManaged((void**)&in1, sizeof(float));
    cudaMallocManaged((void**)&in2, sizeof(float));
    */
    // 申请主机内存
    in0 = (float*)malloc(nBytes);
    in1 = (float*)malloc(sizeof(float));
    in2 = (float*)malloc(sizeof(float));
    out = (float*)malloc(nBytes);
    // 初始化数据
    for(int i = 0; i < ARRAY_SIZE; i++){
        in0[i] = rand()%200;
    }
    in1[0] = 20.0;
    in2[0] = 90.0;
    // 申请device内存
    float *d_x, *d_y, *d_z, *d_out;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, sizeof(float));
    cudaMalloc((void**)&d_z, sizeof(float));
    cudaMalloc((void**)&d_out, nBytes);
    // 将数据从host拷贝到device
    cudaMemcpy((void*)d_x, (void*)in0, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)in1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_z, (void*)in2, sizeof(float), cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((ARRAY_SIZE + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    // 计时
    GpuTimer timer;
    timer.Start();
    //kernelStart = clock();
    UnaryClipCustomKernel<<<gridSize, blockSize>>>(ARRAY_SIZE, d_x, d_y, d_z, d_out);
    timer.Stop();
    printf("\nExecution time in milliseconds = %0.6f ms\n", timer.Elapsed());
    sum += timer.Elapsed();
    // 同步device, 保证结果能正确访问
    cudaDeviceSynchronize();
    //kernelEnd = clock();
    //double kernelTime=(double)(kernelEnd-kernelStart)/CLOCKS_PER_SEC;
    //cout<<"Kernel time:"<<kernelTime*1000<<"ms"<<endl;	//ms为单位
    //sum += kernelTime*1000;
    // 将结果从device拷贝回host
    cudaMemcpy((void*)out, (void*)d_out, nBytes, cudaMemcpyDeviceToHost);
    // 检查执行结果
    /*
    for(int i=0; i<ARRAY_SIZE; i++){
        cout<<out[i]<<endl;
    }
    */
    // 释放内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_out);
    free(in0);
    free(in1);
    free(in2);
    free(out);
}