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

const int ARRAY_SIZE = 5120;
typedef int32_t int32;

template <typename T>
__global__ void UnaryClipCustomKernel(const int32 size_in,
                                      const T *__restrict__ in0,
                                      const T *__restrict__ in1,
                                      const T *__restrict__ in2,
                                      T *__restrict__ out) {
  /*
  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[0] < in0[i] ? in2[0] : in0[i];
    out[i] = value < in1[0] ? in1[0] : value;
  }
  */
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
int main(int argc, char* argv[]){
    // 计时开始
    totalStart = clock();
    int nBytes = ARRAY_SIZE * sizeof(float);
    float *in0, *in1, *in2, *out;
    // 申请托管内存
    cudaMallocManaged((void**)&in0, nBytes);
    cudaMallocManaged((void**)&out, nBytes);
    cudaMallocManaged((void**)&in1, sizeof(float));
    cudaMallocManaged((void**)&in2, sizeof(float));
    // 初始化数据
    // 从文本中读取数据，此处用float类型
    ifstream infile;
    ofstream outfile;
    infile.open("in0.txt",ios::in);
    if(!infile.is_open())
        cout<<"Opening data file fails"<<endl;
    else
        cout<<"Opening data file successes"<<endl;
    for(int i=0; i<ARRAY_SIZE; i++){
        infile>>in0[i];
    }
    in1[0] = 20.0;
    in2[0] = 90.0;
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((ARRAY_SIZE + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    // 计时
    // GpuTimer timer;
    // timer.Start();
    kernelStart = clock();
    UnaryClipCustomKernel<<<gridSize, blockSize>>>(ARRAY_SIZE, in0, in1, in2, out);
    // timer.Stop();
    // printf("\nExecution time in milliseconds = %0.6f ms\n", timer.Elapsed());
    // 同步device, 保证结果能正确访问
    cudaDeviceSynchronize();
    kernelEnd = clock();
    double kernelTime=(double)(kernelEnd-kernelStart)/CLOCKS_PER_SEC;
	cout<<"Kernel time:"<<kernelTime*1000<<"ms"<<endl;	//ms为单位
    // 检查执行结果
    
    for(int i=0; i<ARRAY_SIZE; i++){
        cout<<out[i]<<endl;
    }
    
    // 释放内存
    cudaFree(in0);
    cudaFree(in1);
    cudaFree(in2);
    cudaFree(out);

    // 计时结束
    totalEnd = clock();
    double totalTime=(double)(totalEnd-totalStart)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<totalTime*1000<<"ms"<<endl;	//ms为单位
    return 0;
}