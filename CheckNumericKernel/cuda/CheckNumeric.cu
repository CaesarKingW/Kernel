// includes, system
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

using namespace std;

const int ARRAY_SIZE = 1000000;
typedef int32_t int32;

// 此核函数检查data数组中是否存在无穷数与无效数字
// inf:infinite; nan:not a number
template <typename T>
__global__ void CheckNumericsKernel(const T* __restrict__ data, int size,
                                    int abnormal_detected[2]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;

  while (offset < size) {
    if (isnan(data[offset])) {
      abnormal_detected[0] = 1;
    }
    if (isinf(data[offset])) {
      abnormal_detected[1] = 1;
    }
    offset += total_thread_count;
  }
}

int main(){
    int nBytes = ARRAY_SIZE * sizeof(double);
    double *data;
    int *abnormal_detected;
    // 申请托管内存
    cudaMallocManaged((void**)&data, nBytes);
    cudaMallocManaged((void**)&abnormal_detected,2*sizeof(int));
    // 初始化数据
    // 从文本中读取数据,此处用double类型
    ifstream infile;
    ofstream outfile;
    infile.open("data.txt", ios::in);
    if(!infile.is_open())
        cout<<"Open data file failure"<<endl;
    else
        cout<<"Open data file successfully"<<endl;
    for(int i=0; i<ARRAY_SIZE; i++){
        infile>>data[i];
        cout<<data[i]<<endl;
    }
    abnormal_detected[0] = 0;
    abnormal_detected[1] = 0;
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((ARRAY_SIZE + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    CheckNumericsKernel<< < gridSize, blockSize >> >(data, ARRAY_SIZE, abnormal_detected);
    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    cout<<"是否存在无效数字："<<bool(abnormal_detected[0])<<endl;
    cout<<"是否存在无穷大数字："<<bool(abnormal_detected[1])<<endl;
    //释放内存
    cudaFree(data);
    cudaFree(abnormal_detected);

    return 0;    
}