#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "./ocl_wrapper.hpp"
#include "./device.hpp"

using namespace std;

template <typename Dtype>
void UnaryClipCustomKernel(const int32_t size_in, \
const Dtype* __restrict__ in0, const Dtype* __restrict__ in1, \
const Dtype* __restrict__ in2, Dtype* __restrict__ out){
    std::string kernel_name = "UnaryClipCustomKernel" + get_dtype_suffix<Dtype>();
    /*
    std::map<std::string, cl_kernel>::iterator it = Kernels.find(kernel_name);
    if (it == Kernels.end()) {
        cl_int _err = 0;
        cl_kernel kernel = clCreateKernel(Program, kernel_name.c_str(), &_err);
        OCL_CHECK(_err);
        Kernels[kernel_name] = kernel;
    }
    cl_kernel Kernel = Kernels[kernel_name];
    */
    cl_kernel ker_rand = amdDevice.GetKernel(kernel_name);
    cl_int ret;
    ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &size_in);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &in0);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &in1);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &in2);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &out);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {(size_t) size_in};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(
        clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL);
    )
}

template void UnaryClipCustomKernel<float>(const int32_t size_in, \
const Dtype* __restrict__ in0, const Dtype* __restrict__ in1, \
const Dtype* __restrict__ in2, Dtype* __restrict__ out);
template void UnaryClipCustomKernel<double>(const int32_t size_in, \
const Dtype* __restrict__ in0, const Dtype* __restrict__ in1, \
const Dtype* __restrict__ in2, Dtype* __restrict__ out);
