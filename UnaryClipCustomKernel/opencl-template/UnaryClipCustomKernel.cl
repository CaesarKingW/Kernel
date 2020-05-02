template <class T>
__kernel void UnaryClipCustomKernel(const int32_t size_in,
                                     __global T *__restrict__ in0,
                                     __global T *__restrict__ in1,
                                     __global T *__restrict__ in2,
                                     T *__restrict__ out){
    int32_t index = get_global_id(0);
    if(index < size_in){
        T value = in2[0] < in0[index] ? in2[0] : in0[index];
        out[index] = value < in1[0] ? in1[0] : value;
    }    
}

template __attribute__((mangled_name(UnaryClipCustomKernel_float))) \
__kernel void UnaryClipCustomKernel(const int32_t size_in, __global float* __restrict__ in0 \
__global float* __restrict__ in1, __global float* __restrict__ in2, float* __restrict__ out);

template __attribute__((mangled_name(UnaryClipCustomKernel_double))) \
__kernel void UnaryClipCustomKernel(const int32_t size_in, __global double* __restrict__ in0 \
__global double* __restrict__ in1, __global double* __restrict__ in2, double* __restrict__ out);