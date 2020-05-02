__kernel void UnaryClipCustomKernel(const int size_in,
                                     __global const float* in0,
                                     __global const float* in1,
                                     __global const float* in2,
                                     __global float* out){
    int index = get_global_id(0);
    if(index < size_in){
        float value = in2[0] < in0[index] ? in2[0] : in0[index];
        out[index] = value < in1[0] ? in1[0] : value;
    }    
}