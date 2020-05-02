//For clarity,error checking has been omitted.
#include <CL/cl.h>
#include "tool.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

int main(int argc, char* argv[])
{
    cl_int    status;
    /**Step 1: Getting platforms and choose an available one(first).*/
    cl_platform_id platform;
    getPlatform(platform);

    /**Step 2:Query the platform and choose the first GPU device if has one.*/
    cl_device_id *devices=getCl_device_id(platform);

    /**Step 3: Create context.*/
    cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);

    /**Step 4: Creating command queue associate with the context.*/
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);

    /**Step 5: Create program object */
    const char *filename = "UnaryClipCustomKernel.cl";
    string sourceStr;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

    /**Step 6: Build program. */
    status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);

    /**Step 7: Initial input,output for the host and create memory objects for the kernel*/
    const int NUM=10000;
    float* in0 = new float[NUM];
    for(int i=0;i<NUM;i++)
        in0[i]=i;
    float* out = new float[NUM];
    float in1[] = {20.0};
    float in2[] = {90.0};

    cl_mem in0Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, NUM * sizeof(float), (void *)in0, NULL);
    cl_mem in1Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float), (void *)in1, NULL);
    cl_mem in2Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float), (void *)in2, NULL);
    cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , NUM * sizeof(float), NULL, NULL);

    /**Step 8: Create kernel object */
    cl_kernel kernel = clCreateKernel(program,"UnaryClipCustomKernel", NULL);

    /**Step 9: Sets Kernel arguments.*/
    /*
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
    */
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&NUM);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&in0);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&in1);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&in2);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&out);
    /**Step 10: Running the kernel.*/
    size_t global_work_size[1] = {NUM};
    size_t local_work_size[1] = {256};
    cl_event enentPoint;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
    clWaitForEvents(1,&enentPoint); ///wait
    clReleaseEvent(enentPoint);

    /**Step 11: Read the cout put back to host memory.*/
    status = clEnqueueReadBuffer(commandQueue, outBuffer, CL_TRUE, 0, NUM * sizeof(float), out, 0, NULL, NULL);
    cout<<out[NUM-1]<<endl;

    /**Step 12: Clean the resources.*/
    status = clReleaseKernel(kernel);//*Release kernel.
    status = clReleaseProgram(program);    //Release the program object.
    status = clReleaseMemObject(in0Buffer);//Release mem object.
    status = clReleaseMemObject(in1Buffer);
    status = clReleaseMemObject(in2Buffer);
    status = clReleaseMemObject(outBuffer);
    status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
    status = clReleaseContext(context);//Release context.

    if (out != NULL)
    {
        free(out);
        out = NULL;
    }

    if (devices != NULL)
    {
        free(devices);
        devices = NULL;
    }
    return 0;
}