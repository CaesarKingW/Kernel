#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <string>
#include <vector>
#include <ctime>
#include <typeinfo>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

const int ARRAY_SIZE = 5120000;
clock_t totalStart, totalEnd;
clock_t kernelStart, kernelEnd;

// 获取变量类型
template <typename T>
string getTypeName(T value){
	const char type = typeid(value).name()[0];
	string suffix;
	switch (type) {
	case 'i':
		suffix = "int";
		break;
	case 'd':
		suffix = "double";
		break;
	case 'f':
		suffix = "float";
	case 'c':
		suffix = "char";
	case 's':
		suffix = "string";
	default:
		suffix = "float";
	}
	return suffix;
}
//释放内存
//清理任何创建OpenCL的资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
			 cl_program program, cl_kernel kernel, cl_mem memObjects[0])
{
	for (int i = 0; i < 1; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);
 
	if (kernel != 0)
		clReleaseKernel(kernel);
 
	if (program != 0)
		clReleaseProgram(program);
 
	if (context != 0)
		clReleaseContext(context);
 
}

int main(int argc, char** argv){
	totalStart=clock();
    cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[1] = { 0};
	cl_int errNum;
    // 初始化计算数据
    srand((unsigned)time(NULL));
    float *ptr = new float[ARRAY_SIZE];
    float value = 100.0;
    for(int i=0; i<ARRAY_SIZE; i++)
        ptr[i] = rand()%1000;
    // 选择平台
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return 1;
	}
    // 创建一个opencl上下文，成功则使用GPU上下文，否则使用cpu
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return 1;
		}
	}
    // 在创建的一个上下文中选择第一个可用的设备并创建一个命令队列
    cl_device_id *devices;
    size_t deviceBufferSize = -1;
    //这个clGetContextInfo获得设备缓冲区的大小
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return 1;
	}
 
	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return 1;
	}
    //为设备缓冲区分配内存，这个clGetContextInfo用来获得上下文中所有可用的设备
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return 1;
	}
    // 选择第一个可用的设备
	cl_command_queue_properties props[] = {
		CL_QUEUE_PROPERTIES,
		CL_QUEUE_PROFILING_ENABLE
	};
	commandQueue = clCreateCommandQueueWithProperties(context, devices[0], props, &errNum);
	if (commandQueue == NULL)
	{
		cout<<errNum<<endl;
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return 1;
	}
	device = devices[0];
	delete[] devices;
    // 创建一个程序对象 将核函数作为字符串编译进主程序
	/*
	    STATIC_ASSERT(get_global_size(1) = 1); \
    STATIC_ASSERT(get_global_size(2) = 1); \
	*/
    string typeName = getTypeName(ptr[0]);
    string src = "__kernel void SetToValue(const int count, \
                                     __global " + typeName + " * ptr, \
                                     "+typeName+ " value){ \
	if(get_global_size(1) != 1) return; \
	if(get_global_size(2) != 1) return; \
    int index = get_global_id(0); \
    if(index < count){ \
        ptr[index] = value; \
    } \
}";
    const char *srcStr = src.c_str();
    // 创建程序对象
    program = clCreateProgramWithSource(context, 1, &srcStr, NULL, NULL);
    if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return 1;
	}
    //编译内核源代码
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		cout<<errNum<<endl;
		// 编译失败可以通过clGetProgramBuildInfo获取日志
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);
 
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return 1;
	}
    // 创建内核
    kernel = clCreateKernel(program, "SetToValue", NULL);
	if (kernel == NULL)
	{
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
    // 创建内存对象
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*ARRAY_SIZE, NULL, NULL);
    if (memObjects[0] == NULL)
	{
		std::cerr << "Error creating memory objects." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
    // 设置内核参数、执行内核并读回结果
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_int), &ARRAY_SIZE);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 2 , sizeof(float), &value);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error setting kernel arguments." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

    // 设定工作项大小
    size_t globalWorkSize[1] = { ARRAY_SIZE  };//让之等于数组的大小
	size_t localWorkSize[1] = { 256 }; 
    // 利用命令队列使将在设备执行的内核排队
	cl_event event;
	// kernelStart = clock();
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, &event);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error queuing kernel for execution." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
    clWaitForEvents(1, &event);
	cl_int err_code;
	
  	cl_ulong startTime, endTime;
  	err_code = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
  	err_code |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
	unsigned long elapsed =  (unsigned long)(endTime - startTime);
	double kernelTime = elapsed /1000000.0;
	cout<<"Kernel time:"<<kernelTime<<"ms"<<endl;
	// kernelEnd = clock();
	// double kernelTime=(double)(kernelEnd-kernelStart)/CLOCKS_PER_SEC;
	// cout<<"Kernel time:"<<kernelTime*1000<<"ms"<<endl;
    // 读取结果
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE,
		0, ARRAY_SIZE * sizeof(float), ptr,
		0, NULL, NULL);
    if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
    totalEnd = clock();
	double totalTime=(double)(totalEnd-totalStart)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<totalTime*1000<<"ms"<<endl;
    // 输出结果
	/*
    for(int i=0; i<ARRAY_SIZE;i++)
        cout<<ptr[i]<<endl;
	*/
    cout<<"Executing program succesfully"<<endl;
    // 释放内存
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 0;
}