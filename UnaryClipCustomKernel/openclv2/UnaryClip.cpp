#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <string>
#include <vector>
#include <ctime>

 
using namespace std;
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
 
const int ARRAY_SIZE = 5120000;

 
//  选择平台并创建上下文
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
 
	//选择第一个可用的平台
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
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
			return NULL;
		}
	}
 
	return context;
}
 
 
//选择第一个可用的设备并创建一个命令队列
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;
 
	//这个clGetContextInfo获得设备缓冲区的大小
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}
 
	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}
 
	//为设备缓冲区分配内存，这个clGetContextInfo用来获得上下文中所有可用的设备
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}
	char    deviceName[512];
	char    deviceVendor[512];
	char    deviceVersion[512];
	errNum = clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, sizeof(deviceVendor),
		deviceVendor, NULL);
	errNum |= clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(deviceName),
		deviceName, NULL);
	errNum |= clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, sizeof(deviceVersion),
		deviceVersion, NULL);
 
	printf("OpenCL Device Vendor = %s,  OpenCL Device Name = %s,  OpenCL Device Version = %s\n", deviceVendor, deviceName, deviceVersion);
	// 在这个例子中，我们只选择第一个可用的设备。在实际的程序，你可能会使用所有可用的设备或基于OpenCL设备查询选择性能最高的设备
	cl_queue_properties props[] = {
		CL_QUEUE_PROPERTIES,
		CL_QUEUE_PROFILING_ENABLE,
		1
	};
	commandQueue = clCreateCommandQueueWithProperties(context, devices[0], props, NULL);
	if (commandQueue == NULL)
	{
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}
 
	*device = devices[0];
	delete[] devices;
	return commandQueue;
}
 
//从磁盘加载内核源文件并创建一个程序对象
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;
 
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}
 
	std::ostringstream oss;
	oss << kernelFile.rdbuf();
 
	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	//创建程序对象
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}
	//编译内核源代码
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// 编译失败可以通过clGetProgramBuildInfo获取日志
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);
 
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}
 
	return program;
}
 
 
//创建内存对象
bool CreateMemObjects(cl_context context, cl_mem memObjects[4],
					  float *a, float *b, float *c)
{
	//创建内存对象
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float)* ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) , b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float), c, NULL);
	memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(float)* ARRAY_SIZE, NULL, NULL);
 
	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL || memObjects[3] == NULL)
	{
		std::cerr << "Error creating memory objects." << std::endl;
		return false;
	}
 
	return true;
}
 


//清理任何创建OpenCL的资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
			 cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
	for (int i = 0; i < 3; i++)
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
 
clock_t totalStart, totalEnd;
clock_t kernelStart, kernelEnd;
//主函数
int main(int argc, char** argv)
{
	totalStart=clock();
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[4] = { 0, 0, 0 ,0};
	cl_int errNum;
	
	// 创建opencl上下文和第一个可用平台
	context = CreateContext();
	if (context == NULL)
	{
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return 1;
	}
 
	// 在创建的一个上下文中选择第一个可用的设备并创建一个命令队列
	commandQueue = CreateCommandQueue(context, &device);
	if (commandQueue == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
 
	// 创建一个程序对象 HelloWorld.cl kernel source
	program = CreateProgram(context, device, "UnaryClipCustomKernel.cl");
	if (program == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		system("pause");
		return 1;
	}

	clFinish(commandQueue);
	

	// 创建内核
	kernel = clCreateKernel(program, "UnaryClipCustomKernel", NULL);
	if (kernel == NULL)
	{
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		system("pause");
		return 1;
	}
 
	// 创建一个将用作参数内核内存中的对象。首先创建将被用来将参数存储到内核主机存储器阵列
	float *in0 = new float[ARRAY_SIZE];
	float in1[1];
	float in2[1];
	float *result = new float[ARRAY_SIZE];
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
	/*
	for(int i=0; i<ARRAY_SIZE; i++){
		in0[i] = i;
	}
	*/
	in1[0] = 20.0;
	in2[0] = 90.0;
 
	if (!CreateMemObjects(context, memObjects, in0, in1, in2))
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		system("pause");
		return 1;
	}
 
	// 设置内核参数、执行内核并读回结果
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_int), &ARRAY_SIZE);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[3]);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error setting kernel arguments." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		system("pause");
		return 1;
	}
 
	size_t globalWorkSize[1] = { ARRAY_SIZE  };//让之等于数组的大小
	size_t localWorkSize[1] = { 256 }; 
 
	// 利用命令队列使将在设备执行的内核排队
	cl_event event;
	kernelStart = clock();
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, &event);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error queuing kernel for execution." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		system("pause");
		return 1;
	}
	// 此函数会使主机线程等待event_list中的事件对象所标识的命令完成。
	// 第一个参数表示event_list中event的数量
	// 第二个参数是这个事件列表的指针，对应clEnqueueNDRangeKernel中最后一个参数
	clWaitForEvents(1, &event);
	kernelEnd = clock();
	double kernelTime=(double)(kernelEnd-kernelStart)/CLOCKS_PER_SEC;
	cout<<"Kernel time:"<<kernelTime*1000<<"ms"<<endl;	//ms为单位
	/*
	cl_int err_code;
	
  	cl_ulong startTime, endTime;
  	err_code = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
  	err_code = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
	unsigned long elapsed =  (unsigned long)(endTime - startTime);
	// cout<<startTime<<endl;
	// cout<<endTime<<endl;
	// cout<<elapsed<<endl;
	printf("\nExecution time in milliseconds = %0.6f ms\n", (elapsed / 1000000.0) );
	std::cout << "Executed program succesfully." << std::endl;
	*/

	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE,
		0, ARRAY_SIZE * sizeof(float), result,
		0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		system("pause");
		return 1;
	}
	totalEnd = clock();
	double totalTime=(double)(totalEnd-totalStart)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<totalTime*1000<<"ms"<<endl;	//ms为单位
	 //输出结果
	 /*
	for(int i=0; i<ARRAY_SIZE; i++){
		std::cout<<result[i]<<endl;
	}
	*/
	std::cout << std::endl;
	std::cout << "Executed program succesfully." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);
	system("pause");
	return 0;
}