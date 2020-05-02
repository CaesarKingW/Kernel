#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

class Device {
  public:
    Device()
        : numPlatforms(0), numDevices(0), device_id(INT_MIN) {
    }
    ~Device();
    cl_uint numPlatforms;
    cl_platform_id * platformIDs;
    char platformName[64];
    char openclVersion[64];
    cl_uint numDevices;
    cl_device_id * DeviceIDs;

    cl_context Context;
    cl_command_queue CommandQueue;
    cl_command_queue CommandQueue_helper;
    cl_program Program;
    cl_device_id * pDevices;
    int device_id;

    clblasOrder col;
    clblasOrder row;
    std::map<std::string, cl_kernel> Kernels;

    cl_int Init(int device_id = -1);
    cl_int ConvertToString(std::string pFileName, std::string &Str);
    void DisplayPlatformInfo();
    void DisplayInfo(cl_platform_id id, cl_platform_info name, std::string str);

    void GetDeviceInfo();
    void DeviceQuery();
    int GetDevice() {
      return device_id;
    }
    ;
    void BuildProgram(std::string kernel_dir);

    template <typename T>
    void DisplayDeviceInfo(cl_device_id id, cl_device_info name,
        std::string str);
    template <typename T>
    void appendBitfield(T info, T value, std::string name, std::string &str);

    cl_kernel GetKernel(std::string kernel_name);
    void ReleaseKernels();
};
extern std::string buildOption;
extern Device amdDevice;
