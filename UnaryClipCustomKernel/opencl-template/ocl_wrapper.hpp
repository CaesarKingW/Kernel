#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <logging.h>

#define OCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    CHECK_EQ(error, CL_SUCCESS) << " " << error; \
    if(CL_SUCCESS != error){ \
       LOG(INFO) << "failed";\
    } \
  } while (0)

using namespace std;

std::map<std::string, cl_kernel> Kernels;

template <typename dtype> inline std::string get_dtype_suffix() {
  dtype x;
  const char type = typeid(x).name()[0];
  std::string suffix;
  switch (type) {
  case 'i':
    suffix = "_int";
    break;
  case 'd':
    suffix = "_double";
    break;
  case 'f':
  default:
    suffix = "_float";
  }
  return suffix;
}

template <typename Dtype>
void UnaryClipCustomKernel(const int32_t size_in, const Dtype* __restrict__ in0, const Dtype* __restrict__ in1, const Dtype* __restrict__ in2, Dtype* __restrict__ out);