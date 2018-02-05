// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/blob.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
#ifndef CPU_ONLY
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}

#ifndef CPU_ONLY
using caffe::CAFFE_TEST_CUDA_PROP;
#endif

int main(int argc, char** argv) {
#if defined(DEBUG)
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = 0;
#endif
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  std::vector<int> devices;
  int device_count;

  cudaGetDeviceCount(&device_count);
  cout << "Cuda number of devices: " << device_count << endl;

  if (argc > 1) {
    // Use the given device
    devices.push_back(atoi(argv[1]));
    CUDA_CHECK(cudaSetDevice(devices[0]));
  } else if (CUDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    devices.push_back(CUDA_TEST_DEVICE);
  }

  if (devices.size() == 1) {
    cout << "Setting to use device " << devices[0] << endl;
    CUDA_CHECK(cudaSetDevice(devices[0]));
  } else {
    for (int i = 0; i < device_count; ++i)
      devices.push_back(i);
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cout << "Current device id: " << device << endl;
  CUDA_CHECK(cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device));

  cout << "Current device name: " << CAFFE_TEST_CUDA_PROP.name << endl;
  caffe::Caffe::SetDevice(device);
  caffe::GPUMemory::Scope gpu_memory_scope(devices);

#endif
  // invoke the test.
  int ret = RUN_ALL_TESTS();
  return ret;
}
