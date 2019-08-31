/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <string>
#include "triton/driver/platform.h"
#include "triton/driver/device.h"


namespace triton
{
namespace driver
{


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

std::string cu_platform::version() const{
  int version;
  dispatch::cuDriverGetVersion(&version);
  return std::to_string(version);
}

void cu_platform::devices(std::vector<device *> &devices) const{
  int N;
  dispatch::cuDeviceGetCount(&N);
  for(int i = 0 ; i < N ; ++i){
    CUdevice dvc;
    dispatch::cuDeviceGet(&dvc, i);
    devices.push_back(new driver::cu_device(dvc));
  }
}

/* ------------------------ */
//        OpenCL            //
/* ------------------------ */

std::string cl_platform::version() const {
  size_t size;
  check(dispatch::clGetPlatformInfo(*cl_, CL_PLATFORM_VERSION, 0, nullptr, &size));
  std::string result(size, 0);
  check(dispatch::clGetPlatformInfo(*cl_, CL_PLATFORM_VERSION, size, (void*)&*result.begin(), nullptr));
  return result;
}

void cl_platform::devices(std::vector<device*> &devices) const{
  cl_uint num_devices;
  check(dispatch::clGetDeviceIDs(*cl_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices));
  std::vector<cl_device_id> ids(num_devices);
  check(dispatch::clGetDeviceIDs(*cl_, CL_DEVICE_TYPE_GPU, num_devices, ids.data(), nullptr));
  for(cl_device_id id: ids)
    devices.push_back(new driver::ocl_device(id));
}

/* ------------------------ */
//        Host              //
/* ------------------------ */

std::string host_platform::version() const {
  return "1.0";
}

void host_platform::devices(std::vector<driver::device*> &devices) const {
  devices.push_back(new driver::host_device());
}

/* ------------------------ */
//        Vulkan            //
/* ------------------------ */
std::string vk_platform::version() const {
  return "1.0";
}

unsigned vk_platform::get_compute_queue_idx(VkPhysicalDevice p_device) const {
  uint32_t queueFamilyCount;
  dispatch::vkGetPhysicalDeviceQueueFamilyProperties(p_device, &queueFamilyCount, NULL);
  // Retrieve all queue families.
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  dispatch::vkGetPhysicalDeviceQueueFamilyProperties(p_device, &queueFamilyCount, queueFamilies.data());
  // Now find a family that supports compute.
  uint32_t i = 0;
  for (; i < queueFamilies.size(); ++i) {
    VkQueueFamilyProperties props = queueFamilies[i];
    if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT))
        break;
  }
  if (i == queueFamilies.size())
    throw std::runtime_error("could not find a queue family that supports operations");
  return i;
}

void vk_platform::devices(std::vector<driver::device*> &devices) const {
  uint32_t count;
  dispatch::vkEnumeratePhysicalDevices(*vk_, &count, NULL);
  std::vector<VkPhysicalDevice> p_devices(count);
  dispatch::vkEnumeratePhysicalDevices(*vk_, &count, p_devices.data());
  for(VkPhysicalDevice p_device: p_devices){
    // queue info
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = get_compute_queue_idx(p_device);
    queueCreateInfo.queueCount = 1;
    float queuePriorities = 1.0;
    queueCreateInfo.pQueuePriorities = &queuePriorities;
    // info and features
    VkDeviceCreateInfo deviceCreateInfo = {};
    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = NULL;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    // create
    VkDevice device;
    dispatch::vkCreateDevice(p_device, &deviceCreateInfo, NULL, &device);
    devices.push_back(new driver::vk_device(vk_device_t{p_device, device}));
  }
}


}
}
