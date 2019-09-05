#pragma once

#ifndef _TRITON_DRIVER_PLATFORM_H_
#define _TRITON_DRIVER_PLATFORM_H_

#include <vector>
#include <string>

#include "triton/driver/handle.h"

namespace triton
{

namespace driver
{

class device;

class platform
{
public:
  // Constructor
  platform(const std::string& name): name_(name){ }
  // Accessors
  std::string name() const { return name_; }
  // Virtual methods
  virtual std::string version() const = 0;
  virtual void devices(std::vector<driver::device *> &devices) const = 0;
private:
  std::string name_;
};

// CUDA
class cu_platform: public platform
{
public:
  cu_platform(): platform("CUDA") { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;

private:
  handle<CUPlatform> cu_;
};

// OpenCL
class cl_platform: public platform
{
public:
  cl_platform(cl_platform_id cl): platform("OpenCL"), cl_(cl) { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;

private:
  handle<cl_platform_id> cl_;
};

// Host
class host_platform: public platform
{
public:
  host_platform(): platform("CPU") { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;
};

// Vulkan
class vk_platform: public platform
{
private:
  unsigned get_compute_queue_idx(VkPhysicalDevice p_device) const;

public:
  vk_platform(vk_platform_t vk): platform("Vulkan"), vk_(vk){ }
  std::string version() const;
  void devices(std::vector<device *> &devices) const;

private:
  handle<vk_platform_t> vk_;
};

}

}

#endif
