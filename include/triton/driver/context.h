#pragma once

#ifndef _TRITON_DRIVER_CONTEXT_H_
#define _TRITON_DRIVER_CONTEXT_H_

#include "triton/driver/device.h"
#include "triton/driver/handle.h"

namespace triton
{
namespace driver
{

class context: public polymorphic_resource<CUcontext, cl_context, host_context_t, vk_context_t>{
protected:
  static std::string get_cache_path();

public:
  context(driver::device *dev, CUcontext cu, bool take_ownership);
  context(driver::device *dev, cl_context cl, bool take_ownership);
  context(driver::device *dev, host_context_t hst, bool take_ownership);
  context(driver::device *dev, vk_context_t vk, bool take_ownership);
  driver::device* device() const;
  std::string const & cache_path() const;
  // factory methods
  static context* create(driver::device *dev);

protected:
  driver::device* dev_;
  std::string cache_path_;
};

// Host
class host_context: public context {
public:
  host_context(driver::device* dev);
};

// CUDA
class cu_context: public context {
public:
  class context_switcher{
  public:
      context_switcher(driver::context const & ctx);
      ~context_switcher();
  private:
      driver::cu_context const & ctx_;
  };

private:
  static CUdevice get_device_of(CUcontext);

public:
  //Constructors
  cu_context(CUcontext cu, bool take_ownership = true);
  cu_context(driver::device* dev);
};

// OpenCL
class ocl_context: public context {
public:
  ocl_context(driver::device* dev);
};

// Vulkan
class vk_context: public context {
public:
  vk_context(driver::device* dev);
};



}
}

#endif
