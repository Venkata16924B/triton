#pragma once

#ifndef _TRITON_DRIVER_STREAM_H_
#define _TRITON_DRIVER_STREAM_H_

#include <map>
#include "triton/driver/context.h"
#include "triton/driver/device.h"
#include "triton/driver/handle.h"
#include "triton/driver/buffer.h"

namespace triton
{

namespace driver
{

class kernel;
class event;
class Range;
class cu_buffer;

// Base
class stream: public polymorphic_resource<CUstream, cl_command_queue, host_stream_t, vk_stream_t> {
public:
  stream(driver::context *ctx, CUstream, bool has_ownership);
  stream(driver::context *ctx, cl_command_queue, bool has_ownership);
  stream(driver::context *ctx, host_stream_t, bool has_ownership);
  stream(driver::context *ctx, vk_stream_t, bool has_ownership);
  // factory
  static driver::stream* create(driver::context* ctx);
  // accessors
  driver::context* context() const;
  // methods
  virtual void synchronize() = 0;
  virtual void enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const * = NULL, event *event = NULL) = 0;
  virtual void write(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void const* ptr) = 0;
  virtual void read(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void* ptr) = 0;
  // template helpers
  template<class T> void write(driver::buffer* buf, bool blocking, std::size_t offset, std::vector<T> const & x)
  { write(buf, blocking, offset, x.size()*sizeof(T), x.data()); }
  template<class T> void read(driver::buffer* buf, bool blocking, std::size_t offset, std::vector<T>& x)
  { read(buf, blocking, offset, x.size()*sizeof(T), x.data()); }

protected:
  driver::context *ctx_;
};

// Host
class host_stream: public stream {
public:
  // Constructors
  host_stream(driver::context *ctx);

  // Overridden
  void synchronize();
  void enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event *event);
  void write(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void* ptr);
};

// OpenCL
class cl_stream: public stream {
public:
  // Constructors
  cl_stream(driver::context *ctx);

  // Overridden
  void synchronize();
  void enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event *event);
  void write(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void* ptr);
};

// CUDA
class cu_stream: public stream {
public:
  // Constructors
  cu_stream(CUstream str, bool take_ownership);
  cu_stream(driver::context* context);

  // Overridden
  void synchronize();
  void enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event *event);
  void write(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void* ptr);
};

// Vulkan
class vk_stream: public stream {
public:
  // Constructors
  vk_stream(driver::context* context);

  // Overridden
  void synchronize();
  void enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event *event);
  void write(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void* ptr);

};

}

}

#endif
