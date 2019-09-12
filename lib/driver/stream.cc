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

#include <cassert>
#include <array>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/driver/context.h"
#include "triton/driver/device.h"
#include "triton/driver/event.h"
#include "triton/driver/kernel.h"
#include "triton/driver/buffer.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"

namespace triton
{

namespace driver
{

/* ------------------------ */
//         Base             //
/* ------------------------ */

stream::stream(driver::context *ctx, CUstream cu, bool has_ownership)
  : polymorphic_resource(cu, has_ownership), ctx_(ctx) {
}

stream::stream(driver::context *ctx, cl_command_queue cl, bool has_ownership)
  : polymorphic_resource(cl, has_ownership), ctx_(ctx) {
}

stream::stream(driver::context *ctx, host_stream_t cl, bool has_ownership)
  : polymorphic_resource(cl, has_ownership), ctx_(ctx) {
}

stream::stream(driver::context *ctx, vk_stream_t vk, bool has_ownership)
  : polymorphic_resource(vk, has_ownership), ctx_(ctx) {

}
driver::stream* stream::create(driver::context* ctx) {
  switch(ctx->backend()){
    case CUDA: return new cu_stream(ctx);
    case OpenCL: return new cl_stream(ctx);
    case Host: return new host_stream(ctx);
    case Vulkan: return new vk_stream(ctx);
    default: throw std::runtime_error("unknown backend");
  }
}

driver::context* stream::context() const {
  return ctx_;
}

/* ------------------------ */
//          Host            //
/* ------------------------ */

host_stream::host_stream(driver::context *ctx): stream(ctx, host_stream_t(), true) {

}

void host_stream::synchronize() {

}

void host_stream::enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event* event) {
  driver::host_kernel* hst_kernel = (host_kernel*)kernel;
  llvm::ExecutionEngine* engine = kernel->module()->hst()->engine;
  void (*fn)(char**, int32_t, int32_t, int32_t) = (void(*)(char**, int32_t, int32_t, int32_t))engine->getFunctionAddress("main");
  for(size_t i = 0; i < grid[0]; i++)
    for(size_t j = 0; j < grid[1]; j++)
      for(size_t k = 0; k < grid[2]; k++)
        fn((char**)hst_kernel->params().data(), int32_t(i), int32_t(j), int32_t(k));
}

void host_stream::write(driver::buffer* buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr) {
  std::memcpy((void*)buffer->hst()->data, ptr, size);
}

void host_stream::read(driver::buffer* buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr) {
  std::memcpy(ptr, (const void*)buffer->hst()->data, size);
}


/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

cl_stream::cl_stream(driver::context *ctx): stream(ctx, cl_command_queue(), true) {
  cl_int err;
  *cl_ = dispatch::clCreateCommandQueue(*ctx->cl(), *ctx->device()->cl(), 0, &err);
  check(err);
}

void cl_stream::synchronize() {
  check(dispatch::clFinish(*cl_));
}

void cl_stream::enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event* event) {
  std::array<size_t, 3> global = {grid[0]*block[0], grid[1]*block[1], grid[2]*block[2]};
  check(dispatch::clEnqueueNDRangeKernel(*cl_, *kernel->cl(), grid.size(), NULL, (const size_t*)global.data(), (const size_t*)block.data(), 0, NULL, NULL));
}

void cl_stream::write(driver::buffer* buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr) {
  check(dispatch::clEnqueueWriteBuffer(*cl_, *buffer->cl(), blocking?CL_TRUE:CL_FALSE, offset, size, ptr, 0, NULL, NULL));
}

void cl_stream::read(driver::buffer* buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr) {
  check(dispatch::clEnqueueReadBuffer(*cl_, *buffer->cl(), blocking?CL_TRUE:CL_FALSE, offset, size, ptr, 0, NULL, NULL));
}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */

inline CUcontext get_context() {
  CUcontext result;
  dispatch::cuCtxGetCurrent(&result);
  return result;
}

cu_stream::cu_stream(CUstream str, bool take_ownership):
  stream(backend::contexts::import(get_context()), str, take_ownership) {
}

cu_stream::cu_stream(driver::context *context): stream((driver::cu_context*)context, CUstream(), true) {
  cu_context::context_switcher ctx_switch(*ctx_);
  dispatch::cuStreamCreate(&*cu_, 0);
}

void cu_stream::synchronize() {
  cu_context::context_switcher ctx_switch(*ctx_);
  dispatch::cuStreamSynchronize(*cu_);
}

void cu_stream::enqueue(driver::kernel* kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<event> const *, event* event) {
  driver::cu_kernel* cu_kernel = (driver::cu_kernel*)kernel;
  cu_context::context_switcher ctx_switch(*ctx_);
  if(event)
    dispatch::cuEventRecord(event->cu()->first, *cu_);
  dispatch::cuLaunchKernel(*kernel->cu(), grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, *cu_,(void**)cu_kernel->cu_params(), NULL);
  if(event)
    dispatch::cuEventRecord(event->cu()->second, *cu_);
}

void cu_stream::write(driver::buffer* buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr) {
  cu_context::context_switcher ctx_switch(*ctx_);
  if(blocking)
    dispatch::cuMemcpyHtoD(*buffer->cu() + offset, ptr, size);
  else
    dispatch::cuMemcpyHtoDAsync(*buffer->cu() + offset, ptr, size, *cu_);
}

void cu_stream::read(driver::buffer* buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr) {
  cu_context::context_switcher ctx_switch(*ctx_);
  if(blocking)
    dispatch::cuMemcpyDtoH(ptr, *buffer->cu() + offset, size);
  else
    dispatch::cuMemcpyDtoHAsync(ptr, *buffer->cu() + offset, size, *cu_);
}

/* ------------------------ */
//         Vulkan           //
/* ------------------------ */


vk_stream::vk_stream(driver::context* context):
    stream(context, vk_stream_t(), true){
  VkDevice dev = context->device()->vk()->device;
  unsigned queue_family_idx = 0;
  unsigned queue_idx = 0;
  // create queue
  dispatch::vkGetDeviceQueue(context->device()->vk()->device, queue_family_idx, queue_idx, &vk_->queue);
  // create pool
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = 0;
  pool_info.queueFamilyIndex = 0;
  dispatch::vkCreateCommandPool(dev, &pool_info, nullptr, &vk_->pool);
  // create command buffer
  VkCommandBufferAllocateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  buffer_info.commandPool = vk_->pool;
  buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  buffer_info.commandBufferCount = 1;
  dispatch::vkAllocateCommandBuffers(dev, &buffer_info, &vk_->buffer);
}
  // Overridden
void vk_stream::synchronize() {

}

void vk_stream::enqueue(driver::kernel* kernel, std::array<size_t, 3> grid,
                        std::array<size_t, 3> block, std::vector<event> const *, event *event) {
    driver::vk_kernel* vk_kernel = (driver::vk_kernel*)kernel;
    vk_kernel->initPipeline();
    // start recording
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    dispatch::vkBeginCommandBuffer(vk_->buffer, &begin_info);
    // bind pipeline
    dispatch::vkCmdBindPipeline(vk_->buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                kernel->vk()->pipeline);
    // bind descriptors
    dispatch::vkCmdBindDescriptorSets(vk_->buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      kernel->vk()->pipeline_layout, 0, 1, &kernel->vk()->descriptor_set, 0, nullptr);
    // dispatch
    dispatch::vkCmdDispatch(vk_->buffer, grid[0], grid[1], grid[2]);
}

void vk_stream::write(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void const* ptr) {

}

void vk_stream::read(driver::buffer* buf, bool blocking, std::size_t offset, std::size_t size, void* ptr) {

}



}

}
