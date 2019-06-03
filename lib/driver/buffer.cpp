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

#include <iostream>
#include "triton/driver/stream.h"
#include "triton/driver/buffer.h"
#include "triton/driver/context.h"
#include "triton/driver/dispatch.h"


namespace triton
{

namespace driver
{


//

buffer::buffer(driver::context* ctx, CUdeviceptr cu, bool take_ownership)
  : polymorphic_resource(cu, take_ownership), context_(ctx) { }

buffer::buffer(driver::context* ctx, cl_mem cl, bool take_ownership)
  : polymorphic_resource(cl, take_ownership), context_(ctx) { }

buffer::buffer(driver::context* ctx, host_buffer_t hst, bool take_ownership)
  : polymorphic_resource(hst, take_ownership), context_(ctx) { }

buffer::buffer(driver::context *ctx, vk_buffer_t vk, bool take_ownership)
  : polymorphic_resource(vk, take_ownership), context_(ctx) { }

driver::context* buffer::context() {
  return context_;
}

buffer* buffer::create(driver::context* ctx, size_t size) {
  switch(ctx->backend()){
  case CUDA: return new cu_buffer(ctx, size);
  case OpenCL: return new ocl_buffer(ctx, size);
  case Host: return new host_buffer(ctx, size);
  default: throw std::runtime_error("unknown backend");
  }
}

//

host_buffer::host_buffer(driver::context *context, size_t size)
  :  buffer(context, host_buffer_t(), true){
  hst_->data = new char[size];
}

//

ocl_buffer::ocl_buffer(driver::context* context, size_t size)
  : buffer(context, cl_mem(), true){
  cl_int err;
  *cl_ = dispatch::clCreateBuffer(*context->cl(), CL_MEM_READ_WRITE, size, NULL, &err);
  check(err);
}


//

cu_buffer::cu_buffer(driver::context* context, size_t size)
  : buffer(context, CUdeviceptr(), true) {
  cu_context::context_switcher ctx_switch(*context_);
  dispatch::cuMemAlloc(&*cu_, size);
}

cu_buffer::cu_buffer(driver::context* context, CUdeviceptr cu, bool take_ownership)
  : buffer(context, cu, take_ownership){
}

void cu_buffer::set_zero(driver::stream* queue, size_t size)
{
  cu_context::context_switcher ctx_switch(*context_);
  dispatch::cuMemsetD8Async(*cu_, 0, size, *queue->cu());
}

//
vk_buffer::vk_buffer(driver::context *context, size_t size)
  : buffer(context, vk_buffer_t(), true) {
  driver::vk_device* device = (driver::vk_device*)context->device();
  VkDevice vk_device = context->device()->vk()->device;
  // create buffer
  VkBufferCreateInfo buffer_create_info = {};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size = size;
  buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  dispatch::vkCreateBuffer(vk_device, &buffer_create_info, NULL, &vk_->buffer);
  // allocate memory
  VkMemoryRequirements memory_requirements;
  dispatch::vkGetBufferMemoryRequirements(vk_device, vk_->buffer, &memory_requirements);
  VkMemoryAllocateInfo allocateInfo = {};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = memory_requirements.size; // specify required memory.
  uint32_t idx = device->find_memory_type(memory_requirements.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  allocateInfo.memoryTypeIndex = idx;
  dispatch::vkAllocateMemory(vk_device, &allocateInfo, NULL, &vk_->memory);
  // bind buffer to allocated memory
  dispatch::vkBindBufferMemory(vk_device, vk_->buffer, vk_->memory, 0);
}

}

}
