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

#include <string.h>
#include "triton/driver/kernel.h"
#include "triton/driver/buffer.h"

namespace triton
{

namespace driver
{


/* ------------------------ */
//         Base             //
/* ------------------------ */

kernel::kernel(driver::module *program, CUfunction fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel::kernel(driver::module *program, cl_kernel fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel::kernel(driver::module *program, host_function_t fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel::kernel(driver::module *program, vk_function_t fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel* kernel::create(driver::module* program, const char* name) {
    switch(program->backend()){
    case CUDA: return new cu_kernel(program, name);
    case OpenCL: return new ocl_kernel(program, name);
    case Host: return new host_kernel(program, name);
    case Vulkan: return new vk_kernel(program, name);
    default: throw std::runtime_error("unknown backend");
    }
}

driver::module* kernel::module() {
  return program_;
}

/* ------------------------ */
//         Host             //
/* ------------------------ */

host_kernel::host_kernel(driver::module* program, const char *name): kernel(program, host_function_t(), true) {
  hst_->fn = program->hst()->functions.at(name);
}

void host_kernel::setArg(unsigned int index, std::size_t size, void* ptr){
  if(index + 1> params_store_.size()){
    params_store_.resize(index+1);
    params_.resize(index+1);
  }
  params_store_[index].reset(malloc(size), free);
  memcpy(params_store_[index].get(), ptr, size);
  params_[index] = params_store_[index].get();
}

void host_kernel::setArg(unsigned int index, driver::buffer* buffer){
  if(buffer)
    kernel::setArg(index, (void*)buffer->hst()->data);
  else
    kernel::setArg(index, (std::ptrdiff_t)0);
}

const std::vector<void *> &host_kernel::params(){
  return params_;
}

/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

ocl_kernel::ocl_kernel(driver::module* program, const char* name): kernel(program, cl_kernel(), true) {
//  cl_uint res;
//  check(dispatch::clCreateKernelsInProgram(*program->cl(), 0, NULL, &res));
//  std::cout << res << std::endl;
  cl_int err;
  *cl_ = dispatch::clCreateKernel(*program->cl(), "matmul", &err);
  check(err);
}

void ocl_kernel::setArg(unsigned int index, std::size_t size, void* ptr) {
  check(dispatch::clSetKernelArg(*cl_, index, size, ptr));
}

void ocl_kernel::setArg(unsigned int index, driver::buffer* buffer) {
  if(buffer)
    check(dispatch::clSetKernelArg(*cl_, index, sizeof(cl_mem), (void*)&*buffer->cl()));
  else
    kernel::setArg(index, (std::ptrdiff_t)0);
}


/* ------------------------ */
//         Vulkan           //
/* ------------------------ */

vk_kernel::vk_kernel(driver::module* program, const char * name): kernel(program, vk_function_t(), true){
}

void vk_kernel::initPipeline() {
    VkDevice vk_device = module()->context()->device()->vk()->device;

    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
    // create descriptor pool.
    dispatch::vkCreateDescriptorPool(vk_device, &descriptorPoolCreateInfo, NULL, &vk_->descriptor_pool);

    // With the pool allocated, we can now allocate the descriptor set.
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = vk_->descriptor_pool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = vk_params_.size(); // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts = vk_params_.data();

    // allocate descriptor set.
    dispatch::vkAllocateDescriptorSets(vk_device, &descriptorSetAllocateInfo, &vk_->descriptor_set);


//    // Specify the buffer to bind to the descriptor.
//    VkDescriptorBufferInfo descriptorBufferInfo = {};
//    descriptorBufferInfo.buffer = buffer;
//    descriptorBufferInfo.offset = 0;
//    descriptorBufferInfo.range = bufferSize;

//    VkWriteDescriptorSet writeDescriptorSet = {};
//    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
//    writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
//    writeDescriptorSet.dstBinding = 0; // write to the first, and only binding.
//    writeDescriptorSet.descriptorCount = 1; // update a single descriptor.
//    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
//    writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;

//    // perform the update of the descriptor set.
//    dispatch::vkUpdateDescriptorSets(vk_device, 1, &writeDescriptorSet, 0, NULL);
}

void vk_kernel::setArg(unsigned int index, std::size_t size, void* ptr){
  throw std::runtime_error("not implemented");
}

void vk_kernel::setArg(unsigned int index, driver::buffer* buffer){
  if(index + 1> vk_params_store_.size()){
    vk_params_store_.resize(index+1);
    vk_params_.resize(index+1);
  }

  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = index; // binding = 0
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.bindingCount = 1; // only a single binding in this descriptor set layout.
  create_info.pBindings = &binding;

  VkDescriptorSetLayout *layout = new VkDescriptorSetLayout();
  VkDevice device = module()->context()->device()->vk()->device;
  dispatch::vkCreateDescriptorSetLayout(device, &create_info, nullptr, layout);
  vk_params_store_[index].reset(layout);
  vk_params_[index] = vk_params_store_[index].get();
}


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

cu_kernel::cu_kernel(driver::module *program, const char * name) : kernel(program, CUfunction(), true) {
  cu_params_store_.reserve(64);
  cu_params_.reserve(64);
  dispatch::cuModuleGetFunction(&*cu_, *program->cu(), name);
//  dispatch::cuFuncSetCacheConfig(*cu_, CU_FUNC_CACHE_PREFER_SHARED);
}

void cu_kernel::setArg(unsigned int index, std::size_t size, void* ptr){
  if(index + 1> cu_params_store_.size()){
    cu_params_store_.resize(index+1);
    cu_params_.resize(index+1);
  }
  cu_params_store_[index].reset(malloc(size), free);
  memcpy(cu_params_store_[index].get(), ptr, size);
  cu_params_[index] = cu_params_store_[index].get();
}

void cu_kernel::setArg(unsigned int index, driver::buffer* data){
  if(data)
    kernel::setArg(index, *data->cu());
  else
    kernel::setArg(index, (std::ptrdiff_t)0);
}

void* const* cu_kernel::cu_params() const
{ return cu_params_.data(); }


}

}

