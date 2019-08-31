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
  VkDevice vk_device = program->context()->device()->vk()->device;

  std::vector<VkDescriptorSetLayoutBinding> bindings(2);
  for (uint32_t i = 0; i < bindings.size(); i++) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[i].pImmutableSamplers = nullptr;
  }

  VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  descriptor_layout_info.pBindings = bindings.data();
  descriptor_layout_info.bindingCount = bindings.size();
  dispatch::vkCreateDescriptorSetLayout(vk_device, &descriptor_layout_info, nullptr, &vk_->descriptor_layout);


  VkPipelineLayoutCreateInfo pipeline_layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &vk_->descriptor_layout;
  dispatch::vkCreatePipelineLayout(vk_device, &pipeline_layout_info, nullptr, &vk_->pipeline_layout);


  VkPipelineShaderStageCreateInfo shader_info = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  shader_info.module = *program->vk();
  shader_info.pName = name;
  shader_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  VkComputePipelineCreateInfo pipeline_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = shader_info;
  pipeline_info.layout = vk_->pipeline_layout;
  dispatch::vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_->pipeline);
}

void vk_kernel::setArg(unsigned int index, std::size_t size, void* ptr){
  throw std::runtime_error("not implemented");
}

void vk_kernel::setArg(unsigned int index, driver::buffer* buffer){
  throw std::runtime_error("not implemented");
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

