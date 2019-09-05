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

#include <vector>
#include <stdexcept>
#include "triton/driver/dispatch.h"
#include "triton/driver/backend.h"
#include "triton/driver/buffer.h"
#include "triton/driver/context.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"


namespace triton
{

namespace driver
{

/*-----------------------------------*/
//-----------  Platforms ------------*/
/*-----------------------------------*/

void backend::platforms::init() {
  if(!cache_.empty())
    return;
  //if CUDA is here
//  if(dispatch::cuinit()){
//    cache_.push_back(new cu_platform());
//  }
//  //if OpenCL is here
//  if(dispatch::clinit()){
//    cl_uint num_platforms;
//    dispatch::clGetPlatformIDs(0, nullptr, &num_platforms);
//    std::vector<cl_platform_id> ids(num_platforms);
//    dispatch::clGetPlatformIDs(num_platforms, ids.data(), nullptr);
//    for(cl_platform_id id: ids)
//      cache_.push_back(new cl_platform(id));
//  }
    // vulkan
  if(dispatch::vkinit()) {
    std::vector<platform*> pf;

    /*
    Contains application info. This is actually not that important.
    The only real important field is apiVersion.
    */
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "triton";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_API_VERSION_1_0;;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;

    VkInstance vk;
    dispatch::vkCreateInstance(&createInfo, nullptr, &vk);
    cache_.push_back(new vk_platform(vk));
  }
//  //if host is here
//  bool host_visible = true;
//  if(host_visible){
//    cache_.push_back(new host_platform());
//  }
  if(cache_.empty())
    throw std::runtime_error("Triton: No backend available. Make sure CUDA is available in your library path");
}

void backend::platforms::get(std::vector<platform *> &results) {
  std::copy(cache_.begin(), cache_.end(), std::back_inserter(results));
}

std::vector<driver::platform*> backend::platforms::cache_;


/*-----------------------------------*/
//-----------  Devices --------------*/
/*-----------------------------------*/

void backend::devices::init(std::vector<platform*> const & platforms) {
  if(!cache_.empty())
    return;
  for(driver::platform* pf: platforms)
    pf->devices(cache_);
  if(cache_.empty())
    throw std::runtime_error("Triton: No device available. Make sure that your platform is configured properly");
}

void backend::devices::get(std::vector<device*> &devs) {
  std::copy(cache_.begin(), cache_.end(), std::back_inserter(devs));
}

std::vector<driver::device*> backend::devices::cache_;



/*-----------------------------------*/
//---------- Modules ----------------*/
/*-----------------------------------*/

void backend::modules::release(){
  for(auto & x: cache_)
    delete x.second;
  cache_.clear();
}

driver::module* backend::modules::get(driver::stream* stream, std::string const & name, llvm::Module* src){
  std::tuple<driver::stream*, std::string> key(stream, name);
  if(cache_.find(key)==cache_.end()){
    return &*cache_.insert({key, driver::module::create(stream->context(), src)}).first->second;
  }
  return &*cache_.at(key);
}

std::map<std::tuple<driver::stream*, std::string>, driver::module*>  backend::modules::cache_;

/*-----------------------------------*/
//-----------  Kernels --------------*/
/*-----------------------------------*/

void backend::kernels::release(){
  for(auto & x: cache_)
    delete x.second;
  cache_.clear();
}

driver::kernel* backend::kernels::get(driver::module *mod, std::string const & name){
  std::tuple<driver::module*, std::string> key(mod, name);
  if(cache_.find(key)==cache_.end()){
    return &*cache_.insert({key, driver::kernel::create(mod, name.c_str())}).first->second;
  }
  return cache_.at(key);
}

std::map<std::tuple<driver::module*, std::string>, driver::kernel*> backend::kernels::cache_;

/*-----------------------------------*/
//------------  Queues --------------*/
/*-----------------------------------*/

void backend::streams::init(std::list<driver::context*> const & contexts){
  for(driver::context* ctx : contexts)
    if(cache_.find(ctx)==cache_.end())
      cache_.insert(std::make_pair(ctx, std::vector<driver::stream*>{driver::stream::create(ctx)}));
}

void backend::streams::release(){
  for(auto & x: cache_)
    for(auto & y: x.second)
      delete y;
  cache_.clear();
}

driver::stream* backend::streams::get_default()
{ return get(contexts::get_default(), 0); }

driver::stream* backend::streams::get(driver::context* context, unsigned int id){
  init(std::list<driver::context*>(1,context));
  for(auto & x : cache_)
    if(x.first==context)
      return x.second[id];
  throw;
}

void backend::streams::get(driver::context* context, std::vector<driver::stream*> & queues){
  init(std::list<driver::context*>(1,context));
  queues = cache_.at(context);
}

std::map<driver::context*, std::vector<driver::stream*>> backend::streams::cache_;

/*-----------------------------------*/
//------------  Contexts ------------*/
/*-----------------------------------*/

void backend::contexts::init(std::vector<driver::device*> const & devices){
  for(driver::device* dvc: devices)
    cache_.push_back(driver::context::create(dvc));
}

void backend::contexts::release(){
  for(auto & x: cache_)
    delete x;
  cache_.clear();
}

driver::context* backend::contexts::get_default(){
  backend::init();
  auto it = cache_.begin();
  std::advance(it, default_device);
  return *it;
}

void backend::contexts::get(std::list<driver::context*> & contexts){
  backend::init();
  contexts = cache_;
}

std::list<driver::context*> backend::contexts::cache_;



/*-----------------------------------*/
//------------  General -------------*/
/*-----------------------------------*/

void backend::synchronize(driver::context* context){
  for(driver::stream * queue: streams::cache_.at(context))
    queue->synchronize();
}


void backend::release(){
  backend::kernels::release();
//  backend::programs::release();
  backend::streams::release();
  backend::contexts::release();
}


void backend::init(){
  if(!contexts::cache_.empty())
    return;
  // initialize platforms
  backend::platforms::init();
  // initialize devices
  backend::devices::init(platforms::cache_);
  // initialize contexts
  backend::contexts::init(devices::cache_);
  // initialize streams
  streams::init(contexts::cache_);
}

unsigned int backend::default_device = 0;

}

}
