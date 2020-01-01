#include <numeric>
#include <iostream>
#include "llvm/IR/IRBuilder.h"
#include "triton/codegen/selection/machine_value.h"

namespace triton{
namespace codegen{

using namespace llvm;

/* Distributed Tile */
void distributed_tile::init_indices() {
  std::vector<size_t> id(axes_.size(), 0);
  // create iteration order
  std::vector<size_t> order(id.size());
  std::iota(order.begin(), order.end(), 0);
  auto cmp = [&](int x, int y) {
    return order_[x] < order_[y];
  };
  std::sort(order.begin(), order.end(), cmp);

  // build
  size_t k = 0;
  while(true) {
    indices_t current;
    for(size_t d = 0; d < id.size(); d++)
      current.push_back(axes_[d].values[id[d]]);
    size_t sz = indices_.size();
    indices_[current] = sz;
    values_[current] = nullptr;
    ordered_indices_.push_back(current);
    id[order[0]]++;
    while(id[order[k]] == axes_[order[k]].values.size()){
      if(k == id.size() - 1)
        return;
      id[order[k++]] = 0;
      id[order[k]]++;
    }
    k = 0;
  }
}

llvm::Type *distributed_tile::make_vector_ty(llvm::Type *ty, size_t vector_size) {
  if(vector_size == 1)
    return ty;
  return VectorType::get(ty, vector_size);
}

distributed_tile::distributed_tile(Type *ty, const shapes_t &shapes, const std::vector<int>& order, const axes_t &axes, llvm::IRBuilder<> &builder, bool vectorize)
    : tile(make_vector_ty(ty, vectorize?axes[0].contiguous:1), shapes), axes_(axes), order_(order), builder_(builder) {
  vector_size_ = vectorize?ty_->getVectorNumElements():1;
  init_indices();
}

void distributed_tile::set_value(indices_t idx, Value *x) {
  assert(x->getType() == ty_ && "cannot set a value of different type");
  Value *&result = values_[idx];
  assert(!result && "value cannot be set twice");
  result = x;
}

Value* distributed_tile::get_value(indices_t idx) {
  Value *result = values_.at(idx);
  assert(result && "value has not been set");
  return result;
}

unsigned distributed_tile::get_linear_index(indices_t idx) {
  return indices_[idx];
}

indices_t distributed_tile::get_ordered_indices(unsigned id) {
  return ordered_indices_.at(id);
}


void distributed_tile::for_each(std::function<void (indices_t)> fn) {
  for(unsigned i = 0; i < ordered_indices_.size(); i++){
    if(i % vector_size_ == 0)
      fn(ordered_indices_[i]);
  }
}

/* Shared Tile */
void shared_tile::extract_constant(Value *arg, Value *&non_cst, Value *&cst) {
  BinaryOperator *bin_op = dyn_cast<BinaryOperator>(arg);
  Constant *_0 = ConstantInt::get(Type::getInt32Ty(arg->getContext()), 0);
  if(dyn_cast<Constant>(arg)){
    cst = arg;
    non_cst = _0;
    return;
  }
  if(!bin_op || bin_op->getOpcode() != llvm::BinaryOperator::Add){
    non_cst = arg;
    cst = _0;
    return;
  }
  Constant *cst_lhs = dyn_cast<Constant>(bin_op->getOperand(0));
  Constant *cst_rhs = dyn_cast<Constant>(bin_op->getOperand(1));
  if(cst_lhs && cst_rhs){
    cst = arg;
    non_cst = _0;
  }
  else if(cst_lhs){
    cst = cst_lhs;
    non_cst = bin_op->getOperand(1);
  }
  else if(cst_rhs){
    cst = cst_rhs;
    non_cst = bin_op->getOperand(0);
  }
  else{
    non_cst = arg;
    cst = _0;
  }
}

void shared_tile::extract_constant(const indices_t &arg_idx, indices_t &non_cst_idx, indices_t &cst_idx) {
  non_cst_idx.clear();
  cst_idx.clear();
  for(Value *idx: arg_idx){
    Value *non_cst, *cst;
    extract_constant(idx, non_cst, cst);
    non_cst_idx.push_back(non_cst);
    cst_idx.push_back(cst);
  }
}


Value* shared_tile::shared_offset(size_t stride, llvm::IRBuilder<> &builder, const shapes_t& shapes, const std::vector<int>& perm, const std::vector<int>& order, indices_t idx) {
  // strides
  std::vector<Value*> strides(order.size());
  strides[order[0]] = builder.getInt32(stride);
  for(size_t i = 1; i < idx.size(); i++)
    strides[order[i]] = builder.CreateMul(strides[order[i-1]], builder.getInt32(shapes[order[i-1]]));
  // result
  Value *result = builder.getInt32(0);
  for(size_t i = 0; i < strides.size(); i++)
    result = builder.CreateAdd(result, builder.CreateMul(idx[perm[i]], strides[i]));
  return result;
}

shared_tile::shared_tile(Type *ty, const shapes_t &shapes, const std::vector<int>& order, Value *ptr, llvm::IRBuilder<> &builder, Value *offset, const std::vector<int>& perm):
  tile(ty, shapes), order_(order), ptr_(ptr), builder_(builder), offset_(offset), vector_size_(1), perm_(perm){
  return_vector_ = false;
  if(perm_.empty()){
    perm_.resize(shapes.size());
    std::iota(perm_.begin(), perm_.end(), 0);
  }
}

void shared_tile::set_value(indices_t idx, Value *value) {
  Value *ptr = builder_.CreateAdd(ptr_, shared_offset(ty_->getPrimitiveSizeInBits()/8, builder_, shapes_, perm_, order_, idx));
  unsigned addr_space = 3;
  ptr = builder_.CreateBitCast(ptr, value->getType()->getPointerTo(addr_space));
  builder_.CreateStore(value, ptr);
}

void shared_tile::set_vector_size(unsigned vector_size) {
  vector_size_ = vector_size;
}

void shared_tile::set_return_mode(bool return_vector){
  return_vector_ = return_vector;
}


Value* shared_tile::get_value(indices_t idx) {
  indices_t non_cst_idx, cst_idx;
  extract_constant(idx, non_cst_idx, cst_idx);
  Value *&base_ptr = ptr_cache_[non_cst_idx];
  unsigned vector_size = vector_size_;
  Type *ty = ty_;
  if(ty->isHalfTy() && (vector_size % 2 == 0)){
    ty = IntegerType::get(ty->getContext(), 32);
    vector_size = vector_size / 2;
  }
  Type *ptr_ty = ty->getPointerTo(3);
  if(base_ptr == nullptr)
    base_ptr = builder_.CreateAdd(ptr_, shared_offset(ty_->getPrimitiveSizeInBits()/8, builder_, shapes_, perm_, order_, non_cst_idx));
  if(vector_size_ > 1){
    Type *vec_ty = VectorType::get(ty, vector_size);
    ptr_ty = PointerType::get(vec_ty, 3);
  }

  Value *offset = shared_offset(1, builder_, shapes_, perm_, order_, cst_idx);
  Value *div = offset;
  if(vector_size_ > 1)
    div = builder_.CreateUDiv(offset, builder_.getInt32(vector_size_));
  div = builder_.CreateMul(div, builder_.getInt32(vector_size * ty->getPrimitiveSizeInBits()/8));

  Value *&packed = val_cache_[std::make_pair(base_ptr, div)].first;
  std::vector<Value*> &unpacked = val_cache_[std::make_pair(base_ptr, div)].second;

  if(packed == nullptr){
    Value *ptr = builder_.CreateAdd(base_ptr, div);
    packed = builder_.CreateLoad(builder_.CreateBitCast(ptr, ptr_ty));
    for(int n = 0; n < vector_size_; n++)
      unpacked.push_back(builder_.CreateExtractElement(packed, n));
  }

  if(return_vector_ || vector_size_ == 1)
    return packed;
  Value *rem = builder_.CreateURem(offset, builder_.getInt32(vector_size_));
  return unpacked.at(dyn_cast<ConstantInt>(rem)->getZExtValue());
}



}
}
