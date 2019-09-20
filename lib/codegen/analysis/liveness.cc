#include <iostream>
#include "triton/codegen/instructions.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/transform/cts.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/instructions.h"
#include "triton/ir/value.h"

namespace triton{
namespace codegen{
namespace analysis{

// Entry point
void liveness::run(ir::module &mod) {
  for(ir::function *fn: mod.get_function_list()){
    // Assigns index to each instruction
    slot_index index = 0;
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *instr: block->get_inst_list()){
      index += 1;
      indices_.insert({instr, index});
    }
    // Liveness analysis
    // Creates live intervals
    for(auto i: indices_){
      ir::value *v = i.first;
//      ir::instruction* instr = dynamic_cast<ir::instruction*>(v);
//      if(!instr)
//        continue;
//      if(storage_info.at(instr->get_id()).first != SHARED)
//        continue;
      if(!info_->is_shared(v) || info_->get_reference(v))
        continue;
      unsigned start = i.second;
      unsigned end = start;
      for(ir::value *u: v->get_users()){
        start = std::min(start, indices_.at(u));
        end = std::max(end, indices_.at(u));
      }
      intervals_[v] = segment{start, end};
    }
  }
}

}
}
}