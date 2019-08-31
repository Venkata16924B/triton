#include "triton/codegen/transform/vectorize.h"
#include "triton/codegen/analysis/tune.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"

namespace triton {

namespace codegen{
namespace transform{

void vectorize::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    if(auto *trans = dynamic_cast<ir::trans_inst*>(i)){
      ir::value *x = i->get_operand(0);
      if(trans->get_perm()[0]->get_value() != 0)
        continue;
      builder.set_insert_point(i);
      ir::instruction *rx = (ir::instruction*)builder.create_vectorize(x);
      x->replace_all_uses_with(rx);
      rx->set_operand(0, x);
      params_->copy(rx, x);
    }
    if(dynamic_cast<ir::copy_to_shared_inst*>(i)){
      ir::value *x = i->get_operand(0);
      if(params_->get_param(x, "nts.d0")->get_value() == 1)
        continue;
      builder.set_insert_point(i);
      ir::instruction *rx = (ir::instruction*)builder.create_vectorize(x);
      x->replace_all_uses_with(rx);
      rx->set_operand(0, x);
      params_->copy(rx, x);
    }
  }
}

}
}
}
