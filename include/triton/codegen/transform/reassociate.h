#ifndef TDL_INCLUDE_IR_CODEGEN_REASSOCIATE_H
#define TDL_INCLUDE_IR_CODEGEN_REASSOCIATE_H

#include <map>
#include <set>
#include <vector>

namespace triton {

// forward declaration
namespace ir {
class module;
class value;
class builder;
class instruction;
class getelementptr_inst;
}

namespace codegen{

namespace analysis{
class grids;
class alignment_info;
}

namespace transform{

class reassociate {
  struct cst_info {
    ir::value* dyn_ptr;
    ir::getelementptr_inst* sta_ptr;
  };

private:
  ir::instruction* is_bin_add(ir::value *x);
  ir::value *reassociate_idx(ir::value *value, ir::builder &builder, ir::value *&noncst, ir::value *&cst);
  ir::value *reassociate_ptr(ir::getelementptr_inst* pz, ir::builder &builder, std::map<ir::value*, cst_info> &offsets);

public:
  reassociate(analysis::alignment_info* align, analysis::grids *params);
  void run(ir::module& module);

private:
  analysis::grids* params_;
  analysis::alignment_info* align_;
};

}

}

}

#endif
