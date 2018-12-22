#include <cstring>
#include <cstdio>
#include "ast.h"
#include "codegen.h"

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
using tdl::ast::translation_unit;
extern translation_unit *ast_root;

const char src[] =
"\
void test(fp32 *A, fp32 *B, fp32 *C){\
  int32 i = 0;\
}\
";

int main() {
   YY_BUFFER_STATE buffer = yy_scan_string(src);
   yyparse();
   yy_delete_buffer(buffer);
   translation_unit *program = ast_root;
   tdl::context context;
   tdl::module module("matrix", &context);
   program->codegen(&module);
   return 0;
}