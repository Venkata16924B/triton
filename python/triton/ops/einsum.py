import numpy as np
import torch
from math import ceil, log2
from enum import IntEnum
import triton
from functools import reduce
from operator import mul
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
from collections import OrderedDict
from collections import namedtuple
import re
from sympy.printing.ccode import C89CodePrinter

class TritonCodePrinter(C89CodePrinter):
    
    def __init__(self, axes_0, axes_1, axes_2):
        super(TritonCodePrinter, self).__init__()
        self.axes_0 = axes_0
        self.axes_1 = axes_1
        self.axes_2 = axes_2

    def _print_Symbol(self, expr):
        name = super(C89CodePrinter, self)._print_Symbol(expr)
        if expr in self.axes_0:
            return f'r{name}[:, newaxis, newaxis]'
        if expr in self.axes_1:
            return f'r{name}[newaxis, :, newaxis]'
        if expr in self.axes_2:
            return f'r{name}[newaxis, newaxis, :]'
        return name

    def _print_Indexed(self, expr):
        assert len(expr.indices) == 1
        return "*(%s + %s)" % (self._print(expr.base.label),
                               self._print(expr.indices[0]))
        
def print_triton(expr, axes_0, axes_1, axes_2):
    return TritonCodePrinter(axes_0, axes_1, axes_2).doprint(expr)

class _einsum(triton.function):


    cache = dict()

    #############################
    ## Triton-C code generation
    #############################

    def unpack_cc(tile, axes, prefix):
        ret = ''
        axes = list(map(str, axes))
        for i, d in enumerate(reversed(axes)):
            if i == len(axes) - 1:
                break
            currs = ''.join(axes[: len(axes) - i])
            nexts = ''.join(axes[: len(axes) - (i + 1)])
            ret += f'    int {prefix}{nexts}[{tile}] = r{currs} / dim_{d};\n'
            ret += f'    int {prefix}{d}[{tile}] = r{currs} % dim_{d};\n'
        return ret

    def strides_cc(name, expr):
        ret = [f'stride_{name}_{d}' for d in expr[:-1]] + ['1']
        ret = dict(zip(expr, ret))
        return ret

    def make_kernel(name,
                    expr_a, expr_b, expr_c,
                    axes_m, axes_n, axes_k, axes_b,
                    multipleof_a, multipleof_b, multipleof_c,
                    lut_mode_a, lut_mode_b,
                    subscripted):

        src = f"""
__global__ void {name}(
              TYPE * A __noalias __readonly __aligned(16)
            , TYPE * B __noalias __readonly __aligned(16)
            , TYPE * C
            , int * locks
            , float alpha
            , int matmul_m, int matmul_n, int matmul_k __multipleof(64)
            , int div_m
            , int off_a, int off_b, int off_c
            """
        for dim in [axes_m, axes_n, axes_k, axes_b]:
            for d in dim:
                src += f", int dim_{d}"
        src += "\n            "
        for dim, name, mult in zip([expr_a, expr_b, expr_c],
                                         ['a', 'b', 'c'],
                                         [multipleof_a, multipleof_b, multipleof_c]):
            for d in range(len(dim) - 1):
                attr = f'__multipleof({mult})'
                src += f", int stride_{name}_{d} {attr}"
            src += "\n            "
        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_a_inner __multipleof({multipleof_a})"
        else:
            src += ", int* AD"
        src += "\n            "
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_b_inner __multipleof({multipleof_b})"
        else:
            src += ", int* BD"
        for ptr in subscripted:
            src += f", int* {ptr}"
        src += """) {

    // re-order outer program ids
    int grid_n = (matmul_n + TN - 1) / TN;
    int pid_mn = get_program_id(0) / div_m;
    int pid_n = pid_mn % grid_n;
    int pid_m = (pid_mn / grid_n)*div_m + (get_program_id(0) % div_m);

    // get batch program id
    int pid_b = get_program_id(1);

    // get reduction sub-group program id
    int pid_z = get_program_id(2);
    int grid_z = get_num_programs(2);
    int div_z = matmul_k / TZ;
    int rem_z = matmul_k % TZ;
    int off_k = pid_z * div_z;
    matmul_k = select(pid_z < rem_z, div_z, div_z + rem_z);

    // create ranges
"""
        rk = 'r{}'.format(''.join(map(str,axes_k)))
        src += f"    int {rk}[TK] = off_k + 0 ... TK;\n"

        for axes, tile, pid in zip([axes_m, axes_n, axes_b],
                                   ['TM', 'TN', 'TB'],
                                   ['pid_m', 'pid_n', 'pid_b']):
            currs = ''.join(map(str,axes))
            if axes:
                src += f"    int r{currs}[{tile}] = {pid} * {tile} + 0 ... {tile};\n"
        

        if axes_m:
            src += _einsum.unpack_cc('TM', axes_m, 'r')
        if axes_n:
            src += _einsum.unpack_cc('TN', axes_n, 'r')
        if axes_b:
            src += _einsum.unpack_cc('TB', axes_b, 'r')
        if axes_k:
            src += _einsum.unpack_cc('TK', axes_k, 'r')


        src += """    

    // initialize pointers to A
    TYPE *pa[TM, TK, TB] = A + off_a"""
        for i, sym in enumerate(expr_a):
            ccode = print_triton(sym, axes_m, axes_k, axes_b)
            stride = f'stride_a_{i}' if i < len(expr_a) - 1 else '1'
            src += f" + ({ccode}) * {stride}\n                            "
        src += ';'

        if not lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += f"""
    // initialize pointers to A look-up table
    int *padelta[TK]  = AD  + 0 ... TK;"""
    
        src += """

    // initialize pointers to B
    TYPE *pb[TK, TN, TB] = B + off_b"""
        for i, sym in enumerate(expr_b):
            ccode = print_triton(sym, axes_k, axes_n, axes_b)
            stride = f'stride_b_{i}' if i < len(expr_b) - 1 else '1'
            src += f" + ({ccode}) * {stride}\n                            "
        src += ';'


        if not lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += f"""
    // initialize pointers to B look-up table
    int *pbdelta[TK]  = BD  + 0 ... TK;"""

        #print(axes_k)
        src += f"""
    
    // prefetch
    {rk} -= off_k;
    bool checka[TM, TK, TB] = {rk}[newaxis, :, newaxis] < matmul_k;
    bool checkb[TK, TN, TB] = {rk}[:, newaxis, newaxis] < matmul_k;
    TYPE a[TM, TK, TB] = checka ? *pa : 0;
    TYPE b[TK, TN, TB] = checkb ? *pb : 0;

    // accumulate
    float acc[TM, TN, TB] = 0;
    for(int k = matmul_k; k > 0; k -= TK) {{
        acc += a @ b;"""

        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += """
        pa += stride_a_inner;"""
        else:
            src += """
        pa += (*padelta)[newaxis, :, newaxis];
        padelta += TK;"""

        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += """
        pb += stride_b_inner;"""
        else:
            src += """
        pb += (*pbdelta)[:, newaxis, newaxis];
        pbdelta += TK;"""

        src += f"""
        bool checka[TM, TK, TB] = {rk}[newaxis, :, newaxis] < k - TK;
        bool checkb[TK, TN, TB] = {rk}[:, newaxis, newaxis] < k - TK;
        a = checka ? *pa : 0;
        b = checkb ? *pb : 0;
    }}
    acc = acc * alpha;

 """

    
        src += """

    // initialize pointers to C
    TYPE *pc[TM, TN, TB] = C + off_c"""
        for i, sym in enumerate(expr_c):
            ccode = print_triton(sym, axes_m, axes_n, axes_b)
            stride = f'stride_c_{i}' if i < len(expr_c) - 1 else '1'
            src += f" + ({ccode}) * {stride}\n                            "
        src += ';'
    
        src += """
    // bounds-checking
    bool checkm[TM] = r""" + ''.join(map(str,axes_m)) + """ < matmul_m;
    bool checkn[TN] = r""" + ''.join(map(str,axes_n)) + """ < matmul_n;
    bool checkc[TM, TN, TB] = checkm[:, newaxis, newaxis] && 
                              checkn[newaxis, :, newaxis];

    // write back
    TYPE c[TM, TN, TB] = acc;
#if TZ == 1
    *?(checkc)pc = c;
#else
    int *plock = locks + pid_mn + pid_b * get_num_programs(0);
    int *pcount = plock + 1024*1024;
    // spin
    for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
    int count = *pcount;
    if(count == 0)
      *?(checkc)pc = c;
    else
      *?(checkc)pc = c + *pc;
    atomic_xchg(pcount, (count + 1) % (grid_z));
    atomic_xchg(plock, 0);
#endif
}
"""

        return triton.kernel(src, ['C'])

    ############################
    ## Look-up Table Helper
    ############################

    class LUT_MODE(IntEnum):
        SCALAR = 1
        CONSTANT = 2
        DRAM = 3
    
    def lut_mode(delta):
        if np.min(delta) == np.max(delta):
            return _einsum.LUT_MODE.SCALAR
        if delta.size < 2048:
            return _einsum.LUT_MODE.CONSTANT
        return _einsum.LUT_MODE.DRAM


    ############################
    ## Einsum parsing
    ############################

    def parse_axes(expr_a, expr_b, expr_c, arrays):
        is_index = lambda x: type(x) == sp.indexed.Indexed or str(x) in arrays
        sym_a = [x for s in expr_a for x in s.free_symbols if not is_index(x)]
        sym_b = [x for s in expr_b for x in s.free_symbols if not is_index(x)]
        sym_c = [x for s in expr_c for x in s.free_symbols]
        batch = [d for d in sym_a if d in sym_b and d in sym_c]
        outer = [d for d in sym_a if d not in sym_b and d in sym_c]
        inner = [d for d in sym_a if d in sym_b and d not in sym_c]
        illegal = [d for d in sym_a if d not in sym_b and d not in sym_c]
        if illegal:
            raise ValueError(f"einsum labels {illegal} ({expr_a}) "\
                             f"not present in {expr_b} or {expr_c}")
        return list(set(batch)), list(set(outer)), list(set(inner))

    def unpack_offset(k, axes, dims):
        ret = dict()
        for d in reversed(axes):
            ret[d] = k % dims[d]
            k = k // dims[d]
        return ret

    def symbolic_delta(symbols, axes):
        rank = len(symbols)
        strides = [sp.symbols(f'stride{d}') for d in range(rank)]
        nexts = {s: sp.symbols(f'next{s}') for s in axes}
        delta = 0
        for i in range(rank):
            delta += strides[i] * (symbols[i].subs(nexts) - symbols[i])
        return delta
    
    
    def make_delta(axes, step, stride, dims, symbols, arrays):
        # symbolic pointer increments
        delta = _einsum.symbolic_delta(symbols, axes)
        args =  [f'stride{d}' for d in range(len(stride))]
        args += [f'{sk}' for sk in axes]
        args += [f'next{sk}' for sk in axes]
        args += [f'{sk}' for sk, _ in arrays]
        fn = sp.lambdify(args, delta, 'numpy')
        # inner axes values
        inner = [dims[d] for d in axes]
        k = np.arange(np.prod(inner), dtype=np.int32)
        off      = _einsum.unpack_offset(k, axes, dims)
        nextoff  = _einsum.unpack_offset(k + step, axes, dims)
        # evaluate deltas
        args  = [s for s in stride]
        args += [off[sk] for sk in axes]
        args += [nextoff[sk] for sk in axes]
        args += [x for _, x in arrays]
        delta = fn(*args)
        return delta

    def replace_subscript(expr, arrays):
        # replace array indexing by Indexed()
        indexed = re.findall('([_a-zA-Z][_a-zA-Z0-9]*)\[([_a-z]*)\]', expr)
        for x in indexed:
            arrays.append(x[0])
            expr = expr.replace(f'{x[0]}[{x[1]}]', f'Indexed({x[0]},{x[1]})')
        return expr


    def parse_expr(expr, arrays):
        # extract symbols
        sym = []
        i = 0
        while i < len(expr):
            d = expr[i]
            if d == '(':
                size = expr[i:].find(')')
                d = expr[i : i + size + 1]
                d = _einsum.replace_subscript(d, arrays)
                sym.append(parse_expr(d))
                i += size + 1
            else:
                sym.append(parse_expr(d))
                i += 1
        return sym
  
    @staticmethod
    def divto4(val):
        for N in [4, 3, 5, 2, 7]:
            if val % N == 0:
                return N
        return 1

    @staticmethod
    def pad(tensor, pad):
        pad = pad + [0] *  (2*len(tensor.shape) - len(pad))
        begin = [ x if x > 0 else None for x in pad[-1::-2]]
        end   = [-x if x > 0 else None for x in pad[-2::-2]]
        slices = [slice(b, e) for b, e in zip(begin, end)]
        tensor = torch.nn.functional.pad(tensor, pad, 'constant', 0)
        tensor = tensor[slices]
        return tensor


    @staticmethod
    def forward(ctx, einsum, a, b, shape_c, bench, values):
        # tile sizes
        dtype = a.dtype
        TK = 16 if dtype == triton.fw.torch.float16 else 8
        # parse symbols
        expr_a, expr_bc = einsum.split(",")
        expr_b, expr_c  = expr_bc.split("->")
        subscripted = []
        sym_a = _einsum.parse_expr(expr_a, subscripted)
        sym_b = _einsum.parse_expr(expr_b, subscripted)
        sym_c = _einsum.parse_expr(expr_c, subscripted)
        # parse axes
        axes_b, axes_m, axes_k = _einsum.parse_axes(sym_a, sym_b, sym_c, subscripted)
        _, axes_n, _           = _einsum.parse_axes(sym_b, sym_a, sym_c, subscripted)
        axes = axes_b + axes_m + axes_n + axes_k
        # shapes
        shape_a = np.shape(a)
        shape_b = np.shape(b)
        # check batch and inner dimensions
        dims_a  = dict(zip(sym_a, shape_a))
        dims_b  = dict(zip(sym_b, shape_b))
        dims_c  = dict(zip(sym_c, shape_c))
        for axes in [axes_b, axes_k]:
            for d in axes:
                dim_a = dims_a[d] if d in sym_a else None
                dim_b = dims_b[d] if d in sym_b else None
                if dim_a and dim_b and dim_a != dim_b:
                    raise ValueError(f'incompatible dimension {d}'
                                     f' (a: {dim_a}; b: {dim_b})')
        # extract outer dimensions
        dims = dict()
        dims.update(dims_a)
        dims.update(dims_b)
        dims.update(dims_c)
        # look-up tables
        arrays = [(x, values[x]) for x in subscripted]
        delta_a = _einsum.make_delta(axes_k, TK, a.stride(), dims, sym_a, arrays)
        delta_b = _einsum.make_delta(axes_k, TK, b.stride(), dims, sym_b, arrays)
        # look-up mode
        lut_mode_a = _einsum.lut_mode(delta_a)
        lut_mode_b = _einsum.lut_mode(delta_b)  
        # make kernel
        stride_a_multiple = max([x for x in [1, 2, 4, 8] if shape_a[-1] % x == 0])
        stride_b_multiple = max([x for x in [1, 2, 4, 8] if shape_b[-1] % x == 0])
        stride_c_multiple = max([x for x in [1, 2, 4, 8] if shape_c[-1] % x == 0])
        name = f'{expr_a}_{expr_b}_{expr_c}_{lut_mode_a}_{lut_mode_b}'\
               f'_{stride_a_multiple}_{stride_b_multiple}_{stride_c_multiple}'
        if name not in _einsum.cache:
            cachesize = len(_einsum.cache)
            _einsum.cache[name] = _einsum.make_kernel(f'__einsum{cachesize}', 
                                                     sym_a, sym_b, sym_c, 
                                                     axes_m, axes_n, axes_k, axes_b, 
                                                     stride_a_multiple, stride_b_multiple, stride_c_multiple,
                                                     lut_mode_a, lut_mode_b, subscripted)
        kernel = _einsum.cache[name]
        # allocate buffers
        locks = torch.zeros(2*1024*1024, dtype=torch.int32).cuda()
        c = triton.empty(shape_c, dtype=dtype)
        # execute kernel
        matmul_m = reduce(mul, [dims[d] for d in axes_m], 1)
        matmul_n = reduce(mul, [dims[d] for d in axes_n], 1)
        matmul_k = reduce(mul, [dims[d] for d in axes_k], 1)
        matmul_b = reduce(mul, [dims[d] for d in axes_b], 1)
        alpha = 1.
        div_m = 1
        args  = []
        args += [a, b, c, 
                 locks, 
                 alpha, 
                 matmul_m, matmul_n, matmul_k, 
                 div_m,
                 a.storage_offset(), b.storage_offset(), c.storage_offset()]
        args += [dims[d] for d in axes_m]
        args += [dims[d] for d in axes_n]
        args += [dims[d] for d in axes_k]
        args += [dims[d] for d in axes_b]
        args += list(a.stride())[:-1]
        args += list(b.stride())[:-1]
        args += list(c.stride())[:-1]
        # look-up table for a
        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            args += [delta_a[0]]
        else:
            args += [torch.from_numpy(delta_a).cuda()]
        # look-up table for b
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            args += [delta_b[0]]
        else:
            args += [torch.from_numpy(delta_b).cuda()]
        # external arrays
        for x in subscripted:
            args += [torch.from_numpy(values[x]).cuda()]
        # tile sizes
        TM = [x for x in [16, 32, 64, 128] if x <= matmul_m]
        TN = [x for x in [16, 32, 64, 128] if x <= matmul_n]
        # batch
        TB = [x for x in [1, 2, 4] if x <= matmul_b]
        # reduction-splitting
        MAX_GZ = matmul_k // 2048
        MIN_GM = matmul_m // max(TM)
        MIN_GN = matmul_n // max(TN)
        MIN_GB = matmul_b // max(TB)
        TZ = [x for x in [1, 2, 4, 8, 16, 32] \
                if x < MAX_GZ and x*MIN_GM*MIN_GN*MIN_GB < 256]
        TZ = [1] if not TZ else [TZ[-1], TZ[-1]*2]
        # launch
        args += [lambda opt: [triton.cdiv(matmul_m, opt.d('TM')) * 
                              triton.cdiv(matmul_n, opt.d('TN')),
                              triton.cdiv(matmul_b, opt.d('TB')),
                              opt.d('TZ')]]
        kernel(*args, bench=bench, TM=TM, TN=TN, TK=TK, TZ=TZ, TYPE=dtype, TB=TB)
        # save information in context
        ctx.flops = 2. * matmul_b * matmul_m * matmul_n * matmul_k
        ctx.bench = bench
        ctx.save_for_backward(a, b)
        ctx.sym_a = sym_a
        ctx.sym_b = sym_b
        ctx.sym_c = sym_c
        return c

    @staticmethod
    def sym_invert(sym_c, sym_x, prefix, renamed, inverse):
        for i, expr in enumerate(sym_x):
           if expr.is_symbol:
               continue
           sc = [x for x in expr.free_symbols if x in sym_c][0]
           sx = sp.symbols(f'{prefix}{i}')
           renamed[expr] = sx
           inverse[sc] = sp.solve(sp.Eq(expr, sx), sc)[0]

    @staticmethod
    def sym_to_expr(sym):
        res = [f'({x})' for x in sym]
        res = ''.join(res)
        return res

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        sym_a = ctx.sym_a
        sym_b = ctx.sym_b
        sym_c = ctx.sym_c
        inverse = dict()
        renamed = dict()
        _einsum.sym_invert(sym_c, sym_a, 'a', renamed, inverse)
        _einsum.sym_invert(sym_c, sym_b, 'b', renamed, inverse)
        sym_a = [renamed[x] if x in renamed else x for x in sym_a]
        sym_b = [renamed[x] if x in renamed else x for x in sym_b]
        sym_c = [inverse[x] if x in inverse else x for x in sym_c]
        expr_a =  _einsum.sym_to_expr(sym_a)
        expr_b =  _einsum.sym_to_expr(sym_b)
        expr_c =  _einsum.sym_to_expr(sym_c)
        expr = f'{expr_c},{expr_b}->{expr_a}'
        da = einsum(expr, dy, b, a.shape, False)
        return None, da, None, None, None



einsum = _einsum.apply