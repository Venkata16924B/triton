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
                    lut_mode_a, lut_mode_b):

        src = f"""
__global__ void {name}(
              TYPE * A __noalias __readonly __aligned(16)
            , TYPE * B __noalias __readonly __aligned(16)
            , TYPE * C
            , int * locks
            , float alpha
            , int matmul_m, int matmul_n, int matmul_k __multipleof(64)
            , int div_m
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
            src += ", int* AD, int* ADI"
        src += "\n            "
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_b_inner __multipleof({multipleof_b})"
        else:
            src += ", int* BD, int* BDI"
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
    TYPE *pa[TM, TK, TB] = A"""
        for i, sym in enumerate(expr_a):
            free = sym.free_symbols
            replace = dict()
            replace.update({d: sp.symbol._symbol(f'r{d}[:, newaxis, newaxis]') for d in free if d in axes_m})
            replace.update({d: sp.symbol._symbol(f'r{d}[newaxis, :, newaxis]') for d in free if d in axes_k})
            replace.update({d: sp.symbol._symbol(f'r{d}[newaxis, newaxis, :]') for d in free if d in axes_b})
            sym = sym.subs(replace)
            stride = f'stride_a_{i}' if i < len(expr_a) - 1 else '1'
            src += f" + ({sym}) * {stride}\n                            "
        src += ';'

        if not lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += f"""
    // initialize pointers to A look-up table
    int *padelta[TK]  = AD  + 0 ... TK;
    int *padeltai[TK] = ADI + 0 ... TK;"""
    
        src += """

    // initialize pointers to B
    TYPE *pb[TK, TN, TB] = B"""
        for i, sym in enumerate(expr_b):
            free = sym.free_symbols
            replace = dict()
            replace.update({d: sp.symbol._symbol(f'r{d}[:, newaxis, newaxis]') for d in free if d in axes_k})
            replace.update({d: sp.symbol._symbol(f'r{d}[newaxis, :, newaxis]') for d in free if d in axes_n})
            replace.update({d: sp.symbol._symbol(f'r{d}[newaxis, newaxis, :]') for d in free if d in axes_b})
            sym = sym.subs(replace)
            stride = f'stride_b_{i}' if i < len(expr_b) - 1 else '1'
            src += f" + ({sym}) * {stride}\n                            "
        src += ';'


        if not lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += f"""
    // initialize pointers to B look-up table
    int *pbdelta[TK]  = BD  + 0 ... TK;
    int *pbdeltai[TK] = BDI + 0 ... TK;"""

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
        int adeltai[TK] = *padeltai;
        padelta += adeltai;
        padeltai += adeltai;"""

        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += """
        pb += stride_b_inner;"""
        else:
            src += """
        pb += (*pbdelta)[:, newaxis, newaxis];
        int bdeltai[TK] = *pbdeltai;
        pbdelta += bdeltai;
        pbdeltai += bdeltai;"""

        src += f"""
        bool checka[TM, TK, TB] = {rk}[newaxis, :, newaxis] < k - TK;
        bool checkb[TK, TN, TB] = {rk}[:, newaxis, newaxis] < k - TK;
        a = checka ? *pa : 0;
        b = checkb ? *pb : 0;
    }}
    //acc = acc * alpha;

 """

    
        src += """

    // initialize pointers to C
    TYPE *pc[TM, TN, TB] = C"""
        for i, sym in enumerate(expr_c):
            free = sym.free_symbols
            replace = dict()
            replace.update({d: sp.symbol._symbol(f'r{d}[:, newaxis, newaxis]') for d in free if d in axes_m})
            replace.update({d: sp.symbol._symbol(f'r{d}[newaxis, :, newaxis]') for d in free if d in axes_n})
            replace.update({d: sp.symbol._symbol(f'r{d}[newaxis, newaxis, :]') for d in free if d in axes_b})
            sym = sym.subs(replace)
            stride = f'stride_c_{i}' if i < len(expr_c) - 1 else '1'
            src += f" + ({sym}) * {stride}\n                            "
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
        print(src)
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

    def parse_axes(expr_a, expr_b, expr_c):
        sym_a = [x for s in expr_a for x in s.free_symbols]
        sym_b = [x for s in expr_b for x in s.free_symbols]
        sym_c = [x for s in expr_c for x in s.free_symbols]
        batch = [d for d in sym_a if d in sym_b and d in sym_c]
        outer = [d for d in sym_a if d not in sym_b and d in sym_c]
        inner = [d for d in sym_a if d in sym_b and d not in sym_c]
        illegal = [d for d in sym_a if d not in sym_b and d not in sym_c]
        if illegal:
            raise ValueError(f"einsum labels {illegal} ({expr_a}) "\
                             f"not present in {expr_b} or {expr_c}")
        return batch, outer, inner

    def unpack_offset(k, axes, dims):
        ret = dict()
        for d in reversed(axes):
            ret[d] = k % dims[d]
            k = k // dims[d]
        return ret

    def symbolic_delta(symbols, axes):
        rank = len(symbols)
        # symbolic strides
        shape = [sp.symbols(f'shape{d}') for d in range(rank)]
        strides = [1] * rank
        for i in range(rank - 1, 0, -1):
            strides[i-1] = strides[i] * shape[i] 
        # compute deltas
        nexts = {s: sp.symbols(f'next{s}') for s in axes}
        delta = 0
        for i in range(rank):
            delta += strides[i] * (symbols[i].subs(nexts) - symbols[i])
        return delta
    
    class bound_info:

        def __init__(self, outer, inner, config, keys):
            self.outer = outer
            self.inner = inner
            self.config = config
            self.keys = keys
        
        def inner_at(self, outer):
            idx = [None, None]
            idx[self.outer[0][0]] = outer
            idx[self.inner[0][0]] = slice(0, self.config.shape[self.inner[0][0]])
            return self.config[tuple(idx)]

    def make_delta(axes, step, shape, dims, symbols):
        # symbolic pointer increments
        delta = _einsum.symbolic_delta(symbols, axes)
        args =  [f'shape{d}' for d in range(len(shape))]
        args += [f'{sk}' for sk in axes]
        args += [f'next{sk}' for sk in axes]
        fn = sp.lambdify(args, delta, 'numpy')
        # inner axes values
        inner = [dims[d] for d in axes]
        k = np.arange(np.prod(inner[1:]), dtype=np.int32)
        off      = _einsum.unpack_offset(k, axes, dims)
        nextoff  = _einsum.unpack_offset(k + step, axes, dims)
        # evaluate deltas
        args  = [s for s in shape]
        args += [off[sk] for sk in axes]
        args += [nextoff[sk] for sk in axes]
        delta = fn(*args)
        # bounds-checking
        exprs = [expr for expr in symbols if not expr.is_symbol]
        bounds = []
        d_shape = []
        for i, expr in enumerate(symbols):
            if expr.is_symbol:
                continue
            syms = expr.free_symbols
            fn = sp.lambdify(syms, expr, 'numpy')
            args = []
            for j, s in enumerate(syms):
                idx = np.arange(dims[s])
                view = [1] * len(syms)
                view[j] = idx.size
                idx = np.reshape(idx, view)
                args.append(idx)
            res = fn(*args)
            res = np.logical_or(res < 0, res > shape[i] - 1).astype(np.int32)
            outer = [(j, s) for j, s in enumerate(syms) if s not in axes]
            inner = [(j, s) for j, s in enumerate(syms) if s in axes]
            config, idx, keys = np.unique(res, axis=outer[0][0], return_index=True, return_inverse=True)
            info = _einsum.bound_info(outer = outer, inner = inner, config = config, keys = keys)
            bounds.append(info)
            d_shape.append(len(idx))
        d_shape.append(delta.size)
        deltas = np.empty(d_shape, dtype = np.int32)
        for idx in np.ndindex(deltas.shape):
            mask = np.zeros(k.size, dtype=np.int32)
            view = []
            for i in range(len(idx) - 1):
                is_oob = bounds[i].inner_at(idx[i])
                is_oob = is_oob[off[bounds[i].inner[0][1]]]
                mask = np.logical_or(mask, is_oob)
                view.append(idx[i])
            mask = mask * -2 + 1
            view.append(slice(0, d_shape[-1]))
            deltas[tuple(view)] = delta*mask
        # look-up table increments
        diff = np.arange(delta.size, dtype=np.int32)
        diff = ((diff + step) % diff.size) - diff        
        return delta, diff

    def extract_strides(shape):
        strides = np.cumprod(shape[::-1])[::-1]
        strides = np.concatenate((strides[1:], [1]))
        return strides.tolist()


    def parse_expr(expr):
        sym = []
        i = 0
        while i < len(expr):
            d = expr[i]
            if d == '(':
                size = expr[i:].find(')')
                d = expr[i : i + size + 1]
                sym.append(parse_expr(d))
                i += size + 1
            else:
                sym.append(parse_expr(d))
                i += 1
        return sym
  
    def solve_shape(symbols, target, shape):
        ret = set()
        for d, sym in enumerate(symbols):
            if target in sym.free_symbols:
                res = sp.solve(sp.Eq(sym, 0), target)[0],\
                      sp.solve(sp.Eq(sym, shape[d]), target)[0]
                ret.add(res)
        return ret

    def infer_shape(sym_c, sym_a, sym_b, expr_a, expr_b):
        # create symbol for each shape
        shape_c = []
        shape_a = [sp.symbols(f'shape_a_{d}') for d in range(len(sym_a))]
        shape_b = [sp.symbols(f'shape_b_{d}') for d in range(len(sym_b))]
        in_shape = shape_a + shape_b
        # maximum value of each index symbol
        max_idx = dict()
        max_idx.update({sa: shape_a[d] - 1 for d, sa in enumerate(sym_a) if sa.is_symbol})
        max_idx.update({sb: shape_b[d] - 1 for d, sb in enumerate(sym_b) if sb.is_symbol})
        min_idx = dict()
        min_idx.update({sa: 0 for d, sa in enumerate(sym_a) if sa.is_symbol})
        min_idx.update({sb: 0 for d, sb in enumerate(sym_b) if sb.is_symbol})
        for sc in sym_c:
            # solve value of shape of symbol sc
            # given how it appears in sym_a and sym_b
            current = set()
            current.update(_einsum.solve_shape(sym_a, sc, shape_a))
            current.update(_einsum.solve_shape(sym_b, sc, shape_b))
            # There is conflicting definition for sc
            if len(current) > 1:
                raise ValueError(f"conflicting shape definition for {sc} ({current})")
            # There is no solution for sc
            elif len(current) == 0:
                raise ValueError(f"einsum def for output: {i} ({expr_c})"\
                                 f", not present in either input def ({expr_a}, {expr_b})")
            # There is one unique solution for sc; create a lambda function for it
            else:
                sols = current.pop()
                min_fn = sp.lambdify(in_shape, sp.Min(sols[0].subs(min_idx), sols[0].subs(max_idx)))
                max_fn = sp.lambdify(in_shape, sp.Max(sols[1].subs(min_idx), sols[1].subs(max_idx)))
                shape_c.append((min_fn, max_fn))
        return shape_c

    @staticmethod
    def divto4(val):
        for N in [4, 3, 5, 2, 7]:
            if val % N == 0:
                return N
        return 1

    @staticmethod
    def forward(ctx, einsum, a, b, shape_c, bench):
        dtype = a.dtype
        TK = 16 if dtype == triton.fw.torch.float16 else 8
        # parse symbols
        expr_a, expr_bc = einsum.split(",")
        expr_b, expr_c  = expr_bc.split("->")
        sym_a = _einsum.parse_expr(expr_a)
        sym_b = _einsum.parse_expr(expr_b)
        sym_c = _einsum.parse_expr(expr_c)
        # parse axes
        axes_b, axes_m, axes_k = _einsum.parse_axes(sym_a, sym_b, sym_c)
        _, axes_n, _ = _einsum.parse_axes(sym_b, sym_a, sym_c)
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
        delta_a, diff_a = _einsum.make_delta(axes_k, TK, shape_a, dims, sym_a)
        delta_b, diff_b = _einsum.make_delta(axes_k, TK, shape_b, dims, sym_b)
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
                                                     lut_mode_a, lut_mode_b)
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
        args += [a, b, c, locks, alpha, matmul_m, matmul_n, matmul_k, div_m]
        args += [dims[d] for d in axes_m]
        args += [dims[d] for d in axes_n]
        args += [dims[d] for d in axes_k]
        args += [dims[d] for d in axes_b]
        args += _einsum.extract_strides(shape_a)[:-1]
        args += _einsum.extract_strides(shape_b)[:-1]
        args += _einsum.extract_strides(shape_c)[:-1]
        # look-up table for a
        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            args += [delta_a[0]]
        else:
            args += [torch.from_numpy(delta_a).cuda(), 
                     torch.from_numpy(diff_a).cuda()]
        # look-up table for b
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            args += [delta_b[0]]
        else:
            args += [torch.from_numpy(delta_b).cuda(), 
                     torch.from_numpy(diff_b).cuda()]
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
        return c


einsum = _einsum.apply