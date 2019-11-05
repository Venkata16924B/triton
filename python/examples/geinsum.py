import numpy as np
import torch
from enum import Enum
import triton
from functools import reduce
from operator import mul


class _einsum(triton.function):


    cache = dict()

    #############################
    ## Triton-C code generation
    #############################

    def unpack_cc(tile, axes):
        ret = ''
        for i, d in enumerate(reversed(axes)):
            if i == len(axes) - 1:
                break
            currs = ''.join(axes[: len(axes) - i])
            nexts = ''.join(axes[: len(axes) - (i + 1)])
            ret += f'    int r{nexts}[{tile}] = r{currs} / dim_{d};\n'
            ret += f'    int r{d}[{tile}] = r{currs} % dim_{d};\n'
        return ret

    def strides_cc(name, expr):
        ret = [f'stride_{name}_{d}' for d in expr[:-1]] + ['1']
        ret = dict(zip(expr, ret))
        return ret

    def make_kernel(expr_a, expr_b, expr_c,
                    axes_m, axes_n, axes_k, axes_b,
                    lut_mode_a, lut_mode_b):

        src = """
__global__ void einsumk(TYPE * A
            , TYPE * B
            , TYPE * C
            , int matmul_m, int matmul_n, int matmul_k
            """
        for dim in [axes_m, axes_n, axes_k, axes_b]:
            for d in dim:
                src += f", int dim_{d}"
        src += "\n            "
        for dim, name in zip([expr_a, expr_b, expr_c],
                                ['a', 'b', 'c']):
            for d in dim:
                src += f", int stride_{name}_{d}"
            src += "\n            "
        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += ", int stride_a_inner"
        else:
            src += ", int* AD, int* ADI"
        src += "\n            "
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += ", int stride_b_inner"
        else:
            src += ", int* BD, int* BDI"
        src += """) {

    // create ranges
"""
        for i, (axes, tile) in enumerate(zip([axes_m, axes_n, axes_b, axes_k],
                                           ['TM', 'TN', 'TB', 'TK'])):
            currs = ''.join(axes)
            if axes == axes_k:
                src += f"    int r{currs}[{tile}] = 0 ... {tile};\n"
            else:
                src += f"    int r{currs}[{tile}] = get_program_id({i}) * {tile} + 0 ... {tile};\n"
        
        if axes_m:
            src += _einsum.unpack_cc('TM', axes_m)
        if axes_n:
            src += _einsum.unpack_cc('TN', axes_n)
        if axes_b:
            src += _einsum.unpack_cc('TB', axes_b)
        if axes_k:
            src += _einsum.unpack_cc('TK', axes_k)

        src += """    

    // initialize pointers to A
    TYPE *pa[TM, TK, TB] = A"""
        strides_a = _einsum.strides_cc('a', expr_a)
        for d in axes_m:
            src += f" + r{d}[:, newaxis, newaxis] * {strides_a[d]}\n                            "
        for d in axes_k:
            src += f" + r{d}[newaxis, :, newaxis] * {strides_a[d]}\n                            "
        for d in axes_b:
            src += f" + r{d}[newaxis, newaxis, :] * {strides_a[d]}\n                            "
        src += ";"

        if not lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += """
    // initialize pointers to A look-up table
    int *padelta[TK]  = AD  + 0 ... TK;
    int *padeltai[TK] = ADI + 0 ... TK;""".format(k = ''.join(axes_k))
    
        src += """

    // initialize pointers to B
    TYPE *pb[TK, TN, TB] = B"""
        strides_b = _einsum.strides_cc('b', expr_b)
        for d in axes_n:
            src += f" + r{d}[newaxis, :, newaxis] * {strides_b[d]}\n                            "
        for d in axes_k:
            src += f" + r{d}[:, newaxis, newaxis] * {strides_b[d]}\n                            "
        for d in axes_b:
            src += f" + r{d}[newaxis, newaxis, :] * {strides_b[d]}\n                            "
        src += ";"

        if not lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += """
    // initialize pointers to B look-up table
    int *pbdelta[TK]  = BD  + 0 ... TK;
    int *pbdeltai[TK] = BDI + 0 ... TK;""".format(k = ''.join(axes_k))

        src += """
    
    // accumulate
    float c[TM, TN, TB] = 0;
    TYPE a[TM, TK, TB] = *pa;
    TYPE b[TK, TN, TB] = *pb;
    for(int k = matmul_k; k > 0; k -= TK) {
        c += a @ b;"""

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

        src += """
        bool checka[TM, TK, TB] = k > TK;
        bool checkb[TK, TN, TB] = k > TK;
        a = checka ? *pa : 0;
        b = checkb ? *pb : 0;
    }"""

        src += """

    // initialize pointers to C
    TYPE *pc[TM, TN, TB] = C"""
        strides_c = _einsum.strides_cc('c', expr_c)
        for d in axes_m:
            src += f" + r{d}[:, newaxis, newaxis] * {strides_c[d]}\n                            "
        for d in axes_n:
            src += f" + r{d}[newaxis, :, newaxis] * {strides_c[d]}\n                            "
        for d in axes_b:
            src += f" + r{d}[newaxis, newaxis, :] * {strides_c[d]}\n                            "
        src += """;

    // write-back
    bool checkm[TM] = rm < matmul_m;
    bool checkn[TN] = rn < matmul_n;
    bool checkc[TM, TN, TB] = checkm[:, newaxis, newaxis] && 
                              checkn[newaxis, :, newaxis];
    *?(checkc)pc = (TYPE[TM, TN, TB])c;
}
"""
        return triton.kernel(src, ['C'])

    ############################
    ## Look-up Table Helper
    ############################

    class LUT_MODE(Enum):
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
        batch = [d for d in expr_a if d in expr_b and d in expr_c]
        outer = [d for d in expr_a if d not in expr_b and d in expr_c]
        inner = [d for d in expr_a if d in expr_b and d not in expr_c]
        illegal = [d for d in expr_a if d not in expr_b and d not in expr_c]
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

    def make_delta(axes, step, dims, strides):
        shape = [dims[d] for d in axes]
        if len(axes) <= 1:
            delta = step * strides[axes[0]] * np.ones(step, dtype=np.int32)
            diff = np.zeros(delta.size, dtype=np.int32)
            return delta, diff
        k = np.arange(np.prod(shape[1:]), dtype=np.int32)
        off      = _einsum.unpack_offset(k, axes, dims)
        next_off = _einsum.unpack_offset(k + step, axes, dims)
        # pointer deltas
        delta = np.zeros(k.size, dtype=np.int32)
        for d in axes:
            delta += (next_off[d] - off[d])*strides[d]
        # delta increments
        diff = np.arange(delta.size, dtype=np.int32)
        diff = ((diff + step) % diff.size) - diff
        return delta, diff


    def extract_strides(shape):
        strides = np.cumprod(shape[::-1])[::-1]
        strides = np.concatenate((strides[1:], [1]))
        return strides


    def extract_symbols(expr):
        sym = []
        i = 0
        while i < len(expr):
            d = expr[i]
            if d == '(':
                size = expr[i:].find(')')
                sym.append(expr[i : i + size + 1])
                i += size
            else:
                sym.append(d)
                i += 1
        return sym
  

    @staticmethod
    def forward(ctx, einsum, a, b):
        # parse symbols
        expr_a, expr_bc = einsum.split(",")
        expr_b, expr_c  = expr_bc.split("->")
        sym_a = _einsum.extract_symbols(expr_a)
        sym_b = _einsum.extract_symbols(expr_b)
        sym_c = _einsum.extract_symbols(expr_c)
        # input shapes
        shape_a = np.shape(a)
        shape_b = np.shape(b)
        # extract output shape
        shape_c = []
        dims_a  = dict(zip(sym_a, shape_a))
        dims_b  = dict(zip(sym_b, shape_b))
        strides_a = dict(zip(sym_a, _einsum.extract_strides(shape_a)))
        strides_b = dict(zip(sym_b, _einsum.extract_strides(shape_b)))
        for i in sym_c:
            if i in sym_a:
                shape_c.append(dims_a[i])
            elif i in sym_b:
                shape_c.append(dims_b[i])
            else:
                raise ValueError(f"einsum def for output: {i} ({sym_c})"\
                                ", not present in either input def ({sym_a}, {sym_b})")
        strides_c = dict(zip(sym_c, _einsum.extract_strides(shape_c)))
        # extract axes
        batch_a, outer_a, inner_a = _einsum.parse_axes(sym_a, sym_b, sym_c)
        batch_b, outer_b, inner_b = _einsum.parse_axes(sym_b, sym_a, sym_c)
        # extract batch dimensions
        dim_b_a = {d: dims_a[d] for d in batch_a}
        dim_b_b = {d: dims_b[d] for d in batch_b}
        if dim_b_a != dim_b_b:
            raise ValueError(f'incompatible batch dimensions {dim_b_a} and {dim_b_b}')
        dim_b = dim_b_a
        # extract inner dimensions
        dim_k_a = {d: dims_a[d] for d in inner_a}
        dim_k_b = {d: dims_b[d] for d in inner_b}
        if dim_k_a != dim_k_b:
            raise ValueError(f'incompatible inner dimensions {dim_k_a} and {dim_k_b}')
        dim_k = dim_k_a
        # extract outer dimensions
        dim_m = {d: dims_a[d] for d in outer_a}
        dim_n = {d: dims_b[d] for d in outer_b}
        # axes
        axes_k = sorted(dim_k.keys(), key = lambda d: sym_a.index(d))
        axes_m = sorted(dim_m.keys(), key = lambda d: sym_a.index(d))
        axes_b = sorted(dim_b.keys(), key = lambda d: sym_a.index(d))
        axes_n = sorted(dim_n.keys(), key = lambda d: sym_b.index(d))
        # look-up tables
        TK = 8
        delta_a, diff_a = _einsum.make_delta(axes_k, TK, dims_a, strides_a)
        delta_b, diff_b =  _einsum.make_delta(axes_k, TK, dims_b, strides_b)
        # look-up mode
        lut_mode_a = _einsum.lut_mode(delta_a)
        lut_mode_b = _einsum.lut_mode(delta_b)  
        # make kernel
        key = (expr_a, expr_b, expr_c, lut_mode_a, lut_mode_b)
        if key not in _einsum.cache:
            _einsum.cache[key] = _einsum.make_kernel(sym_a, sym_b, sym_c, 
                                                     axes_m, axes_n, axes_k, axes_b, 
                                                     lut_mode_a, lut_mode_b)
        kernel = _einsum.cache[key]
        # execute kernel
        c = triton.empty(shape_c, a.dtype)
        matmul_m = reduce(mul, dim_m.values(), 1)
        matmul_n = reduce(mul, dim_n.values(), 1)
        matmul_k = reduce(mul, dim_k.values(), 1)
        matmul_b = reduce(mul, dim_b.values(), 1)
        args  = []
        args += [a, b, c]
        args += [matmul_m, matmul_n, matmul_k]
        # dims
        args += [dim_m[d] for d in axes_m]
        args += [dim_n[d] for d in axes_n]
        args += [dim_k[d] for d in axes_k]
        args += [dim_b[d] for d in axes_b]
        # strides
        args += [strides_a[d] for d in sym_a]
        args += [strides_b[d] for d in sym_b]
        args += [strides_c[d] for d in sym_c]
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
        # grid
        args += [lambda opt: [triton.cdiv(matmul_m, opt.d('TM')), 
                              triton.cdiv(matmul_n, opt.d('TN')),
                              triton.cdiv(matmul_b, opt.d('TB'))]]
        kernel(*args, TM=[64], TN=[64], TK=TK, TYPE='float', TB=1)
        return c


einsum = _einsum.apply

for a_shape, b_shape, expr in [([4, 64, 24, 16, 16], [16, 4, 24, 64], 'bmkqi,ibkn->bmqn')]:
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    rc = np.einsum(expr, a, b)
    tc = einsum(expr, torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda())
    tc = tc.cpu().numpy()
    print(np.max(np.abs(tc - rc)))