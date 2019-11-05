import numpy as np
import torch
import triton


class _einsum(triton.function):

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

    @staticmethod
    def forward(ctx, einsum, a, b):
        expr_a, expr_bc = einsum.split(",")
        expr_b, expr_c  = expr_bc.split("->")
        shape_a = np.shape(a)
        shape_b = np.shape(b)
        # extract output shape
        shape_c = []
        dims_a  = dict(zip(expr_a, shape_a))
        dims_b  = dict(zip(expr_b, shape_b))
        strides_a = dict(zip(expr_a, _einsum.extract_strides(shape_a)))
        strides_b = dict(zip(expr_b, _einsum.extract_strides(shape_b)))
        for i in expr_c:
            if i in expr_a:
                shape_c.append(dims_a[i])
            elif i in expr_b:
                shape_c.append(dims_b[i])
            else:
                raise ValueError(f"einsum def for output: {i} ({expr_c})"\
                                ", not present in either input def ({expr_a}, {expr_b})")
        strides_c = dict(zip(expr_c, _einsum.extract_strides(shape_c)))
        # extract axes
        batch_a, outer_a, inner_a = _einsum.parse_axes(expr_a, expr_b, expr_c)
        batch_b, outer_b, inner_b = _einsum.parse_axes(expr_b, expr_a, expr_c)
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
        axes_k = sorted(dim_k.keys(), key = lambda d: expr_a.index(d))
        axes_m = sorted(dim_m.keys(), key = lambda d: expr_a.index(d))
        axes_b = sorted(dim_b.keys(), key = lambda d: expr_a.index(d))
        axes_n = sorted(dim_n.keys(), key = lambda d: expr_b.index(d))
        # look-up tables
        TK = 8
        delta_a, diff_a = _einsum.make_delta(axes_k, TK, dims_a, strides_a)
        delta_b, diff_b =  _einsum.make_delta(axes_k, TK, dims_b, strides_b)
        # find if delta is scalar
        is_delta_a_scalar = np.min(delta_a) == np.max(delta_a)
        delta_a_scalar = np.min(delta_a)
        is_delta_b_scalar = np.min(delta_b) == np.max(delta_b)
        delta_b_scalar = np.min(delta_b)
        # un-packing outer dimensions
        INIT_RM = _einsum.unpack_cc('TM', axes_m)
        INIT_RN = _einsum.unpack_cc('TN', axes_n)
        INIT_RB = _einsum.unpack_cc('TB', axes_b)

        cc_strides_a = [f'stride_a_{d}' for d in expr_a[:-1]] + ['1']
        cc_strides_b = [f'stride_b_{d}' for d in expr_b[:-1]] + ['1']
        cc_strides_c = [f'stride_c_{d}' for d in expr_c[:-1]] + ['1']
        cc_strides_a = dict(zip(expr_a, cc_strides_a))
        cc_strides_b = dict(zip(expr_b, cc_strides_b))
        cc_strides_c = dict(zip(expr_c, cc_strides_c))

        src = """
void einsumk(TYPE * A
            , TYPE * B
            , TYPE * C
            , int matmul_m, int matmul_n, int matmul_k
            """
        for dim in [dim_m, dim_n, dim_k, dim_b]:
            for d in dim:
                src += f", int dim_{d}"
        src += "\n            "
        for stride, name in zip([strides_a, strides_b, strides_c],
                                ['a', 'b', 'c']):
            for s in stride:
                src += f", int stride_{name}_{s}"
            src += "\n            "
        if is_delta_a_scalar:
            src += ", int stride_a_inner"
        else:
            src += ", int* AD, int* ADI"
        src += "\n            "
        if is_delta_b_scalar:
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
        for d in axes_m:
            src += f" + r{d}[:, newaxis, newaxis] * {cc_strides_a[d]}\n                            "
        for d in axes_k:
            src += f" + r{d}[newaxis, :, newaxis] * {cc_strides_a[d]}\n                            "
        for d in axes_b:
            src += f" + r{d}[newaxis, newaxis, :] * {cc_strides_a[d]}\n                            "
        src += ";"

        if not is_delta_a_scalar:
            src += """
    // initialize pointers to A look-up table
    int *pad[TK] = AD + 0 ... TK;
    int *padi[TK] = ADI + 0 ... TK;""".format(k = ''.join(axes_k))
    
        src += """
    // initialize pointers to B
    TYPE *pb[TK, TN, TB] = B"""
        for d in axes_n:
            src += f" + r{d}[newaxis, :, newaxis] * {cc_strides_b[d]}\n                            "
        for d in axes_k:
            src += f" + r{d}[:, newaxis, newaxis] * {cc_strides_b[d]}\n                            "
        for d in axes_b:
            src += f" + r{d}[newaxis, newaxis, :] * {cc_strides_b[d]}\n                            "
        src += ";"

        if not is_delta_b_scalar:
            src += """
    // initialize pointers to B look-up table
    int *pbd[TK] = BD + 0 ... TK;
    int *pbdi[TK] = BDI + 0 ... TK;""".format(k = ''.join(axes_k))

        src += """
    
    // accumulate
    float c[TM, TN, TB] = 0;
    TYPE a[TM, TK, TB] = *pa;
    TYPE b[TK, TN, TB] = *pb;
    for(int k = matmul_k; k > 0; k -= TK) {
        c += a @ b;"""

        if is_delta_a_scalar:
            src += """
        pa += stride_a_inner;"""
        else:
            src += """
        pa += (*pad)[newaxis, :, newaxis];
        int adi[TK] = *padi;
        pad += adi;
        padi += adi;"""

        if is_delta_b_scalar:
            src += """
        pb += stride_b_inner;"""
        else:
            src += """
        pb += (*pbd)[:, newaxis, newaxis];
        int bdi[TK] = *pbdi;
        pbd += bdi;
        pbdi += bdi;"""

        src += """
        bool checka[TM, TK, TB] = k > TK;
        bool checkb[TK, TN, TB] = k > TK;
        a = checka ? *pa : 0;
        b = checkb ? *pb : 0;
    }"""

        src += """

    // initialize pointers to C
    TYPE *pc[TM, TN, TB] = C"""
        for d in axes_m:
            src += f" + r{d}[:, newaxis, newaxis] * {cc_strides_c[d]}\n                            "
        for d in axes_n:
            src += f" + r{d}[newaxis, :, newaxis] * {cc_strides_c[d]}\n                            "
        for d in axes_b:
            src += f" + r{d}[newaxis, newaxis, :] * {cc_strides_c[d]}\n                            "
        src += """;

    // write-back
    bool checkm[TM] = rm < matmul_m;
    bool checkn[TN] = rn < matmul_n;
    bool checkc[TM, TN, TB] = checkm[:, newaxis, newaxis] && 
                              checkn[newaxis, :, newaxis];
    *?(checkc)pc = (TYPE[TM, TN, TB])c;
}
"""
        kernel = triton.kernel(src, ['C'])

        print(src)

        c = triton.empty(shape_c, a.dtype)
        delta_a = torch.from_numpy(delta_a).cuda()
        diff_a = torch.from_numpy(diff_a).cuda()
        delta_b = torch.from_numpy(delta_b).cuda()
        diff_b = torch.from_numpy(diff_b).cuda()
        matmul_m = int(np.prod(list(dim_m.values())))
        matmul_n = int(np.prod(list(dim_n.values())))
        matmul_k = int(np.prod(list(dim_k.values())))
        matmul_b = int(np.prod(list(dim_b.values())))
        args  = []
        args += [a, b, c]
        args += [matmul_m, matmul_n, matmul_k]
        args += list(dim_m.values())
        args += list(dim_n.values())
        args += list(dim_k.values())
        args += list(dim_b.values())
        args += list(strides_a.values())
        args += list(strides_b.values())
        args += list(strides_c.values())
        if is_delta_a_scalar:
            args += [delta_a_scalar]
        else:
            args += [delta_a, diff_a]
        if is_delta_b_scalar:
            args += [delta_b_scalar]
        else:
            args += [delta_b, diff_b]
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