import triton
import torch
from torch.utils.cpp_extension import load
import numpy as np
#import utils
from time import time

#torch.backends.cudnn.benchmark = True

configs = []

# Matrix multiplication
MNK = [
    #   (128, 128, 128),
    #   (512, 512 ,512), 
    #   (2048, 2048, 2048),
    (12600, 512, 3456), 
    #   (8192, 8192, 8192),

    #    (64, 64, 64000),
    #    (64, 64, 128000),
    #    (256, 256, 64000),
    #    (256, 256, 128000),

    #    (1536, 16, 1536),
    #    (1536, 32, 1536),
    #    (1536, 64, 1536),
    #    (1536, 128, 1536),
    #    (4096, 16, 4096),
    #    (4096, 32, 4096),
    #    (4096, 64, 4096),
    #    (4096, 128, 4096)
      ]
for M, N, K in MNK:
    matmul = lambda a, b: torch.matmul(a, b)
    configs += [([M, K], [K, N], [M, N], matmul, 'mk,kn->mn', dict())]
#for M, N, K in MNK:
#    matmul = lambda a, b: torch.matmul(a.t(), b)
#    configs += [([M, K], [M, N], [M, N], None, 'mk,mn->kn', dict())]
#for M, N, K in MNK:
#    matmul = lambda a, b: torch.matmul(a, b.t())
#    configs += [([M, N], [K, N], [M, N], None, 'mn,kn->mk', dict())]

# Relative attention
NTHSE = [
        #  (16, 512, 1, 32, 64), 
        #  (16, 512, 1, 128, 128),
        #  (16, 512, 1, 256, 256),
        # (16, 512, 1, 256, 512),
        #  (16, 512, 8, 64, 64), 
        #  (16, 512, 8, 128, 128),
        #  (16, 512, 8, 256, 256),
        #  (16, 512, 8, 256, 512),

        #  (64, 1024, 1, 64, 64), 
        #  (64, 1024, 1, 128, 128),
        #  (64, 1024, 1, 256, 256),
        #  (64, 1024, 1, 256, 512),
        #  (64, 1024, 8, 64, 64), 
        #  (64, 1024, 8, 128, 128),
        #  (64, 1024, 8, 256, 256),
        #  (64, 1024, 8, 256, 512),

        #  (128, 1024, 1, 64, 64), 
        #  (128, 1024, 1, 128, 128),
        #  (128, 1024, 1, 256, 256),
        #  (128, 1024, 1, 256, 512),
        #  (128, 1024, 8, 64, 64), 
        #  (128, 1024, 8, 128, 128),
        #  (128, 1024, 8, 256, 256),
        #  (128, 1024, 8, 256, 512)
        ]
# for N, T, H, S, E in NTHSE:
#     configs += [([N, T, H, S], [H, E, S], [N, H, T, E], None, 'nths,hes->nhte', dict())]
# for N, T, H, S, E in NTHSE:
#     configs += [([N, H, T, E], [N, T, H, S], [H, E, S], None, 'nhte,nths->hes', dict())]
# for N, T, H, S, E in NTHSE:
#     configs += [([N, H, T, E], [H, E, S], [N, T, H, S], None, 'nhte,hes->nths', dict())]

# 1D Dense convolution
NCHKR = [
         #   (1, 1152, 12602, 512, 3)
          ]
for N, C, H, K, R in NCHKR:
    torch_fn = lambda a, b: torch.nn.functional.conv1d(a, b.permute(2, 0, 1))
    configs += [([N, C, H], 
                 [C, R, K], 
                 [N, K, H - R + 1], 
                 torch_fn, 
                 'nc(h+r),crk->nkh',
                 dict())]

# 2D Dense convolution
NCHWKRS = [
           (1, 384, 422, 32, 512, 3, 3)
          ]
for N, C, H, W, K, R, S in NCHWKRS:
    torch_fn = lambda a, b: torch.nn.functional.conv2d(a, b.permute(3, 0, 1, 2))
    configs += [([N, C, H, W], 
                  [C, R, S, K], 
                  [N, K, H - R + 1, W - R + 1], 
                  torch_fn, 
                  'nc(h+r)(w+s),crsk->nkhw',
                  dict())]

# 3D Dense Convolution
NCDHWKTRS = [
         #  (1, 128, 16, 32, 32, 512, 3, 3, 3)
          ]
for N, C, D, H, W, K, T, R, S in NCDHWKTRS:
    torch_fn = lambda a, b: torch.nn.functional.conv3d(a, b.permute(4, 0, 1, 2, 3))
    configs += [([N, C, D, H, W], 
                 [C, T, R, S, K], 
                 [N, K, D - T + 1, H - R + 1, W - R + 1], 
                 torch_fn, 
                 'nc(d+t)(h+r)(w+s),ctrsk->nkdhw',
                 dict())]


# Shift convolution
shift_cuda = torch.utils.cpp_extension.load(
    'shift_cuda', ['kernels/shift_cuda.cpp', 
                   'kernels/shift_cuda_kernel.cu'],
    extra_cflags=['-O3'])
class shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift):
        ctx.save_for_backward(shift)
        return shift_cuda.forward(x, shift)

    @staticmethod
    def backward(ctx, grad_output):
        shift, = ctx.saved_tensors
        grad_output = shift_cuda.backward(grad_output, shift)

        return grad_output, None

NCHWKRS = [
           (1, 384*9, 420, 30, 512, 3, 3)
          ]
for N, C, H, W, K, R, S in NCHWKRS:
    shift_h = np.random.randint(R, size=C, dtype=np.int32) - R//2
    shift_w = np.random.randint(S, size=C, dtype=np.int32) - S//2
    shift_torch =  np.column_stack((shift_w*-1, shift_h*-1))
    shift_torch = torch.from_numpy(shift_torch).cuda()
    def shift_conv(a, b):
        a = shift.apply(a, shift_torch)
        b = b.permute(1, 0).reshape(K, C, 1, 1)
        return torch.nn.functional.conv2d(a, b)
    configs += [([N, C, H, W], 
                  [C, K], 
                  [N, K, H, W], 
                  shift_conv, 
                  'nc(h + sh[c])(w + sw[c]),ck->nkhw',
                  {'sh': shift_h, 'sw': shift_w})]

# Benchmark
torch.set_num_threads(1)
for a_shape, b_shape, c_shape, torch_fn, expr, arrays in configs:
    # initialize input tensors
    a = np.random.randn(*a_shape).astype(np.float16)
    b = np.random.randn(*b_shape).astype(np.float16)
    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()
    ta = triton.ops._einsum.pad(a, [16,16,16,16])
    # triton output
    tc = triton.ops.einsum(expr, ta, b, c_shape, arrays = arrays, bench = True)
    # reference output
    if torch_fn:
        rc = torch_fn(a, b)
    else:
        rc = torch.einsum(expr, a, b)
    # test and benchmark
    bench = triton.ctx_registry[tc].flops / triton.bench_registry[tc] * 1e-3
    diff = (tc - rc).abs().max() / rc.abs().max()
    print(f'{expr:>15}; {str(a_shape):>20}; {str(b_shape):>20};          {bench:4.2f};          {diff:4.2f}')
