import triton
import torch
import numpy as np

configs = []

# Matrix multiplication
MNK = [
       (512, 512 ,512), 
       (2048, 2048, 2048), 
       (8192, 8192, 8192),

       (64, 64, 64000),
       (64, 64, 128000),
       (256, 256, 64000),
       (256, 256, 128000),

       (1536, 16, 1536),
       (1536, 32, 1536),
       (1536, 64, 1536),
       (1536, 128, 1536),
       (4096, 16, 4096),
       (4096, 32, 4096),
       (4096, 64, 4096),
       (4096, 128, 4096)
      ]
#for M, N, K in MNK:
#    configs += [([M, K], [K, N], [M, N], None, 'mk,kn->mn')]
#for M, N, K in MNK:
#    configs += [([M, K], [M, N], [M, N], None, 'mk,mn->kn')]
#for M, N, K in MNK:
#    configs += [([M, N], [K, N], [M, N], None, 'mn,kn->mk')]

# Relative attention
NTHSE = [
         (16, 512, 1, 32, 64), 
         (16, 512, 1, 128, 128),
         (16, 512, 1, 256, 256),
         (16, 512, 1, 256, 512),
         (16, 512, 8, 64, 64), 
         (16, 512, 8, 128, 128),
         (16, 512, 8, 256, 256),
         (16, 512, 8, 256, 512),

         (64, 1024, 1, 64, 64), 
         (64, 1024, 1, 128, 128),
         (64, 1024, 1, 256, 256),
         (64, 1024, 1, 256, 512),
         (64, 1024, 8, 64, 64), 
         (64, 1024, 8, 128, 128),
         (64, 1024, 8, 256, 256),
         (64, 1024, 8, 256, 512),

         (128, 1024, 1, 64, 64), 
         (128, 1024, 1, 128, 128),
         (128, 1024, 1, 256, 256),
         (128, 1024, 1, 256, 512),
         (128, 1024, 8, 64, 64), 
         (128, 1024, 8, 128, 128),
         (128, 1024, 8, 256, 256),
         (128, 1024, 8, 256, 512)
        ]
#for N, T, H, S, E in NTHSE:
#    configs += [([N, T, H, S], [H, E, S], [N, H, T, E], None, 'nths,hes->nhte')]
#for N, T, H, S, E in NTHSE:
#    configs += [([N, H, T, E], [N, T, H, S], [H, E, S], None, 'nhte,nths->hes')]
#for N, T, H, S, E in NTHSE:
#    configs += [([N, H, T, E], [H, E, S], [N, T, H, S], None, 'nhte,hes->nths')]

# Dense convolution
NCHWKRS = [
           (16, 16, 16, 16, 16, 3, 3)
          ]
for N, C, H, W, K, R, S in NCHWKRS:
    configs += [([N, C, H, W], 
                 [K, C, R, S], 
                 [N, K, H - R + 1, W - S + 1], 
                 torch.nn.functional.conv2d, 'nc(h+r)(w+s),kcrs->nkhw')]

# Shift convolution
#for N, C, H, W, K, R, S in NCHWKRS:
#    shift_h = np.random.randint(3, size=C)
#    shift_w = np.random.randint(3, size=C)
#    configs += [([N, C, H, W], 
#                 [K, C], 
#                 [N, K, H, W], 
#                 torch.nn.functional.conv2d, 'nc(h+sh[c])(w+sw[c]),kc->nkhw')]

# Benchmark
for a_shape, b_shape, c_shape, torch_fn, expr in configs:
    # initialize input tensors
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()
    # triton output
    tc = triton.ops.einsum(expr, a, b, c_shape, dict(), bench = True)
    # reference output
    if torch_fn:
        rc = torch_fn(a, b)
    else:
        rc = torch.einsum(expr, a, b)
    # test and benchmark
    bench = triton.ctx_registry[tc].flops / triton.bench_registry[tc] * 1e-3
    diff = (tc - rc).abs().max() / rc.abs().max()
    print(f'{expr:>15}; {str(a_shape):>20}; {str(b_shape):>20};          {bench:4.2f};          {diff:4.2f}')