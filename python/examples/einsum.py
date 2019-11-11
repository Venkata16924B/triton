import triton
import torch
import numpy as np

configs = []

# Matrix multiplication
MNK = [
       (512 , 512 , 512 ), 
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
       (4096, 64, 4096)
      ]
for M, N, K in MNK:
    configs += [([M, K], [K, N], 'mk,kn->mn')]
for M, N, K in MNK:
    configs += [([M, K], [M, N], 'mk,mn->kn')]
for M, N, K in MNK:
    configs += [([M, N], [K, N], 'mn,kn->mk')]

# Relative attention
NTHSE = [
         (16, 512, 1, 64, 64),
         (16, 256, 2, 128, 128),
         (64, 2048, 8, 64, 64),
         (128, 1024, 4, 64, 128)
        ]
for N, T, H, S, E in NTHSE:
    configs += [([N, T, H, S], [H, E, S], 'nths,hes->nhte')]
for N, T, H, S, E in NTHSE:
    configs += [([N, H, T, E], [N, T, H, S], 'nhte,nths->hes')]
for N, T, H, S, E in NTHSE:
    configs += [([N, H, T, E], [H, E, S], 'nhte,hes->nths')]

# Benchmark
for a_shape, b_shape, expr in configs:
    a = np.random.randn(*a_shape).astype(np.float16)
    b = np.random.randn(*b_shape).astype(np.float16)
    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()
    rc = torch.einsum(expr, a, b)
    tc = triton.ops.einsum(expr, a, b, True)
    bench = triton.ctx_registry[tc].flops / triton.bench_registry[tc] * 1e-3
    diff = (tc - rc).abs().max() / rc.abs().max()
    print(f'{expr:>15}; {str(a_shape):>20}; {str(b_shape):>20}; {bench:4.2f}, {diff:4.2f}')