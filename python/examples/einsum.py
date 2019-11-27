import triton
import torch
import numpy as np

configs = []

# Matrix multiplication
MNK = [
       (512, 512 ,512), 
       #(2048, 2048, 2048), 
       #(8192, 8192, 8192),

       #(64, 64, 64000),
       #(64, 64, 128000),
       #(256, 256, 64000),
       #(256, 256, 128000),

       #(1536, 16, 1536),
       #(1536, 32, 1536),
       #(1536, 64, 1536),
       #(1536, 128, 1536),
       #(4096, 16, 4096),
       #(4096, 32, 4096),
       #(4096, 64, 4096),
       #(4096, 128, 4096)
      ]
#for M, N, K in MNK:
#    configs += [([M, K], [K, N], 'mk,kn->mn')]
#for M, N, K in MNK:
#    configs += [([M, K], [M, N], 'mk,mn->kn')]
#for M, N, K in MNK:
#    configs += [([M, N], [K, N], 'mn,kn->mk')]

# Relative attention
NTHSE = [
         (16, 512, 1, 32, 64), 
         #(16, 512, 1, 128, 128),
         #(16, 512, 1, 256, 256),
         #(16, 512, 1, 256, 512),
         #(16, 512, 8, 64, 64), 
         #(16, 512, 8, 128, 128),
         #(16, 512, 8, 256, 256),
         #(16, 512, 8, 256, 512),

         #(64, 1024, 1, 64, 64), 
         #(64, 1024, 1, 128, 128),
         #(64, 1024, 1, 256, 256),
         #(64, 1024, 1, 256, 512),
         #(64, 1024, 8, 64, 64), 
         #(64, 1024, 8, 128, 128),
         #(64, 1024, 8, 256, 256),
         #(64, 1024, 8, 256, 512),

         #(128, 1024, 1, 64, 64), 
         #(128, 1024, 1, 128, 128),
         #(128, 1024, 1, 256, 256),
         #(128, 1024, 1, 256, 512),
         #(128, 1024, 8, 64, 64), 
         #(128, 1024, 8, 128, 128),
         #(128, 1024, 8, 256, 256),
         #(128, 1024, 8, 256, 512)
        ]
#for N, T, H, S, E in NTHSE:
#    configs += [([N, T, H, S], [H, E, S], 'nths,hes->nhte')]
#for N, T, H, S, E in NTHSE:
#    configs += [([N, H, T, E], [N, T, H, S], 'nhte,nths->hes')]
#for N, T, H, S, E in NTHSE:
#    configs += [([N, H, T, E], [H, E, S], 'nhte,hes->nths')]

# Convolution
NCHWKRS = [
           (16, 16, 16, 16, 16, 3, 3)
          ]
for N, C, H, W, K, R, S in NCHWKRS:
    configs += [([N, C, H, W], [K, C, R, S], 'nc(p+r-1)(q+s-1),kcrs->nkpq')]

def conv_shape(a_shape, b_shape, pad_h, pad_w):
    N, C, H, W = a_shape
    K, C, R, S = b_shape
    P = H - R + 2*pad_h + 1
    Q = W - S + 2*pad_w + 1
    return [N, K, P, Q]

# Benchmark
for a_shape, b_shape, expr in configs:
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()
    if '(' in expr:
        rc = torch.nn.functional.conv2d(a, b, padding=1)
        print(rc.shape)
    else:
        rc = torch.einsum(expr, a, b)
    pad_h = 1
    pad_w = 1
    shape_c = conv_shape(a_shape, b_shape, pad_h, pad_w)
    tc = triton.ops.einsum(expr, a, b, shape_c, True)
    bench = triton.ctx_registry[tc].flops / triton.bench_registry[tc] * 1e-3
    print(tc[0,0,0,0])
    print(rc[0,0,0,0])
    diff = (tc - rc).abs().max() / rc.abs().max()
    print(f'{expr:>15}; {str(a_shape):>20}; {str(b_shape):>20};          {bench:4.2f};          {diff:4.2f}')