import torch
import triton

N, C, K = 32, 8, 32
H, W = 16, 16
R, S = 3, 3
torch.manual_seed(0)
a = torch.rand(N, C, H, W).cuda()
b = torch.rand(K, R, S, C).cuda()
a.requires_grad_(True)

rc = torch.nn.functional.conv2d(a, b.permute(0, 3, 1, 2))
tc = triton.ops.conv(a, b)
#print(rc)
#print(tc)
#print(rc.shape)
#print(tc.shape)
print((rc - tc).abs().max())

# gradients
#dc = torch.randn(rc.shape).cuda()
#rc.backward(dc)
#rda = a.grad
#print(rda)

#print((rc[:30,:30,:,:] - tc[:30, :30, :, :]).abs().max())
#print(tc[31, 31,:,:])