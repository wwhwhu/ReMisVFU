import torch
from torch.autograd import grad

w = torch.tensor([1.0, 2.0], requires_grad=True)
A_loss = (w**2).sum()
B_loss = (3 * w).sum()
alpha = 0.000001

# 分开求导
grad_A = grad(A_loss, w, retain_graph=True)[0]
grad_B = grad(B_loss, w, retain_graph=True)[0]
combined = grad_A + alpha * grad_B

# 一起求导
total_loss = A_loss + alpha * B_loss
grad_total = grad(total_loss, w)[0]

# 验证是否相等
print(torch.allclose(grad_total, combined))  # True
