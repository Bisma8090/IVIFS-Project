# src/losses.py  (fusion-only)
import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_loss(img):
    def gradient(x):
        D_dx = x[:, :, :, :-1] - x[:, :, :, 1:]
        D_dy = x[:, :, :-1, :] - x[:, :, 1:, :]
        return D_dx, D_dy
    dx, dy = gradient(img)
    return (dx.abs().mean() + dy.abs().mean())

class FusionLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_grad=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_l1 = w_l1
        self.w_grad = w_grad

    def forward(self, fused, vis, ir, alpha):
        if ir.shape[1] == 1:
            ir3 = ir.repeat(1, 3, 1, 1)
        else:
            ir3 = ir

        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=fused.device)
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
        if alpha.dim() == 1:
            alpha = alpha.view(-1,1,1,1)

        target = alpha * vis + (1 - alpha) * ir3
        loss_l1 = self.l1(fused, target)
        loss_grad = gradient_loss(fused)

        return self.w_l1 * loss_l1 + self.w_grad * loss_grad
