import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_loss(img):
    def gradient(x):
        if x.dim() == 3:  # [C,H,W] → add batch dim
            x = x.unsqueeze(0)
        D_dx = x[:, :, :, :-1] - x[:, :, :, 1:]
        D_dy = x[:, :, :-1, :] - x[:, :, 1:, :]
        return D_dx, D_dy
    dx, dy = gradient(img)
    return (dx.abs().mean() + dy.abs().mean())

class FusionLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_grad=1.0, w_seg=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_seg = w_seg
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, fused, vis, ir, alpha, seg_logits=None, seg_target=None):
        # ---------------------------
        # Fusion loss
        # ---------------------------
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

        target = alpha * vis + (1 - alpha) * (ir3 * 0.6)
        loss_l1 = self.l1(fused, target)
        loss_grad = gradient_loss(fused)

        loss = self.w_l1 * loss_l1 + self.w_grad * loss_grad

        # ---------------------------
        # Segmentation loss (2-class)
        # ---------------------------
        if (seg_logits is not None) and (seg_target is not None):
            # BCE for single-channel OR pick first channel if seg_logits has 2 channels
            if seg_logits.shape[1] != 1:
                seg_logits = seg_logits[:,0:1,:,:]
            seg_target = seg_target.unsqueeze(1).float()  # [B,1,H,W]
            loss_seg = self.bce(seg_logits, seg_target)
            loss += self.w_seg * loss_seg

        return loss
