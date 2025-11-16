# src/models/cvifsm.py  (fusion-only version)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
    def forward(self, x): return self.pool(x)

class AttModalInteraction(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch//8, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch//8, 4), ch, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, f_vis, f_ir, alpha):
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(float(alpha), device=f_vis.device)
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
        if alpha.dim() == 1:
            alpha = alpha.view(-1,1,1,1)

        v = alpha * f_vis
        r = (1.0 - alpha) * f_ir
        combined = v + r

        ca = self.channel_fc(combined)
        combined_att = combined * ca
        g = torch.sigmoid(self.spatial(combined_att))
        f_vis_mod = f_vis * g + v * (1 - g)
        f_ir_mod  = f_ir  * (1 - g) + r * g

        return f_vis_mod, f_ir_mod

class FeatureAggregator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(x)

class FusionDecoder(nn.Module):
    def __init__(self, in_ch, mid_ch=64, out_ch=3):
        super().__init__()
        self.dec = nn.Sequential(
            ConvBlock(in_ch, mid_ch),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(mid_ch, max(mid_ch//2, 8)),
            nn.Conv2d(max(mid_ch//2,8), out_ch, 1)
        )
    def forward(self, x): return self.dec(x)

class CVIFSM(nn.Module):
    """
    Fusion-only CVIFSM
    """
    def __init__(self, base_ch=32, att_enable=True, use_mask=True):
        super().__init__()
        self.att_enable = att_enable
        self.use_mask = use_mask

        # encoder stages
        self.e_vis_0 = ConvBlock(3, base_ch)
        self.e_ir_0  = ConvBlock(1, base_ch)
        self.down0   = Downsample()

        self.e_vis_1 = ConvBlock(base_ch, base_ch*2)
        self.e_ir_1  = ConvBlock(base_ch, base_ch*2)
        self.down1   = Downsample()

        self.e_vis_2 = ConvBlock(base_ch*2, base_ch*4)
        self.e_ir_2  = ConvBlock(base_ch*2, base_ch*4)

        self.att0 = AttModalInteraction(base_ch) if att_enable else None
        self.att1 = AttModalInteraction(base_ch*2) if att_enable else None
        self.att2 = AttModalInteraction(base_ch*4) if att_enable else None

        self.agg = FeatureAggregator(base_ch*8, base_ch*4)

        self.fusion_head = FusionDecoder(base_ch*4, mid_ch=base_ch*4, out_ch=3)

    def forward(self, vi, ir, alpha=0.5, mask=None):
        B,_,H,W = vi.shape

        f_v0 = self.e_vis_0(vi)
        f_i0 = self.e_ir_0(ir)
        if self.att_enable:
            f_v0, f_i0 = self.att0(f_v0, f_i0, alpha)
        v1 = self.down0(f_v0); i1 = self.down0(f_i0)

        f_v1 = self.e_vis_1(v1)
        f_i1 = self.e_ir_1(i1)
        if self.att_enable:
            f_v1, f_i1 = self.att1(f_v1, f_i1, alpha)
        v2 = self.down1(f_v1); i2 = self.down1(f_i1)

        f_v2 = self.e_vis_2(v2)
        f_i2 = self.e_ir_2(i2)
        if self.att_enable:
            f_v2, f_i2 = self.att2(f_v2, f_i2, alpha)

        cat = torch.cat([f_v2, f_i2], dim=1)
        feat = self.agg(cat)

        if (mask is not None) and self.use_mask:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask_feat = F.interpolate(mask.float(), size=feat.shape[2:], mode='bilinear', align_corners=False)
            feat = feat * mask_feat + feat * (1 - mask_feat)

        fused = self.fusion_head(feat)
        fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

        return fused
