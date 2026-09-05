import torch
import torch.nn.functional as F

def _haar_kernels(device, dtype):
    k_ll = torch.tensor([[1,  1],
                         [1,  1]], device=device, dtype=dtype) / 2.0
    k_lh = torch.tensor([[1, -1],
                         [1, -1]], device=device, dtype=dtype) / 2.0
    k_hl = torch.tensor([[ 1,  1],
                         [-1, -1]], device=device, dtype=dtype) / 2.0
    k_hh = torch.tensor([[ 1, -1],
                         [-1,  1]], device=device, dtype=dtype) / 2.0
    return torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0)  # [4,2,2]

def _prepare_group_weights(K, C):
    w = torch.zeros(4*C, 1, 2, 2, device=K.device, dtype=K.dtype)
    for c in range(C):
        w[4*c:4*c+4, 0] = K
    return w

def _even_pad_bchw(x):
    _, _, H, W = x.shape
    pad_h = H & 1  # H % 2
    pad_w = W & 1
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, pad_h, pad_w

def haar_dwt2(x):
    x, ph, pw = _even_pad_bchw(x)
    B, C, H, W = x.shape
    K = _haar_kernels(x.device, x.dtype)
    w = _prepare_group_weights(K, C)  # [4C,1,2,2]
    y = F.conv2d(x, w, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
    y = y.view(B, C, 4, H//2, W//2)
    LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
    return (LL, LH, HL, HH), (ph, pw)

# ---------------- 辅助：鲁棒 σ（MAD） ----------------
def _mad_sigma(x, per_channel=True, eps=1e-8):
    if per_channel:
        med = x.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
        sigma = 1.4826 * mad + eps
        return sigma
    else:
        med = x.median()
        mad = (x - med).abs().median()
        sigma = 1.4826 * mad + eps
        return sigma.view(1,1,1,1)

# ---------------- 小波 L1 正则（只对 LH/HL/HH） ----------------
def wavelet_l1_loss_bchw(residual_bchw: torch.Tensor,
                         normalize=True,
                         per_channel=True,
                         band_weights=(1.0, 1.0, 1.0)):

    (LL, LH, HL, HH), _ = haar_dwt2(residual_bchw)  # 各 [B,C,h,w]
    w_lh, w_hl, w_hh = band_weights

    if normalize:
        Hcat = torch.cat([LH, HL, HH], dim=1)  # [B,3C,h,w]
        if per_channel:
            B, C, h, w = LH.shape
            H3 = torch.stack([LH, HL, HH], dim=2).reshape(B, C, -1)  # [B,C,3*h*w]
            med = H3.median(dim=-1, keepdim=True).values
            mad = (H3 - med).abs().median(dim=-1, keepdim=True).values
            sigma = 1.4826 * mad  # [B,C,1]
            sigma = sigma.view(B, C, 1, 1) + 1e-8
        else:
            sigma = _mad_sigma(Hcat, per_channel=False)  # [1,1,1,1]
        LHn, HLn, HHn = LH / sigma, HL / sigma, HH / sigma

        high_loss = (w_lh * LHn.abs().mean() +
                w_hl * HLn.abs().mean() +
                w_hh * HHn.abs().mean())
    else:
        sigma = 0
        high_loss = (w_lh * LH.abs().mean() +
                w_hl * HL.abs().mean() +
                w_hh * HH.abs().mean())


        
    low_loss = LL.abs().mean()  

    return high_loss, low_loss, sigma

def wavelet_l1_loss_hwc(residual_hwc: torch.Tensor, **kwargs):
    assert residual_hwc.dim() == 3, "expect [H,W,C]"
    x = residual_hwc.permute(2,0,1).unsqueeze(0).contiguous()  # [1,C,H,W]
    return wavelet_l1_loss_bchw(x, **kwargs)
