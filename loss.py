import torch
import torch.nn as nn
import pytorch_msssim
import math
import torch.nn.functional as F


class Spatial_Loss(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Loss, self).__init__()
        self.res_scale = in_channels

        self.make_PAN = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        self.L1_loss = nn.L1Loss().cuda()

    def forward(self, ref_HS, pred_HS):
        pan_pred = self.make_PAN(pred_HS)
        with torch.no_grad():
            pan_ref = self.make_PAN(ref_HS)
        spatial_loss = self.L1_loss(pan_pred, pan_ref)

        return spatial_loss


class ECCharbonnierLoss(nn.Module):
    def __init__(self):
        super(ECCharbonnierLoss, self).__init__()
        self._alpha = 0.45
        self._epsilon = 1e-3
        self.EC_weight = 5.
        self.input = 3

    def forward(self, pred_cloudfree, cloudfree, cloudmask):
        batch_size = pred_cloudfree.shape[0]
        loss = 0.
        for i in range(batch_size):
            _pred_cloudfree = pred_cloudfree[i, ...]
            _cloudfree = cloudfree[i, ...]
            _cloudmask = cloudmask[i, ...]
            _weight = torch.ones_like(_cloudmask) + self.EC_weight * _cloudmask
            loss += torch.mean(
                _weight * torch.pow(((_pred_cloudfree - _cloudfree) ** 2 + self._epsilon ** 2), self._alpha))
        return loss / batch_size


class sl1_ssim_loss(nn.Module):
    def __init__(self, c=13):
        super(sl1_ssim_loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.ssim_loss = pytorch_msssim.MS_SSIM(data_range=1, channel=c)
        self.input = 2

    def forward(self, loc_pred, loc_target):
        l1_loss = self.smooth_l1_loss(loc_pred, loc_target).reshape([-1, 1])
        ssim_loss = 1 - self.ssim_loss(loc_pred, loc_target)
        return 0.2 * l1_loss + 0.8 * ssim_loss


class sl1_ssim_sam_loss(nn.Module):
    def __init__(self, c=13):
        super(sl1_ssim_sam_loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.ssim_loss = pytorch_msssim.MS_SSIM(data_range=1, channel=c)
        self.sam_loss = SAMLoss()
        self.input = 2

    def forward(self, loc_pred, loc_target):
        l1_loss = self.smooth_l1_loss(loc_pred, loc_target)
        ssim_loss = 1 - self.ssim_loss(loc_pred, loc_target)
        sam_loss = self.sam_loss(loc_pred, loc_target)

        total_loss = 0.2 * l1_loss + 0.8 * ssim_loss + 0.005 * sam_loss
        return total_loss


def _sam(x1, x2, eps=1e-6):
    B, N, _, _ = x1.shape
    x1_ = x1.reshape(B * N, -1)
    x2_ = x2.reshape(B * N, -1)

    dot = torch.sum(x1_ * x2_, dim=1)
    norm1 = torch.sqrt(torch.sum(x1_ ** 2, dim=1)) + eps
    norm2 = torch.sqrt(torch.sum(x2_ ** 2, dim=1)) + eps

    cos = dot / (norm1 * norm2)

    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)

    SAM = torch.acos(cos) * 180 / math.pi
    return torch.mean(SAM)


class SAMLoss(nn.Module):

    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, pred, target):
        return _sam(pred, target)


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()
        self.L1_Loss = nn.L1Loss()
        self.input = 2

    def forward(self, output, label):
        loss = self.L1_Loss(output, label)
        return loss




class CARL_Loss(nn.Module):
    def __init__(self):
        super(CARL_Loss, self).__init__()
        self.input = 4

    def forward(self, output, target, mask, cloud):
        clear_mask = torch.ones_like(mask) - mask
        return torch.mean(
            clear_mask * torch.abs(output - cloud) + mask * torch.abs(output - target)) + 1.0 * torch.mean(
            torch.abs(output - target))



def normalize_feat(x, eps=1e-8):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def get_adaptive_K(t, eps=1e-6, K_min=4, K_max=32):
    n = t.numel()

    q1 = torch.quantile(t, 0.25)
    q3 = torch.quantile(t, 0.75)
    iqr = q3 - q1

    bin_width = 2 * iqr / (n ** (1 / 3) + eps)
    data_range = t.max() - t.min()

    K = (data_range / (bin_width + eps)).round()
    K = torch.clamp(K, min=K_min, max=K_max)

    return K.long()


def soft_spatial_cluster_kd(f_s, f_t, eps=1e-6):
    B, C, H, W = f_s.shape

    s_mean = f_s.mean(dim=1).view(B, -1)
    t_mean = f_t.mean(dim=1).view(B, -1)

    loss_all = torch.zeros(B, device=f_s.device)

    for b in range(B):
        t = t_mean[b]
        s = s_mean[b]

        K = get_adaptive_K(t)

        t_min, t_max = t.min(), t.max()
        centers = torch.linspace(t_min, t_max, K, device=t.device)

        sigma = t.std() + eps

        dist = (t.unsqueeze(1) - centers.unsqueeze(0)) ** 2
        weights = torch.exp(-dist / (2 * sigma ** 2 + eps))

        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)

        sum_w = weights.sum(dim=0) + eps

        mu_t = (weights * t.unsqueeze(1)).sum(dim=0) / sum_w
        mu_s = (weights * s.unsqueeze(1)).sum(dim=0) / sum_w

        loss_all[b] = F.mse_loss(mu_s, mu_t, reduction='mean')

    return loss_all


def otsu_threshold(wi_flat, bins=64):
    B, N = wi_flat.shape
    tau = []

    for b in range(B):
        x = wi_flat[b]
        x = x - x.min()
        x = x / (x.max() + 1e-8)

        hist = torch.histc(x, bins=bins, min=0.0, max=1.0)
        prob = hist / hist.sum()

        omega = torch.cumsum(prob, dim=0)
        mu = torch.cumsum(prob * torch.arange(bins, device=x.device), dim=0)
        mu_t = mu[-1]

        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-8)
        idx = torch.argmax(sigma_b)
        tau.append(idx.float() / bins)

    tau = torch.stack(tau).view(B, 1)
    return tau


def cosine_loss(f_s, f_t):
    # C normal 
    f_s = f_s.flatten(2)
    f_t = f_t.flatten(2)

    f_s = F.normalize(f_s, p=2, dim=1)
    f_t = F.normalize(f_t, p=2, dim=1) 

    cos_sim = torch.sum(f_s * f_t, dim=1).mean(dim=1)
    return 1 - cos_sim


def cosine_loss_6(f_s, f_t, wi, T=0.2, k=10.0):
    B, C, H, W = f_s.shape

    f_s_flat = f_s.view(B, C, -1)
    f_t_flat = f_t.view(B, C, -1)

    f_s_flat = F.normalize(f_s_flat, p=2, dim=1)
    f_t_flat = F.normalize(f_t_flat, p=2, dim=1)

    cos_sim_map = torch.sum(f_s_flat * f_t_flat, dim=1)

    wi_flat = wi.view(B, -1).detach()

    # wi_norm = wi_flat - wi_flat.min(dim=1, keepdim=True)[0]
    # wi_norm = wi_norm / (wi_norm.max(dim=1, keepdim=True)[0] + 1e-8)

    # tau = otsu_threshold(wi_norm)
    tau = otsu_threshold(wi_flat)

    attn = F.softmax(wi_flat / T, dim=1)

    gate = torch.sigmoid(k * (wi_flat - tau))

    weight = attn * gate
    weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)

    weighted_cos_sim = cos_sim_map * weight
    cos_sim = weighted_cos_sim.sum(dim=1)

    loss = 1 - cos_sim
    return loss




class KD_loss_nl2_select_sarea(nn.Module):
    def __init__(self, c=13):
        super(KD_loss_nl2_select_sarea, self).__init__()
        self.s3_loss = sl1_ssim_sam_loss(c)
        self.input = 'KD'
        self.l1_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, output, true, pred_KD, gt_KD, ist=False, pred_KD_s=None, gt_KD_s=None, epoch=None):
        s3_loss = self.s3_loss(output, true)

        n = len(pred_KD)
        w = []
        sum_w = 0
        total_loss = 0
        loss = []

        pred_last = pred_KD[-1]
        gt_last = gt_KD[-1]

        mae_map_1 = torch.mean(torch.abs(gt_last - pred_last), dim=1, keepdim=True)
        mae_pred = torch.mean(torch.abs(pred_last - true), dim=1, keepdim=True)
        mae_gt = torch.mean(torch.abs(gt_last - true), dim=1, keepdim=True)

        mae_two = mae_map_1.mean(dim=[1, 2, 3])
        mae = mae_pred.mean(dim=[1, 2, 3]) - mae_gt.mean(dim=[1, 2, 3])
        mae_map = mae_pred - mae_gt
        mae_map = mae_map.detach()

        mae_map_1 = mae_map_1.detach()

        if ist:
            n = n - 1
            if n!=0:
                for i in range(n):
                    feat_loss = cosine_loss(pred_KD[i].detach(), gt_KD[i])
                    T = 0.2
                    wb = F.softmax(mae_two.detach() / T, dim=0)
                    feat_loss = feat_loss * wb
                    feat_loss = feat_loss.sum()
                    loss.append(feat_loss)
                T = 0.2
                loss = torch.stack(loss)
                swi = F.softmax(loss.detach() / T, dim=0)
                total_loss = 0.2 * torch.sum(swi * loss)
                #total_loss = 0.2*torch.mean(loss)
        else:
            for i in range(n):
                mae_resized = F.interpolate(mae_map, size=pred_KD[i].shape[2:], mode='bilinear', align_corners=False)
                feat_loss = cosine_loss_6(pred_KD[i], gt_KD[i].detach(), mae_resized)
                T = 0.2
                wb = F.softmax(mae.detach() / T, dim=0)
                feat_loss = feat_loss * wb
                feat_loss = feat_loss.sum()

                loss.append(feat_loss)
            T = 0.2
            loss = torch.stack(loss)
            swi = F.softmax(loss.detach() / T, dim=0)
            total_loss = 0.1 * torch.sum(swi * loss)

        total_loss = s3_loss + total_loss
        return total_loss



