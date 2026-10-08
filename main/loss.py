import torch
import torch.nn as nn
import pytorch_msssim
import math
import torch.nn.functional as F


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
    f_t = F.normalize(f_t, p=2, dim=1)  # torch.norm(f_t[0, :, 0, 0], p=2)=1

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
        self.epoch_max = 10
        self.l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.bank_size = 100
        self.momentum = 0.9
        self.register_buffer("mae_two_bank", torch.zeros(self.bank_size))
        self.register_buffer("mae_bank", torch.zeros(self.bank_size))
        self.ptr = 0
        self.full = False

    def update_bank(self, mae_two_mean, mae_mean):
        self.mae_two_bank[self.ptr] = mae_two_mean.detach()
        self.mae_bank[self.ptr] = mae_mean.detach()

        self.ptr += 1
        if self.ptr >= self.bank_size:
            self.ptr = 0
            self.full = True

    def get_bank_mean(self):
        if self.full:
            return self.mae_two_bank.mean(), self.mae_bank.mean()
        else:
            valid = self.ptr
            if valid == 0:
                return None, None
        return self.mae_two_bank[:valid].mean(), self.mae_bank[:valid].mean()

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
                    w_i = 1 / n
                    feat_loss = cosine_loss(pred_KD[i].detach(), gt_KD[i])
                    T = 0.2
                    wb = F.softmax(mae_two.detach() / T, dim=0)
                    feat_loss = feat_loss * wb
                    feat_loss = feat_loss.sum()

                    # total_loss += 2*w_i * feat_loss
                    loss.append(feat_loss)
                T = 0.2
                loss = torch.stack(loss)
                swi = F.softmax(loss.detach() / T, dim=0)
                total_loss = 0.2 * torch.sum(swi * loss)
        else:
            for i in range(n):
                w_i = 1 / n
                mae_resized = F.interpolate(mae_map, size=pred_KD[i].shape[2:], mode='bilinear', align_corners=False)
                feat_loss = cosine_loss_6(pred_KD[i], gt_KD[i].detach(), mae_resized)
                T = 0.2
                wb = F.softmax(mae.detach() / T, dim=0)
                feat_loss = feat_loss * wb
                feat_loss = feat_loss.sum()

                # total_loss += w_i * feat_loss

                loss.append(feat_loss)
            T = 0.2
            loss = torch.stack(loss)
            swi = F.softmax(loss.detach() / T, dim=0)
            total_loss = 0.1 * torch.sum(swi * loss)

        total_loss = s3_loss + total_loss
        return total_loss
