import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange as rearrange
import numbers



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention_C_M(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(Attention_C_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_A = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_A = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.qkv_B = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_B = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.normA = LayerNorm(dim, LayerNorm_type)
        self.normB = LayerNorm(dim, LayerNorm_type)
        self.gate_A = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.gate_B = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, A, B):
        b, c, h, w = A.shape
        A_1 = self.normA(A)
        g_A = self.gate_A(A_1)
        B_1 = self.normB(B)
        g_B = self.gate_B(B_1)

        qkv = self.qkv_dwconv_A(self.qkv_A(A_1))
        q_A, k_A, v_A = qkv.chunk(3, dim=1)

        qkv = self.qkv_dwconv_B(self.qkv_B(B_1))
        q_B, k_B, v_B = qkv.chunk(3, dim=1)

        q_A = rearrange(q_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_A = rearrange(k_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_A = rearrange(v_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_B = rearrange(q_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_B = rearrange(k_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_B = rearrange(v_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_A = torch.nn.functional.normalize(q_A, dim=-1)
        q_B = torch.nn.functional.normalize(q_B, dim=-1)
        k_A = torch.nn.functional.normalize(k_A, dim=-1)
        k_B = torch.nn.functional.normalize(k_B, dim=-1)

        scale = (q_A.shape[2]) ** -0.5
        
        attn_A = torch.matmul(q_A,k_B.transpose(-2, -1)) * scale
        # attn = attn.softmax(dim=-1)
        attn_A = attn_A.softmax(dim=-1)
        # attn = torch.square(attn)

        out_A = torch.matmul(attn_A, v_A)

        attn_B = torch.matmul(q_B, k_A.transpose(-2, -1)) * scale
        # attn = attn.softmax(dim=-1)
        attn_B = attn_B.softmax(dim=-1)
        # attn = torch.square(attn)

        out_B = torch.matmul(attn_B, v_B)

        out_A = rearrange(out_A, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_B = rearrange(out_B, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_A = out_A * g_A
        out_B = out_B * g_B
        out = out_A + out_B
        # out = x+ out
        return out


class GatedDilatedConv(nn.Module):
    def __init__(self, channels, dilation):
        super(GatedDilatedConv, self).__init__()
        self.conv_feat = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_gate = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = self.sigmoid(self.conv_gate(x))
        return feat * gate


class EdgeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kH, kW = kernel_size
        assert kH % 2 == 1 and kW % 2 == 1, "卷积核高宽必须是奇数"
        assert out_channels % 4 == 0, "out_channels 必须能被4整除"

        self.kH, self.kW = kH, kW
        self.stride = stride
        self.padding = kH // 2
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.per_group = out_channels // 4

        # ---- trainable positive parameters ----
        self.w_top = nn.Parameter(torch.randn(self.per_group, in_channels, kH // 2, kW))
        self.w_left = nn.Parameter(torch.randn(self.per_group, in_channels, kH, kW // 2))
        self.w_lt = nn.Parameter(torch.randn(self.per_group, in_channels, kH // 2, kW // 2))
        self.w_rt = nn.Parameter(torch.randn(self.per_group, in_channels, kH // 2, kW // 2))

        # ---- learnable sign scalars ----
        self.sign_top = nn.Parameter(torch.ones(self.per_group, in_channels, 1, 1))
        self.sign_left = nn.Parameter(torch.ones(self.per_group, in_channels, 1, 1))
        self.sign_lt = nn.Parameter(torch.ones(self.per_group, in_channels, 1, 1))
        self.sign_rt = nn.Parameter(torch.ones(self.per_group, in_channels, 1, 1))

        # ---- prepare static index masks ----
        self.register_buffer("mask_top", self._make_mask("top", kH, kW))
        self.register_buffer("mask_left", self._make_mask("left", kH, kW))
        self.register_buffer("mask_lt", self._make_mask("lt", kH, kW))
        self.register_buffer("mask_rt", self._make_mask("rt", kH, kW))

    def positive(self, w):
        return F.softplus(w)

    def _make_mask(self, mode, kH, kW):
        """
        mode ∈ {top, left, lt, rt}, 生成对应的填充 mask
        mask.shape = (kH, kW), 值为：
          0 = 永远固定0
          1 = 正区域
         -1 = 由正区域推导出的负区域
        """
        mid_h, mid_w = kH // 2, kW // 2
        mask = torch.zeros(kH, kW, dtype=torch.int8)

        if mode == "top":
            mask[:mid_h, :] = 1  # 上半
            mask[mid_h + 1:, :] = -1  # 下半
        elif mode == "left":
            mask[:, :mid_w] = 1  # 左半
            mask[:, mid_w + 1:] = -1  # 右半
        elif mode == "lt":
            for i in range(mid_h):
                for j in range(mid_w):
                    mask[i, j] = 1  # 左上
                    mask[kH - 1 - i, kW - 1 - j] = -1  # 右下
        elif mode == "rt":
            for i in range(mid_h):
                for j in range(mid_w):
                    mask[i, kW - 1 - j] = 1  # 右上
                    mask[kH - 1 - i, j] = -1  # 左下
        return mask

    def build_weight(self, base, mask, sign):
        """
        base: (B, C, h, w) 正权重
        mask: (kH, kW) {0,1,-1}
        sign: (B,1,1,1)
        """
        kH, kW = self.kH, self.kW
        full = torch.zeros(base.size(0), base.size(1), kH, kW, device=base.device, dtype=base.dtype)
        pos_idx = (mask == 1)
        neg_idx = (mask == -1)
        full[:, :, pos_idx] = base.reshape(base.size(0), base.size(1), -1)
        full[:, :, neg_idx] = -base.reshape(base.size(0), base.size(1), -1)
        return full * sign

    def forward(self, x):
        w1 = self.build_weight(self.positive(self.w_top), self.mask_top, self.sign_top)
        w2 = self.build_weight(self.positive(self.w_left), self.mask_left, self.sign_left)
        w3 = self.build_weight(self.positive(self.w_lt), self.mask_lt, self.sign_lt)
        w4 = self.build_weight(self.positive(self.w_rt), self.mask_rt, self.sign_rt)

        weight = torch.cat([w1, w2, w3, w4], dim=0)
        return F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.padding)


class Encoder_SAR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_SAR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        mid = max(1, in_channels // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, 3, 1)   # 3 kernels → 3 weights
        )

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # branch convs
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)

        # stack: (B,3,C,H,W)
        s = torch.stack([f3, f5, f7], dim=1)

        # gating weights: (B,3,H,W)
        g = self.gate(x)
        a = F.softmax(g, dim=1).unsqueeze(2)  # (B,3,1,H,W)

        # weighted fusion: (B,C,H,W)
        fused = (s * a).sum(dim=1)

        # residual (保证输出shape不变)
        out = self.relu(0.1 * fused + x)

        return out


class Encoder_RGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_RGB, self).__init__()

        self.gdc1 = GatedDilatedConv(in_channels, dilation=4)
        self.gdc2 = GatedDilatedConv(in_channels, dilation=2)
        self.gdc3 = GatedDilatedConv(in_channels, dilation=1)

        self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x1 = self.relu(self.gdc1(x)) + x
        x1 = self.relu(self.gdc2(x1)) +x1
        x1 = self.relu(self.gdc3(x1)) +x1
        x = self.fuse(x+x1)
        x = self.relu(x)
        return x


class Edge_Pro(nn.Module):
    def __init__(self, out_channels=1):
        super(Edge_Pro, self).__init__()
        self.Edge_sar = EdgeConv2d(1, 4, 7)
        self.Edge_rgb = EdgeConv2d(1, 4, 7)

        self.Edge_out = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, rgb, sar):
        sar = torch.mean(sar, dim=1, keepdim=True)
        rgb = torch.mean(rgb, dim=1, keepdim=True)
        sar = self.Edge_sar(sar)
        rgb = self.Edge_rgb(rgb)
        self.sar =sar

        Edge = self.Edge_out(torch.concat([sar, rgb], dim=1))
        return Edge


class Edge_Mix_Down(nn.Module):
    def __init__(self, channels, channels_out):
        super(Edge_Mix_Down, self).__init__()
        edge_channels = int(channels / 8)
        #self.Edge = Edge_Pro(edge_channels)

        self.Down_sar = nn.Conv2d(channels, channels_out, kernel_size=3, stride=2, padding=1)
        self.Down_rgb = nn.Conv2d(channels, channels_out, kernel_size=3, stride=2, padding=1)
        self.idx = 0

    def forward(self, rgb, sar):
        #edge = self.Edge(rgb, sar)
        #self.sar = self.Edge.sar
        rgb = self.Down_rgb(rgb)
        sar = self.Down_sar(sar)
        if self.idx % 1000 == 0:
            print(f"rgb:{rgb.mean()},sar:{sar.mean()}")
        self.idx += 1
        return rgb, sar


class fusion(nn.Module):
    def __init__(self, channels_in, channels_out, act=nn.ReLU(inplace=True)):
        super(fusion, self).__init__()
        p = OrderedDict()
        p['conv1'] = nn.Conv2d(channels_in, channels_in, kernel_size=3, bias=False, stride=1, padding=1)
        p['relu1'] = act
        self.pre = nn.Sequential(p)
        e = OrderedDict()
        e['conv1'] = nn.Conv2d(channels_in * 2 + 4, channels_in, kernel_size=3, bias=False, stride=1, padding=1)
        e['relu1'] = act
        e['deconv2'] = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1)
        e['relu2'] = act
        self.end = nn.Sequential(e)

        self.Edge = EdgeConv2d(1, 4, 7)

        self.att = Attention_C_M(channels_in)

    def forward(self, input, S, H):
        input = self.pre(input)
        e = torch.mean(input, dim=1, keepdim=True)
        self.Edge_F = self.Edge(e)
        SH = self.att(S, H)
        output = self.end(torch.concat([input, SH, self.Edge_F], dim=1))
        return output


class Net(nn.Module):
    def __init__(self, optical_channels, sar_channels,iss=False,ist=False):
        super(Net, self).__init__()
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.KD_label = []
        self.iss = iss
        self.ist = ist

        layer_num = [64, 64, 64, 128, 256, 512]
        out_num = [256, 128, 64, 64, 64]

        self.RGB_pre = nn.Conv2d(self.optical_channels, layer_num[0], kernel_size=3, stride=1, padding=1)
        self.SAR_pre = nn.Conv2d(self.sar_channels, layer_num[0], kernel_size=3, stride=1, padding=1)

        self.encoder_list_sar = nn.ModuleList()
        self.encoder_list_rgb = nn.ModuleList()
        self.processor_list_edge = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        if self.iss or self.ist:
            self.proj_list = nn.ModuleList()

        for i in range(len(layer_num) - 1):
            self.encoder_list_sar.append(Encoder_SAR(layer_num[i], layer_num[i]))
            self.encoder_list_rgb.append(Encoder_RGB(layer_num[i], layer_num[i]))
            self.processor_list_edge.append(Edge_Mix_Down(layer_num[i], layer_num[i + 1]))
            if self.iss or self.ist:
                proj = nn.Conv2d(layer_num[i + 1], layer_num[i + 1], 1)
                nn.init.dirac_(proj.weight)
                nn.init.zeros_(proj.bias)
                self.proj_list.append(proj)

        mid = OrderedDict()
        mid['deconv'] = nn.ConvTranspose2d(layer_num[-1], out_num[0], kernel_size=4, stride=2, padding=1)
        mid['relu'] = self.act
        self.mid = nn.Sequential(mid)

        self.att = Attention_C_M(layer_num[-1])

        for i in range(len(out_num) - 1):
            self.decoder_list.append(fusion(out_num[i], out_num[i + 1], act=self.act))

        self.out = nn.Sequential(
            nn.Conv2d(out_num[-1], self.optical_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.Tanh()
        )
        mask_pro = OrderedDict()
        mask_pro['conv1'] = nn.Conv2d(self.optical_channels*2, layer_num[0], 3, 1, 1)
        mask_pro['relu2'] = self.act
        mask_pro['conv3'] = nn.Conv2d(layer_num[0], layer_num[0], 3, 1, 1)
        mask_pro['relu4'] = self.act
        mask_pro['conv5'] = nn.Conv2d(layer_num[0], 1, 3, 1, 1)
        mask_pro['sig6'] = nn.Sigmoid()
        self.mask_pro = nn.Sequential(mask_pro)

    def _split(self, feature):
        RGB = feature[:, :self.optical_channels, :, :]
        SAR = feature[:, self.optical_channels:self.sar_channels + self.optical_channels, :, :]
        return RGB, SAR

    def forward(self, feature):
        self.KD_label = []
        self.D_KD_label = []
        RGB, SAR = self._split(feature)

        # 输入预处理
        x1 = self.RGB_pre(RGB)
        x2 = self.SAR_pre(SAR)

        x1_enc, x2_enc = [], []

        # 编码部分
        for i, (encoder_sar, encoder_rgb, processor_edge) in enumerate(
                zip(self.encoder_list_sar, self.encoder_list_rgb, self.processor_list_edge)
        ):
            x1 = encoder_rgb(x1)
            x2 = encoder_sar(x2)
            x1, x2 = processor_edge(x1, x2)

            x1_enc.append(x1)
            x2_enc.append(x2)
            if self.ist:
                self.D_KD_label.append(self.proj_list[i](x2))
            else:
                self.D_KD_label.append(x2)
            if self.iss:
                self.KD_label.append(self.proj_list[i](x2))
            else:
                self.KD_label.append(x2)
            #self.D_KD_label.append(processor_edge.sar)
            
        #self.KD_label.append(x2)
        
        # 注意力 + 中间层
        att_out = self.att(x1_enc.pop(), x2_enc.pop())

        output = self.mid(att_out)
        self.KD_label.append(output)

        # 解码部分
        for i, decoder in enumerate(self.decoder_list):
            output = decoder(output, x1_enc.pop(), x2_enc.pop())
            self.KD_label.append(output)
            #self.KD_label.append(decoder.Edge_F)

        # 输出层
        output = self.out(output)
        #self.KD_label.append(output)
        mask = self.mask_pro(torch.concat([RGB, output], dim=1))
        self.KD_label.append(mask)
        output = mask * RGB + (1 - mask) * output
        self.KD_label.append(output)
        self.D_KD_label.append(output)
        return output


if __name__ == "__main__":
    optical_channels = 13
    sar_channels = 2

    model = Net(optical_channels=optical_channels, sar_channels=sar_channels)

    x = torch.randn(1, optical_channels + sar_channels, 256, 256)

    with torch.no_grad():
        out = model(x)

    print(f"Output shape: {out.shape}")
