# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba
from models.csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

class Block(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type = None,
                 k_group = None,
                 dim=None,
                 dt_rank = None,
                 d_inner = None,
                 d_state = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type

        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=(1, 1, 1), ** kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def flatten_spectral_spatial(self, x):
        x = rearrange(x, 'b c t h w -> b c (h w) t')  # [10, 192, 64, 28]
        x = rearrange(x, 'b c n m -> b c (n m)')  # [10, 192, 1792]
        return x
    def flatten_spatial_spectral(self, x):
        x = rearrange(x, 'b c t h w -> b c t (h w)')  # [10, 192, 28, 64]
        x = rearrange(x, 'b c n m -> b c (n m)')  # [10, 192, 1792]
        return x
    def reshape_spectral_spatial(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view(B, H * W, T, -1)
        y = rearrange(y, 'b o t c -> b t o c')
        y = y.view(B, T, H, W, -1)
        return y
    def reshape_spatial_spectral(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view(B, T, H, W, -1)
        return y

    def scan(self, x, scan_type=None, group_type=None):
        if scan_type == 'Spectral-priority':
            x = self.flatten_spectral_spatial(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Spatial-priority':
            x = self.flatten_spatial_spectral(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Cross spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spa, dims=[-1])], dim=1)
        elif scan_type == 'Cross spatial-spectral':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spa, torch.flip(x_spe, dims=[-1])], dim=1)
        elif scan_type == 'Parallel spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spe, dims=[-1]), x_spa, torch.flip(x_spa, dims=[-1])], dim=1)
        return xs

    def forward(self, x: Tensor, SelectiveScan = SelectiveScanMamba):
        x = self.in_proj(x)  # d_inner=192  [10, 64, 96] -> [10, 64, 384]    [10, 8, 8, 96]->[10, 8, 8, 384]
        x, z = x.chunk(2, dim=-1)  # [10, 64, 192]  [10, 8, 8, 192]
        z = self.act(z)  # [10, 64, 192]   [10, 8, 8, 192]

        # forward con1d
        x = x.permute(0, 4, 1, 2, 3).contiguous() #[64, 192, 28, 8, 8]
        x = self.conv3d(x) #[64, 192, 28, 8, 8]
        x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, T, H, W = x.shape
        L = T * H * W
        D, N = self.A_logs.shape  # D 768   N 16
        K, D, R = self.dt_projs_weight.shape  # 4   192    6

        # scan
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight) #[10, 2, 64, 64] einsum指定输入张量和输出张量之间的维度关系，你可以定义所需的运算操作
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  #[10, 2, 32, 64]  [10, 2, 16, 64]  [10, 2, 16, 64]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  #[10, 2, 192, 64]

        xs = xs.view(B, -1, L) # [10, 384, 64]  [10, 768, 64]
        dts = dts.contiguous().view(B, -1, L) # [10, 768, 64] .contiguous()是一个用于确保张量存储连续性
        Bs = Bs.contiguous()  #[10, 2, 16, 64]   [10, 4, 16, 64]
        Cs = Cs.contiguous()  #[10, 2, 16, 64]   [10, 4, 16, 64]

        As = -torch.exp(self.A_logs.float())   # [384, 16]  [768, 16]
        Ds = self.Ds.float() # (k * d)  [384]  [768]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # [384]  [768]

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)  #[10, 384, 64]->[10, 2, 192, 64]    [10, 4, 192, 64]
        assert out_y.dtype == torch.float

        if self.group_type == 'Cube':
            if self.scan_type == 'Spectral-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
                y = self.reshape_spectral_spatial(y, B, H, W, T)
                y = self.out_norm(y)  # [10, 64, 192]
                ###### equal to above ####
                # y_fwd = out_y[:, 0]
                # y_fwd = self.reshape_spectral_spatial(y_fwd, B, H, W, T)
                # y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                # y_rvs = self.reshape_spectral_spatial(y_rvs, B, H, W, T)
                # y = y_fwd + y_rvs
                # y = self.out_norm(y)  # [10, 64, 192]
            elif self.scan_type == 'Spatial-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
                y = self.reshape_spatial_spectral(y, B, H, W, T)
                y = self.out_norm(y)  # [10, 64, 192]
            elif self.scan_type == 'Cross spectral-spatial':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spectral_spatial(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spatial_spectral(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Cross spatial-spectral':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spatial_spectral(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spectral_spatial(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Parallel spectral-spatial':
                ye = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
                ye = self.reshape_spectral_spatial(ye, B, H, W, T)
                ya = out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1])  # [10, 192, 64]
                ya = self.reshape_spatial_spectral(ya, B, H, W, T)
                y = ye + ya
                y = self.out_norm(y)  # [10, 64, 192]

        y = y * z   #[10, 64, 192]   [10, 8, 8, 192]
        out = self.dropout(self.out_proj(y))  #[10, 64, 96]   [10, 8, 8, 96]

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    

class VisionMamba(nn.Module):
    def __init__(
            self,
            group_type=None,
            k_group=None,
            depth=None,
            embed_dim=None,
            dt_rank: int = None,
            d_inner: int = None,
            d_state: int = None,
            num_classes: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            fused_add_norm=False,
            residual_in_fp32=True,
            bimamba=True,
            # video
            fc_drop_rate=0.,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            Pos_Cls = False,
            scan_type=None,
            pos: str = None,
            cls: str = None,
            conv3D_channel: int = None,
            conv3D_kernel: int = None,
            dim_patch: int = None,
            dim_linear: int = None,
            **kwargs,
        ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.Pos_Cls = Pos_Cls
        self.scan_type = scan_type
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.k_group = k_group
        self.group_type = group_type

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )

        self.embedding_spatial_spectral = nn.Sequential(nn.Linear(conv3D_channel, embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                Block(
                    scan_type=scan_type,
                    group_type=group_type,
                    k_group=k_group,
                    dim=embed_dim,
                    d_state=d_state,
                    d_inner=d_inner,
                    dt_rank=dt_rank,
                    bimamba=bimamba,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

    def forward_features(self, x, inference_params=None):
        x = self.conv3d_features(x)  #[10, 1, 30, 15, 15]->[10, 32, 28, 8, 8]
        #scan
        x = rearrange(x, 'b c t h w -> b t h w c')  # [64, 28, 8, 8, 32]
        x = self.embedding_spatial_spectral(x)  # [10, 8, 8, 96]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1, 1, 64]  stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_token, x), dim=1)  ##[10, 1793, 64]
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        # mamba impl
        for idx, layer in enumerate(self.layers):  ##24
            x = x + self.drop_path(layer(self.norm(x)))

        return self.flatten(self.avgpool(x.permute(0, 4, 1, 2, 3)).mean(dim=2)) #[64, 28, 8, 8, 32]->[64, 32, 28, 8, 8]->[10, 32]

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)  ##[10, 192]
        x = self.head(self.head_drop(feature))   #[10, 9]
        return x, feature


