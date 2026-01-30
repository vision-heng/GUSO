import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
from functools import partial
from einops import rearrange
from kornia.utils import create_meshgrid

# Import necessary components from timm
from timm.layers import (
    trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp,
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, to_ntuple
)

# =========================================================================
# 1. Base Components (Wavelet & Block)
# =========================================================================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """Creates wavelet decomposition and reconstruction filters."""
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    
    # Decomposition filters (LL, LH, HL, HH)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    
    # Reconstruction filters
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class WTConv2d(nn.Module):
    """Wavelet Transform Convolutional Layer."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', 
                                   stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', 
                      stride=1, groups=in_channels*4, bias=False) 
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels*4, 1, 1], init_scale=0.1) 
            for _ in range(self.wt_levels)
        ])

        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride) if stride > 1 else None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        # Recursive Multi-level Wavelet Decomposition
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # Recursive Reconstruction
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        self.conv = create_conv2d(in_chs, out_chs, 1, stride=1) if in_chs != out_chs else nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class WTCBlock(nn.Module):
    """Wavelet Transform Convolutional Block."""
    def __init__(self, in_chs, out_chs=None, kernel_size=5, stride=1, dilation=(1, 1), mlp_ratio=4,
                 conv_mlp=False, conv_bias=True, use_grn=False, ls_init_value=1e-6, act_layer='gelu',
                 norm_layer=None, drop_path=0., wt_levels=0):
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        
        self.use_conv_mlp = conv_mlp
        self.conv_dw = WTConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, bias=conv_bias, wt_levels=wt_levels)
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        
        x = self.drop_path(x) + self.shortcut(shortcut)
        return x

class WTCStage(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size=5, stride=2, depth=2, dilation=(1, 1),
                 drop_path_rates=None, ls_init_value=1.0, conv_mlp=False, conv_bias=True,
                 use_grn=False, act_layer='gelu', norm_layer=None, norm_layer_cl=None, wt_levels=0):
        super().__init__()
        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(in_chs, out_chs, kernel_size=ds_ks, stride=stride, dilation=dilation[0], padding=pad, bias=conv_bias),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(WTCBlock(
                in_chs=in_chs, out_chs=out_chs, kernel_size=kernel_size, dilation=dilation[1],
                drop_path=drop_path_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                conv_bias=conv_bias, use_grn=use_grn, act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl, wt_levels=wt_levels
            ))
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# =========================================================================
# 2. WFENet
# =========================================================================

class WFENet(nn.Module):
    def __init__(self, depths=(3, 3), dims=(96, 192), wt_levels=(5, 4), kernel_sizes=(5, 5)):
        super().__init__()
        
        # Part 1: Backbone construction
        patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            LayerNorm2d(dims[0]),
        )
        stem_stride = patch_size

        self.stages = nn.Sequential()
        # Drop path rates initialized to 0
        dp_rates = [x.tolist() for x in torch.linspace(0, 0.0, sum(depths)).split(depths)] 
        
        prev_chs = dims[0]
        curr_stride = stem_stride
        
        for i in range(2):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            curr_stride *= stride
            out_chs = dims[i]
            
            self.stages.add_module(str(i), WTCStage(
                prev_chs, out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=1e-6,
                conv_bias=True,
                act_layer='gelu',
                norm_layer=LayerNorm2d,
                norm_layer_cl=LayerNorm,
                wt_levels=wt_levels[i]
            ))
            prev_chs = out_chs
            
        self.apply(self._init_weights)
        
        # Part 2: Channel dimension conversion
        self.dimconvert_4x = nn.Conv2d(96, 128, 1)
        self.dimconvert_8x = nn.Conv2d(192, 256, 1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, data):
        B, _, H, W = data['image_opt'].shape
        # Concatenate Optical and SAR images along batch dimension
        x = torch.cat([data['image_opt'], data['image_sar']], 0)
        
        x = self.stem(x)        # -> [2B, 96, H/4, W/4]
        x = self.stages[0](x)   # -> [2B, 96, H/4, W/4]
        feat_raw_4 = x
        
        x = self.stages[1](x)   # -> [2B, 192, H/8, W/8]
        feat_raw_8 = x

        # Projection and splitting Optical/SAR back into separate tensors
        feat_8_0, feat_8_1 = self.dimconvert_8x(feat_raw_8).split(B)
        feat_4_0, feat_4_1 = self.dimconvert_4x(feat_raw_4).split(B)

        # Grid Generation
        scale = 8
        h_8, w_8 = H // scale, W // scale
        device = data['image_opt'].device
        
        grid_base = (create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0)
        grid_flat = rearrange(grid_base, 'h w t -> (h w) t')
        grid_8 = torch.stack([grid_flat] * B, 0)

        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],
            'h_8': h_8,
            'w_8': w_8,
            'hw_8': h_8 * w_8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })
        return data
