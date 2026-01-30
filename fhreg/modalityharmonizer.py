import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class ModalityHarmonizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mamba_attention = MambaAttention(*args, **kwargs)
        self.cross_attn = CrossAttention(*args, **kwargs)

    def forward(self, desc0, desc1, position_encoding0, position_encoding1):
        desc0 = self.mamba_attention(desc0, position_encoding0)
        desc1 = self.mamba_attention(desc1, position_encoding1)
        desc0, desc1 = self.cross_attn(desc0, desc1)
        return desc0, desc1


class MambaAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        self.mh_attn = MultiHeadAttention()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.mamba_twopath = MambaTwoPath(self.embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
    
    def rotary_emb(self, freqs, q, k):
        q = (q * freqs[0]) + (rotate_half(q) * freqs[1])
        k = (k * freqs[0]) + (rotate_half(k) * freqs[1])
        return q, k
    
    def forward(self, x, position_encoding):
        x_ori = x 
        qkv = self.to_qkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q, k = self.rotary_emb(position_encoding, q, k)
        o_x = self.mh_attn(q, k, v)
        o_x = self.out_proj(o_x.transpose(1, 2).flatten(start_dim=-2))

        s_x, c_x = self.mamba_twopath(x)

        x = self.ffn(torch.cat([x, o_x, s_x, c_x], -1))

        return x + x_ori
    

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def map_(self, func, x0, x1):
        return func(x0), func(x1)

    def forward(self, x0, x1):
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2), (qk0, qk1, v0, v1))
        
        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
        m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
           
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1
    

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")

    def forward(self, q, k, v):
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            return q.new_zeros((*q.shape[:-1], v.shape[-1]))
        
        args = [x.contiguous() for x in [q, k, v]]
        v = F.scaled_dot_product_attention(*args, attn_mask=None)
        return v
        
        

class MambaTwoPath(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=5, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, \
                        dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True, layer_idx=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(self.d_inner // 2, **factory_kwargs)*(math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner // 2,).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner // 2, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(in_channels=self.d_inner // 2, out_channels=self.d_inner // 2, bias=conv_bias // 2, \
                                  kernel_size=d_conv, groups=self.d_inner // 2, **factory_kwargs)
        self.conv1d_z = nn.Conv1d(in_channels=self.d_inner // 2, out_channels=self.d_inner // 2, bias=conv_bias // 2, \
                                  kernel_size=d_conv, groups=self.d_inner // 2, **factory_kwargs)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        """
        (_, seqlen, _,) = (hidden_states.shape)  # [B, 512, D]     512(L) : max_num_keypoints, D : 256 (usually)
        xz = self.in_proj(hidden_states)  # [B, L, 2*D]
        xz = rearrange(xz, "b l d -> b d l")  # [B, 2D, L]
        x, z = xz.chunk(2, dim=1)  # [B, D, L], [B, D, L]
        A = -torch.exp(self.A_log.float())  # [256]
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding="same", groups=self.d_inner//2))  # [B, D, L]
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding="same", groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # [B*D, L]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)  # [BL, dt_rank], [16384, 16], [16384, 16]
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)  # [B, dt_rank, L]
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # [B, dt_state, L]
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # [B, dt_state, L]
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=None)  # [B, D, L]

        y = rearrange(y, "b d l -> b l d")  # [B, L, D]
        z = rearrange(z, "b d l -> b l d")  # [B, L, D]

        out_y = self.out_proj(y)
        out_z = self.out_proj(z)
        return out_y, out_z  # [B, 512, 256], [B, 512, 256]


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class FourierPositionalEmbedding(nn.Module):
    def __init__(self, M, dim, F_dim=None, gamma=1.0):
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x):
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)



