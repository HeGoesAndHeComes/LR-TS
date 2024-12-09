import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from model.LMA.local_attention import LocalAttention
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

class LocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head=64,
        heads=8,
        dropout=0.,
        causal=False,
        prenorm=False,
        qk_rmsnorm=False,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
        gate_values_per_head=False,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=causal,
            autopad=True,
            scale=(qk_scale if qk_rmsnorm else None),
            exact_windowsize=default(exact_windowsize, True),
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, attn_bias=None):

        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask=mask, attn_bias=attn_bias)

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
