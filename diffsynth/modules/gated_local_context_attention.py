import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except Exception:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
) -> torch.Tensor:
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class GatedLocalContextAttention(nn.Module):
    """Gated Local-Context Attention (GLCA).

    This module performs framewise cross-attention from video tokens (Q) to local
    audio tokens (K/V) with a strict per-frame constraint, and returns a gated
    residual (caller does `x = x + glca(...)`).

    Inputs:
      - x: (B, L, dim)
      - audio_tokens: (B, T_latent, w, dim)
      - video_frame_lens: (B, N_groups)
      - audio_frame_lens: (B, N_groups)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 5,
        gate_init: float = 1e-5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={self.dim}, heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        self.window_size = int(window_size)

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.o_proj = nn.Linear(self.dim, self.dim)
        self.norm_q = RMSNorm(self.dim, eps=eps)
        self.norm_k = RMSNorm(self.dim, eps=eps)

        # Learnable gate controls the residual audio-attention strength.
        self.gate = nn.Parameter(torch.full((self.dim,), float(gate_init)))

    def forward(
        self,
        x: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        video_frame_lens: torch.Tensor | None,
        audio_frame_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if audio_tokens is None or video_frame_lens is None or audio_frame_lens is None:
            return x.new_zeros(x.shape)

        if x.dim() != 3:
            raise ValueError(f"x must be (B,L,D), got {tuple(x.shape)}")
        if audio_tokens.dim() != 4:
            raise ValueError(f"audio_tokens must be (B,T,w,D), got {tuple(audio_tokens.shape)}")

        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"x dim mismatch: expected {self.dim}, got {D}")
        if audio_tokens.shape[0] != B or audio_tokens.shape[-1] != D:
            raise ValueError(
                f"audio_tokens shape mismatch: expected (B,T,w,{D}), got {tuple(audio_tokens.shape)}"
            )

        if video_frame_lens.dim() != 2 or audio_frame_lens.dim() != 2:
            raise ValueError(
                f"frame_lens must be (B,N), got video={tuple(video_frame_lens.shape)} audio={tuple(audio_frame_lens.shape)}"
            )
        if video_frame_lens.shape != audio_frame_lens.shape:
            raise ValueError(
                f"video_frame_lens and audio_frame_lens must have same shape, got {tuple(video_frame_lens.shape)} vs {tuple(audio_frame_lens.shape)}"
            )

        # Compute framewise video-to-audio local attention.
        if int((audio_frame_lens > 0).sum().item()) == 0:
            return x.new_zeros(x.shape)

        out = x.new_zeros((B, L, D))

        for b in range(B):
            v_lens = video_frame_lens[b].tolist()
            a_lens = audio_frame_lens[b].tolist()
            if sum(v_lens) != L:
                raise ValueError(f"sum(video_frame_lens) must equal L={L}, got {sum(v_lens)}")

            audio_idx = 0
            offset = 0
            for vl, al in zip(v_lens, a_lens):
                vl = int(vl)
                al = int(al)
                if vl < 0 or al < 0:
                    raise ValueError("frame lens must be non-negative")
                if vl == 0:
                    continue

                if al > 0:
                    if al != self.window_size:
                        raise ValueError(
                            f"audio_frame_lens must be 0 or window_size={self.window_size}, got {al}"
                        )
                    if audio_idx >= audio_tokens.shape[1]:
                        raise ValueError("audio_tokens length is smaller than nonzero audio_frame_lens")

                    q = x[b, offset : offset + vl].unsqueeze(0)
                    kv = audio_tokens[b, audio_idx].unsqueeze(0)
                    audio_idx += 1

                    q = self.norm_q(self.q_proj(q))
                    k = self.norm_k(self.k_proj(kv))
                    v = self.v_proj(kv)

                    y = flash_attention(q, k, v, num_heads=self.num_heads)
                    out[b, offset : offset + vl] = self.o_proj(y[0])

                offset += vl

            if audio_idx != audio_tokens.shape[1]:
                raise ValueError(
                    f"audio_tokens T={audio_tokens.shape[1]} must match count(audio_frame_lens>0)={audio_idx}"
                )

        return out * self.gate
