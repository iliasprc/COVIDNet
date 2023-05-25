import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum

"""
Vision Transformer implementation in PyTorch.

This module contains the implementation of Vision Transformer (ViT), a transformer model for image classification tasks.
The module also contains implementation for the following components of the transformer model:
    1. Residual connection (Residual)
    2. Pre-normalization (PreNorm)
    3. FeedForward network (FeedForward)
    4. Multihead Self-attention mechanism (Attention)
    5. Transformer block (Transformer)

Classes:
    Residual(nn.Module): Implements a residual connection.
    PreNorm(nn.Module): Implements pre-normalization.
    FeedForward(nn.Module): Implements a feed forward network.
    Attention(nn.Module): Implements the multihead self-attention mechanism.
    Transformer(nn.Module): Implements the Transformer block.
    ViT(nn.Module): Implements the Vision Transformer (ViT).
"""


class Residual(nn.Module):
    """Implements a residual connection.

    Args:
        fn (nn.Module): The function to apply the residual connection to.

    Returns:
        x (torch.Tensor): The output of the function added with the input.
    """

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Implements pre-normalization.

    Args:
        dim (int): The dimension of the input.
        fn (nn.Module): The function to apply pre-normalization to.

    Returns:
        (torch.Tensor): The function applied on the normalized input.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Implements a feed forward network.

    Args:
        dim (int): The dimension of the input.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float, optional): The dropout rate.

    Returns:
        (torch.Tensor): The output of the feed forward network.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Implements the multihead self-attention mechanism.

    Args:
        dim (int): The dimension of the input.
        heads (int, optional): The number of heads in the multihead attention.
        dim_head (int, optional): The dimension of each head.
        dropout (float, optional): The dropout rate.

    Returns:
        out (torch.Tensor): The output of the multihead self-attention mechanism.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """Implements the Transformer block.

    Args:
        dim (int): The dimension of the input.
        depth (int): The number of layers.
        heads (int): The number of heads in the multihead attention.
        dim_head (int): The dimension of each head.
        mlp_dim (int): The dimension of the feed forward network.
        dropout (float, optional): The dropout rate.

    Returns:
        x (torch.Tensor): The output of the Transformer block.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):

        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    """Implements the Vision Transformer (ViT).

    Args:
        image_size (int): The size of the input images.
        patch_size (int): The size of each patch.
        num_classes (int): The number of classes.
        dim (int): The dimension of the input.
        depth (int): The number of layers.
        heads (int): The number of heads in the multihead attention.
        mlp_dim (int): The dimension of the feed forward network.
        pool (str, optional): The type of pooling ('cls' or 'mean').
        channels (int, optional): The number of channels in the input.
        dim_head (int, optional): The dimension of each head.
        dropout (float, optional): The dropout rate.
        emb_dropout (float, optional): The dropout rate for embeddings.

    Returns:
        (torch.Tensor): The output of the ViT model.
    """

    def __init__(self, *, image_size: int, patch_size: int, num_classes: int, dim: int, depth: int, heads: int,
                 mlp_dim: int, pool: str = 'cls', channels: int = 3, dim_head: int = 64, dropout: float = 0.,
                 emb_dropout: float = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img: torch.Tensor, mask=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
