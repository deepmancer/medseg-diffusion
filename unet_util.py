import copy
import math
from collections import namedtuple
from functools import partial
from random import random

import cv2
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.autograd import Variable
from torch.fft import fft2, ifft2
from torch.nn import Module, ModuleList, functional as F

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim=True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Upsample, self).__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Downsample, self).__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
        self.conv = nn.Conv2d(dim * 4, dim_out, 1)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.conv(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(Module):
    def __init__(self, input_channels, output_channels, time_embedding_dim=None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, output_channels * 2)
        ) if exists(time_embedding_dim) else None

        self.block1 = Block(input_channels, output_channels, groups=groups)
        self.block2 = Block(output_channels, output_channels, groups=groups)
        self.res_conv = nn.Conv2d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, x, time_embedding = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_embedding):
            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, 'b c -> b c 1 1')
            scale_shift = time_embedding.chunk(2, dim = 1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Upsample, self).__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Downsample, self).__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
        self.conv = nn.Conv2d(dim * 4, dim_out, 1)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.conv(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, self.inner_dim, 1)
        self.conv2 = nn.Conv2d(self.inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, dim_head = 32, heads = 4, depth = 1):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Residual(Attention(dim, dim_head = dim_head, heads = heads)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
