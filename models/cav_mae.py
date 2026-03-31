# -*- coding: utf-8 -*-

import os
os.environ['TORCH_HOME'] = './pretrained_model'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0., type='nofuse'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if type == 'fuse':
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.first_batch = True

    def forward(self, x, ft=False):
        if ft:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn

        B, N, C = x.shape

        # multi-modal fusion path (N > 512)
        if N > 512:
            if self.first_batch:
                self.q.weight.data = self.qkv.weight.data[:self.dim, :]
                self.q.bias.data   = self.qkv.bias.data[:self.dim]
                self.k.weight.data = self.qkv.weight.data[self.dim:self.dim * 2, :]
                self.k.bias.data   = self.qkv.bias.data[self.dim:self.dim * 2]
                self.v.weight.data = self.qkv.weight.data[self.dim * 2:, :]
                self.v.bias.data   = self.qkv.bias.data[self.dim * 2:]
                self.first_batch = False
            qkv = torch.cat(
                [self.q(x), self.k(x), self.v(x)], dim=-1
            ).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        vis_attn = attn.detach().mean(0).mean(0) if N > 512 else None

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, vis_attn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, type='nofuse'):
        super().__init__()
        self.norm1   = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, type=type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2   = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None, ft=False):
        if modality is None:
            output, attn = self.attn(self.norm1(x), ft=ft)
            x = x + self.drop_path(output)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            output, attn = self.attn(self.norm1_a(x), ft=ft)
            x = x + self.drop_path(output)
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            output, attn = self.attn(self.norm1_v(x), ft=ft)
            x = x + self.drop_path(output)
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x, attn


# ─────────────────────────────────────────────────────────────────
#  CAVMAEFT
# ─────────────────────────────────────────────────────────────────

class CAVMAEFT(nn.Module):
    def __init__(self, label_dim, img_size=224, audio_length=1024,
                 patch_size=16, in_chans=3, embed_dim=768,
                 modality_specific_depth=11, num_heads=12, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True):
        super().__init__()
        timm.models.vision_transformer.Block    = Block
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        print('Use norm_pix_loss: ', norm_pix_loss)

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(
            self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(
            torch.zeros(1, self.patch_embed_a.num_patches, embed_dim),
            requires_grad=tr_pos)
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.patch_embed_v.num_patches, embed_dim),
            requires_grad=tr_pos)

        self.blocks_a = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        self.blocks_v = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        self.blocks_u = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer, type='fuse')
            for _ in range(12 - modality_specific_depth)
        ])

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm   = norm_layer(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, label_dim),
        )

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    # ── weight init ───────────────────────────────────────────────

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(
            self.pos_embed_a.shape[-1], 8,
            int(self.patch_embed_a.num_patches / 8), cls_token=False)
        self.pos_embed_a.data.copy_(
            torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(
            self.pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches ** .5),
            int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(
            torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        torch.nn.init.xavier_uniform_(
            self.patch_embed_a.proj.weight.data.view(
                [self.patch_embed_a.proj.weight.data.shape[0], -1]))
        torch.nn.init.xavier_uniform_(
            self.patch_embed_v.proj.weight.data.view(
                [self.patch_embed_v.proj.weight.data.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_patch_num(self, input_shape, stride):
        test_input  = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj   = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    # ── training forward ──────────────────────────────────────────

    def forward(self, a, v, mode):
        if mode == 'multimodal':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v

            for blk in self.blocks_a:
                a, _ = blk(a, ft=True)
            for blk in self.blocks_v:
                v, _ = blk(v, ft=True)

            x = torch.cat((a, v), dim=1)
            for blk in self.blocks_u:
                x, _ = blk(x, ft=True)
            x = self.norm(x)
            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x

        elif mode == 'audioonly':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a
            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            return x

        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v
            for blk in self.blocks_v:
                v = blk(v)
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            return x

        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a
            for blk in self.blocks_a:
                a = blk(a)
            u = a
            for blk in self.blocks_u:
                u = blk(u)
            u = self.norm(u)
            u = u.mean(dim=1)
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            a = a.mean(dim=1)
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v
            for blk in self.blocks_v:
                v = blk(v)
            u = v
            for blk in self.blocks_u:
                u = blk(u)
            u = self.norm(u)
            u = u.mean(dim=1)
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            v = v.mean(dim=1)
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # ── original TTA forward (READ) ───────────────────────────────

    def forward_eval(self, a, v, mode):
        if mode == 'multimodal':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v

            for blk in self.blocks_a:
                a, _ = blk(a)
            for blk in self.blocks_v:
                v, _ = blk(v)

            x = torch.cat((a, v), dim=1)
            for blk in self.blocks_u:
                x, attn = blk(x, ft=False)
            x = self.norm(x)
            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x, attn

    # ── CR-RTTA forward ───────────────────────────────────────────

    def forward_eval_crrtta(self, a, v, mode: str = "multimodal") -> dict:
        """
        Returns per-modality logits, GAP features, and fused logits.
        Keys: z_a, z_v, z_fused, f_a, f_v, attn
        """
        # ── audio branch ─────────────────────────────────────────
        a_ = a.unsqueeze(1).transpose(2, 3)
        a_ = self.patch_embed_a(a_)
        a_ = a_ + self.pos_embed_a + self.modality_a

        for blk in self.blocks_a:
            a_, _ = blk(a_, ft=False)

        # modality-specific upper blocks → audio-only representation
        a_ms = a_
        for blk in self.blocks_u:
            a_ms, _ = blk(a_ms, modality='a', ft=False)
        a_ms = self.norm_a(a_ms)
        f_a  = a_ms.mean(dim=1)          # (B, D)
        z_a  = self.mlp_head(f_a)        # (B, C)

        # ── video branch ─────────────────────────────────────────
        v_ = self.patch_embed_v(v)
        v_ = v_ + self.pos_embed_v + self.modality_v

        for blk in self.blocks_v:
            v_, _ = blk(v_, ft=False)

        # modality-specific upper blocks → video-only representation
        v_ms = v_
        for blk in self.blocks_u:
            v_ms, _ = blk(v_ms, modality='v', ft=False)
        v_ms = self.norm_v(v_ms)
        f_v  = v_ms.mean(dim=1)          # (B, D)
        z_v  = self.mlp_head(f_v)        # (B, C)

        # ── fusion branch ────────────────────────────────────────
        x    = torch.cat((a_, v_), dim=1)
        attn = None
        for blk in self.blocks_u:
            x, attn = blk(x, ft=False)
        x = self.norm(x)
        f_fused = x.mean(dim=1)          # (B, D)
        z_fused = self.mlp_head(f_fused) # (B, C)

        return dict(
            z_a=z_a,
            z_v=z_v,
            z_fused=z_fused,
            f_a=f_a,
            f_v=f_v,
            attn=attn,
        )