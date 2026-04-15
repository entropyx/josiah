"""Channel-token transformer for proper PFN MMM decomposition.

Architecture:
  1. Project channel and global features to d_model.
  2. Build a sequence of tokens: [global_week_1, ..., global_week_T, ch_1_week_1, ..., ch_C_week_T]
  3. Channel tokens have NO positional encoding across the channel dimension
     (only the time dimension), ensuring permutation invariance over channels.
  4. Temporal encoding added to both streams for time ordering.
  5. Output per channel per week (shares) + per week (baseline, non_media).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ProperPFNModel(nn.Module):
    """Transformer with channel-token permutation invariance.

    Sequence layout:
      - Global tokens (one per week): encode y, context, time — index [0 .. T-1]
      - Channel tokens (one per (channel, week)): index [T .. T + T*max_channels - 1]
    Total sequence length: T + T * max_channels.

    Channel tokens have NO channel-positional encoding (only time encoding),
    ensuring permutation invariance over channels.
    """

    def __init__(
        self,
        channel_feat_dim: int,
        global_feat_dim: int,
        max_channels: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 300,
    ) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.d_model = d_model

        self.channel_proj = nn.Linear(channel_feat_dim, d_model)
        self.global_proj = nn.Linear(global_feat_dim, d_model)

        # Temporal encoding shared between global and channel streams.
        self.temporal_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Type embeddings: distinguish global tokens from channel tokens.
        self.global_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.channel_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.channel_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.baseline_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.non_media_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        channel_tokens: Tensor,
        global_tokens: Tensor,
        channel_pad_mask: Tensor,
        time_pad_mask: Tensor,
    ) -> dict[str, Tensor]:
        B, T, C, _ = channel_tokens.shape

        ch_emb = self.channel_proj(channel_tokens)  # (B, T, C, d)
        gl_emb = self.global_proj(global_tokens)    # (B, T, d)

        temp = self.temporal_encoding[:, :T, :]  # (1, T, d)
        ch_emb = ch_emb + temp.unsqueeze(2)      # (B, T, C, d)
        gl_emb = gl_emb + temp                   # (B, T, d)

        ch_emb = ch_emb + self.channel_type_emb
        gl_emb = gl_emb + self.global_type_emb

        ch_seq = ch_emb.reshape(B, T * C, self.d_model)
        seq = torch.cat([gl_emb, ch_seq], dim=1)  # (B, T + T*C, d)

        # Padding mask: True=ignore
        global_mask_flat = time_pad_mask  # (B, T)
        channel_mask_2d = time_pad_mask.unsqueeze(-1) | channel_pad_mask.unsqueeze(1)  # (B, T, C)
        channel_mask_flat = channel_mask_2d.reshape(B, T * C)
        attn_mask = torch.cat([global_mask_flat, channel_mask_flat], dim=1)

        out = self.transformer(seq, src_key_padding_mask=attn_mask)

        out_global = out[:, :T, :]
        out_channel = out[:, T:, :].reshape(B, T, C, self.d_model)

        channel_shares = self.channel_head(out_channel).squeeze(-1)  # (B, T, C)
        baseline_share = self.baseline_head(out_global).squeeze(-1)  # (B, T)
        non_media_share = self.non_media_head(out_global).squeeze(-1)  # (B, T)

        active_mask = (~channel_pad_mask).float()
        channel_shares = channel_shares * active_mask.unsqueeze(1)

        return {
            "channel_shares": channel_shares,
            "baseline_share": baseline_share,
            "non_media_share": non_media_share,
        }
