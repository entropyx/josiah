"""Embedding networks for the Demantiq neural inference engine.

Architecture:
1. Per-channel temporal encoder (shared dilated 1D CNN)
2. Channel-type embeddings (learnable, one per channel name)
3. Cross-channel Set Transformer attention
4. Context encoder for outcome y + auxiliary variables
5. Summary MLP producing fixed-size embedding for the density estimator
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.neural.utils import NUM_CHANNEL_TYPES, MAX_CHANNELS
from demantiq.orchestration.training_format import MAX_CONTEXT_COLS


class DilatedConv1DEncoder(nn.Module):
    """Dilated 1D CNN for encoding time series into a fixed-size vector.

    Uses exponentially increasing dilation to capture multi-scale temporal
    patterns across 26-260 timesteps.

    Args:
        in_channels: Number of input channels (1 for univariate).
        hidden_dim: Hidden dimension of conv layers.
        output_dim: Output embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 64,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3, dilation=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=4, dilation=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hidden_dim * 2, output_dim, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(0.01),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a time series.

        Args:
            x: (batch, T) or (batch, T, C) time series.

        Returns:
            (batch, output_dim) embedding.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, T)
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)  # (batch, C, T)
        h = self.conv_layers(x)
        h = self.pool(h).squeeze(-1)  # (batch, output_dim)
        return h

    def forward_temporal(self, x: Tensor) -> Tensor:
        """Encode a time series preserving temporal dimension (no pooling).

        Args:
            x: (batch, T) or (batch, T, C) time series.

        Returns:
            (batch, output_dim, T) temporal feature map.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, T)
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)  # (batch, C, T)
        return self.conv_layers(x)  # (batch, output_dim, T)


class MultiheadSetAttention(nn.Module):
    """Multi-head self-attention over a set of elements.

    Implements the core Set Transformer attention mechanism for
    learning cross-element interactions in a permutation-equivariant way.

    Args:
        d_model: Dimension of each element.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Apply self-attention over set elements.

        Args:
            x: (batch, N, d_model) set elements.
            key_padding_mask: (batch, N) True for padded positions.

        Returns:
            (batch, N, d_model) attended elements.
        """
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class PoolingByMultiheadAttention(nn.Module):
    """PMA: Pooling by Multihead Attention from Set Transformer.

    Learns k seed vectors that attend to the set, producing k output vectors.
    For aggregation, k=1 gives a single set-level summary.

    Args:
        d_model: Dimension of set elements.
        n_heads: Number of attention heads.
        k: Number of seed (output) vectors.
    """

    def __init__(self, d_model: int, n_heads: int = 4, k: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Pool set into k summary vectors.

        Args:
            x: (batch, N, d_model) set elements.
            key_padding_mask: (batch, N) True for padded positions.

        Returns:
            (batch, k, d_model) pooled summaries.
        """
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        h, _ = self.attn(seeds, x, x, key_padding_mask=key_padding_mask)
        return self.norm(seeds + h)


class SetTransformerEncoder(nn.Module):
    """Set Transformer for encoding a variable-size set of channel embeddings.

    Applies self-attention to learn cross-channel interactions, then pools
    into a fixed-size summary via PMA.

    Args:
        d_model: Dimension of each channel embedding.
        n_heads: Number of attention heads.
        n_layers: Number of self-attention layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 80,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiheadSetAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.pma = PoolingByMultiheadAttention(d_model, n_heads, k=1)

    def forward(
        self, x: Tensor, key_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Encode a set of channel embeddings.

        Args:
            x: (batch, N, d_model) channel embeddings.
            key_padding_mask: (batch, N) True for padded/inactive channels.

        Returns:
            Tuple of:
                set_summary: (batch, d_model) aggregated summary.
                per_element: (batch, N, d_model) per-channel attended embeddings.
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        per_element = x
        summary = self.pma(x, key_padding_mask=key_padding_mask).squeeze(1)  # (batch, d_model)
        return summary, per_element


class EmbeddingNetwork(nn.Module):
    """Full embedding network for Demantiq neural inference.

    Produces both a global summary embedding (for global parameter inference)
    and per-channel embeddings (for compositional per-channel inference).

    Architecture:
        1. Shared DilatedConv1D encodes each channel's spend → (64,)
        2. Learnable channel-type embeddings → (16,)
        3. Per-channel: concat(temporal(64), type(16)) → (80,)
        4. Set Transformer over channels → set_summary(80) + per_channel(80)
        5. DilatedConv1D encodes y → (64,)
        6. DilatedConv1D encodes business context (price, distribution, etc.) → (64,)
        7. Global summary MLP: concat(set_summary, y_emb, ctx_emb, n_channels) → (256,)
        8. Per-channel summary: concat(per_channel_emb, global_context) → (336,) each

    Args:
        temporal_dim: Output dim of the temporal encoder.
        type_embed_dim: Dimension of channel-type embeddings.
        n_attn_heads: Number of attention heads in Set Transformer.
        n_attn_layers: Number of Set Transformer layers.
        global_summary_dim: Output dimension of the global summary.
        dropout: Dropout rate.
        n_context_cols: Number of business context columns.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        type_embed_dim: int = 16,
        n_attn_heads: int = 4,
        n_attn_layers: int = 2,
        global_summary_dim: int = 256,
        dropout: float = 0.1,
        n_context_cols: int = MAX_CONTEXT_COLS,
        n_fixed_channels: int | None = None,
    ):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.type_embed_dim = type_embed_dim
        self.channel_dim = temporal_dim + type_embed_dim  # 80
        self.global_summary_dim = global_summary_dim
        self.n_context_cols = n_context_cols

        # Shared temporal encoder for channel spend
        self.channel_temporal_encoder = DilatedConv1DEncoder(
            in_channels=1, hidden_dim=32, output_dim=temporal_dim
        )

        # Temporal encoder for y (outcome)
        self.y_encoder = DilatedConv1DEncoder(
            in_channels=1, hidden_dim=32, output_dim=temporal_dim
        )

        # Context encoder for business variables (price, distribution, competition, macro)
        self.context_encoder = DilatedConv1DEncoder(
            in_channels=n_context_cols, hidden_dim=32, output_dim=temporal_dim
        )

        # Learnable channel-type embeddings
        self.channel_type_embedding = nn.Embedding(NUM_CHANNEL_TYPES, type_embed_dim)

        # Set Transformer for cross-channel attention
        self.set_transformer = SetTransformerEncoder(
            d_model=self.channel_dim,
            n_heads=n_attn_heads,
            n_layers=n_attn_layers,
            dropout=dropout,
        )

        # Global summary MLP
        # Input: set_summary(channel_dim) + y_emb(temporal_dim) + ctx_emb(temporal_dim) + n_channels(1)
        global_input_dim = self.channel_dim + temporal_dim + temporal_dim + 1
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, global_summary_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_summary_dim, global_summary_dim),
            nn.ReLU(),
        )

        # Per-channel projection (for compositional inference)
        # Input: per_channel_emb(channel_dim) + global_summary(global_summary_dim)
        self.per_channel_dim = self.channel_dim + global_summary_dim

        # Temporal projection for per-period embeddings (decomposition mode)
        # When n_fixed_channels is set: concatenate per-channel features (no averaging)
        # so the decoder sees each channel's spend pattern individually.
        # When None: fall back to masked mean (loses channel identity).
        self.n_fixed_channels = n_fixed_channels
        if n_fixed_channels is not None:
            # Concatenate all channels: n_ch * channel_dim + y + ctx + n_ch_scalar
            temporal_input_dim = n_fixed_channels * self.channel_dim + temporal_dim + temporal_dim + 1
        else:
            temporal_input_dim = self.channel_dim + temporal_dim + temporal_dim + 1
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_input_dim, global_summary_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_summary_dim, global_summary_dim),
            nn.ReLU(),
        )
        self.temporal_embedding_dim = global_summary_dim

    def forward(
        self,
        y: Tensor,
        spend: Tensor,
        n_channels: Tensor,
        channel_type_ids: Tensor,
        context: Tensor | None = None,
        time_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute global and per-channel embeddings.

        Args:
            y: (batch, T) outcome time series.
            spend: (batch, T, max_channels) spend matrix.
            n_channels: (batch,) number of active channels.
            channel_type_ids: (batch, max_channels) channel type indices.
            context: (batch, T, n_context_cols) business context matrix (optional).
                     Contains price, promo, distribution, competition, macro variables.
            time_mask: (batch, T) True for valid timesteps (optional).

        Returns:
            Tuple of:
                global_summary: (batch, global_summary_dim)
                per_channel_summary: (batch, max_channels, per_channel_dim)
                channel_mask: (batch, max_channels) True for active channels
        """
        batch_size = y.shape[0]
        max_ch = spend.shape[2]

        # Build channel padding mask (True = padded/inactive)
        ch_indices = torch.arange(max_ch, device=n_channels.device).unsqueeze(0)
        channel_mask = ch_indices < n_channels.unsqueeze(1)  # True = active
        padding_mask = ~channel_mask  # True = padded (for attention)

        # 1. Encode each channel's spend through shared temporal encoder
        # Reshape: (batch * max_ch, T)
        spend_flat = spend.permute(0, 2, 1).reshape(batch_size * max_ch, -1)
        channel_temporal = self.channel_temporal_encoder(spend_flat)  # (batch*max_ch, temporal_dim)
        channel_temporal = channel_temporal.reshape(batch_size, max_ch, self.temporal_dim)

        # 2. Get channel-type embeddings
        type_emb = self.channel_type_embedding(channel_type_ids)  # (batch, max_ch, type_embed_dim)

        # 3. Concatenate temporal + type → per-channel embedding
        channel_emb = torch.cat([channel_temporal, type_emb], dim=-1)  # (batch, max_ch, channel_dim)

        # Zero out inactive channels
        channel_emb = channel_emb * channel_mask.unsqueeze(-1).float()

        # 4. Set Transformer: cross-channel attention + pooling
        set_summary, per_element = self.set_transformer(
            channel_emb, key_padding_mask=padding_mask
        )

        # 5. Encode y
        y_emb = self.y_encoder(y)  # (batch, temporal_dim)

        # 6. Encode business context (price, distribution, competition, macro)
        if context is not None:
            ctx_emb = self.context_encoder(context)  # (batch, temporal_dim)
        else:
            ctx_emb = torch.zeros(batch_size, self.temporal_dim, device=y.device)

        # 7. Global summary
        n_ch_normalized = n_channels.float().unsqueeze(-1) / MAX_CHANNELS
        global_input = torch.cat([set_summary, y_emb, ctx_emb, n_ch_normalized], dim=-1)
        global_summary = self.global_mlp(global_input)  # (batch, global_summary_dim)

        # 8. Per-channel summary: concat per-element attended emb with global context
        global_expanded = global_summary.unsqueeze(1).expand(-1, max_ch, -1)
        per_channel_summary = torch.cat([per_element, global_expanded], dim=-1)
        per_channel_summary = per_channel_summary * channel_mask.unsqueeze(-1).float()

        return global_summary, per_channel_summary, channel_mask

    def forward_temporal(
        self,
        y: Tensor,
        spend: Tensor,
        n_channels: Tensor,
        channel_type_ids: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """Compute per-period embeddings preserving temporal dimension.

        Instead of pooling across time, this produces one embedding per timestep
        for use by the temporal decoder.

        Args:
            y: (batch, T) outcome time series.
            spend: (batch, T, max_channels) spend matrix.
            n_channels: (batch,) number of active channels.
            channel_type_ids: (batch, max_channels) channel type indices.
            context: (batch, T, n_context_cols) business context matrix (optional).

        Returns:
            (batch, T, temporal_embedding_dim) per-period embeddings.
        """
        batch_size = y.shape[0]
        T = y.shape[1]
        max_ch = spend.shape[2]

        # Slice channel_type_ids to match spend's channel dimension
        channel_type_ids = channel_type_ids[:, :max_ch]

        # Build channel mask
        ch_indices = torch.arange(max_ch, device=n_channels.device).unsqueeze(0)
        channel_mask = ch_indices < n_channels.unsqueeze(1)  # (batch, max_ch) True = active

        # 1. Normalize spend globally (preserve relative magnitudes across channels)
        # Without this, raw spend (10K-80K) causes dying ReLU in the CNN.
        spend_scale = spend.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1.0)  # (B, 1, 1)
        spend_normed = spend / spend_scale  # now in [-1, 1], relative magnitudes preserved

        # Per-channel temporal features (no pooling)
        spend_flat = spend_normed.permute(0, 2, 1).reshape(batch_size * max_ch, T)
        ch_temporal = self.channel_temporal_encoder.forward_temporal(spend_flat)  # (B*C, D, T')
        T_out = ch_temporal.shape[-1]  # may differ from T due to conv padding
        ch_temporal = ch_temporal.reshape(batch_size, max_ch, self.temporal_dim, T_out)

        # 2. Add channel-type embeddings (broadcast across time)
        type_emb = self.channel_type_embedding(channel_type_ids)  # (B, max_ch, type_dim)
        type_emb_t = type_emb.unsqueeze(-1).expand(-1, -1, -1, T_out)  # (B, max_ch, type_dim, T_out)
        ch_combined = torch.cat([ch_temporal, type_emb_t], dim=2)  # (B, max_ch, channel_dim, T_out)

        # 3. Aggregate channels per timestep
        if self.n_fixed_channels is not None:
            # Concatenate all per-channel features so decoder sees each channel individually
            # ch_combined: (B, max_ch, channel_dim, T_out) → flatten channels
            ch_combined = ch_combined * channel_mask.float().unsqueeze(-1).unsqueeze(-1)
            # Take first n_fixed_channels and flatten: (B, n_fixed * channel_dim, T_out)
            ch_agg = ch_combined[:, :self.n_fixed_channels].reshape(
                batch_size, self.n_fixed_channels * self.channel_dim, T_out
            )
        else:
            # Fall back to masked mean (loses per-channel identity)
            mask = channel_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            n_active = mask.sum(dim=1).clamp(min=1)  # (B, 1, 1)
            ch_agg = (ch_combined * mask).sum(dim=1) / n_active  # (B, channel_dim, T_out)

        # 4. Y temporal features (no pooling) — normalize to prevent dying ReLU
        y_scale = y.abs().amax(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        y_normed = y / y_scale
        y_temporal = self.y_encoder.forward_temporal(y_normed)  # (B, D, T_out)

        # 5. Context temporal features (no pooling) — normalize per column
        if context is not None:
            ctx_scale = context.abs().amax(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, C_ctx)
            ctx_normed = context / ctx_scale
            ctx_temporal = self.context_encoder.forward_temporal(ctx_normed)  # (B, D, T_out)
        else:
            ctx_temporal = torch.zeros(
                batch_size, self.temporal_dim, T_out, device=y.device
            )

        # 6. n_channels broadcast to all timesteps
        n_ch_norm = (n_channels.float() / MAX_CHANNELS).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        n_ch_t = n_ch_norm.expand(-1, -1, T_out)  # (B, 1, T_out)

        # 7. Concatenate per-timestep features
        combined = torch.cat([ch_agg, y_temporal, ctx_temporal, n_ch_t], dim=1)  # (B, D_total, T_out)
        combined = combined.permute(0, 2, 1)  # (B, T_out, D_total)

        # 8. Project to embedding dim
        temporal_emb = self.temporal_proj(combined)  # (B, T_out, temporal_embedding_dim)

        return temporal_emb
