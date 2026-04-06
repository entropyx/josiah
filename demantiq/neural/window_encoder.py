"""Window-based per-channel contribution model for Marketing Mix Modeling.

Architecture:
    Per-Channel CNN  →  Cross-Channel Attention  →  Per-Channel Prediction

Takes a W-week window of marketing data and predicts per-channel contributions
plus baseline and non-media effects such that they sum to observed y.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.neural.encoders import MultiheadSetAttention


class ChannelFeatureCNN(nn.Module):
    """Shared CNN for encoding a single channel's spend over a time window.

    Normalizes spend per sample, concatenates a log-magnitude feature, then
    applies two Conv1d layers to produce temporal features.

    Args:
        feature_dim: Output feature dimension.
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(32)
        self.conv2 = nn.Conv1d(32, feature_dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.act = nn.GELU()

    def forward(self, spend: Tensor, global_scale: Tensor | None = None) -> Tensor:
        """Encode one channel's spend time series.

        Args:
            spend: (B, W) raw spend values for one channel.
            global_scale: (B, 1) shared normalization scale across all channels.
                If None, falls back to per-channel normalization.

        Returns:
            (B, W, feature_dim) temporal features.
        """
        B, W = spend.shape

        if global_scale is not None:
            scale = global_scale
        else:
            scale = spend.abs().amax(dim=1, keepdim=True).clamp(min=1.0)

        spend_normed = spend / scale  # (B, W) — preserves relative magnitude across channels

        log_mag = torch.log1p(scale / 1000.0).expand(B, W)  # (B, W)

        x = torch.stack([spend_normed, log_mag], dim=1)  # (B, 2, W)

        h = self.act(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))  # (B, 32, W)
        h = self.act(self.norm2(self.conv2(h).transpose(1, 2)).transpose(1, 2))  # (B, feature_dim, W)

        return h.transpose(1, 2)  # (B, W, feature_dim)


class YEncoder(nn.Module):
    """CNN for encoding the observed demand time series.

    Args:
        feature_dim: Output feature dimension.
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(32)
        self.conv2 = nn.Conv1d(32, feature_dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.act = nn.GELU()

    def forward(self, y: Tensor) -> Tensor:
        """Encode the demand time series.

        Args:
            y: (B, W) observed demand.

        Returns:
            (B, W, feature_dim) temporal features.
        """
        max_abs = y.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
        y_normed = (y / max_abs).unsqueeze(1)  # (B, 1, W)

        h = self.act(self.norm1(self.conv1(y_normed).transpose(1, 2)).transpose(1, 2))
        h = self.act(self.norm2(self.conv2(h).transpose(1, 2)).transpose(1, 2))

        return h.transpose(1, 2)  # (B, W, feature_dim)


class ContextEncoder(nn.Module):
    """Pointwise encoder for context variables and their presence flags.

    Args:
        n_context_dims: Number of context variable dimensions.
        feature_dim: Output feature dimension.
    """

    def __init__(self, n_context_dims: int = 10, feature_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(n_context_dims * 2, feature_dim)
        self.act = nn.GELU()

    def forward(self, context: Tensor, presence_mask: Tensor) -> Tensor:
        """Encode context variables with their presence flags.

        Args:
            context: (B, W, n_context_dims) context variables.
            presence_mask: (B, n_context_dims) 1 for present, 0 for missing.

        Returns:
            (B, W, feature_dim) context features.
        """
        B, W, D = context.shape
        mask = presence_mask.unsqueeze(1).expand(B, W, D)  # (B, W, D)
        x = torch.cat([context, mask], dim=-1)  # (B, W, 2*D)
        return self.act(self.proj(x))  # (B, W, feature_dim)


class CrossChannelAttention(nn.Module):
    """Per-timestep attention across channels, y, context, and a baseline token.

    At each timestep, assembles a set of tokens (one per channel plus y,
    context, and a learnable baseline) and applies multi-layer self-attention.

    Args:
        feature_dim: Token dimension.
        n_heads: Number of attention heads.
        n_layers: Number of MultiheadSetAttention layers.
        max_channels: Maximum number of channel slots.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_channels: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_channels = max_channels
        self.baseline_token = nn.Parameter(torch.randn(feature_dim) * 0.02)
        self.layers = nn.ModuleList([
            MultiheadSetAttention(feature_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        channel_features: Tensor,
        y_features: Tensor,
        context_features: Tensor,
        channel_pad_mask: Tensor,
    ) -> Tensor:
        """Apply cross-token attention at each timestep.

        Args:
            channel_features: (B, W, C, feature_dim) per-channel features.
            y_features: (B, W, feature_dim) y token features.
            context_features: (B, W, feature_dim) context token features.
            channel_pad_mask: (B, C) True for padded channel positions.

        Returns:
            (B, W, C+3, feature_dim) updated token features.
        """
        B, W, C, D = channel_features.shape

        baseline = self.baseline_token.view(1, 1, 1, D).expand(B, W, 1, D)
        y_tok = y_features.unsqueeze(2)          # (B, W, 1, D)
        ctx_tok = context_features.unsqueeze(2)  # (B, W, 1, D)

        tokens = torch.cat([channel_features, y_tok, ctx_tok, baseline], dim=2)  # (B, W, C+3, D)

        # Build key_padding_mask for (B*W, C+3)
        # Padded channels → True; y, context, baseline tokens → False
        pad_bw = channel_pad_mask.unsqueeze(1).expand(B, W, C).reshape(B * W, C)
        extra_false = torch.zeros(B * W, 3, dtype=torch.bool, device=channel_features.device)
        key_padding_mask = torch.cat([pad_bw, extra_false], dim=1)  # (B*W, C+3)

        tokens_flat = tokens.reshape(B * W, C + 3, D)

        for layer in self.layers:
            tokens_flat = layer(tokens_flat, key_padding_mask=key_padding_mask)

        return tokens_flat.reshape(B, W, C + 3, D)


def _make_prediction_mlp(feature_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.GELU(),
        nn.Linear(feature_dim, 1),
    )


class WindowContributionModel(nn.Module):
    """Full window-based per-channel contribution model.

    Takes a W-week window of spend, y, and context, and decomposes y into
    per-channel contributions, a baseline, and non-media effects.

    Args:
        feature_dim: Internal token/feature dimension.
        n_attention_heads: Number of attention heads.
        n_attention_layers: Number of cross-channel attention layers.
        max_channels: Maximum number of channel slots (padded to this size).
        n_context_dims: Number of context variable dimensions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        n_attention_heads: int = 4,
        n_attention_layers: int = 2,
        max_channels: int = 8,
        n_context_dims: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_channels = max_channels
        self.feature_dim = feature_dim

        self.channel_cnn = ChannelFeatureCNN(feature_dim)
        self.y_encoder = YEncoder(feature_dim)
        self.context_encoder = ContextEncoder(n_context_dims, feature_dim)

        # Project temporal summary stats (3 per channel) into feature space
        self.summary_proj = nn.Linear(3, feature_dim)

        self.cross_attention = CrossChannelAttention(
            feature_dim, n_attention_heads, n_attention_layers, max_channels, dropout
        )

        self.channel_prediction_mlp = _make_prediction_mlp(feature_dim)
        self.baseline_mlp = _make_prediction_mlp(feature_dim)
        self.non_media_mlp = _make_prediction_mlp(feature_dim)

    def forward(
        self,
        spend: Tensor,
        y: Tensor,
        context: Tensor,
        context_mask: Tensor,
        channel_pad_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Decompose observed demand into channel contributions and baseline.

        Args:
            spend: (B, W, max_channels) per-channel spend, zeros for padded.
            y: (B, W) observed demand.
            context: (B, W, n_context_dims) context variables.
            context_mask: (B, n_context_dims) presence flags (1=present, 0=missing).
            channel_pad_mask: (B, max_channels) True for padded channel slots.

        Returns:
            Dict with keys:
                'channel_contributions': (B, W, max_channels)
                'baseline': (B, W)
                'non_media': (B, W)
                'y_hat': (B, W)
        """
        B, W, C = spend.shape

        # Compute y scale for output denormalization — MLPs output in [-1, 1] range,
        # multiply by y_scale to reach the right magnitude
        y_scale = y.abs().mean(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        active_mask = (~channel_pad_mask).float()  # (B, C)

        # 0. Compute per-channel temporal summary stats: corr(spend_i, y), mean, std
        # These capture the temporal relationship OLS uses for identification
        y_centered = y - y.mean(dim=1, keepdim=True)  # (B, W)
        y_std = y.std(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)

        summary_stats = []
        for c in range(C):
            sp_c = spend[:, :, c]  # (B, W)
            sp_mean = sp_c.mean(dim=1)  # (B,)
            sp_std = sp_c.std(dim=1).clamp(min=1e-6)  # (B,)
            sp_centered = sp_c - sp_c.mean(dim=1, keepdim=True)
            # Pearson correlation between spend_c and y across the window
            corr = (sp_centered * y_centered).mean(dim=1) / (sp_std * y_std.squeeze(1))  # (B,)
            # Normalize mean/std to [0, 1] range
            sp_mean_norm = sp_mean / sp_mean.abs().clamp(min=1.0)
            sp_std_norm = sp_std / sp_std.abs().clamp(min=1.0)
            summary_stats.append(torch.stack([corr, sp_mean_norm, sp_std_norm], dim=-1))  # (B, 3)

        summary_stats = torch.stack(summary_stats, dim=1)  # (B, C, 3)
        summary_stats = summary_stats * active_mask.unsqueeze(-1)  # zero padded
        summary_features = self.summary_proj(summary_stats)  # (B, C, feature_dim)

        # 1. Encode each channel through shared CNN
        # Global scale: max abs spend across ALL channels — preserves relative magnitudes
        global_spend_scale = spend.abs().amax(dim=(1, 2), keepdim=False).clamp(min=1.0)  # (B,)
        global_spend_scale = global_spend_scale.unsqueeze(1)  # (B, 1)

        channel_features_list = []
        for c in range(C):
            ch_feat = self.channel_cnn(spend[:, :, c], global_scale=global_spend_scale)  # (B, W, feature_dim)
            channel_features_list.append(ch_feat)

        channel_features = torch.stack(channel_features_list, dim=2)  # (B, W, C, feature_dim)

        # Add temporal summary features to each channel (broadcast across W)
        channel_features = channel_features + summary_features.unsqueeze(1)

        # Zero out padded channel features
        channel_features = channel_features * active_mask.unsqueeze(1).unsqueeze(-1)

        # 2. Encode y
        y_features = self.y_encoder(y)  # (B, W, feature_dim)

        # 3. Encode context
        context_features = self.context_encoder(context, context_mask)  # (B, W, feature_dim)

        # 4. Cross-channel attention
        tokens = self.cross_attention(
            channel_features, y_features, context_features, channel_pad_mask
        )  # (B, W, C+3, feature_dim)

        # 5. Extract and decode token outputs
        channel_tokens = tokens[:, :, :C, :]  # (B, W, C, feature_dim)
        context_token = tokens[:, :, C + 1, :]  # (B, W, feature_dim)
        baseline_token = tokens[:, :, C + 2, :]  # (B, W, feature_dim)

        # Per-channel predictions via shared MLP, scaled by y magnitude
        ch_preds = self.channel_prediction_mlp(channel_tokens).squeeze(-1)  # (B, W, C)
        ch_preds = ch_preds * y_scale.unsqueeze(-1)  # scale to demand units
        ch_preds = ch_preds * active_mask.unsqueeze(1)  # zero out padded

        baseline = self.baseline_mlp(baseline_token).squeeze(-1) * y_scale  # (B, W)
        non_media = self.non_media_mlp(context_token).squeeze(-1) * y_scale  # (B, W)

        # 6. Reconstruct y
        y_hat = baseline + ch_preds.sum(dim=-1) + non_media

        return {
            "channel_contributions": ch_preds,
            "baseline": baseline,
            "non_media": non_media,
            "y_hat": y_hat,
        }
