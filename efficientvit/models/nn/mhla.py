"""Token-grouped Multi-Head Linear Attention (MHLA).

This is an adaptation of DAGroup-PKU/MHLA's non-causal vision operator to
EfficientViT's pre-projected Q/K/V tensors.  EfficientViT already supplies
feature/channel heads; MHLA adds the paper's second kind of head by grouping
the *spatial token* dimension and mixing one KV summary per token group.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiHeadLinearAttention"]


def _factor_grid(num_groups: int) -> tuple[int, int]:
    """Return the most nearly square exact 2-D factorization of num_groups."""
    grid_h = int(math.sqrt(num_groups))
    while num_groups % grid_h != 0:
        grid_h -= 1
    return grid_h, num_groups // grid_h


def _cosine_distance_mixing(num_groups: int) -> torch.Tensor:
    """Match the locality-biased cosine initialization in the MHLA release."""
    grid_h, grid_w = _factor_grid(num_groups)
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32) + 0.5,
        torch.arange(grid_w, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    centers = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
    distance = torch.cdist(centers, centers)
    max_distance = distance.max()
    if max_distance == 0:
        return torch.ones((1, 1), dtype=torch.float32)
    weights = torch.cos(distance / max_distance * (math.pi / 4))
    return weights / weights.sum(dim=0, keepdim=True)


class MultiHeadLinearAttention(nn.Module):
    """MHLA mixing over non-overlapping 2-D spatial token groups.

    Args:
        num_groups: Number of token-dimension heads (``M`` in the paper).
        eps: Denominator stabilizer used by normalized linear attention.

    Input Q/K/V tensors use ``[batch, feature_heads, head_dim, height, width]``.
    The output has the same shape as V.  The learnable ``M x M`` matrix is
    shared across EfficientViT's feature heads, as in the authors' vision code.

    We intentionally do not port the reference implementation's optional LePE
    depthwise convolution: E2 must contain no convolutional compensation after
    the QKV projection, so the experiment isolates token-head mixing.
    """

    def __init__(self, num_groups: int = 4, eps: float = 1.0e-15) -> None:
        super().__init__()
        if num_groups < 1:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        self.num_groups = num_groups
        self.eps = eps
        self.mixing_weight = nn.Parameter(_cosine_distance_mixing(num_groups))

    @staticmethod
    def _spatial_group_ids(height: int, width: int, num_groups: int, device: torch.device) -> torch.Tensor:
        """Assign flattened row-major tokens to rectangular 2-D groups."""
        grid_h, grid_w = _factor_grid(num_groups)
        # Put the longer group grid axis along the longer feature-map axis.
        if height > width:
            grid_h, grid_w = grid_w, grid_h
        rows = torch.div(torch.arange(height, device=device) * grid_h, height, rounding_mode="floor")
        cols = torch.div(torch.arange(width, device=device) * grid_w, width, rounding_mode="floor")
        return (rows[:, None] * grid_w + cols[None, :]).flatten()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 5:
            raise ValueError(
                "q, k, and v must have matching [B, heads, dim, H, W] shapes; "
                f"got {tuple(q.shape)}, {tuple(k.shape)}, and {tuple(v.shape)}"
            )

        batch, feature_heads, head_dim, height, width = q.shape
        num_tokens = height * width
        active_groups = min(self.num_groups, num_tokens)
        group_ids = self._spatial_group_ids(height, width, active_groups, q.device)
        grid_h, grid_w = _factor_grid(active_groups)
        if height > width:
            grid_h, grid_w = grid_w, grid_h

        # [B, feature_heads, N, D].  Flattening is row-major, preserving the
        # feature map's H/W order used to construct group_ids above.
        q = q.flatten(3).transpose(-1, -2)
        k = k.flatten(3).transpose(-1, -2)
        v = v.flatten(3).transpose(-1, -2)
        equal_sized_groups = height % grid_h == 0 and width % grid_w == 0
        if equal_sized_groups:
            group_height = height // grid_h
            group_width = width // grid_w
            tokens_per_group = num_tokens // active_groups

            def group_tokens(tokens: torch.Tensor) -> torch.Tensor:
                return (
                    tokens.reshape(batch, feature_heads, height, width, head_dim)
                    .reshape(batch, feature_heads, grid_h, group_height, grid_w, group_width, head_dim)
                    .permute(0, 1, 2, 4, 3, 5, 6)
                    .reshape(batch, feature_heads, active_groups, tokens_per_group, head_dim)
                )

            q_grouped = group_tokens(q)
            k_grouped = group_tokens(k)
            v_grouped = group_tokens(v)
            summaries = torch.einsum("bhmnd,bhmne->bhmde", k_grouped, v_grouped)
            normalizers = k_grouped.sum(dim=3)
        else:
            # Unequal edge groups are uncommon for model feature maps, but this
            # path keeps MHLA valid for arbitrary spatial resolutions.
            assignment = F.one_hot(group_ids, num_classes=active_groups).to(dtype=k.dtype)
            summaries = torch.einsum("bhnd,bhne,nm->bhmde", k, v, assignment)
            normalizers = torch.einsum("bhnd,nm->bhmd", k, assignment)

        mixing = self.mixing_weight[:active_groups, :active_groups].to(dtype=summaries.dtype)
        mixed_summaries = torch.einsum("im,bhmde->bhide", mixing, summaries)
        mixed_normalizers = torch.einsum("im,bhmd->bhid", mixing, normalizers)

        if equal_sized_groups:
            numerator = torch.einsum("bhmnd,bhmde->bhmne", q_grouped, mixed_summaries)
            denominator = torch.einsum("bhmnd,bhmd->bhmn", q_grouped, mixed_normalizers).unsqueeze(-1)
            out = (
                (numerator / (denominator + self.eps))
                .reshape(batch, feature_heads, grid_h, grid_w, group_height, group_width, head_dim)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .reshape(batch, feature_heads, num_tokens, head_dim)
            )
        else:
            # Group sizes can differ, so compute each query group separately.
            out = torch.zeros_like(v)
            for group_idx in range(active_groups):
                token_idx = torch.nonzero(group_ids == group_idx, as_tuple=False).flatten()
                q_group = q.index_select(2, token_idx)
                numerator = torch.einsum("bhnd,bhde->bhne", q_group, mixed_summaries[:, :, group_idx])
                denominator = torch.einsum(
                    "bhnd,bhd->bhn", q_group, mixed_normalizers[:, :, group_idx]
                ).unsqueeze(-1)
                group_out = numerator / (denominator + self.eps)
                out.index_copy_(2, token_idx, group_out)

        return out.transpose(-1, -2).reshape(batch, feature_heads, head_dim, height, width)
