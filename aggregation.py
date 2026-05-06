"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import numpy as np

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    """

    # from tensors to numoy arrays
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.cpu().numpy()

    # if batch, extract first 
    if hidden_states.ndim == 4:
        hidden_states = hidden_states[0]  # (n_layers, seq_len, hidden_dim)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[0]  # (seq_len,)

    n_layers, seq_len, hidden_dim = hidden_states.shape
    assert attention_mask.shape == (seq_len,), f"attention_mask shape {attention_mask.shape} != (n_layers,)"

    # 1. Mean for real tokens
    mask = attention_mask.astype(np.float32)  # (seq_len, )
    mask_sum = np.sum(mask, axis=0)
    if mask_sum == 0:
        mask_sum = 1.0

    # Mean over tokens for each layer
    layer_features = []
    for l in range(n_layers):
        h = hidden_states[l]  # (seq_len, hidden_dim)
        weighted_h = h * mask[:, None]  # (seq_len, hidden_dim)
        mean_h = np.sum(weighted_h, axis=0) / mask_sum  # (hidden_dim,)
        layer_features.append(mean_h)

    layer_features = np.array(layer_features)  # (n_layers, hidden_dim)


    scores = np.linalg.norm(layer_features, axis=1)
    k = min(3, n_layers)
    topk_indices = np.argsort(scores)[-k:]
    fused_features = np.mean(layer_features[topk_indices], axis=0)

    return fused_features
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).

    if hidden_states.ndim == 4:
        hidden_states = hidden_states[0]  # (n_layers, seq_len, hidden_dim)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[0]  # -> (seq_len,)

    n_layers, seq_len, hidden_dim = hidden_states.shape
    assert attention_mask.shape == (seq_len,), f"attention_mask shape {attention_mask.shape} != (seq_len,)"

    # Masked mean по токенам для каждого слоя
    mask = attention_mask.to(hidden_states.device).float()  # (seq_len,)
    mask_sum = mask.sum()
    if mask_sum == 0.0:
        mask_sum = 1.0

    layer_norms = []
    for l in range(n_layers):
        h = hidden_states[l]  # (seq_len, hidden_dim)
        h_masked = h * mask[:, None]  # (seq_len, hidden_dim)
        mean_h = h_masked.sum(dim=0) / mask_sum  # (hidden_dim,)
        norm = torch.norm(mean_h, p=2)
        layer_norms.append(norm)

    x = torch.stack(layer_norms, dim=0)  # (n_geometric_features,)

    return x

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
