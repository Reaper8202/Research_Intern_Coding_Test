"""Random-init decoder-only transformer (model plan section 9.1).

* Token embedding plus a linear map of the continuous age features (``tokenizer.age_features``).
* Pre-LayerNorm blocks with causal multi-head attention, rotary positions (RoPE), and a GELU MLP.
* Pretraining head: next-token logits with weights tied to the token embedding. Input-only
  symbols (padding, structure, and summary tokens) are removed from the output distribution.
* Prediction head: the last clinical-event state, joined with the head features, then dropout and
  one linear layer with 40 logits. A patient without clinical events gets a zero state.
* Optional readout ``last_mean`` (D36 round): the last-event state joined with the mean state of
  all clinical events, so the head also sees a summary of the whole retained history.
* Image encoder (D34, only when the tokenizer has ``[IMG]``): a four-layer 3-D CNN from random
  weights maps a standardized brain volume to one ``d_model`` vector. That vector replaces the
  ``[IMG]`` token embedding, so the image enters attention like any other input token.

Sequences are right-padded. Causal attention then keeps every real position independent of the
padding after it. No external weights are loaded anywhere.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ehrforecast.tokenizer import N_AGE_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    head_dropout: float = 0.1
    n_targets: int = 40
    n_head_features: int = 4
    rope_base: float = 10_000.0
    image_token_id: int = -1  # -1: no image encoder
    readout: str = "last"  # "last" or "last_mean"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer for RoPE")
        if self.readout not in {"last", "last_mean"}:
            raise ValueError(f"unknown readout {self.readout}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def rotary(x: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate pairs of channels by position-dependent angles. ``x`` is (B, heads, T, head_dim)."""
    t, dim = x.shape[-2], x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads, self.rope_base, self.p = config.n_heads, config.rope_base, config.dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc_in = nn.Linear(d, config.ff_mult * d)
        self.fc_out = nn.Linear(config.ff_mult * d, d)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        q, k = rotary(q, self.rope_base), rotary(k, self.rope_base)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0
        )
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.fc_out(F.gelu(self.fc_in(self.ln2(x)))))


class ImageEncoder(nn.Module):
    """(N, 1, S, S, S) standardized volumes -> (N, d_model). Strided convolutions, mean pooling."""

    channels = (1, 8, 16, 32, 64)

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for c_in, c_out in itertools.pairwise(self.channels):
            layers += [nn.Conv3d(c_in, c_out, 3, stride=2, padding=1), nn.GELU()]
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(self.channels[-1], d_model)

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        return self.project(self.drop(self.features(volumes).mean(dim=(2, 3, 4))))


class EHRTransformer(nn.Module):
    output_mask: torch.Tensor
    blocks: nn.ModuleList
    image_encoder: ImageEncoder | None

    def __init__(self, config: ModelConfig, input_only_ids: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, d)
        self.age_embedding = nn.Linear(N_AGE_FEATURES, d)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_final = nn.LayerNorm(d)
        self.head_dropout = nn.Dropout(config.head_dropout)
        state_size = 2 * d if config.readout == "last_mean" else d
        self.classifier = nn.Linear(state_size + config.n_head_features, config.n_targets)
        self.image_encoder = ImageEncoder(d, config.dropout) if config.image_token_id >= 0 else None
        mask = torch.zeros(config.vocab_size, dtype=torch.bool)
        if input_only_ids is not None:
            mask[input_only_ids] = True
        self.register_buffer("output_mask", mask)
        self.apply(self._init)
        for block in self.blocks:  # scaled residual projections (GPT-2)
            assert isinstance(block, Block)
            nn.init.normal_(block.proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(block.fc_out.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def body_modules(self) -> list[nn.Module]:
        body = [self.token_embedding, self.age_embedding, self.blocks, self.ln_final]
        return body + ([self.image_encoder] if self.image_encoder is not None else [])

    def body_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.body_modules() for p in m.parameters()]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.classifier.parameters())

    def hidden(
        self, tokens: torch.Tensor, ages: torch.Tensor, images: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, T) token IDs and (B, T, age features) -> (B, T, d_model).

        ``images`` holds one volume per ``[IMG]`` token, in row order (at most one per row).
        """
        x = self.token_embedding(tokens)
        if self.image_encoder is not None:
            rows, cols = torch.nonzero(tokens == self.config.image_token_id, as_tuple=True)
            n_images = 0 if images is None else images.shape[0]
            if len(rows) != n_images:
                raise ValueError(f"{len(rows)} [IMG] tokens but {n_images} image volumes")
            if images is not None and n_images:
                x = x.index_put((rows, cols), self.image_encoder(images).to(x.dtype))
        elif images is not None:
            raise ValueError("image volumes were given to a model without an image encoder")
        x = self.drop(x + self.age_embedding(ages))
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)

    def next_token_logits(self, states: torch.Tensor) -> torch.Tensor:
        """(N, d_model) states -> (N, vocab) logits; input-only symbols get -inf."""
        logits = states @ self.token_embedding.weight.T
        return logits.masked_fill(self.output_mask, float("-inf"))

    def classify(
        self,
        states: torch.Tensor,
        readout: torch.Tensor,
        head_features: torch.Tensor,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read the last clinical event (zero state when ``readout`` is -1); return 40 logits."""
        rows = torch.arange(states.shape[0], device=states.device)
        picked = states[rows, readout.clamp(min=0)]
        picked = picked * (readout >= 0).unsqueeze(1).to(picked.dtype)
        if self.config.readout == "last_mean":
            if event_mask is None:
                raise ValueError("the last_mean readout needs the event mask")
            weights = event_mask.unsqueeze(2).to(states.dtype)
            mean = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            picked = torch.cat([picked, mean], dim=1)
        return self.classifier(self.head_dropout(torch.cat([picked, head_features], dim=1)))

    def forward(
        self,
        tokens: torch.Tensor,
        ages: torch.Tensor,
        readout: torch.Tensor,
        head_features: torch.Tensor,
        images: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self.hidden(tokens, ages, images)
        return self.classify(states, readout, head_features, event_mask)
