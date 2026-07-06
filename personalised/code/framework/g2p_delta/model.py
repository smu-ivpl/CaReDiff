"""Generic-to-Personal residual denoising adapter.

The verified Generic Offline diffusion model remains frozen.  A listener
condition produces a bounded, zero-initialised correction at the denoiser's
existing ``to_emotion_feat`` projection, so a fresh model is exactly the
Generic model while retaining its prior and EEG paths.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonalityEncoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, personality: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(personality.float()), dim=-1)


class HistoryEncoder(nn.Module):
    """Checkpoint-free LHFB encoder for a 58-D historical 3DMM sequence."""

    def __init__(self, output_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(58, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3 or history.shape[-1] != 58:
            raise ValueError(f"LHFB must have shape (B,T,58), got {tuple(history.shape)}")
        features = self.temporal(history.float().transpose(1, 2))
        pooled = torch.cat(
            (features.mean(dim=-1), features.std(dim=-1, unbiased=False)), dim=-1
        )
        return F.normalize(self.project(pooled), dim=-1)


class ConditionFusion(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, history: torch.Tensor, personality: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat((history, personality), dim=-1))
        return F.normalize(self.out(gate * history + (1.0 - gate) * personality), dim=-1)


class ResidualDenoisingAdapter(nn.Module):
    """Low-rank correction applied at every reverse-diffusion denoiser call."""

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        output_dim: int = 25,
        rank: int = 128,
        max_scale: float = 0.15,
    ):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_down = nn.Linear(hidden_dim, rank)
        self.condition = nn.Linear(condition_dim, rank)
        self.output = nn.Linear(rank, output_dim)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.max_scale = float(max_scale)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # hidden: (T, B*num_preds, D); one listener condition is broadcast to samples.
        batch = hidden.shape[1]
        if condition.shape[0] == 1 and batch != 1:
            condition = condition.expand(batch, -1)
        elif condition.shape[0] != batch:
            raise ValueError(
                f"Condition batch {condition.shape[0]} cannot broadcast to denoiser batch {batch}"
            )
        joint = self.hidden_down(self.hidden_norm(hidden))
        joint = joint + self.condition(condition).unsqueeze(0)
        delta = self.output(F.gelu(joint))
        scale = self.max_scale * torch.sigmoid(self.gate_logit)
        return scale * delta


class CoarsePriorAdapter(nn.Module):
    """Small zero-init listener correction to the Generic coarse expression plan."""

    def __init__(self, condition_dim: int, coarse_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, coarse_hidden),
            nn.GELU(),
            nn.Linear(coarse_hidden, 2 * coarse_hidden),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return 0.1 * torch.tanh(gamma), 0.1 * torch.tanh(beta)


class G2PDeltaModel(nn.Module):
    """Thin personalised extension around the final Generic Offline model."""

    VALID_MODES = {"3dmm_only", "personality_only", "3dmm_personality"}

    def __init__(self, cfg, main_net: nn.Module):
        super().__init__()
        self.main_net = main_net
        args = cfg.main_model.args
        self.personal_condition_mode = str(
            args.get("personal_condition_mode", "personality_only")
        )
        if self.personal_condition_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported personal condition: {self.personal_condition_mode}")

        embed_dim = int(args.get("embed_dim", 256))
        dropout = float(args.get("condition_dropout", 0.1))
        self.history_encoder = (
            HistoryEncoder(embed_dim, int(args.get("history_hidden_dim", 128)), dropout)
            if self.personal_condition_mode in {"3dmm_only", "3dmm_personality"}
            else None
        )
        self.personality_encoder = (
            PersonalityEncoder(embed_dim, int(args.get("personality_hidden_dim", 128)), dropout)
            if self.personal_condition_mode in {"personality_only", "3dmm_personality"}
            else None
        )
        self.condition_fusion = (
            ConditionFusion(embed_dim, dropout)
            if self.personal_condition_mode == "3dmm_personality"
            else None
        )

        modules = OrderedDict(self.main_net.named_modules())
        denoiser = modules.get("diffusion_decoder.model")
        if denoiser is None:
            raise ValueError("Generic model has no diffusion_decoder.model")
        if not hasattr(denoiser, "to_emotion_feat"):
            raise ValueError("Generic denoiser has no to_emotion_feat projection")
        self._denoiser = denoiser
        hidden_dim = int(getattr(denoiser, "latent_dim", 512))
        self.delta_adapter = ResidualDenoisingAdapter(
            hidden_dim=hidden_dim,
            condition_dim=embed_dim,
            output_dim=25,
            rank=int(args.get("delta_rank", 128)),
            max_scale=float(args.get("delta_max_scale", 0.15)),
        )
        self.coarse_adapter = (
            CoarsePriorAdapter(embed_dim, denoiser.coarse_gru.hidden_size)
            if bool(args.get("personalize_coarse", True)) and hasattr(denoiser, "coarse_gru")
            else None
        )
        self.anchor_weight = float(args.get("anchor_weight", 1.0e-3))
        self._current_condition: Optional[torch.Tensor] = None
        self._last_delta_energy = torch.tensor(0.0)
        self._hook = denoiser.to_emotion_feat.register_forward_hook(self._delta_hook)

        # Freeze the verified Generic backbone. EEG is selectively unfrozen by the trainer.
        for parameter in self.main_net.parameters():
            parameter.requires_grad = False

        # Compatibility with the inherited personalised trainer.
        self.person_encoder = None

    def _delta_hook(self, _module, inputs, output):
        if self._current_condition is None:
            self._last_delta_energy = output.new_tensor(0.0)
            return output
        hidden = inputs[0]
        delta = self.delta_adapter(hidden, self._current_condition)
        self._last_delta_energy = delta.square().mean()
        return output + delta

    def encode_person_condition(self, p=None, personality=None):
        history_embedding = personality_embedding = None
        if self.history_encoder is not None:
            if p is None or p.numel() == 0:
                raise ValueError("LHFB condition requires non-empty personal 3DMM history")
            history_embedding = self.history_encoder(p)
        if self.personality_encoder is not None:
            if personality is None or personality.numel() == 0:
                raise ValueError("Personality condition requires the listener Big-Five vector")
            if personality.dim() == 1:
                personality = personality.unsqueeze(0)
            personality_embedding = self.personality_encoder(personality)
        if self.personal_condition_mode == "3dmm_only":
            return history_embedding
        if self.personal_condition_mode == "personality_only":
            return personality_embedding
        return self.condition_fusion(history_embedding, personality_embedding)

    def set_person_condition(self, p=None, personality=None):
        condition = self.encode_person_condition(p=p, personality=personality)
        self._current_condition = condition
        if self.coarse_adapter is not None:
            self._denoiser._person_coarse_film = self.coarse_adapter(condition)
        return condition

    def clear_person_condition(self):
        self._current_condition = None
        if self.coarse_adapter is not None:
            self._denoiser._person_coarse_film = None

    def forward(self, x, p=None, personality=None):
        self.set_person_condition(p=p, personality=personality)
        try:
            output = self.main_net(**x)
            regular = self.anchor_weight * self._last_delta_energy
        finally:
            self.clear_person_condition()
        return output, regular

    def eeg_head(self):
        return getattr(self.main_net, "eeg_head", None)

    def has_eeg_head(self):
        return self.eeg_head() is not None

    def set_eeg_head_requires_grad(self, requires_grad=True):
        if not self.has_eeg_head():
            raise RuntimeError("Generic EEG prediction head is disabled")
        for parameter in self.eeg_head().parameters():
            parameter.requires_grad = requires_grad

    def modifier_parameters(self, include_eeg_head=False):
        modules = [
            self.history_encoder,
            self.personality_encoder,
            self.condition_fusion,
            self.delta_adapter,
            self.coarse_adapter,
        ]
        for module in modules:
            if module is not None:
                yield from module.parameters()
        if include_eeg_head:
            self.set_eeg_head_requires_grad(True)
            yield from self.eeg_head().parameters()

    def modifier_state_dict(self, include_eeg_head=False):
        names = (
            "history_encoder",
            "personality_encoder",
            "condition_fusion",
            "delta_adapter",
            "coarse_adapter",
        )
        state = {}
        for name in names:
            module = getattr(self, name)
            if module is not None:
                state.update({f"{name}.{k}": v for k, v in module.state_dict().items()})
        if include_eeg_head and self.has_eeg_head():
            state.update({f"eeg_head.{k}": v for k, v in self.eeg_head().state_dict().items()})
        return state

    def load_modifier_state_dict(self, state_dict):
        for name in (
            "history_encoder",
            "personality_encoder",
            "condition_fusion",
            "delta_adapter",
            "coarse_adapter",
        ):
            module = getattr(self, name)
            selected = {
                key[len(name) + 1 :]: value
                for key, value in state_dict.items()
                if key.startswith(f"{name}.")
            }
            if selected:
                if module is None:
                    raise ValueError(f"Checkpoint contains {name}, but it is disabled")
                module.load_state_dict(selected)
        eeg_state = {
            key[len("eeg_head.") :]: value
            for key, value in state_dict.items()
            if key.startswith("eeg_head.")
        }
        if eeg_state:
            self.eeg_head().load_state_dict(eeg_state)

    def train(self, mode: bool = True):
        super().train(mode)
        # The final Generic model stays deterministic/frozen; only adapters train.
        self.main_net.eval()
        return self
