"""
CausalTransformerDenoiser
=========================
Non-invasive subclass of `TransformerDenoiser` that adds, for the OFFLINE task:

  1. Per-timestep speaker fusion: audio_t, 3dmm_t, emotion_t -> one fused speaker
     token s_t (a clean, time-aligned speaker sequence of length T instead of a
     3*T token bag).
  2. Causal lead-lag cross-attention bias: listener query at time t attends to
     speaker key at time tau with an additive, per-head learnable bias b_h(t - tau)
     and a hard causal mask (future speaker frames, tau > t + lookahead, are -inf).
     Different heads specialise in different reaction delays.
  3. Coarse-to-fine conditioning: a causal GRU over the fused speaker sequence
     predicts the listener's 8-class facial-expression "plan" per timestep
     (`_coarse_logits`, surfaced for an explicit CE loss) and FiLM-modulates the
     listener tokens before the decoder.

Everything else (diffusion wrapper, EEG head, loss plumbing for the other terms,
checkpointing) is inherited unchanged. The coarse logits are stashed on
`self._coarse_logits` so the matcher can surface them to the loss without
touching the model's tensor return signature.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser import (
    TransformerDenoiser,
    lengths_to_mask,
)


class CausalTransformerDenoiser(TransformerDenoiser):
    def __init__(
        self,
        num_heads: int = 4,
        lag_max: int = 60,
        lag_lookahead: int = 0,
        coarse_classes: int = 8,
        coarse_hidden: int = 256,
        coarse_emo_start: int = 17,
        use_lag_bias: bool = True,
        use_coarse: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(num_heads=num_heads, **kwargs)

        self.num_heads = num_heads
        self.lag_max = int(lag_max)
        self.lag_lookahead = int(lag_lookahead)
        self.coarse_classes = int(coarse_classes)
        self.coarse_emo_start = int(coarse_emo_start)
        self.use_lag_bias = bool(use_lag_bias)
        self.use_coarse = bool(use_coarse)

        d = self.latent_dim

        # (1) per-timestep speaker fusion: [audio | 3dmm | emotion] (3d) -> d
        self.fuse_proj = nn.Linear(3 * d, d)

        # (2) causal lead-lag attention bias table (per head).
        # buckets: [-lookahead .. lag_max] exact, plus one "far past" bucket.
        self.num_lag_buckets = self.lag_lookahead + self.lag_max + 1 + 1
        self.lag_bias = nn.Parameter(torch.zeros(num_heads, self.num_lag_buckets))

        # (3) coarse-to-fine head: causal GRU over fused speaker -> per-t plan.
        self.coarse_gru = nn.GRU(d, coarse_hidden, num_layers=1, batch_first=False)
        self.coarse_out = nn.Linear(coarse_hidden, self.coarse_classes)
        self.coarse_film = nn.Linear(coarse_hidden, 2 * d)
        # start FiLM near identity (gamma~0, beta~0)
        nn.init.zeros_(self.coarse_film.weight)
        nn.init.zeros_(self.coarse_film.bias)

        self._coarse_logits: Optional[torch.Tensor] = None

        # (P4) optional per-person FiLM over the coarse GRU hidden state. Set
        # externally (by the personalized modifier) to a (gamma, beta) tuple of
        # broadcastable tensors right before the forward; None => generic plan.
        self._person_coarse_film = None

    # ------------------------------------------------------------------ utils
    def _fuse_speaker(self, audio, mm3d, emotion, T, bs, device, dtype):
        """Fuse the three per-frame speaker encodings into one token / frame.

        Each input is (Ts, bs, d) or length-0 (dropped). Missing channels are
        replaced by zeros so the fusion projection always sees 3*d.
        """
        def _fix(x):
            if x is None or x.shape[0] == 0:
                return torch.zeros(T, bs, self.latent_dim, device=device, dtype=dtype)
            return x

        audio, mm3d, emotion = _fix(audio), _fix(mm3d), _fix(emotion)
        fused = self.fuse_proj(torch.cat([audio, mm3d, emotion], dim=-1))  # (T, bs, d)
        return fused

    def _build_lag_mask(self, T, Ts, n_global, bs, device, dtype, offset=0):
        """Additive cross-attention mask (bs*num_heads, T, n_global + Ts).

        Global tokens (time / latent / past) get bias 0 and are always visible.
        Speaker columns get per-head lag bias b_h(t-tau); future (tau > t +
        lookahead) is -inf (hard causality).
        """
        # `offset` aligns the listener window to the speaker window in absolute
        # frame index: listener frame t corresponds to speaker frame (t + offset).
        # offline: Ts == T, offset == 0.  online: speaker leads by offset = Ts - T.
        t_idx = (torch.arange(T, device=device) + offset).unsqueeze(1)  # (T, 1)
        tau = torch.arange(Ts, device=device).unsqueeze(0)       # (1, Ts)
        delta = t_idx - tau                                      # (T, Ts)

        future = delta < (-self.lag_lookahead)                   # speaker in the future
        far = delta > self.lag_max                               # very old speaker
        bucket = (delta.clamp(min=-self.lag_lookahead, max=self.lag_max)
                  + self.lag_lookahead)                          # [0 .. lookahead+lag_max]
        bucket = bucket.masked_fill(far, self.num_lag_buckets - 1)
        bucket = bucket.clamp(0, self.num_lag_buckets - 1).long()

        bias = self.lag_bias[:, bucket]                          # (num_heads, T, Ts)
        bias = bias.masked_fill(future.unsqueeze(0), float("-inf"))

        gbias = torch.zeros(self.num_heads, T, n_global, device=device, dtype=bias.dtype)
        full = torch.cat([gbias, bias], dim=2)                   # (num_heads, T, S_mem)
        full = full.unsqueeze(0).expand(bs, -1, -1, -1).reshape(
            bs * self.num_heads, T, n_global + Ts)
        return full.to(dtype)

    # --------------------------------------------------------------- forward
    def _forward(
        self,
        sample,
        time_embed,
        speaker_audio_encodings,
        speaker_latent_embed,
        speaker_3dmm_encodings,
        speaker_emotion_encodings,
        past_listener_emotion,
        motion_length=None,
    ):
        # Causal path supports offline (full sequence, Ts == T) and online
        # (windowed: speaker s_w leads listener l_w by offset = Ts - T frames).
        if self.arch != "trans_dec" or self.task not in ("offline", "online"):
            return super()._forward(
                sample, time_embed, speaker_audio_encodings, speaker_latent_embed,
                speaker_3dmm_encodings, speaker_emotion_encodings,
                past_listener_emotion, motion_length)

        device = time_embed.device
        dtype = sample.dtype

        sample = self.to_emotion_embed(sample)                  # (T, bs, d)
        T, bs, _ = sample.shape

        # (1) fuse speaker channels into a time-aligned sequence ----------------
        Ts = speaker_audio_encodings.shape[0]
        if Ts == 0:
            Ts = max(speaker_3dmm_encodings.shape[0], speaker_emotion_encodings.shape[0], T)
        fused = self._fuse_speaker(
            speaker_audio_encodings, speaker_3dmm_encodings, speaker_emotion_encodings,
            Ts, bs, device, dtype)                              # (Ts, bs, d)
        offset = Ts - T                                         # speaker leads listener (online: 30, offline: 0)

        # (3) coarse-to-fine plan (causal GRU over fused speaker) ---------------
        # The GRU runs over all speaker frames; the listener-aligned hidden states
        # are the last T (indices [offset : offset+T]) -> one plan per listener frame.
        if self.use_coarse:
            h, _ = self.coarse_gru(fused)                       # (Ts, bs, H)
            h_l = h[offset:offset + T]                          # (T, bs, H) aligned to listener
            # (P4) personalize the plan: FiLM the hidden state with the person's
            # embedding so BOTH the 8-class logits and the downstream fine FiLM
            # reflect "how this person reacts" (zero-init => identity at start).
            pcf = self._person_coarse_film
            if pcf is not None:
                p_gamma, p_beta = pcf                           # each (·, H) broadcastable
                if p_gamma.shape[0] == 1 and bs != 1:
                    p_gamma = p_gamma.expand(bs, -1)
                    p_beta = p_beta.expand(bs, -1)
                elif p_gamma.shape[0] != bs:
                    raise ValueError(
                        f"Personal coarse batch {p_gamma.shape[0]} cannot broadcast "
                        f"to denoiser batch {bs}"
                    )
                h_l = (1.0 + p_gamma.view(1, bs, -1)) * h_l + p_beta.view(1, bs, -1)
            coarse_logits = self.coarse_out(h_l)                # (T, bs, C)
            self._coarse_logits = coarse_logits.permute(1, 0, 2).contiguous()  # (bs, T, C)
            film = self.coarse_film(h_l)                        # (T, bs, 2d)
            gamma, beta = film.chunk(2, dim=-1)                 # (T, bs, d) each
        else:
            self._coarse_logits = None
            gamma = beta = None

        # listener tokens + positional encoding + FiLM conditioning ------------
        sample = self.query_pos(sample)
        if self.use_coarse and gamma is not None:
            sample = (1.0 + gamma) * sample + beta

        # (2) assemble memory: [global tokens ... | fused speaker seq] ----------
        # globals (time / speaker-latent / past-listener) are always visible
        # (past-listener frames all precede the target window -> no causal mask).
        global_tokens = [time_embed]
        if speaker_latent_embed is not None and speaker_latent_embed.shape[0] > 0:
            global_tokens.append(speaker_latent_embed)
        if past_listener_emotion is not None and past_listener_emotion.shape[0] > 0:
            global_tokens.append(past_listener_emotion)
        n_global = sum(g.shape[0] for g in global_tokens)

        memory = torch.cat(global_tokens + [fused], dim=0)      # (n_global + Ts, bs, d)
        memory = self.mem_pos(memory)

        # padding masks --------------------------------------------------------
        tgt_key_padding_mask = None
        memory_key_padding_mask = None
        if motion_length is not None:
            l_valid = lengths_to_mask(motion_length, device=device, max_len=T)        # (bs, T)
            # speaker validity: the leading `offset` context frames are always
            # valid; the last T speaker frames share the listener's validity.
            if offset > 0:
                s_valid = torch.cat(
                    [torch.ones(bs, offset, dtype=torch.bool, device=device), l_valid], dim=1)
            else:
                s_valid = l_valid                                                    # (bs, Ts==T)
            tgt_key_padding_mask = ~l_valid
            g_pad = torch.zeros(bs, n_global, dtype=torch.bool, device=device)       # globals never padded
            memory_key_padding_mask = torch.cat([g_pad, ~s_valid], dim=1)            # (bs, n_global+Ts)

        # (2) causal lead-lag additive mask -----------------------------------
        memory_mask = None
        if self.use_lag_bias:
            memory_mask = self._build_lag_mask(T, Ts, n_global, bs, device, dtype, offset=offset)

        sample = self.decoder(
            tgt=sample, memory=memory,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ).squeeze(0)

        sample = self.to_emotion_feat(sample)
        sample = sample.permute(1, 0, 2)
        return sample
