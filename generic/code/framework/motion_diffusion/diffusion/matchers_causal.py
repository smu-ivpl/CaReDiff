"""
Causal / coarse-to-fine matcher wrappers
=========================================
Non-invasive subclasses that swap the baseline `TransformerDenoiser` for the
`CausalTransformerDenoiser` and surface the coarse 8-class logits to the loss.

  * `CoarseDecoderLatentMatcher` rebuilds `self.model` as the causal denoiser
    (reusing the already-resolved `init_params`) and, during training, copies
    the stashed `_coarse_logits` into the output dict.
  * `CausalLatentMatcher` is a thin `LatentMatcher` that uses the coarse decoder
    wrapper while keeping the prior / EEG head / checkpoint loading identical.

No original file is modified; select these via Hydra `_target_` in a new config.
"""
import os

from omegaconf import DictConfig

from framework.motion_diffusion.diffusion.matchers import (
    DecoderLatentMatcher,
    LatentMatcher,
)
from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser_causal import (
    CausalTransformerDenoiser,
)
from framework.utils.util import from_pretrained_checkpoint


class CoarseDecoderLatentMatcher(DecoderLatentMatcher):
    def __init__(self, conf: DictConfig = None, **kwargs):
        super().__init__(conf, **kwargs)
        cfg = conf.args
        coarse_kwargs = dict(
            lag_max=int(cfg.get("lag_max", 60)),
            lag_lookahead=int(cfg.get("lag_lookahead", 0)),
            coarse_classes=int(cfg.get("coarse_classes", 8)),
            coarse_hidden=int(cfg.get("coarse_hidden", 256)),
            coarse_emo_start=int(cfg.get("coarse_emo_start", 17)),
            use_lag_bias=bool(cfg.get("use_lag_bias", True)),
            use_coarse=bool(cfg.get("use_coarse", True)),
        )
        # Rebuild the denoiser as the causal+coarse variant, reusing the
        # init_params resolved by the parent (latent_dim, num_heads, drop probs ...).
        self.model = CausalTransformerDenoiser(**self.init_params, **coarse_kwargs)

    def _forward(self, **kwargs):
        out = super()._forward(**kwargs)
        if self.stage != "test":
            coarse_logits = getattr(self.model, "_coarse_logits", None)
            if coarse_logits is not None:
                # (bs*num_preds, T, C) -> (bs, num_preds, T, C)
                out["coarse_logits"] = coarse_logits.view(
                    -1, self.num_preds, *coarse_logits.shape[1:])
        return out


class CausalLatentMatcher(LatentMatcher):
    def __init__(
        self,
        task: str = "online",
        stage: str = "fit",
        device: str = None,
        diffusion_prior: DictConfig = None,
        diffusion_decoder: DictConfig = None,
        latent_embedder: DictConfig = None,
        audio_encoder: DictConfig = None,
        eeg_head: DictConfig = None,
        resumed_training: bool = False,
        auto_load_ckpt: bool = True,
        **kwargs,
    ):
        import torch

        # Build everything via the parent but defer checkpoint loading so we can
        # swap in the coarse decoder first.
        super().__init__(
            task=task, stage=stage, device=device,
            diffusion_prior=diffusion_prior, diffusion_decoder=diffusion_decoder,
            latent_embedder=latent_embedder, audio_encoder=audio_encoder,
            eeg_head=eeg_head, resumed_training=resumed_training,
            auto_load_ckpt=False, **kwargs,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        module_dict_cfg = DictConfig(
            {"latent_embedder": latent_embedder, "audio_encoder": audio_encoder})
        self.diffusion_decoder = CoarseDecoderLatentMatcher(
            self.diffusion_decoder_cfg, task=task, stage=stage,
            module_dict_cfg=module_dict_cfg, **kwargs)

        if auto_load_ckpt:
            self._load_causal_checkpoints(device, resumed_training, stage)

    def _load_causal_checkpoints(self, device, resumed_training, stage):
        load_ckpt = bool(resumed_training) or stage == "test"
        if not load_ckpt:
            return
        want_last = bool(resumed_training)
        # published generic checkpoints are named checkpoint_120.pth (not checkpoint_best.pth)
        test_epoch = 120 if stage == "test" else None
        want_best = False

        ckpt_path = self.get_ckpt_path(
            self.diffusion_decoder.model, runid="resume_runid",
            epoch=test_epoch, best=want_best, last=want_last)
        from_pretrained_checkpoint(str(ckpt_path), self.diffusion_decoder.model, device)

        if self.diffusion_prior is not None:
            prior_ckpt_path = self.get_ckpt_path(
                self.diffusion_prior.model, runid="resume_runid",
                epoch=None, best=want_best, last=want_last, create_dir=False)
            if os.path.exists(prior_ckpt_path):
                from_pretrained_checkpoint(str(prior_ckpt_path), self.diffusion_prior.model, device)
            elif resumed_training:
                raise FileNotFoundError(
                    f"Missing prior checkpoint for resumed training: {prior_ckpt_path}")

        if self.eeg_head is not None:
            eeg_ckpt_path = self.get_ckpt_path(
                self.eeg_head, runid="resume_runid",
                epoch=None, best=want_best, last=want_last, create_dir=False)
            if os.path.exists(eeg_ckpt_path):
                from_pretrained_checkpoint(str(eeg_ckpt_path), self.eeg_head, device)
            elif resumed_training:
                raise FileNotFoundError(
                    f"Missing EEG head checkpoint for resumed training: {eeg_ckpt_path}")
