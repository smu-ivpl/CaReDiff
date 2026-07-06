"""
DataParallel wrapper for motion_diffusion Trainer.
Overrides main_diffusion() to use nn.DataParallel when multiple GPUs are available.
All other logic (test, val, data_resample, etc.) is inherited unchanged.
"""
import torch
import torch.nn as nn
import logging
from hydra.utils import instantiate
from framework.utils.util import from_pretrained_checkpoint

from trainer.motion_diffusion import Trainer as _BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(_BaseTrainer):

    def main_diffusion(self, stage):
        if self.train_eeg_head_only and self.resumed_training:
            raise ValueError(
                "train_eeg_head_only=True should be launched with resume=false, "
                "so the optimizer and old EEG head checkpoint are not restored."
            )

        model = instantiate(self.model_cfg.diff_model,
                            stage=stage,
                            resumed_training=self.resumed_training,
                            latent_embedder=self.model_cfg.latent_embedder \
                                if hasattr(self.model_cfg, "latent_embedder") else None,
                            audio_encoder=self.model_cfg.audio_encoder \
                                if hasattr(self.model_cfg, "audio_encoder") else None,
                            **self.kwargs,
                            _recursive_=False)
        model.to(self.device)
        self._load_pretrained_motion_diffusion(model)

        # --- DataParallel ---
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.info(f"Using DataParallel across {n_gpus} GPUs")
            model = nn.DataParallel(model)
        model_raw = model.module if isinstance(model, nn.DataParallel) else model
        # --------------------

        optimizer_params = model_raw.parameters()
        if self.train_eeg_head_only:
            model_raw.freeze_except_eeg_head()
            trainable_params = [p for p in model_raw.parameters() if p.requires_grad]
            trainable_names = [n for n, p in model_raw.named_parameters() if p.requires_grad]
            frozen_count = self._count_parameters(
                p for p in model_raw.parameters() if not p.requires_grad
            )
            trainable_count = self._count_parameters(trainable_params)
            if len(trainable_params) == 0:
                raise RuntimeError("No trainable parameters found for EEG head-only training.")
            optimizer_params = trainable_params
            logger.info(
                "EEG head-only training enabled. "
                f"Trainable parameters: {trainable_count}; frozen parameters: {frozen_count}"
            )
            logger.info(f"Trainable parameter tensors: {trainable_names}")

        optimizer = instantiate(self.optim_cfg, lr=self.trainer_cfg.lr, params=optimizer_params)
        if self.resumed_training:
            checkpoint_path = model_raw.get_ckpt_path(
                model_raw.diffusion_decoder.model, runid="resume_runid", last=True
            )
            best_validation_loss, self.start_epoch = (
                from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            )
            logger.info(f"Resume training from epoch {self.start_epoch}")
        else:
            best_validation_loss = float('inf')
        print(f"Best validation loss: {best_validation_loss}")

        scheduler = instantiate(self.kwargs.pop("scheduler"), optimizer, len(self.train_loader))
        selected_loss_name = "loss_eeg" if self.train_eeg_head_only else "diff_loss"

        for epoch in range(self.start_epoch, self.epochs):
            if self.train_eeg_head_only:
                model_raw.set_eeg_head_train_mode()
            else:
                model.train()

            diffusion_loss, prior_loss, au_rec_loss, va_rec_loss, em_rec_loss, eeg_rec_loss, eeg_valid_ratio = (
                self.train_diffusion(model, self.train_loader, optimizer, scheduler,
                                     self.criterion, epoch, self.writer, self.device))
            logging.info(f"Epoch: {epoch + 1}  train_{selected_loss_name}: {diffusion_loss:.5f}  "
                         f"prior_loss: {prior_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                         f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}"
                         f"  eeg_rec_loss: {eeg_rec_loss:.5f}  eeg_valid_ratio: {eeg_valid_ratio:.5f}")
            if self.writer is not None:
                self.writer.add_scalar("Epoch/train_loss", diffusion_loss, epoch + 1)

            if (epoch + 1) % self.val_period == 0:
                diffusion_loss, prior_loss, au_rec_loss, va_rec_loss, em_rec_loss, eeg_rec_loss, eeg_valid_ratio = (
                    self.val_diffusion(model, self.val_loader, self.criterion, self.device))
                logging.info(f"Epoch: {epoch + 1}  val_{selected_loss_name}: {diffusion_loss:.5f}  "
                             f"prior_loss: {prior_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                             f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}"
                             f"  eeg_rec_loss: {eeg_rec_loss:.5f}  eeg_valid_ratio: {eeg_valid_ratio:.5f}")
                if self.writer is not None:
                    self.writer.add_scalar("Epoch/val_loss", diffusion_loss, epoch + 1)
                    self.writer.add_scalar("Epoch/val_prior_loss", prior_loss, epoch + 1)
                    self.writer.add_scalar("Epoch/val_au_rec_loss", au_rec_loss, epoch + 1)
                    self.writer.add_scalar("Epoch/val_va_rec_loss", va_rec_loss, epoch + 1)
                    self.writer.add_scalar("Epoch/val_em_rec_loss", em_rec_loss, epoch + 1)

                if diffusion_loss < best_validation_loss:
                    best_validation_loss = diffusion_loss
                    logging.info(
                        f"New best {selected_loss_name} ({best_validation_loss:.5f}) at epoch {epoch + 1}, "
                        f"saving checkpoint"
                    )
                    model_raw.save_ckpt(optimizer, best=True, epoch=(epoch + 1), best_loss=best_validation_loss)

                model_raw.save_ckpt(optimizer, epoch=(epoch + 1), best_loss=best_validation_loss)
                model_raw.save_ckpt(optimizer, last=True, epoch=(epoch + 1), best_loss=best_validation_loss)
