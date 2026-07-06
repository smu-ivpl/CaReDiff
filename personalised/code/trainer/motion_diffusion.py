import math
import os
import random
from einops import rearrange
import torch
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_eeg_metrics, compute_metrics
from framework.utils.util import from_pretrained_checkpoint
from utils.util import AverageMeter, get_lr
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate, to_absolute_path
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 generic: DictConfig = None,
                 renderer: DictConfig = None,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):
        # # current working directory: outputs/${trainer.task_name}/${data.data_name}/${run_id}
        # folder: save/${trainer.task_name}/${data.data_name}  # ckpt_name: checkpoint.pth
        # # last ckpt directory
        # ckpt_dir: ${get_last_checkpoint:${trainer.folder}}  # ${trainer.run_id}
        # # for example, ckpt_dir: save/motion_diffusion/react_2024/checkpoints
        # resume_run_id: ${old_run_id}

        super().__init__()
        self.resumed_training = resumed_training
        self.renderer = renderer
        self.model_cfg = model
        self.criterion_cfg = criterion

        if torch.cuda.device_count() > 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.kwargs = kwargs
        self.trainer_cfg = generic
        self.optim_cfg = kwargs.pop("optim")
        self.task = kwargs.get("task")
        self.train_eeg_head_only = self._as_bool(
            self.trainer_cfg.get("train_eeg_head_only", False)
        )

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _count_parameters(parameters):
        return sum(parameter.numel() for parameter in parameters)

    @staticmethod
    def _resolve_checkpoint_path(path):
        if path is None or str(path).strip() == "":
            return None

        path = str(path)
        if os.path.isabs(path):
            return path
        return to_absolute_path(path)

    def _load_pretrained_motion_diffusion(self, model):
        decoder_checkpoint = self._resolve_checkpoint_path(
            self.trainer_cfg.get("pretrained_decoder_checkpoint", "")
        )
        prior_checkpoint = self._resolve_checkpoint_path(
            self.trainer_cfg.get("pretrained_prior_checkpoint", "")
        )
        load_prior = self._as_bool(self.trainer_cfg.get("pretrained_load_prior", True))

        if not self.train_eeg_head_only and decoder_checkpoint is None and prior_checkpoint is None:
            return

        if self.train_eeg_head_only and decoder_checkpoint is None:
            raise ValueError(
                "train_eeg_head_only=True requires trainer.generic.pretrained_decoder_checkpoint. "
                "Use resume=false and point it to a pretrained TransformerDenoiser checkpoint."
            )

        if decoder_checkpoint is not None:
            if not os.path.exists(decoder_checkpoint):
                raise FileNotFoundError(f"Missing pretrained decoder checkpoint: {decoder_checkpoint}")
            from_pretrained_checkpoint(decoder_checkpoint, model.diffusion_decoder.model, self.device)
            logger.info(f"Loaded pretrained decoder checkpoint: {decoder_checkpoint}")

        if not load_prior or model.diffusion_prior is None:
            if self.train_eeg_head_only:
                model.diffusion_prior = None
            return

        if prior_checkpoint is None:
            if self.train_eeg_head_only:
                logger.warning(
                    "pretrained_load_prior=True but pretrained_prior_checkpoint is empty; skip prior loading."
                )
                model.diffusion_prior = None
            return
        if not os.path.exists(prior_checkpoint):
            logger.warning(f"Missing pretrained prior checkpoint; skip prior loading: {prior_checkpoint}")
            if self.train_eeg_head_only:
                model.diffusion_prior = None
            return

        from_pretrained_checkpoint(prior_checkpoint, model.diffusion_prior.model, self.device)
        logger.info(f"Loaded pretrained prior checkpoint: {prior_checkpoint}")

    def set_data_module(self, data_module):
        self.data_module = data_module

    def data_resample(self,
                      speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips,
                      listener_video_clips, listener_emotion_clips, listener_3dmm_clips,
                      speaker_seq_lengths, listener_seq_lengths,
                      listener_eeg_clips=None, listener_eeg_masks=None):

        s_ratio = self.trainer_cfg.s_ratio
        window_size = self.trainer_cfg.window_size
        clip_length = self.trainer_cfg.clip_length
        s_window_size = s_ratio * window_size
        l_window_size = window_size

        if self.task == 'offline':
            stack = lambda clips: torch.stack(clips, dim=0)
            speaker_audio, speaker_emotion, speaker_3dmm = (
                stack(clips) for clips in (speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips))
            listener_video, listener_emotion, listener_3dmm = (
                stack(clips) for clips in (listener_video_clips, listener_emotion_clips, listener_3dmm_clips))
            past_listener_emotion = past_listener_3dmm = None
            seq_lengths = torch.tensor(speaker_seq_lengths).clamp(max=clip_length)
            listener_eeg = listener_eeg_mask = None
            # Tensor([58, 750, 632, ...])

        elif self.task == "online":
            def get_padded(clip: torch.Tensor, length: int, target_len: int) -> torch.Tensor:
                clip = clip[:length]
                if length < target_len:
                    pad_shape = (target_len - length, *clip.shape[1:])
                    clip = torch.cat([clip, clip.new_zeros(pad_shape)], dim=0)
                return clip

            speaker_audio, speaker_emotion, speaker_3dmm = [], [], []
            listener_video, listener_emotion, listener_3dmm = [], [], []
            past_listener_emotion, past_listener_3dmm = [], []
            listener_eeg, listener_eeg_mask = [], []
            has_eeg = listener_eeg_clips is not None and listener_eeg_masks is not None
            eeg_clips = listener_eeg_clips if has_eeg else [None] * len(speaker_audio_clips)
            eeg_masks = listener_eeg_masks if has_eeg else [None] * len(speaker_audio_clips)

            for (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip, speaker_seq_length,
                 listener_video_clip, listener_emotion_clip, listener_3dmm_clip, listener_seq_length,
                 listener_eeg_clip, listener_eeg_mask_clip) in \
                    zip(speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips, speaker_seq_lengths,
                        listener_video_clips, listener_emotion_clips, listener_3dmm_clips, listener_seq_lengths,
                        eeg_clips, eeg_masks):
                seq_length = speaker_seq_length
                assert speaker_seq_length == listener_seq_length, "Sequence length not equal"

                speaker_audio_clip = get_padded(speaker_audio_clip, seq_length, s_window_size)
                speaker_emotion_clip = get_padded(speaker_emotion_clip, seq_length, s_window_size)
                speaker_3dmm_clip = get_padded(speaker_3dmm_clip, seq_length, s_window_size)
                listener_video_clip = get_padded(listener_video_clip, seq_length, s_window_size)
                listener_emotion_clip = get_padded(listener_emotion_clip, seq_length, s_window_size)
                listener_3dmm_clip = get_padded(listener_3dmm_clip, seq_length, s_window_size)
                if has_eeg:
                    listener_eeg_clip = get_padded(listener_eeg_clip, seq_length, s_window_size)
                    listener_eeg_mask_clip = get_padded(listener_eeg_mask_clip, seq_length, s_window_size)

                if seq_length < clip_length:
                    cp = random.randint(0, seq_length - s_window_size) if seq_length > s_window_size else 0
                else:
                    cp = random.randint(0, clip_length - s_window_size)

                du = cp + s_window_size
                speaker_audio_clip = speaker_audio_clip[cp:du]
                speaker_emotion_clip = speaker_emotion_clip[cp:du]
                speaker_3dmm_clip = speaker_3dmm_clip[cp:du]
                listener_video_clip = listener_video_clip[du - l_window_size:du]

                # past = the K listener windows immediately before the target window,
                # i.e. frames [du-(K+1)*lw : du-lw]; front-pad with zeros when the
                # sequence does not reach that far back (K=1 -> original behaviour).
                K = int(self.trainer_cfg.get("n_past_win", 1))
                pe = du - l_window_size
                ps = du - (K + 1) * l_window_size

                def _past_slice(clip, _ps=ps, _pe=pe):
                    if _ps >= 0:
                        return clip[_ps:_pe]
                    pad = clip.new_zeros((-_ps, *clip.shape[1:]))
                    return torch.cat([pad, clip[0:_pe]], dim=0)

                past_listener_emotion_clip = _past_slice(listener_emotion_clip)
                past_listener_3dmm_clip = _past_slice(listener_3dmm_clip)
                listener_emotion_clip = listener_emotion_clip[(du - l_window_size): du]
                listener_3dmm_clip = listener_3dmm_clip[(du - l_window_size): du]
                if has_eeg:
                    listener_eeg.append(listener_eeg_clip[du - 1])
                    listener_eeg_mask.append(listener_eeg_mask_clip[du - 1])

                speaker_audio.append(speaker_audio_clip)
                speaker_emotion.append(speaker_emotion_clip)
                speaker_3dmm.append(speaker_3dmm_clip)
                listener_video.append(listener_video_clip)
                listener_emotion.append(listener_emotion_clip)
                listener_3dmm.append(listener_3dmm_clip)
                past_listener_emotion.append(past_listener_emotion_clip)
                past_listener_3dmm.append(past_listener_3dmm_clip)

            speaker_audio = torch.stack(speaker_audio, dim=0)  # (bs, s_w, d)
            speaker_emotion = torch.stack(speaker_emotion, dim=0)  # (bs, s_w, 25)
            speaker_3dmm = torch.stack(speaker_3dmm, dim=0)  # (bs, s_w, 58)
            listener_video = torch.stack(listener_video, dim=0)  # (bs, l_w, 3, 224, 224)
            listener_emotion = torch.stack(listener_emotion, dim=0)  # (bs, l_w, 25)
            listener_3dmm = torch.stack(listener_3dmm, dim=0)  # (bs, l_w, 58)
            past_listener_emotion = torch.stack(past_listener_emotion, dim=0)  # (bs, l_w, 25)
            past_listener_3dmm = torch.stack(past_listener_3dmm, dim=0)  # (bs, l_w, 58)
            if has_eeg:
                listener_eeg = torch.stack(listener_eeg, dim=0)  # (bs, d_eeg)
                listener_eeg_mask = torch.stack(listener_eeg_mask, dim=0)  # (bs, d_eeg)
            else:
                listener_eeg = listener_eeg_mask = None
            seq_lengths = None
        else:
            raise ValueError("Unknown task type")

        return (speaker_audio, speaker_emotion, speaker_3dmm, listener_video, listener_emotion,
                listener_3dmm, past_listener_emotion, past_listener_3dmm, seq_lengths,
                listener_eeg, listener_eeg_mask)

    def fit(self):
        """
        # relative directory
        root_dir = save/${trainer.task_name}/${data.data_name}/${folder_name}
        # absolute directory
        saving_dir = Path(hydra.utils.to_absolute_path(root_dir))
        # get saving path
        saving_path = str(saving_dir / ...)
        """

        self.start_epoch = self.trainer_cfg.start_epoch
        self.epochs = self.trainer_cfg.epochs
        self.tb_dir = self.trainer_cfg.tb_dir
        self.clip_grad = self.trainer_cfg.clip_grad
        self.val_period = self.trainer_cfg.val_period
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Data module loaded")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        logger.info("Criterion loaded")

        logger.info("Loading writer")
        self.writer = SummaryWriter(self.tb_dir)
        logger.info(f"Writer loaded: {self.tb_dir}")
        self.main_diffusion(stage)

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

        optimizer_params = model.parameters()
        if self.train_eeg_head_only:
            model.freeze_except_eeg_head()
            trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            frozen_count = self._count_parameters(
                parameter for parameter in model.parameters() if not parameter.requires_grad
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
            print(
                "EEG head-only training enabled. "
                f"Trainable parameters: {trainable_count}; frozen parameters: {frozen_count}"
            )
            print(f"Trainable parameter tensors: {trainable_names}")

        # load optimizer
        optimizer = instantiate(self.optim_cfg, lr=self.trainer_cfg.lr, params=optimizer_params)
        if self.resumed_training:
            checkpoint_path = model.get_ckpt_path(model.diffusion_decoder.model, runid="resume_runid", last=True)
            best_validation_loss, self.start_epoch = (
                from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            )
            logger.info(f"Resume training from epoch {self.start_epoch}")
        else:
            best_validation_loss = float('inf')
        print(f"Best validation loss: {best_validation_loss}")

        # load scheduler
        scheduler = instantiate(self.kwargs.pop("scheduler"), optimizer, len(self.train_loader))
        selected_loss_name = "loss_eeg" if self.train_eeg_head_only else "diff_loss"

        for epoch in range(self.start_epoch, self.epochs):
            diffusion_loss, prior_loss, au_rec_loss, va_rec_loss, em_rec_loss, eeg_rec_loss, eeg_valid_ratio = (
                self.train_diffusion(model, self.train_loader, optimizer, scheduler,
                                     self.criterion, epoch, self.writer, self.device))
            logging.info(f"Epoch: {epoch + 1}  train_{selected_loss_name}: {diffusion_loss:.5f}  "
                         f"prior_loss: {prior_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                         f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}"
                         f"  eeg_rec_loss: {eeg_rec_loss:.5f}  eeg_valid_ratio: {eeg_valid_ratio:.5f}")
            # epoch-aligned train curve (Train/loss above is per-iteration; this is per-epoch
            # so it can be overlaid with Epoch/val_loss to read convergence/overfit).
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
                    model.save_ckpt(optimizer, best=True, epoch=(epoch+1), best_loss=best_validation_loss)

                model.save_ckpt(optimizer, epoch=(epoch + 1), best_loss=best_validation_loss)
                model.save_ckpt(optimizer, last=True, epoch=(epoch+1), best_loss=best_validation_loss)

    def train_diffusion(self, model, data_loader, optimizer, scheduler,
                        criterion, epoch, writer, device):
        whole_losses = AverageMeter()
        prior_losses = AverageMeter()
        au_rec_losses = AverageMeter()
        va_rec_losses = AverageMeter()
        em_rec_losses = AverageMeter()
        eeg_rec_losses = AverageMeter()
        eeg_valid_ratios = AverageMeter()

        if self.train_eeg_head_only:
            model.set_eeg_head_train_mode()
        else:
            model.train()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                listener_video_clip,
                listener_emotion_clip,
                listener_3dmm_clip,
                speaker_clip_length,
                listener_clip_length,
            ) = batch[:9]
            listener_eeg_clip = listener_eeg_mask = None
            if len(batch) > 9:
                listener_eeg_clip, listener_eeg_mask = batch[9:11]

            (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
             listener_video_clip, listener_emotion_clip, listener_3dmm_clip,
             past_listener_emotion, past_listener_3dmm, motion_lengths,
             listener_eeg_clip, listener_eeg_mask) = self.data_resample(
                    speaker_audio_clips=speaker_audio_clip, speaker_emotion_clips=speaker_emotion_clip,
                    speaker_3dmm_clips=speaker_3dmm_clip, listener_video_clips=listener_video_clip,
                    listener_emotion_clips=listener_emotion_clip, listener_3dmm_clips=listener_3dmm_clip,
                    speaker_seq_lengths=speaker_clip_length, listener_seq_lengths=listener_clip_length,
                    listener_eeg_clips=listener_eeg_clip, listener_eeg_masks=listener_eeg_mask)

            (speaker_audio_clip,  # (78-d)
             speaker_emotion_clip,  # (25-d)
             speaker_3dmm_clip,  # (58-d)
             listener_video_clip,
             listener_emotion_clip,  # (25-d)
             ) = (speaker_audio_clip.to(device),
                 speaker_emotion_clip.to(device),
                 speaker_3dmm_clip.to(device),
                 listener_video_clip.to(device),
                 listener_emotion_clip.to(device))
            if listener_eeg_clip is not None:
                listener_eeg_clip = listener_eeg_clip.to(device)
                listener_eeg_mask = listener_eeg_mask.to(device)
            batch_size = speaker_audio_clip.shape[0]

            # ---- scheduled sampling (online only): with prob ss_p, replace the GT past
            # listener window with the model's OWN 1-step x̂₀ prediction of that window, so
            # training matches the autoregressive inference distribution (closes exposure
            # bias). Window-A speaker = [zero history | concurrent frames] (= the test's
            # first-window condition); its GT target is exactly past_listener_emotion. ----
            past_for_B = past_listener_emotion
            ss_p = 0.0
            if (self.task == "online" and past_listener_emotion is not None
                    and self._as_bool(self.trainer_cfg.get("scheduled_sampling", False))):
                p_max = float(self.trainer_cfg.get("ss_p_max", 0.5))
                ramp = max(1, int(self.trainer_cfg.get("ss_ramp_epochs", self.epochs)))
                ss_p = p_max * min(1.0, epoch / ramp)
            if ss_p > 0.0:
                bs_a = speaker_audio_clip.shape[0]
                lw = int(self.trainer_cfg.window_size)       # single window length (e.g. 30)
                s_w_a = speaker_audio_clip.shape[1]

                def _winA(x):
                    hist = x.new_zeros(bs_a, s_w_a - lw, x.shape[-1])
                    return torch.cat([hist, x[:, :lw]], dim=1)

                with torch.no_grad():
                    out_A = model(
                        speaker_audio_input=_winA(speaker_audio_clip),
                        speaker_emotion_input=_winA(speaker_emotion_clip),
                        speaker_3dmm_input=_winA(speaker_3dmm_clip),
                        listener_emotion_input=past_listener_emotion[:, -lw:],   # newest past window only
                        listener_eeg_input=None, listener_eeg_mask=None,
                        past_listener_emotion=None,
                        motion_length=None,
                    )
                dec_A = out_A.get("output_decoder", out_A)
                x0_A = dec_A["prediction_emotion"].detach()                  # (bs, np, lw, 25)
                npred = x0_A.shape[1]
                gt_past_exp = past_listener_emotion.to(x0_A.device).repeat_interleave(npred, dim=0)  # (bs*np, K*lw, 25)
                # replace ONLY the newest window with the model's own x̂₀ (older windows stay GT)
                self_full = gt_past_exp.clone()
                self_full[:, -lw:, :] = x0_A.reshape(bs_a * npred, lw, x0_A.shape[-1])
                use_self = (torch.rand(bs_a, device=x0_A.device) < ss_p).repeat_interleave(npred)
                past_for_B = torch.where(use_self[:, None, None], self_full, gt_past_exp)

            outputs = model(
                speaker_audio_input=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                listener_eeg_input=listener_eeg_clip,
                listener_eeg_mask=listener_eeg_mask,
                past_listener_emotion=past_for_B,
                motion_length=motion_lengths,
            )
            # outputs['prediction_emotion'].shape: [bs, k, l_w, 25]
            # outputs['target_emotion'].shape: [bs, k, l_w, 25]

            output = criterion(outputs)
            loss = output["loss_eeg"] if self.train_eeg_head_only else output["loss"]
            if self.train_eeg_head_only and not loss.requires_grad:
                raise RuntimeError(
                    "loss_eeg has no gradient. Check that EEG labels are enabled and prediction_eeg is returned."
                )

            iteration = batch_idx + len(data_loader) * epoch
            if writer is not None:
                writer.add_scalar("Train/loss", loss.data.item(), iteration)
                writer.add_scalar("Train/loss_total", output["loss"].data.item(), iteration)
                writer.add_scalar("Train/loss_prior", output["loss_prior"].data.item(), iteration)
                writer.add_scalar("Train/loss_eeg", output["loss_eeg"].data.item(), iteration)
                writer.add_scalar("Train/eeg_valid_ratio", output["eeg_valid_ratio"].data.item(), iteration)
                # writer.add_scalar("Train/temporal_loss", temporal_loss.data.item(), iteration)

            whole_losses.update(loss.data.item(), batch_size)
            prior_losses.update(output["loss_prior"].data.item(), batch_size)
            au_rec_losses.update(output["loss_au"].data.item(), batch_size)
            va_rec_losses.update(output["loss_va"].data.item(), batch_size)
            em_rec_losses.update(output["loss_em"].data.item(), batch_size)
            eeg_rec_losses.update(output["loss_eeg"].data.item(), batch_size)
            eeg_valid_ratios.update(output["eeg_valid_ratio"].data.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None and (epoch + 1) >= 5:
            scheduler.step()
        lr = get_lr(optimizer=optimizer)
        if writer is not None:
            writer.add_scalar("Train/lr", lr, epoch)

        return (whole_losses.avg, prior_losses.avg, au_rec_losses.avg,
                va_rec_losses.avg, em_rec_losses.avg, eeg_rec_losses.avg,
                eeg_valid_ratios.avg)

    def val_diffusion(self, model, val_loader, criterion, device):
        whole_losses = AverageMeter()
        prior_losses = AverageMeter()
        au_rec_losses = AverageMeter()
        va_rec_losses = AverageMeter()
        em_rec_losses = AverageMeter()
        eeg_rec_losses = AverageMeter()
        eeg_valid_ratios = AverageMeter()

        model.eval()
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                listener_video_clip,
                listener_emotion_clip,
                listener_3dmm_clip,
                speaker_clip_length,
                listener_clip_length,
            ) = batch[:9]
            listener_eeg_clip = listener_eeg_mask = None
            if len(batch) > 9:
                listener_eeg_clip, listener_eeg_mask = batch[9:11]

            (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
             listener_video_clip, listener_emotion_clip, listener_3dmm_clip,
             past_listener_emotion, past_listener_3dmm, motion_lengths,
             listener_eeg_clip, listener_eeg_mask) = self.data_resample(
                    speaker_audio_clips=speaker_audio_clip, speaker_emotion_clips=speaker_emotion_clip,
                    speaker_3dmm_clips=speaker_3dmm_clip, listener_video_clips=listener_video_clip,
                    listener_emotion_clips=listener_emotion_clip, listener_3dmm_clips=listener_3dmm_clip,
                    speaker_seq_lengths=speaker_clip_length, listener_seq_lengths=listener_clip_length,
                    listener_eeg_clips=listener_eeg_clip, listener_eeg_masks=listener_eeg_mask)

            (speaker_audio_clip,  # (78-d)
             speaker_emotion_clip,  # (25-d)
             speaker_3dmm_clip,  # (58-d)
             listener_video_clip,
             listener_emotion_clip,  # (25-d)
             ) = (speaker_audio_clip.to(device),
                 speaker_emotion_clip.to(device),
                 speaker_3dmm_clip.to(device),
                 listener_video_clip.to(device),
                 listener_emotion_clip.to(device))
            if listener_eeg_clip is not None:
                listener_eeg_clip = listener_eeg_clip.to(device)
                listener_eeg_mask = listener_eeg_mask.to(device)
            batch_size = speaker_audio_clip.shape[0]

            with torch.no_grad():
                outputs = model(
                    speaker_audio_input=speaker_audio_clip,
                    speaker_emotion_input=speaker_emotion_clip,
                    speaker_3dmm_input=speaker_3dmm_clip,
                    listener_emotion_input=listener_emotion_clip,
                    listener_eeg_input=listener_eeg_clip,
                    listener_eeg_mask=listener_eeg_mask,
                    past_listener_emotion=past_listener_emotion,
                    motion_length=motion_lengths,
                )

                output = criterion(outputs)
                loss = output["loss_eeg"] if self.train_eeg_head_only else output["loss"]
            whole_losses.update(loss.data.item(), batch_size)
            prior_losses.update(output["loss_prior"].data.item(), batch_size)
            au_rec_losses.update(output["loss_au"].data.item(), batch_size)
            va_rec_losses.update(output["loss_va"].data.item(), batch_size)
            em_rec_losses.update(output["loss_em"].data.item(), batch_size)
            eeg_rec_losses.update(output["loss_eeg"].data.item(), batch_size)
            eeg_valid_ratios.update(output["eeg_valid_ratio"].data.item(), batch_size)

        return (whole_losses.avg, prior_losses.avg, au_rec_losses.avg,
                va_rec_losses.avg, em_rec_losses.avg, eeg_rec_losses.avg,
                eeg_valid_ratios.avg)

    @staticmethod
    def _eeg_targets_from_motion_lengths(listener_eeg, listener_eeg_mask, motion_lengths):
        if listener_eeg is None or listener_eeg.numel() == 0:
            return None, None
        if listener_eeg_mask is None or listener_eeg_mask.numel() == 0:
            listener_eeg_mask = torch.ones_like(listener_eeg)

        indices = []
        offset = 0
        total_length = listener_eeg.shape[0]
        for motion_length in motion_lengths:
            length = int(motion_length.item() if torch.is_tensor(motion_length) else motion_length)
            last_idx = min(max(offset + max(length, 1) - 1, 0), total_length - 1)
            indices.append(last_idx)
            offset += max(length, 0)
        if not indices:
            return None, None
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return (
            listener_eeg[index_tensor].unsqueeze(0).float(),
            listener_eeg_mask[index_tensor].unsqueeze(0).float(),
        )
    
    def _frrea_render_sample(self, renderer, latent_embedder, pred_listener_emotion,
                             listener_video_clips, sample_idx, fake_dir, real_dir,
                             stride, shard_idx, batch_idx):
        """Render one sample's generated (fake) + real listener frames for FRRea (FID).

        Renders only prediction #0. The full listener video is kept on CPU (test clips can
        be very long); only the reference frame and the subsampled real frames are moved to
        the GPU. Frame filenames are shard-unique so multiple GPU shards can write to the
        same fake/real directories without collisions.
        """
        import cv2
        lv = listener_video_clips[sample_idx]
        if isinstance(lv, (list, tuple)):
            lv = lv[0]
        if not torch.is_tensor(lv) or lv.numel() == 0:
            return
        reference = lv[0].to(self.device)  # (3, H, W); whole video stays on CPU

        emotion = pred_listener_emotion[0].to(self.device).float()  # (clip_len, 25)
        with torch.no_grad():
            coeff_3dmm = latent_embedder.decode_coeff(emotion)      # (clip_len, 58)

        fake_np, real_np = renderer.render_frames_for_fid(
            coeff_3dmm, reference, lv, fake_stride=stride)

        prefix = f"sh{shard_idx}_b{batch_idx}_s{sample_idx}"
        for i in range(fake_np.shape[0]):
            cv2.imwrite(os.path.join(fake_dir, f"{prefix}_f{i}.png"), fake_np[i])
        for i in range(real_np.shape[0]):
            cv2.imwrite(os.path.join(real_dir, f"{prefix}_f{i}.png"), real_np[i])

    def test(self):
        stage = "test"
        data_clamp = self.kwargs.pop("data_clamp")
        eval_eeg = self._as_bool(self.trainer_cfg.get("eval_eeg", False))
        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Test data module loaded")
        clip_len = self.trainer_cfg.clip_length
        w = self.trainer_cfg.window_size
        s_ratio = self.trainer_cfg.s_ratio
        s_w = s_ratio * w

        model = instantiate(self.model_cfg.diff_model,
                            stage=stage,
                            latent_embedder=self.model_cfg.latent_embedder \
                                if hasattr(self.model_cfg, "latent_embedder") else None,
                            audio_encoder=self.model_cfg.audio_encoder \
                                if hasattr(self.model_cfg, "audio_encoder") else None,
                            **self.kwargs,
                            _recursive_=False)
        model.to(self.device)
        model.eval()
        if eval_eeg:
            if getattr(model, "eeg_head", None) is None:
                raise RuntimeError(
                    "trainer.generic.eval_eeg=True but configs/<task-section>/model/motion_diffusion.yaml has no enabled eeg_head."
                )
            eeg_ckpt_path = model.get_ckpt_path(
                model.eeg_head,
                runid="resume_runid",
                epoch=None,
                best=True,
                last=False,
                create_dir=False,
            )
            if not os.path.exists(eeg_ckpt_path):
                raise FileNotFoundError(
                    "trainer.generic.eval_eeg=True requires a trained EEGPredictionHead checkpoint. "
                    f"Missing: {eeg_ckpt_path}"
                )

        logger.info("Loading post processor")
        post_processor = Processor(config_name=self.kwargs.pop("post_config_name"),
                                   clip_len_test=self.kwargs.pop("post_clip_length"),
                                   device=self.device,)
        logger.info("Post processor loaded")

        GT_listener_emotions_all = []
        pred_listener_emotions_all = []
        input_speaker_emotions_all = []
        GT_listener_eeg_all = []
        pred_listener_eeg_all = []
        listener_eeg_mask_all = []

        # ---- FRRea (FID) frame rendering setup (optional, gated by compute_frrea) ----
        compute_frrea = self._as_bool(self.kwargs.get("compute_frrea", False))
        frrea_renderer = frrea_latent_embedder = None
        frrea_fake_dir = frrea_real_dir = None
        frrea_stride = int(self.kwargs.get("frrea_stride", 30))
        frrea_shard_idx = int(os.environ.get("EVAL_SHARD_IDX", "0"))
        if compute_frrea:
            if self.renderer is None:
                raise RuntimeError("compute_frrea=True requires trainer.renderer config.")
            logger.info("Instantiating renderer for FRRea")
            frrea_renderer = instantiate(self.renderer, device=self.device)
            frrea_latent_embedder = model.diffusion_decoder.latent_embedder
            tag = os.environ.get("FRREA_TAG") or str(self.kwargs.get("resume_runid", "run"))
            base = os.path.join(to_absolute_path("frrea_frames"), str(self.task), tag)
            frrea_fake_dir = os.path.join(base, "fake")
            frrea_real_dir = os.path.join(base, "real")
            os.makedirs(frrea_fake_dir, exist_ok=True)
            os.makedirs(frrea_real_dir, exist_ok=True)
            logger.info(f"FRRea frames -> {base} (shard {frrea_shard_idx}, stride {frrea_stride})")

        for batch_idx, batch in enumerate(tqdm(test_loader)):
            (
                speaker_audio_clips,
                speaker_video_clips,
                speaker_emotion_clips,
                speaker_3dmm_clips,
                listener_video_clips,
                listener_emotion_clips,
                _,
                speaker_seq_lengths,
                listener_seq_lengths,
            ) = batch[:9]
            listener_eeg_clips = listener_eeg_masks = None
            if len(batch) > 9:
                listener_eeg_clips, listener_eeg_masks = batch[9:11]
            if eval_eeg and listener_eeg_clips is None:
                raise RuntimeError("trainer.generic.eval_eeg=True but the test dataloader did not return EEG labels.")

            # listener_emotion_clips: List: [[Tensor([l, d]), Tensor([l', d]), ...], ...]
            for em in listener_emotion_clips:
                GT_listener_emotions_all.append([em] if isinstance(em, torch.Tensor) else em)
            input_speaker_emotions_all.extend(speaker_emotion_clips)

            clip_batch_size = 8  # in case too long data sequence
            speaker_audios = []
            speaker_emotions = []
            speaker_3dmms = []
            motion_lengths = []
            sample_batch_size = []
            sample_eeg_targets = []
            sample_eeg_masks = []
            eeg_clips = listener_eeg_clips if eval_eeg else [None] * len(speaker_audio_clips)
            eeg_masks = listener_eeg_masks if eval_eeg else [None] * len(speaker_audio_clips)

            for (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
                 speaker_seq_length, listener_eeg_clip, listener_eeg_mask) in zip(
                    speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips,
                    speaker_seq_lengths, eeg_clips, eeg_masks):
                length = int(speaker_seq_length.item() if torch.is_tensor(speaker_seq_length) else speaker_seq_length)

                # Align all speaker clips to exactly `length` frames
                # (.npy feature files may have slightly different frame counts than the video)
                def _align_clip(clip, tgt_len):
                    if clip.dim() < 1 or clip.shape[0] == tgt_len:
                        return clip
                    if clip.shape[0] > tgt_len:
                        return clip[:tgt_len]
                    return torch.cat([clip, torch.zeros(tgt_len - clip.shape[0], clip.shape[-1])], dim=0)

                speaker_audio_clip   = _align_clip(speaker_audio_clip,   length)
                speaker_emotion_clip = _align_clip(speaker_emotion_clip, length)
                speaker_3dmm_clip    = _align_clip(speaker_3dmm_clip,    length)

                if self.task == "offline":
                    remain_length = length % clip_len
                    b = max(math.ceil(length / clip_len), 1)
                    final_length = remain_length if remain_length != 0 else clip_len
                    lengths = torch.tensor([clip_len] * (b - 1) + [final_length])
                    sample_batch_size.append(b)
                    pad_length = b * clip_len - length

                    speaker_audio_clip = torch.cat((speaker_audio_clip,
                                                    torch.zeros(
                                                        size=(pad_length, speaker_audio_clip.shape[-1]))),
                                                   dim=0)
                    speaker_audio_clip = rearrange(speaker_audio_clip, '(b l) d -> b l d', b=b)

                    speaker_emotion_clip = torch.cat((speaker_emotion_clip,
                                                      torch.zeros(size=(pad_length,
                                                                        speaker_emotion_clip.shape[-1]))), dim=0)
                    speaker_emotion_clip = rearrange(speaker_emotion_clip, '(b l) d -> b l d', b=b)

                    speaker_3dmm_clip = torch.cat((speaker_3dmm_clip,
                                                   torch.zeros(
                                                       size=(pad_length, speaker_3dmm_clip.shape[-1]))),
                                                  dim=0)
                    speaker_3dmm_clip = rearrange(speaker_3dmm_clip, '(b l) d -> b l d', b=b)

                    speaker_audios.append(speaker_audio_clip)
                    speaker_emotions.append(speaker_emotion_clip)
                    speaker_3dmms.append(speaker_3dmm_clip)
                    motion_lengths.append(lengths)
                    if eval_eeg:
                        eeg_target, eeg_mask = self._eeg_targets_from_motion_lengths(
                            listener_eeg_clip, listener_eeg_mask, lengths)
                        sample_eeg_targets.append(eeg_target)
                        sample_eeg_masks.append(eeg_mask)

                else:  # online task
                    num_windows = math.ceil(length / w)
                    sample_batch_size.append(num_windows)

                    speaker_audio_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_audio_clip.shape[-1])),
                         speaker_audio_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_audio_clip.shape[-1]))), dim=0)
                    speaker_emotion_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_emotion_clip.shape[-1])),
                         speaker_emotion_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_emotion_clip.shape[-1]))), dim=0)
                    speaker_3dmm_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_3dmm_clip.shape[-1])),
                         speaker_3dmm_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_3dmm_clip.shape[-1]))), dim=0)

                    motion_length_list = []
                    speaker_audio_clip_list = []
                    speaker_emotion_clip_list = []
                    speaker_3dmm_clip_list = []
                    for i in range(num_windows):
                        motion_length_list.append(w) if i < num_windows - 1 else motion_length_list.append(
                            length - i * w)
                        speaker_audio_clip_list.append(speaker_audio_clip[i*w: i*w + s_w])
                        speaker_emotion_clip_list.append(speaker_emotion_clip[i*w: i*w + s_w])
                        speaker_3dmm_clip_list.append(speaker_3dmm_clip[i*w: i*w + s_w])

                    motion_length = torch.tensor(motion_length_list)
                    speaker_audio_clip = torch.stack(speaker_audio_clip_list, dim=0)
                    speaker_emotion_clip = torch.stack(speaker_emotion_clip_list, dim=0)
                    speaker_3dmm_clip = torch.stack(speaker_3dmm_clip_list, dim=0)

                    motion_lengths.append(motion_length)
                    speaker_audios.append(speaker_audio_clip)
                    speaker_emotions.append(speaker_emotion_clip)
                    speaker_3dmms.append(speaker_3dmm_clip)
                    if eval_eeg:
                        eeg_target, eeg_mask = self._eeg_targets_from_motion_lengths(
                            listener_eeg_clip, listener_eeg_mask, motion_length)
                        sample_eeg_targets.append(eeg_target)
                        sample_eeg_masks.append(eeg_mask)

            motion_lengths = torch.cat(motion_lengths, dim=0)
            speaker_audios = torch.cat(speaker_audios, dim=0)
            speaker_emotions = torch.cat(speaker_emotions, dim=0)
            speaker_3dmms = torch.cat(speaker_3dmms, dim=0)
            sample_batch_size = torch.tensor(sample_batch_size)

            pred_listener_emotions = []
            pred_listener_eegs = []
            frrea_skip = False
            all_batch_size = speaker_audios.shape[0]

            if self.task == "online" and getattr(self, "online_autoregressive", True):
                # Autoregressive cross-window continuity: the listener windows of one
                # sample are generated sequentially, each conditioned on the PREVIOUS
                # window's own prediction (per-prediction past_listener). This restores
                # temporal continuity across the stitched sequence instead of generating
                # every 30-frame window independently. Window 0 has no past (None).
                win_bounds = torch.cat(
                    (torch.tensor([0]), torch.cumsum(sample_batch_size, dim=0)), dim=0)
                K = int(self.trainer_cfg.get("n_past_win", 1))
                lw_t = int(self.trainer_cfg.window_size)
                for si in range(len(sample_batch_size)):
                    a, b = int(win_bounds[si]), int(win_bounds[si + 1])
                    buf = []   # rolling buffer of own generated windows, each (num_preds, lw, 25)
                    for wi in range(a, b):
                        spk_a = speaker_audios[wi:wi + 1].to(self.device)
                        spk_e = speaker_emotions[wi:wi + 1].to(self.device)
                        spk_3 = speaker_3dmms[wi:wi + 1].to(self.device)
                        ml = motion_lengths[wi:wi + 1].to(self.device)
                        # past = last K generated windows, front-padded with zeros to K*lw
                        # (matches training); None for the very first window.
                        if len(buf) == 0:
                            past = None
                        else:
                            cat_buf = torch.cat(buf[-K:], dim=1)               # (np, k*lw, 25)
                            need = K * lw_t - cat_buf.shape[1]
                            if need > 0:
                                cat_buf = torch.cat(
                                    [cat_buf.new_zeros(cat_buf.shape[0], need, cat_buf.shape[2]), cat_buf], dim=1)
                            past = cat_buf
                        try:
                            with torch.no_grad():
                                outputs = model(
                                    speaker_audio_input=spk_a,
                                    speaker_emotion_input=spk_e,
                                    speaker_3dmm_input=spk_3,
                                    motion_length=ml,
                                    past_listener_emotion=past,
                                )
                        except RuntimeError as e:
                            if compute_frrea:
                                logger.warning(f"FRRea: skipping sample (batch {batch_idx}) due to: {e}")
                                torch.cuda.empty_cache()
                                frrea_skip = True
                                break
                            raise
                        pred = outputs["prediction_emotion"]       # (1, num_preds, l_w, 25)
                        pred_listener_emotions.append(pred.detach().cpu())
                        if eval_eeg:
                            if "prediction_eeg" not in outputs:
                                raise RuntimeError("trainer.generic.eval_eeg=True but the model did not return prediction_eeg.")
                            pred_listener_eegs.append(outputs["prediction_eeg"].detach().cpu())
                        buf.append(pred[0].detach())               # (num_preds, lw, 25)
                        if len(buf) > K:
                            buf.pop(0)
                    if frrea_skip:
                        break
            else:
                for i in range(math.ceil(all_batch_size / clip_batch_size)):
                    speaker_audio_clip = speaker_audios[i * clip_batch_size: (i + 1) * clip_batch_size]
                    speaker_emotion_clip = speaker_emotions[i * clip_batch_size: (i + 1) * clip_batch_size]
                    speaker_3dmm_clip = speaker_3dmms[i * clip_batch_size: (i + 1) * clip_batch_size]
                    motion_length = motion_lengths[i * clip_batch_size: (i + 1) * clip_batch_size]

                    (speaker_audio_clip,
                     speaker_emotion_clip,
                     speaker_3dmm_clip) = (
                        speaker_audio_clip.to(self.device),
                        speaker_emotion_clip.to(self.device),
                        speaker_3dmm_clip.to(self.device))
                    # speaker_audio_clip: (bsz, s_w, d_audio)
                    # speaker_emotion_clip: (bsz, s_w, d_emotion)
                    # speaker_3dmm_clip: (bsz, s_w, d_3dmm)

                    try:
                        with torch.no_grad():
                            outputs = model(
                                speaker_audio_input=speaker_audio_clip,
                                speaker_emotion_input=speaker_emotion_clip,
                                speaker_3dmm_input=speaker_3dmm_clip,
                                motion_length=motion_length,
                            )
                    except RuntimeError as e:
                        # A single pathological sample (e.g. a very long clip causing GPU OOM)
                        # should not abort the whole FRRea render run; skip it.
                        if compute_frrea:
                            logger.warning(f"FRRea: skipping sample (batch {batch_idx}) due to: {e}")
                            torch.cuda.empty_cache()
                            frrea_skip = True
                            break
                        raise

                    pred_listener_emotions.append(outputs["prediction_emotion"].detach().cpu())
                    if eval_eeg:
                        if "prediction_eeg" not in outputs:
                            raise RuntimeError("trainer.generic.eval_eeg=True but the model did not return prediction_eeg.")
                        pred_listener_eegs.append(outputs["prediction_eeg"].detach().cpu())
            if frrea_skip:
                torch.cuda.empty_cache()
                continue
            pred_listener_emotions = torch.cat(pred_listener_emotions, dim=0)  # (L', num_preds, l_w, 25)
            pred_listener_eegs = torch.cat(pred_listener_eegs, dim=0) if eval_eeg else None

            bounds = torch.cat((torch.tensor([0]), torch.cumsum(sample_batch_size, dim=0)), dim=0)
            intervals = list(zip(bounds[:-1], bounds[1:]))
            for sample_idx, (l, r) in enumerate(intervals):
                pred_listener_emotion = pred_listener_emotions[l:r]  # (b', num_preds, l_w, 25)
                motion_length = motion_lengths[l:r]
                clip_length = int(torch.sum(motion_length, dim=0, keepdim=False).item())
                pred_listener_emotion = rearrange(pred_listener_emotion,
                                                  'b n w d -> n (b w) d')[:, :clip_length]

                if data_clamp:
                    pred_listener_emotion[:, :, :15] = torch.round(pred_listener_emotion[:, :, :15])

                pred_listener_emotions_all.append(pred_listener_emotion)
                if compute_frrea:
                    self._frrea_render_sample(
                        frrea_renderer, frrea_latent_embedder, pred_listener_emotion,
                        listener_video_clips, sample_idx, frrea_fake_dir, frrea_real_dir,
                        frrea_stride, frrea_shard_idx, batch_idx)
                if eval_eeg:
                    pred_listener_eeg = rearrange(pred_listener_eegs[l:r], 'b n d -> n b d')
                    pred_listener_eeg_all.append(pred_listener_eeg)
                    GT_listener_eeg_all.append(sample_eeg_targets[sample_idx])
                    listener_eeg_mask_all.append(sample_eeg_masks[sample_idx])

        # pred_listener_emotions_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]
        # GT_listener_emotions_all
        # List: 750 [List: [(l', 25), (l'', 25), ...], List: [(l''', 25), (l'''', 25)], ...]
        if len(pred_listener_emotions_all):
            GT_listener_emotions_all = post_processor.forward(
                prediction_list=pred_listener_emotions_all,
                target_list=GT_listener_emotions_all,)
        # GT_listener_emotions_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]

        try:
            result_dict = {'GT': GT_listener_emotions_all, 'PRED': pred_listener_emotions_all}
            if eval_eeg:
                result_dict.update({
                    'GT_EEG': GT_listener_eeg_all,
                    'PRED_EEG': pred_listener_eeg_all,
                    'EEG_MASK': listener_eeg_mask_all,
                })
            torch.save(result_dict, f'results.pt')
            print("Successfully saved Tensor List")
        except Exception:
            print("Failed to save Tensor List")

        if compute_frrea:
            # FRRea render run uses num_preds=1, for which the per-sample emotion metrics
            # (e.g. S-MSE diversity) are undefined; skip them and only emit the frames.
            results = {}
        else:
            results = compute_metrics(
                input_speaker_emotions_all,
                pred_listener_emotions_all,
                GT_listener_emotions_all,
            )
            if eval_eeg:
                results.update(compute_eeg_metrics(
                    pred_listener_eeg_all,
                    GT_listener_eeg_all,
                    listener_eeg_mask_all,
                ))
            logger.info(results)
        if compute_frrea:
            logger.info(
                "FRRea frames written. After all shards finish, compute FID with:\n"
                f"  python -m framework.metrics.FID --fake {frrea_fake_dir} --real {frrea_real_dir}"
            )
