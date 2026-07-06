"""Training/evaluation glue for G2P-Delta on the personalised MARS loader."""

import logging

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from framework.g2p_delta import G2PDeltaModel
from trainer.perfrdiff_rewrite_weight import Trainer as _PersonalisedTrainer
from utils.util import AverageMeter

logger = logging.getLogger(__name__)


def _ccc_loss(prediction, target, eps=1.0e-6):
    prediction = prediction.reshape(-1, prediction.shape[-2], prediction.shape[-1])
    target = target.reshape(-1, target.shape[-2], target.shape[-1])
    pred_mean = prediction.mean(dim=1)
    target_mean = target.mean(dim=1)
    pred_centered = prediction - pred_mean.unsqueeze(1)
    target_centered = target - target_mean.unsqueeze(1)
    covariance = (pred_centered * target_centered).mean(dim=1)
    pred_var = pred_centered.square().mean(dim=1)
    target_var = target_centered.square().mean(dim=1)
    ccc = 2.0 * covariance / (
        pred_var + target_var + (pred_mean - target_mean).square() + eps
    )
    return 1.0 - ccc.mean()


def _dynamics_loss(prediction, target):
    prediction = prediction.reshape(-1, prediction.shape[-2], prediction.shape[-1])
    target = target.reshape(-1, target.shape[-2], target.shape[-1])
    pred_velocity = prediction[:, 1:] - prediction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity = F.mse_loss(pred_velocity, target_velocity)
    if prediction.shape[1] < 3:
        return velocity
    pred_accel = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
    return velocity + 0.5 * F.mse_loss(pred_accel, target_accel)


def _coarse_loss(output_decoder):
    logits = output_decoder.get("coarse_logits")
    if logits is None:
        return output_decoder["prediction_emotion"].new_tensor(0.0)
    target = output_decoder["target_emotion"][..., 17:25]
    denom = target.sum(dim=-1, keepdim=True)
    valid = (denom > 0.5).to(logits.dtype)
    probabilities = target / denom.clamp_min(1.0e-6)
    cross_entropy = -(probabilities * F.log_softmax(logits, dim=-1)).sum(
        dim=-1, keepdim=True
    )
    return (cross_entropy * valid).sum() / valid.sum().clamp_min(1.0)


class Trainer(_PersonalisedTrainer):
    def _build_model(self, stage):
        diffusion = self._build_diffusion(stage)
        model = G2PDeltaModel(self._rewrite_cfg(), diffusion)
        model.to(self.device)
        return model

    def _build_optimizer(self, model):
        args = self.main_model_cfg.optimizer_hypernet.args
        adapter_params = list(model.modifier_parameters(include_eeg_head=False))
        groups = [{"params": adapter_params, "lr": float(args.lr)}]
        if self.train_eeg:
            model.set_eeg_head_requires_grad(True)
            groups.append(
                {
                    "params": list(model.eeg_head().parameters()),
                    "lr": float(self.main_model_cfg.args.get("eeg_lr", 1.0e-5)),
                }
            )
        return optim.AdamW(groups, weight_decay=float(args.weight_decay))

    def _batch_loss(
        self,
        model,
        criterion,
        speaker_audio,
        speaker_emotion,
        speaker_3dmm,
        listener_emotion,
        past_listener_emotion,
        motion_length,
        personal_3dmm,
        listener_personality,
        listener_eeg,
        listener_eeg_mask,
        counterfactual,
    ):
        input_dict = {
            "speaker_audio_input": speaker_audio,
            "speaker_emotion_input": speaker_emotion,
            "speaker_3dmm_input": speaker_3dmm,
            "listener_emotion_input": listener_emotion,
            "past_listener_emotion": past_listener_emotion,
            "motion_length": motion_length,
            "listener_eeg_input": listener_eeg,
            "listener_eeg_mask": listener_eeg_mask,
        }
        personal = personal_3dmm if personal_3dmm.numel() > 0 else None
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        outputs, regular = model(
            x=input_dict, p=personal, personality=listener_personality
        )
        output_prior, output_decoder = self._split_outputs(outputs)
        losses = criterion(output_prior, output_decoder)
        args = self.main_model_cfg.args
        coarse = _coarse_loss(output_decoder)
        ccc = _ccc_loss(
            output_decoder["prediction_emotion"], output_decoder["target_emotion"]
        )
        dynamics = _dynamics_loss(
            output_decoder["prediction_emotion"], output_decoder["target_emotion"]
        )
        total = (
            losses["loss"]
            + float(args.get("coarse_weight", 0.5)) * coarse
            + float(args.get("ccc_weight", 0.1)) * ccc
            + float(args.get("dynamics_weight", 0.02)) * dynamics
            + regular
        )

        cf_loss = total.new_tensor(0.0)
        cf_weight = float(args.get("counterfactual_weight", 0.1))
        if counterfactual and cf_weight > 0 and speaker_audio.shape[0] > 1:
            torch.random.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, self.device)
            negative_personal = (
                personal.roll(1, dims=0) if personal is not None else None
            )
            negative_personality = listener_personality.roll(1, dims=0)
            negative_outputs, _ = model(
                x=input_dict,
                p=negative_personal,
                personality=negative_personality,
            )
            _, negative_decoder = self._split_outputs(negative_outputs)
            negative_losses = criterion(output_prior, negative_decoder)
            margin = float(args.get("counterfactual_margin", 0.02))
            cf_loss = F.relu(
                margin + losses["decoded"] - negative_losses["decoded"]
            )
            total = total + cf_weight * cf_loss

        losses["loss"] = total
        losses["loss_coarse"] = coarse
        losses["loss_ccc"] = ccc
        losses["loss_dynamics"] = dynamics
        losses["loss_counterfactual"] = cf_loss
        return total, losses, regular

    def _run_epoch(self, model, data_loader, criterion, optimizer, writer, epoch, train=True):
        meters = [AverageMeter() for _ in range(6)]
        whole, prior, decoded, eeg, eeg_valid, regular_meter = meters
        model.train(train)
        if train and self.train_eeg:
            model.eeg_head().train()
        max_batches = int(self.trainer_cfg.get("max_train_batches", 0)) if train else int(
            self.trainer_cfg.get("max_val_batches", 0)
        )

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if len(batch) != 12:
                raise ValueError(
                    "G2P-Delta training requires the 12-item personalised batch "
                    "with personality and EEG tensors."
                )
            (
                speaker_audio,
                _,
                speaker_emotion,
                speaker_3dmm,
                _,
                listener_emotion,
                _,
                personal_3dmm,
                listener_personality,
                listener_eeg,
                listener_eeg_mask,
                _,
            ) = batch
            tensors = [
                speaker_audio,
                speaker_emotion,
                speaker_3dmm,
                listener_emotion,
                personal_3dmm,
                listener_personality,
                listener_eeg,
                listener_eeg_mask,
            ]
            (
                speaker_audio,
                speaker_emotion,
                speaker_3dmm,
                listener_emotion,
                personal_3dmm,
                listener_personality,
                listener_eeg,
                listener_eeg_mask,
            ) = [tensor.to(self.device) for tensor in tensors]
            (
                speaker_audio,
                speaker_emotion,
                speaker_3dmm,
                listener_emotion,
                past_listener_emotion,
                motion_length,
                listener_eeg,
                listener_eeg_mask,
            ) = self._resample_train_batch(
                speaker_audio,
                speaker_emotion,
                speaker_3dmm,
                listener_emotion,
                listener_eeg=listener_eeg,
                listener_eeg_mask=listener_eeg_mask,
            )

            # ---- scheduled sampling (online only; generic online_ss recipe): with prob
            # ss_p, replace the GT past window with the model's own 1-step x-hat-0 of that
            # window under the first-window condition (zero history + its concurrent
            # speaker frames), so training matches the autoregressive inference
            # distribution (closes exposure bias). One AR chain per parallel pred. ----
            ss_p = 0.0
            if (train and self.task == "online" and past_listener_emotion is not None
                    and bool(self.trainer_cfg.get("scheduled_sampling", False))):
                p_max = float(self.trainer_cfg.get("ss_p_max", 0.5))
                ramp = max(1, int(self.trainer_cfg.get("ss_ramp_epochs", self.trainer_cfg.epochs)))
                ss_p = p_max * min(1.0, epoch / ramp)
            if batch_idx == 0 and train and self.task == "online":
                logger.info("scheduled sampling: epoch=%s ss_p=%.4f", epoch, ss_p)
            if ss_p > 0.0:
                lw = int(self.trainer_cfg.window_size)

                def _win_a(x):
                    hist = x.new_zeros(x.shape[0], x.shape[1] - lw, x.shape[-1])
                    return torch.cat([hist, x[:, :lw]], dim=1)

                win_a_dict = {
                    "speaker_audio_input": _win_a(speaker_audio),
                    "speaker_emotion_input": _win_a(speaker_emotion),
                    "speaker_3dmm_input": _win_a(speaker_3dmm),
                    "listener_emotion_input": past_listener_emotion,
                    "past_listener_emotion": None,
                    "motion_length": None,
                    "listener_eeg_input": None,
                    "listener_eeg_mask": None,
                }
                personal_a = personal_3dmm if personal_3dmm.numel() > 0 else None
                with torch.no_grad():
                    outputs_a, _ = model(
                        x=win_a_dict, p=personal_a, personality=listener_personality
                    )
                _, decoder_a = self._split_outputs(outputs_a)
                x0_a = decoder_a["prediction_emotion"].detach()  # (bs, np, lw, d)
                bs_a, npred = x0_a.shape[0], x0_a.shape[1]
                gt_past = past_listener_emotion.repeat_interleave(npred, dim=0)
                self_past = x0_a.reshape(bs_a * npred, lw, x0_a.shape[-1])
                use_self = (torch.rand(bs_a, device=x0_a.device) < ss_p).repeat_interleave(npred)
                past_listener_emotion = torch.where(
                    use_self[:, None, None], self_past, gt_past
                )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                loss, loss_dict, regular = self._batch_loss(
                    model,
                    criterion,
                    speaker_audio,
                    speaker_emotion,
                    speaker_3dmm,
                    listener_emotion,
                    past_listener_emotion,
                    motion_length.to(self.device) if motion_length is not None else None,
                    personal_3dmm,
                    listener_personality,
                    listener_eeg,
                    listener_eeg_mask,
                    counterfactual=train,
                )
                if train:
                    loss.backward()
            if train:
                if self.trainer_cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                optimizer.step()

            batch_size = speaker_audio.shape[0]
            whole.update(loss.detach().item(), batch_size)
            prior.update(loss_dict["encoded"].detach().item(), batch_size)
            decoded.update(loss_dict["decoded"].detach().item(), batch_size)
            eeg.update(loss_dict["loss_eeg"].detach().item(), batch_size)
            eeg_valid.update(loss_dict["eeg_valid_ratio"].detach().item(), batch_size)
            regular_meter.update(regular.detach().item(), batch_size)
            if writer is not None:
                step = batch_idx + len(data_loader) * epoch
                prefix = "Train" if train else "Val"
                writer.add_scalar(f"{prefix}/loss", loss.detach().item(), step)
                writer.add_scalar(
                    f"{prefix}/loss_ccc", loss_dict["loss_ccc"].detach().item(), step
                )
                writer.add_scalar(
                    f"{prefix}/loss_counterfactual",
                    loss_dict["loss_counterfactual"].detach().item(),
                    step,
                )
        return whole.avg, prior.avg, decoded.avg, eeg.avg, eeg_valid.avg

    def _single_loss(
        self,
        model,
        criterion,
        speaker_audio,
        speaker_emotion,
        speaker_3dmm,
        listener_emotion,
        past_listener_emotion,
        motion_length,
        personal_3dmm,
        listener_personality,
        listener_eeg,
        listener_eeg_mask,
        idx,
    ):
        input_dict = {
            "speaker_audio_input": speaker_audio[idx : idx + 1],
            "speaker_emotion_input": speaker_emotion[idx : idx + 1],
            "speaker_3dmm_input": speaker_3dmm[idx : idx + 1],
            "listener_emotion_input": listener_emotion[idx : idx + 1],
            "past_listener_emotion": (
                past_listener_emotion[idx : idx + 1]
                if past_listener_emotion is not None
                else None
            ),
            "motion_length": (
                motion_length[idx : idx + 1].to(self.device)
                if motion_length is not None
                else None
            ),
            "listener_eeg_input": (
                listener_eeg[idx : idx + 1] if listener_eeg is not None else None
            ),
            "listener_eeg_mask": (
                listener_eeg_mask[idx : idx + 1]
                if listener_eeg_mask is not None
                else None
            ),
        }
        personal = personal_3dmm[idx : idx + 1] if personal_3dmm.numel() > 0 else None
        personality = listener_personality[idx : idx + 1]
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        outputs, regular = model(x=input_dict, p=personal, personality=personality)
        output_prior, output_decoder = self._split_outputs(outputs)
        losses = criterion(output_prior, output_decoder)

        args = self.main_model_cfg.args
        coarse = _coarse_loss(output_decoder)
        ccc = _ccc_loss(
            output_decoder["prediction_emotion"], output_decoder["target_emotion"]
        )
        dynamics = _dynamics_loss(
            output_decoder["prediction_emotion"], output_decoder["target_emotion"]
        )
        total = (
            losses["loss"]
            + float(args.get("coarse_weight", 0.5)) * coarse
            + float(args.get("ccc_weight", 0.1)) * ccc
            + float(args.get("dynamics_weight", 0.02)) * dynamics
            + regular
        )

        # Same-context listener swap. RNG is restored so matched and swapped
        # conditions see exactly the same diffusion noise.
        counterfactual = total.new_tensor(0.0)
        cf_weight = float(args.get("counterfactual_weight", 0.1))
        if (
            torch.is_grad_enabled()
            and cf_weight > 0
            and speaker_audio.shape[0] > 1
        ):
            negative_idx = (idx + 1) % speaker_audio.shape[0]
            negative_personal = (
                personal_3dmm[negative_idx : negative_idx + 1]
                if personal_3dmm.numel() > 0
                else None
            )
            negative_personality = listener_personality[
                negative_idx : negative_idx + 1
            ]
            torch.random.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, self.device)
            negative_outputs, _ = model(
                x=input_dict,
                p=negative_personal,
                personality=negative_personality,
            )
            _, negative_decoder = self._split_outputs(negative_outputs)
            negative_losses = criterion(output_prior, negative_decoder)
            margin = float(args.get("counterfactual_margin", 0.02))
            counterfactual = F.relu(
                margin + losses["decoded"] - negative_losses["decoded"]
            )
            total = total + cf_weight * counterfactual

        losses["loss"] = total
        losses["loss_coarse"] = coarse
        losses["loss_ccc"] = ccc
        losses["loss_dynamics"] = dynamics
        losses["loss_counterfactual"] = counterfactual
        return total, losses, regular

    def _apply_personalization(self, model, personal_3dmm, listener_personality):
        # Short-triage control: "identity" reproduces the frozen Generic
        # backbone exactly (no listener condition ever set); "matched" and
        # "shuffled" both flow through the normal set_person_condition path
        # and only differ in which personality/history the eval dataset
        # handed us (see scripts/build_subset_eval_root.py, which builds a
        # "shuffled" personality.csv variant so the swap happens at the data
        # layer, not here).
        mode = str(self.trainer_cfg.get("eval_condition_mode", "matched"))
        if mode == "identity":
            model.clear_person_condition()
            return
        if mode not in {"matched", "shuffled"}:
            raise ValueError(f"Unknown trainer.generic.eval_condition_mode: {mode}")
        personal = (
            personal_3dmm.unsqueeze(0).to(self.device)
            if personal_3dmm.numel() > 0
            else None
        )
        personality = listener_personality.unsqueeze(0).to(self.device)
        model.set_person_condition(p=personal, personality=personality)
