"""Velocity-space variant of the PerFRDiff latent matcher.

Only the *decoder* diffusion is reparametrised to operate in frame-delta
(velocity) space; the diffusion prior, EEG head and everything else are left
exactly as in the baseline ``matchers.py`` (the original file is untouched).

Rationale: the listener reaction is modelled by predicting frame-to-frame change
(velocity) rather than absolute emotion values. In CCC terms (the FRC metric) this
directly pushes the temporal-correlation term and discourages variance collapse,
and it tends to yield smoother, more natural reactions — analogous to predicting
epsilon/v instead of x0 in standard diffusion.

The only behavioural change lives in ``VelocityDecoderLatentMatcher._forward``:
  * training: the diffusion target ``x_start`` becomes ``to_delta(listener_emotion)``
    so the network is supervised on velocity (the existing MSE loss then compares
    predicted vs. ground-truth deltas, no loss-file change required);
  * inference: the sampled velocity is integrated back with ``from_delta`` (cumsum)
    before being returned as the emotion sequence, so the rest of the pipeline
    (post-processing, metrics, rendering) sees ordinary emotion values.

``from_delta`` is applied per generated chunk (full sequence for the offline task,
per 30-frame window for online), each anchored by its own learned first frame.
"""

import torch
from einops import rearrange

from framework.motion_diffusion.diffusion.matchers import LatentMatcher, DecoderLatentMatcher
from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser import lengths_to_mask
from framework.motion_diffusion.diffusion.velocity_transform import to_delta, from_delta


class VelocityDecoderLatentMatcher(DecoderLatentMatcher):
    """Decoder matcher whose diffusion data space is the frame-delta sequence.

    Identical to :class:`DecoderLatentMatcher` except for the two velocity
    transforms marked below; carries no extra state, so the parent can be
    promoted to this class in-place (see :class:`VelocityLatentMatcher`).
    """

    def _forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
    ):
        with torch.no_grad():
            s_audio_encodings = self.audio_encoder._encode(speaker_audio_input)
            s_audio_encodings = s_audio_encodings.repeat_interleave(self.num_preds, dim=0)

            # freeze latent RNN_VAE embedder to extract speaker latent embedding
            s_latent_embed = self.latent_embedder.encode(speaker_emotion_input).unsqueeze(1)
            s_latent_embed = s_latent_embed.repeat_interleave(self.num_preds, dim=0)

            s_3dmm_encodings = speaker_3dmm_input.repeat_interleave(self.num_preds, dim=0)

            s_emotion_encodings = speaker_emotion_input.repeat_interleave(self.num_preds, dim=0)

            past_listener_emotion = past_listener_emotion.repeat_interleave(
                self.num_preds, dim=0) if past_listener_emotion is not None else None

            motion_length = motion_length.repeat_interleave(
                self.num_preds, dim=0) if motion_length is not None else None

            model_kwargs = {
                "speaker_audio_encodings": s_audio_encodings,
                "speaker_latent_embed": s_latent_embed,
                "speaker_3dmm_encodings": s_3dmm_encodings,
                "speaker_emotion_encodings": s_emotion_encodings,
                "past_listener_emotion": past_listener_emotion,
                "motion_length": motion_length,
            }

        if self.stage == "test":
            bs, l, _ = s_audio_encodings.shape  # bz * num_preds
            with torch.no_grad():
                output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                    matcher=self,
                    model=self.model,
                    model_kwargs=model_kwargs,
                    shape=(bs, self.window_size if self.task == "online" else l, self.emotion_dim),
                )][-1]  # get last output

            # --- velocity -> emotion: the diffusion sampled deltas; integrate over time. ---
            output_listener_emotion = from_delta(output["sample_enc"])  # (bz * num_preds, w, d=25)
            output_listener_emotion = rearrange(output_listener_emotion,
                                                "(b n) w d -> b n w d", n=self.num_preds)
            output_whole = {"prediction_emotion": output_listener_emotion}

        else:
            listener_emotion_input = listener_emotion_input.repeat_interleave(self.num_preds, dim=0)
            # --- emotion -> velocity: supervise the diffusion on frame deltas. ---
            x_start_selected = to_delta(listener_emotion_input)  # (bs * num_preds, l_w, ...)

            t, _ = self.schedule_sampler.sample(x_start_selected.shape[0], x_start_selected.device)
            timesteps = t.long()

            output_whole = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                          model_kwargs=model_kwargs)
            if motion_length is not None:  # offline task zero masking
                device = x_start_selected.get_device()
                output_mask = lengths_to_mask(motion_length, device=device, max_len=x_start_selected.shape[1])
                output_whole["prediction_emotion"] = (output_whole["prediction_emotion"]
                                                      * output_mask.float().unsqueeze(-1))

            output_whole = {k: v.view(-1, self.num_preds, *output_whole[k].shape[1:]) for k, v in output_whole.items()}
        return output_whole


class VelocityLatentMatcher(LatentMatcher):
    """``LatentMatcher`` whose decoder diffusion runs in velocity (frame-delta) space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The velocity decoder only overrides ``_forward`` and introduces no extra
        # state, so promoting the already-built (and, for test/resume, already
        # checkpoint-loaded) decoder matcher in place is safe and keeps the base
        # init — including weight loading — untouched.
        self.diffusion_decoder.__class__ = VelocityDecoderLatentMatcher
