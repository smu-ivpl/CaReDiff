"""
Audio feature extraction for test split.
Same logic as audio_feature_extraction.py, but:
  - uses video-face-crop instead of video-raw (test set has no video-raw)
  - bypasses audio_separator (not installed); uses wav2vec directly
"""
import argparse
import math
import os
import traceback

import cv2
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from framework.feature_extractor.wav2vec import Wav2VecModel


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_encoder = Wav2VecModel.from_pretrained(
        args.wav2vec_model_path, local_files_only=True
    ).to(device)
    audio_encoder.eval()
    audio_encoder.feature_extractor._freeze_parameters()

    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.wav2vec_model_path, local_files_only=True
    )
    print("Wav2Vec model loaded.")

    video_paths = []
    input_paths = []
    output_paths = []

    for root, _, files in os.walk(args.root_dir):
        for file in sorted(files):
            if not file.endswith(".wav"):
                continue

            input_paths.append(os.path.join(root, file))

            # replace only the first occurrence to avoid corrupt paths
            output_dir = root.replace("/audio/", "/audio-features/", 1)
            os.makedirs(output_dir, exist_ok=True)
            output_paths.append(os.path.join(output_dir, file.replace(".wav", ".npy")))

            video_dir = root.replace("/audio/", "/video-face-crop/", 1)
            video_paths.append(os.path.join(video_dir, file.replace(".wav", ".mp4")))

    print(f"Found {len(input_paths)} audio files")

    saved = 0
    skipped = 0
    for input_path, output_path, video_path in tqdm(
        zip(input_paths, output_paths, video_paths), total=len(input_paths)
    ):
        if os.path.exists(output_path):
            skipped += 1
            continue

        try:
            # get frame count from video-face-crop
            cap = cv2.VideoCapture(video_path)
            seq_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # fallback: compute from audio duration if OpenCV returns 0
            if seq_len == 0:
                print(f"Warning: OpenCV returned 0 frames for {video_path}, computing from audio")
                y_tmp, _ = librosa.load(input_path, sr=args.sample_rate, duration=1.0)
                duration = librosa.get_duration(path=input_path)
                seq_len = math.ceil(duration * args.fps)

            if seq_len == 0:
                print(f"Error: could not determine seq_len for {input_path}, skipping")
                continue

            # load audio
            speech_array, sampling_rate = librosa.load(input_path, sr=args.sample_rate)
            audio_feature = np.squeeze(
                wav2vec_feature_extractor(
                    speech_array, sampling_rate=sampling_rate
                ).input_values
            )

            audio_feature_t = torch.from_numpy(audio_feature).float().unsqueeze(0).to(device)

            with torch.no_grad():
                embeddings = audio_encoder(
                    audio_feature_t, seq_len=seq_len, output_hidden_states=True
                )

            audio_emb = embeddings.last_hidden_state.squeeze().cpu().detach()  # tensor

            np.save(output_path, audio_emb)
            saved += 1

        except Exception:
            print(f"Error processing {input_path}:")
            traceback.print_exc()

    print(f"Done. Saved: {saved}, Skipped (already exist): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Feature Extraction for Test Split")
    parser.add_argument("--root_dir", type=str,
                        default="/mnt/HDD1/MARS/test/audio/speaker")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--wav2vec_model_path", type=str,
                        default="./pretrained_models/wav2vec/wav2vec2-base-960h")
    args = parser.parse_args()
    main(args)
