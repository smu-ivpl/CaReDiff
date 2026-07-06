"""FRRea metric: Frechet Inception Distance (FID) over rendered facial reaction frames.

Realism of generated facial reaction video clips is assessed with FID (denoted FRRea):
features are extracted from generated ("fake") and real listener frames with a
pretrained InceptionV3, then the Frechet distance between the two Gaussians is computed.

FID is a distribution-level metric (mean + covariance over the whole frame set), so it
is NOT a per-sample average and cannot be merged across data shards by averaging. When
evaluation is sharded across GPUs, each shard should only dump its frames to a shared
directory; FID is then computed once over all frames with this module.

Usage (standalone, over two image directories):
    python -m framework.metrics.FID --fake <fake_dir> --real <real_dir>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

try:
    import cv2
except ImportError:
    cv2 = None

from torchvision.models import inception_v3, Inception_V3_Weights

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


class InceptionFeatures(nn.Module):
    """InceptionV3 pool3 (2048-d) feature extractor, standard for FID."""

    def __init__(self, device='cpu'):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           aux_logits=True, transform_input=True)
        net.fc = nn.Identity()  # output the 2048-d pooled features
        net.eval()
        self.net = net.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 3, H, W) float in [0, 1]
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        feat = self.net(x)
        if isinstance(feat, tuple):  # eval() should disable aux, but be safe
            feat = feat[0]
        return feat


def _list_images(directory):
    files = []
    for root, _, names in os.walk(directory):
        for n in sorted(names):
            if n.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, n))
    return sorted(files)


def _load_batch(paths):
    """Load images as RGB float tensor in [0, 1]. Frames are saved by cv2 (BGR)."""
    imgs = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    if not imgs:
        return None
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def extract_features_from_dir(directory, extractor, batch_size=64):
    paths = _list_images(directory)
    if not paths:
        raise ValueError(f"No images found in {directory}")
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = _load_batch(paths[i:i + batch_size])
        if batch is None:
            continue
        batch = batch.to(extractor.device)
        feats.append(extractor(batch).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    return feats, len(paths)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # add small jitter to the diagonal to make the product positive definite
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _stats(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def compute_fid_from_dirs(fake_dir, real_dir, device=None, batch_size=64):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = InceptionFeatures(device=device)
    fake_feats, n_fake = extract_features_from_dir(fake_dir, extractor, batch_size)
    real_feats, n_real = extract_features_from_dir(real_dir, extractor, batch_size)
    mu_f, sig_f = _stats(fake_feats)
    mu_r, sig_r = _stats(real_feats)
    fid = frechet_distance(mu_f, sig_f, mu_r, sig_r)
    return fid, n_fake, n_real


def main():
    ap = argparse.ArgumentParser(description="Compute FRRea (FID) between fake and real frame dirs.")
    ap.add_argument('--fake', required=True, help='directory of generated (fake) frames')
    ap.add_argument('--real', required=True, help='directory of real listener frames')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    if cv2 is None:
        raise ImportError("opencv (cv2) is required to read frames for FID.")

    fid, n_fake, n_real = compute_fid_from_dirs(args.fake, args.real, args.device, args.batch_size)
    print(f"fake frames: {n_fake}, real frames: {n_real}")
    print(f"FRRea (FID) = {fid:.6f}")


if __name__ == '__main__':
    main()
