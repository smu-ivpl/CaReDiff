from .FaceVerseModel import FaceVerseModel
import numpy as np

def get_faceverse(**kargs):
    # Use the caller-provided absolute path when available; the hardcoded relative path
    # breaks once Hydra changes the working directory to the run output dir.
    path = kargs.pop('path', 'external/FaceVerse/data/faceverse_simple_v2.npy')
    faceverse_dict = np.load(path, allow_pickle=True).item()
    faceverse_model = FaceVerseModel(faceverse_dict, **kargs)
    return faceverse_model, faceverse_dict



