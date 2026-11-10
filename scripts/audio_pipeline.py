"""Audio Processing Pipeline
Loads every member's voice clips from the audio folder, applies augmentations,
extracts features (MFCCs, spectral roll-off, energy and more) and saves them
to data/audio_features.csv. Run from anywhere:
    python scripts/audio_pipeline.py

File naming: audio/<member>_<phrase>.<ext>, for example audio/selena_yes_approve.mp4.
Files whose name starts with "unauthorized" are left out of the feature file,
because they are only used to test that the voice model rejects strangers.
"""

import subprocess
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# librosa 1.0 prints deprecation notices from inside its own time-stretch code
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "audio"
FEATURES_PATH = ROOT / "data" / "audio_features.csv"

SAMPLE_RATE = 16000          # 16 kHz is enough for speech and keeps files small
N_MFCC = 13
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".opus", ".webm"}
UNAUTHORIZED_PREFIX = "unauthorized"


# ---------------------------------------------------------------- loading

def _decode_with_ffmpeg(path):
    """Decode formats librosa cannot read on its own (m4a, mp4, aac) using the
    ffmpeg program that ships with the imageio-ffmpeg package."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def load_audio(path, trim=True):
    """Load a clip as mono 16 kHz audio, cut the silence at the start and end,
    and scale it so the loudest point is 1. This makes clips recorded on
    different phones comparable."""
    path = Path(path)
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    else:
        y = _decode_with_ffmpeg(path)
    if trim:
        y, _ = librosa.effects.trim(y, top_db=30)
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 0:
        y = y / peak
    return y


def parse_name(path):
    """'selena_yes_approve.mp4' -> ('selena', 'yes_approve')."""
    member, _, phrase = Path(path).stem.lower().partition("_")
    return member, phrase


def list_clips(include_unauthorized=False):
    """All audio files in the audio folder, sorted by name."""
    clips = []
    for p in sorted(AUDIO_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        is_unauth = p.stem.lower().startswith(UNAUTHORIZED_PREFIX)
        if is_unauth and not include_unauthorized:
            continue
        clips.append(p)
    return clips


# ----------------------------------------------------------- augmentation

def add_noise(y, snr_db=20, seed=0):
    """Add background hiss at a set signal-to-noise ratio (20 dB = clearly
    audible but the voice is still easy to hear)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return y + rng.normal(0, np.sqrt(noise_power), len(y)).astype(np.float32)


def augment(y, sr=SAMPLE_RATE):
    """Return the original clip plus five changed copies. Each copy imitates
    something that happens in real use: a slightly higher or lower voice
    (a cold, tiredness), speaking faster or slower, and a noisy room."""
    return {
        "original": y,
        "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5),
        "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5),
        "time_fast": librosa.effects.time_stretch(y, rate=1.2),
        "time_slow": librosa.effects.time_stretch(y, rate=0.8),
        "noise": add_noise(y, snr_db=20),
    }


# ------------------------------------------------------- feature extraction

def extract_features(y, sr=SAMPLE_RATE):
    """Turn one clip into a fixed-length row of numbers.
    - MFCCs (13): the shape of the voice's sound spectrum, the main voiceprint cue
    - spectral roll-off: the frequency below which 85% of the energy sits
    - RMS energy: how loud the voice is over time
    - spectral centroid: the 'brightness' of the voice
    - zero-crossing rate: how noisy or hissy the sound is
    Each is measured over short frames, then summarised by its mean and
    standard deviation across the clip."""
    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    frame_features = {
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0],
        "rms_energy": librosa.feature.rms(y=y)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
    }
    for name, values in frame_features.items():
        feats[f"{name}_mean"] = float(np.mean(values))
        feats[f"{name}_std"] = float(np.std(values))

    return feats


def features_for_file(path):
    """Features for one clip with no augmentation, used by the command-line app."""
    return extract_features(load_audio(path))


def build_feature_table():
    """Load every member clip, augment it, extract features and return one
    row per (clip, augmentation)."""
    rows = []
    for path in list_clips():
        member, _ = parse_name(path)
        y = load_audio(path)
        for aug_name, y_aug in augment(y).items():
            row = {"member": member, "sample_id": path.stem.lower(), "augmentation": aug_name}
            row.update(extract_features(y_aug))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    clips = list_clips()
    if not clips:
        raise SystemExit(f"No audio clips found in {AUDIO_DIR}")
    print(f"Found {len(clips)} clips:")
    for p in clips:
        print("  ", p.name)

    table = build_feature_table()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved {table.shape[0]} rows x {table.shape[1]} columns to {FEATURES_PATH.relative_to(ROOT)}")
    print("Rows per member:")
    print(table["member"].value_counts().to_string())


if __name__ == "__main__":
    main()
