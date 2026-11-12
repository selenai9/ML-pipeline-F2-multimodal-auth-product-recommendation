"""
Image pipeline for the facial recognition model (Formative 2, Part 2).

Loading, augmentation and feature extraction. Used by the notebook and by the
command-line app, so both process images in exactly the same way.

Photos live in images/ and are named <member>_<expression>.jpg, for example
selena_neutral.jpg, selena_smiling.jpg, selena_surprised.jpg.
Files starting with "unauthorized" belong to people outside the team and are
only used to test that strangers are rejected.

Needs: pillow, numpy, pandas, scikit-image   (opencv-python is optional and
is only used to find the face; without it we crop the centre of the photo).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "images"
FEATURES_PATH = ROOT / "data" / "image_features.csv"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

FACE_SIZE = 128      # size of the cropped face used for display and augmentation
FEATURE_SIZE = 96    # size the face is shrunk to before measuring features
EXPRESSIONS = ("neutral", "smiling", "surprised")

try:  # face detector (OpenCV 4.x). If it is missing we use a centre crop.
    import cv2
    _CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    cv2, _CASCADE = None, None


# ------------------------------------------------------------------ loading
def parse_name(path):
    """selena_smiling.jpg -> ('selena', 'smiling')"""
    stem = Path(path).stem.lower()
    member, _, rest = stem.partition("_")
    if "smil" in rest:
        rest = "smiling"
    elif "surpris" in rest:
        rest = "surprised"
    elif "neutral" in rest:
        rest = "neutral"
    return member, rest or "unknown"


def list_images(include_unauthorized=False):
    """All photos in images/, sorted. Strangers are left out unless asked for."""
    paths = sorted(p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not include_unauthorized:
        paths = [p for p in paths if parse_name(p)[0] != "unauthorized"]
    return paths


def _crop_face(img):
    """Crop the largest face; fall back to the centre square of the photo."""
    if _CASCADE is not None:
        gray = np.array(img.convert("L"))
        faces = _CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.1 * w)   # a little margin around the face
            return img.crop((max(x - pad, 0), max(y - pad, 0),
                             x + w + pad, y + h + pad))
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def load_image(path):
    """Open a photo, fix its rotation, crop the face, resize to 128 x 128 RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")   # phones store rotation in EXIF
    img.thumbnail((640, 640))                           # keeps detection fast
    return _crop_face(img).resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)


# ------------------------------------------------------------- augmentation
def _rotate(img, angle):
    fill = tuple(int(v) for v in np.array(img).reshape(-1, 3).mean(axis=0))
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)


def augment(img):
    """The original plus 6 changed copies. Returns {name: PIL image}."""
    return {
        "original": img,
        "rotate_left": _rotate(img, 15),
        "rotate_right": _rotate(img, -15),
        "flip": ImageOps.mirror(img),
        "grayscale": img.convert("L").convert("RGB"),
        "brighter": ImageEnhance.Brightness(img).enhance(1.4),
        "darker": ImageEnhance.Brightness(img).enhance(0.6),
    }


# ----------------------------------------------------------------- features
def extract_features(img):
    """One face -> one row of numbers (948 values).

    900 HOG values: the pattern of edges (eyes, nose, mouth, jaw outline).
     48 colour values: a 16-bin histogram for each of red, green, blue.
    """
    small = img.resize((FEATURE_SIZE, FEATURE_SIZE), Image.LANCZOS)
    arr = np.array(small)
    gray = np.array(small.convert("L"))
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm="L2-Hys")
    hist = np.concatenate([np.histogram(arr[..., c], bins=16, range=(0, 256))[0]
                           for c in range(3)]).astype(float)
    hist /= hist.sum()
    return np.concatenate([hog_vec, hist])


def feature_names():
    return ([f"hog_{i}" for i in range(900)] +
            [f"hist_{c}_{i}" for c in "rgb" for i in range(16)])


def build_feature_table():
    """Every member photo, original and augmented -> one DataFrame.

    First three columns: member, sample_id, augmentation (same layout as
    audio_features.csv). The rest are the 948 feature columns.
    """
    rows = []
    for path in list_images():
        member, _ = parse_name(path)
        sample_id = path.stem.lower()
        for aug_name, aug_img in augment(load_image(path)).items():
            rows.append([member, sample_id, aug_name, *extract_features(aug_img)])
    return pd.DataFrame(rows, columns=["member", "sample_id", "augmentation"] + feature_names())
