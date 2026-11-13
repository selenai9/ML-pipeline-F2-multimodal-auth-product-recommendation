from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets Pillow open .heic files

images_dir = Path(__file__).resolve().parent.parent / "images"
heic_dir = images_dir / "original_heic"   # the phone's original HEIC files are kept here

# old file name -> new file name
renames = {
    "Neutral_celine.heic": "celine_neutral.jpg",
    "Smiling_celine.HEIC": "celine_smiling.jpg",
    "Shocked_Celine.HEIC": "celine_surprised.jpg",
}

for old, new in renames.items():
    img = Image.open(heic_dir / old)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.save(images_dir / new, quality=95)
    print("Saved", new, img.size)