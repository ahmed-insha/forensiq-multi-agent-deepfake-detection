
"""
forensiq/data/image_degradation.py

Image-specific degradation presets, extending the project's existing
degradation module (previously built for video/audio via ffmpeg) to
standalone images. Uses the same preset-naming convention (clean,
mild_compress, heavy_compress, low_res, compound_severe) for direct
comparability with the Video and Audio agents' degradation evaluation
structure, closing a gap where the Image and Structural agents had not
previously been tested under degraded conditions.
"""
import io

from PIL import Image, ImageFilter


def apply_jpeg_compression(img, quality):
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resize_degradation(img, scale):
    """Downscale then upscale back to original size, simulating resolution loss."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


IMAGE_PRESETS = {
    "clean": lambda img: img,
    "mild_compress": lambda img: apply_jpeg_compression(img, quality=50),
    "heavy_compress": lambda img: apply_jpeg_compression(img, quality=15),
    "low_res": lambda img: apply_resize_degradation(img, scale=0.5),
    "compound_severe": lambda img: apply_jpeg_compression(
        apply_blur(apply_resize_degradation(img, scale=0.6), radius=1.5), quality=20
    ),
}


def apply_image_preset(img, preset_name):
    if preset_name not in IMAGE_PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Options: {list(IMAGE_PRESETS)}")
    return IMAGE_PRESETS[preset_name](img)
