
"""
forensiq/models/frequency_detector.py

Frequency-domain AI-generated image detector, based on the established
finding that GAN/diffusion upsampling operations leave periodic,
grid-like artifacts in an image's frequency spectrum that are not
present in genuine camera-captured photographs (Frank et al., 2020,
"Leveraging frequency analysis for deep fake image recognition").

This is a classical signal-processing technique, not a trained neural
network -- it provides a genuinely different, complementary signal to
UnivFD's learned CLIP-feature-based detection, as the second "expert"
in the multi-detector fusion design (informed by AgentFoX and AIFo,
see methodology log).
"""
import numpy as np
from PIL import Image


def compute_frequency_score(image_path, patch_size=256):
    """
    Computes a "syntheticness" score based on high-frequency spectral
    periodicity. Real photographs have naturally decaying, roughly
    isotropic frequency spectra; GAN/diffusion upsampling introduces
    periodic peaks, especially in the high-frequency range, from the
    repeated convolutional upsampling operations used during
    generation.

    Returns a score roughly in [0, 1], where higher indicates stronger
    periodic artifacts (more likely AI-generated). This is a heuristic
    signal, not a calibrated probability -- it is combined with other
    detectors' outputs by the fusion/reasoning layer, not used alone.
    """
    img = Image.open(image_path).convert("L")  # grayscale; artifacts
                                                  # are present in the
                                                  # luminance channel

    # Center-crop to a fixed patch size for consistent frequency
    # resolution across images of different sizes.
    w, h = img.size
    left = max(0, (w - patch_size) // 2)
    top = max(0, (h - patch_size) // 2)
    img = img.crop((left, top, left + patch_size, top + patch_size))

    arr = np.array(img, dtype=np.float32)
    if arr.shape != (patch_size, patch_size):
        # Image smaller than patch_size in some dimension; pad.
        padded = np.zeros((patch_size, patch_size), dtype=np.float32)
        padded[:arr.shape[0], :arr.shape[1]] = arr
        arr = padded

    # 2D FFT, shifted so the zero-frequency (DC) component is centered.
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log1p(magnitude)  # log scale for stability

    # Focus on the high-frequency ring (outer portion of the spectrum),
    # where upsampling artifacts are most concentrated -- exclude the
    # low-frequency center, which is dominated by overall image
    # structure/content rather than generation artifacts.
    center = patch_size // 2
    y, x = np.ogrid[:patch_size, :patch_size]
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    high_freq_mask = dist_from_center > (patch_size * 0.3)

    high_freq_region = magnitude_log[high_freq_mask]

    # Periodicity signal: real photos have smoothly decaying spectra
    # (low variance in the high-frequency ring); synthetic images with
    # upsampling artifacts show sharp, localized peaks (high variance
    # relative to the mean).
    mean_energy = high_freq_region.mean()
    std_energy = high_freq_region.std()
    peakiness = std_energy / (mean_energy + 1e-6)

    # Normalize to roughly [0, 1] using an empirically reasonable
    # range; this is a heuristic calibration, not a trained threshold.
    score = np.clip((peakiness - 0.3) / 0.7, 0.0, 1.0)

    return float(score)
