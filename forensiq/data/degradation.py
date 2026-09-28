"""
Reusable video/audio degradation module for ForensiQ.
"""
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DegradationConfig:
    crf: Optional[int] = None
    resize_scale: Optional[float] = None
    blur_strength: Optional[int] = None
    noise_strength: Optional[int] = None
    target_fps: Optional[float] = None
    audio_bitrate_k: Optional[int] = None
    audio_lowpass_hz: Optional[int] = None


def _run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1500:]}")
    return result


def degrade_video(input_path, output_path, config: DegradationConfig):
    input_path, output_path = str(input_path), str(output_path)
    vf = []
    if config.resize_scale:
        vf.append(f"scale=trunc(iw*{config.resize_scale}/2)*2:trunc(ih*{config.resize_scale}/2)*2")
    if config.blur_strength:
        vf.append(f"boxblur={config.blur_strength}:1")
    if config.noise_strength:
        vf.append(f"noise=alls={config.noise_strength}:allf=t+u")
    if config.target_fps:
        vf.append(f"fps={config.target_fps}")

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-crf", str(config.crf if config.crf is not None else 23)]

    af = []
    if config.audio_lowpass_hz:
        af.append(f"lowpass=f={config.audio_lowpass_hz}")
    if af:
        cmd += ["-af", ",".join(af)]
    if config.audio_bitrate_k:
        cmd += ["-c:a", "aac", "-b:a", f"{config.audio_bitrate_k}k"]
    else:
        cmd += ["-c:a", "copy"]

    cmd += [output_path]
    _run(cmd)
    return output_path


def degrade_audio(input_path, output_path, config: DegradationConfig):
    """
    Bitrate-based degradation requires transcoding through an actual
    lossy codec (mp3), then decoding back to a soundfile-readable
    format. FLAC is lossless, so ffmpeg cannot apply a bitrate target
    directly to a FLAC output -- this previously failed silently for
    every preset with audio_bitrate_k set.
    """
    input_path, output_path = str(input_path), str(output_path)
    af = []
    if config.audio_lowpass_hz:
        af.append(f"lowpass=f={config.audio_lowpass_hz}")

    if config.audio_bitrate_k:
        tmp_lossy = str(Path(output_path).with_suffix(".mp3"))
        cmd1 = ["ffmpeg", "-y", "-i", input_path]
        if af:
            cmd1 += ["-af", ",".join(af)]
        cmd1 += ["-c:a", "libmp3lame", "-b:a", f"{config.audio_bitrate_k}k", tmp_lossy]
        _run(cmd1)

        cmd2 = ["ffmpeg", "-y", "-i", tmp_lossy, output_path]
        _run(cmd2)
        os.remove(tmp_lossy)
    else:
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += [output_path]
        _run(cmd)

    return output_path


PRESETS = {
    "clean":            DegradationConfig(),
    "mild_compress":    DegradationConfig(crf=28, audio_bitrate_k=96),
    "heavy_compress":   DegradationConfig(crf=40, audio_bitrate_k=32),
    "low_res":          DegradationConfig(resize_scale=0.5, crf=30),
    "social_media":     DegradationConfig(crf=32, resize_scale=0.75, target_fps=24, audio_bitrate_k=64),
    "noisy_blurry":     DegradationConfig(blur_strength=2, noise_strength=15, crf=28),
    "phone_call_audio": DegradationConfig(audio_bitrate_k=24, audio_lowpass_hz=3400),
    "compound_severe":  DegradationConfig(crf=36, resize_scale=0.6, blur_strength=1, noise_strength=10, target_fps=20, audio_bitrate_k=48),
}


def apply_preset(input_path, output_dir, preset_name, is_audio=False):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Options: {list(PRESETS)}")
    config = PRESETS[preset_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(input_path).suffix
    out_path = output_dir / f"{Path(input_path).stem}__{preset_name}{ext}"
    fn = degrade_audio if is_audio else degrade_video
    return fn(input_path, out_path, config)
