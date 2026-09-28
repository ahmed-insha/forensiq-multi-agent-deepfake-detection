
"""
forensiq/data/datasets/build_synthetic_timeline_testset.py

Builds in-domain synthetic test clips with KNOWN, controlled timestamps
at which video and/or audio switch between real and fake.

FIX (see methodology log): the original version derived video and
audio switch points independently from each source's own natural
duration, then muxed with ffmpeg's -shortest flag. Since ASVspoof
audio clips are typically much shorter than FaceForensics video
segments, -shortest silently truncated most final clips before the
scheduled video switch point was ever reached, meaning most built
"test cases" never actually tested the video switch at all. This
version instead builds both tracks to an explicit TARGET_DURATION,
looping audio if needed to reach it, guaranteeing both switch points
actually occur within the final playable clip.
"""
import os
import subprocess

TARGET_DURATION = 10.0  # seconds; both video and audio are built to
                          # exactly this length, so switch points chosen
                          # within it are guaranteed to actually appear
                          # in the final output


def _run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")
    return result


def get_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


def trim_video(input_path, start, duration, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-t", str(duration),
           "-c:v", "libx264", "-an", output_path]
    _run(cmd)
    return output_path


def concat_videos(video_paths, output_path):
    list_path = output_path + "_list.txt"
    with open(list_path, "w") as f:
        for p in video_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
    _run(cmd)
    os.remove(list_path)
    return output_path


def loop_audio_to_duration(input_path, target_duration, output_path):
    """
    Loops an audio clip (repeating it) until it reaches at least
    target_duration, then trims to exactly that length. Needed because
    ASVspoof clips are typically much shorter than TARGET_DURATION / 2.
    """
    cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", input_path,
           "-t", str(target_duration), "-ar", "16000", "-ac", "1", output_path]
    _run(cmd)
    return output_path


def mux_video_audio(video_path, audio_path, output_path):
    # No reliance on -shortest for correctness here: both tracks are
    # already built to exactly TARGET_DURATION. -shortest is kept only
    # as a safety net for small rounding mismatches.
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path]
    _run(cmd)
    return output_path


def build_synthetic_test_clip(
    real_video_path, fake_video_path,
    real_audio_path, fake_audio_path,
    video_switch_frac,   # fraction (0-1) of TARGET_DURATION where video switches real->fake
    audio_switch_frac,   # fraction (0-1) of TARGET_DURATION where audio switches real->fake
    output_dir, sample_name,
):
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, f"_tmp_{sample_name}")
    os.makedirs(tmp_dir, exist_ok=True)

    video_switch_time = TARGET_DURATION * video_switch_frac
    audio_switch_time = TARGET_DURATION * audio_switch_frac

    source_video_duration = get_duration(real_video_path)
    usable_duration = min(TARGET_DURATION, source_video_duration)
    video_switch_time = min(video_switch_time, usable_duration - 0.5)

    real_seg = trim_video(real_video_path, 0, video_switch_time, f"{tmp_dir}/real_seg.mp4")
    fake_seg = trim_video(fake_video_path, video_switch_time, usable_duration - video_switch_time, f"{tmp_dir}/fake_seg.mp4")
    combined_video_raw = concat_videos([real_seg, fake_seg], f"{tmp_dir}/combined_video_raw.mp4")

    actual_video_duration = get_duration(combined_video_raw)
    if actual_video_duration < TARGET_DURATION - 0.5:
        _run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", combined_video_raw,
              "-t", str(TARGET_DURATION), "-c:v", "libx264", "-an", f"{tmp_dir}/combined_video.mp4"])
        combined_video = f"{tmp_dir}/combined_video.mp4"
    else:
        combined_video = combined_video_raw

    real_audio_looped = loop_audio_to_duration(real_audio_path, audio_switch_time, f"{tmp_dir}/real_audio_looped.wav")
    fake_audio_looped = loop_audio_to_duration(fake_audio_path, TARGET_DURATION - audio_switch_time, f"{tmp_dir}/fake_audio_looped.wav")

    list_path = f"{tmp_dir}/audio_list.txt"
    with open(list_path, "w") as f:
        f.write(f"file '{os.path.abspath(real_audio_looped)}'\n")
        f.write(f"file '{os.path.abspath(fake_audio_looped)}'\n")
    combined_audio = f"{tmp_dir}/combined_audio.wav"
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", combined_audio])

    final_path = os.path.join(output_dir, f"{sample_name}.mp4")
    mux_video_audio(combined_video, combined_audio, final_path)

    ground_truth = {
        "sample_name": sample_name,
        "target_duration": TARGET_DURATION,
        "video_switch_time": video_switch_time,
        "video_before_switch": "real", "video_after_switch": "fake",
        "audio_switch_time": audio_switch_time,
        "audio_before_switch": "real", "audio_after_switch": "fake",
        "output_path": final_path,
    }

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return ground_truth


def build_test_set(
    faceforensics_dir, asvspoof_flac_dir, good_audio_keys, audio_labels,
    output_dir, n_samples=10, compression="c23", seed=42,
):
    import random
    random.seed(seed)

    real_dir = os.path.join(faceforensics_dir, "manipulated_sequences", "Deepfakes", compression, "videos")
    original_dir = os.path.join(faceforensics_dir, "original_sequences", "youtube", compression, "videos")

    fake_files = os.listdir(real_dir)
    random.shuffle(fake_files)

    bonafide_keys = [k for k in good_audio_keys
                      if audio_labels.get(k) == 1
                      and os.path.exists(os.path.join(asvspoof_flac_dir, f"{k}.flac"))]
    spoof_keys = [k for k in good_audio_keys
                   if audio_labels.get(k) == 0
                   and os.path.exists(os.path.join(asvspoof_flac_dir, f"{k}.flac"))]
    random.shuffle(bonafide_keys)
    random.shuffle(spoof_keys)

    print(f"Locally available: {len(bonafide_keys)} bonafide, {len(spoof_keys)} spoof")

    ground_truths = []
    i = 0
    for fake_file in fake_files:
        if len(ground_truths) >= n_samples:
            break
        source_id = fake_file.split("_")[0]
        original_path = os.path.join(original_dir, f"{source_id}.mp4")
        fake_path = os.path.join(real_dir, fake_file)

        if not os.path.exists(original_path):
            continue
        if i >= len(bonafide_keys) or i >= len(spoof_keys):
            print(f"Stopping: exhausted available audio pairs at i={i}")
            break

        real_audio_path = os.path.join(asvspoof_flac_dir, f"{bonafide_keys[i]}.flac")
        fake_audio_path = os.path.join(asvspoof_flac_dir, f"{spoof_keys[i]}.flac")

        video_switch_frac = random.uniform(0.3, 0.7)
        audio_switch_frac = random.uniform(0.2, 0.8)

        try:
            gt = build_synthetic_test_clip(
                original_path, fake_path, real_audio_path, fake_audio_path,
                video_switch_frac, audio_switch_frac, output_dir, f"synth_{len(ground_truths):03d}",
            )
            ground_truths.append(gt)
            print(f"Built {gt['sample_name']}: video switches at {gt['video_switch_time']:.1f}s, "
                  f"audio switches at {gt['audio_switch_time']:.1f}s (target duration {TARGET_DURATION}s)")
        except Exception as e:
            print(f"[WARN] Failed to build sample from {fake_file}: {e}")

        i += 1

    return ground_truths
