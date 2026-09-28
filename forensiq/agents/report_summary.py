"""
forensiq/agents/report_summary.py
"""
import cv2


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def find_first_sustained_switch(timeline, prob_key, threshold=0.5, min_consecutive=2):
    consecutive = 0
    for b in timeline:
        if b[prob_key] is not None and b[prob_key] > threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return b["bin_start"] - (min_consecutive - 1) * (timeline[1]["bin_start"] - timeline[0]["bin_start"])
        else:
            consecutive = 0
    return None


def compute_fake_duration_verdict(timeline, duration, threshold=0.5):
    audio_bins = [b for b in timeline if b["audio_fake_prob"] is not None]
    video_bins = [b for b in timeline if b["video_fake_prob"] is not None]

    audio_fake_fraction = (sum(1 for b in audio_bins if b["audio_fake_prob"] > threshold) / len(audio_bins)) if audio_bins else None
    video_fake_fraction = (sum(1 for b in video_bins if b["video_fake_prob"] > threshold) / len(video_bins)) if video_bins else None

    return {
        "audio_fake_fraction": audio_fake_fraction,
        "video_fake_fraction": video_fake_fraction,
        "audio_switch_time": find_first_sustained_switch(timeline, "audio_fake_prob"),
        "video_switch_time": find_first_sustained_switch(timeline, "video_fake_prob"),
    }


def build_clear_summary(orchestrator_result, timeline=None, video_path=None, duration=None):
    summary = {"disagreement": orchestrator_result["disagreement_detected"]}

    if timeline and duration:
        fake_duration = compute_fake_duration_verdict(timeline, duration)
        summary.update(fake_duration)

        summary["summary_text"] = []
        if fake_duration["video_fake_fraction"] is not None:
            pct = fake_duration["video_fake_fraction"] * 100
            summary["summary_text"].append(f"Video: {pct:.0f}% of duration classified as manipulated"
                                             + (f", starting at {fake_duration['video_switch_time']:.1f}s" if fake_duration["video_switch_time"] is not None else ""))
        if fake_duration["audio_fake_fraction"] is not None:
            pct = fake_duration["audio_fake_fraction"] * 100
            summary["summary_text"].append(f"Audio: {pct:.0f}% of duration classified as manipulated"
                                             + (f", starting at {fake_duration['audio_switch_time']:.1f}s" if fake_duration["audio_switch_time"] is not None else ""))

        if fake_duration["audio_switch_time"] is not None:
            t = fake_duration["audio_switch_time"]
            summary["audio_evidence_window"] = {"start": t, "end": min(t + 1.0, duration)}
        if fake_duration["video_switch_time"] is not None:
            t = fake_duration["video_switch_time"]
            summary["video_evidence_window"] = {"start": t, "end": min(t + 1.0, duration)}

    # FIX: explicitly report the intraframe (splice/clone/inpaint)
    # score as its own finding, not only interframe duplication
    # candidates -- previously, "Structural findings:" could show
    # empty even when structural flagged fake, if that verdict came
    # from the intraframe branch rather than a duplication finding.
    structural = orchestrator_result.get("agent_summary", {}).get("structural")
    if structural:
        exp = structural.get("explanation_data", {})
        intraframe_score = exp.get("intraframe_score")
        summary["structural_intraframe_score"] = intraframe_score

        findings_text = []
        if intraframe_score is not None and intraframe_score > 0.5:
            findings_text.append(
                f"Intraframe analysis (splice/clone/inpaint localization): FAKE, confidence {intraframe_score*100:.1f}%"
            )

        interframe_findings = []
        candidates = exp.get("interframe_candidates", [])
        if candidates and video_path:
            fps = get_video_fps(video_path)
            for c in candidates:
                interframe_findings.append({"type": "frame_duplication",
                    "original_time": c["original_start"] / fps, "duplicate_time": c["duplicate_start"] / fps,
                    "similarity": c["mean_similarity"]})
                findings_text.append(
                    f"Interframe analysis: frame duplication detected, content at "
                    f"{c['original_start']/fps:.1f}s duplicated at {c['duplicate_start']/fps:.1f}s"
                )

        summary["structural_tampering_detected"] = len(findings_text) > 0
        summary["structural_findings"] = interframe_findings
        summary["structural_findings_text"] = findings_text

    return summary


def print_clear_summary(summary):
    print("Timeline-based verdict (not a single blended confidence):")
    for line in summary.get("summary_text", []):
        print(f"  {line}")
    print(f"Agent disagreement: {summary['disagreement']}")
    if "audio_evidence_window" in summary:
        w = summary["audio_evidence_window"]
        print(f"Audio evidence window: {w['start']:.1f}s-{w['end']:.1f}s")
    if "video_evidence_window" in summary:
        w = summary["video_evidence_window"]
        print(f"Video evidence window: {w['start']:.1f}s-{w['end']:.1f}s")
    print(f"Structural tampering detected: {summary.get('structural_tampering_detected', False)}")
    for line in summary.get("structural_findings_text", []):
        print(f"  - {line}")
