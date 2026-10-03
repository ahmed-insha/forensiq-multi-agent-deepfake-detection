# forensiq-multi-agent-deepfake-detection
Multi-agent forensic media analysis system for detecting AI-generated and manipulated audio, video, and image content, with a focus on real-world degradation robustness.

# ForensiQ

A multi-agent forensic media analysis system for detecting AI-generated and manipulated audio, video, and image content, with a focus on how detection performance changes when media quality is reduced.

## Overview

ForensiQ combines four specialist detection agents, coordinated through a LangGraph-based orchestrator:

* **Audio Agent**: Wav2Vec 2.0-based detection of synthetic speech and voice spoofing
* **Video Agent**: EfficientNet-B0-based detection of face-swap and reenactment deepfakes
* **Structural Agent**: MVSS-Net-based detection of splice, clone, and inpaint manipulation, along with a custom frame-duplication detector
* **Image Agent**: A two-detector approach combining a calibrated CLIP-based classifier (UnivFD) with independent LLM visual reasoning for detecting fully AI-generated images

The orchestrator combines the results using confidence-weighted fusion, reports disagreements between agents, and flags low-confidence cases as inconclusive for human review instead of forcing a final verdict.

## Key Findings

* Degradation-aware fine-tuning improved the Audio Agent's performance across all tested degradation conditions.
* The same approach did not improve the Video Agent. A validation-split leakage issue was identified as the reason and was documented rather than hidden.
* The Image Agent's two-detector fusion improved accuracy by around 17 percentage points and was reproduced on two independently sourced datasets.
* Cross-dataset testing using FakeAVCeleb and DigiFakeAV showed the real-world generalisation gap that motivated this project, using the models trained as part of ForensiQ.

Full methodology, results, and limitations are documented in the dissertation report.

## Repository Structure

```text
forensiq-multi-agent-deepfake-detection/
├── app.py                                   Streamlit application entry point
├── requirements.txt
├── Dockerfile
├── README.md
├── LICENSE
├── .gitignore
│
├── forensiq/
│   ├── __init__.py
│   ├── config.py                            Paths, constants shared across the project
│   │
│   ├── agents/
│   │   ├── agents.py                        Unified per-agent interface (run_audio_agent, run_video_agent, etc.)
│   │   ├── orchestrator.py                  LangGraph orchestration + fusion logic
│   │   ├── image_fusion.py                  Calibrated UnivFD + Gemini fusion (validated, +17pp)
│   │   ├── video_fusion.py                  Video fusion re-test (tested, not integrated)
│   │   ├── dual_video_agent.py
│   │   ├── localization.py                  Sliding-window / frame scoring, timeline building
│   │   ├── report_summary.py                Clear-summary generation for the app
│   │   └── narration.py                     Gemini-based grounded narration
│   │
│   ├── models/
│   │   ├── audio_model.py                   Wav2Vec2SpoofClassifier
│   │   ├── video_model.py                   EfficientNetFrameClassifier
│   │   ├── image_model.py                   UnivFD model wrapper
│   │   ├── interframe_detector.py           Self-similarity frame-duplication detector
│   │   ├── interframe_motion_detector.py
│   │   ├── frequency_detector.py            Frequency-domain detector (tested, rejected)
│   │   └── temporal_consistency.py          Optical-flow jitter signal (tested, rejected)
│   │
│   ├── data/
│   │   ├── degradation.py                   DeeperForensics-1.0 degradation protocol
│   │   ├── image_degradation.py
│   │   └── datasets/
│   │       ├── asvspoof.py
│   │       ├── asvspoof_keys_cache.json
│   │       ├── asvspoof_validation_cache.json
│   │       ├── audio_holdout_keys.json
│   │       ├── audio_phase_b_dataset.py
│   │       ├── build_audio_phase_b.py
│   │       ├── build_phase_b_data.py
│   │       ├── build_synthetic_timeline_testset.py
│   │       ├── faceforensics.py
│   │       ├── fakeavceleb.py
│   │       ├── htvd_clone_finetune.py
│   │       ├── htvd_intraframe.py
│   │       ├── image_gen_dataset.py
│   │       └── validate_flac.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── search_audio.py                  Hyperparameter search (Audio)
│   │   ├── search_video.py                  Hyperparameter search (Video)
│   │   ├── train_audio.py                   Phase A
│   │   ├── train_audio_phase_b.py
│   │   ├── train_audio_phase_c.py           Class-weighted corrective fine-tune
│   │   ├── train_video.py                   Phase A
│   │   ├── train_video_phase_b.py
│   │   └── finetune_mvssnet_clone.py        Structural Agent Clone fine-tune
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── image_metrics.py
│   │   ├── degradation_eval.py
│   │   └── audio_degradation_eval.py
│   │
│   └── explainability/
│       ├── __init__.py
│       ├── gradcam.py
│       ├── audio_explain.py
│       └── image_explain.py
│
└── checkpoints/                             Not included, see Model Weights in README
```

## Model Weights

The trained checkpoints are not included in this repository because of their file size. They are hosted separately on Hugging Face Hub:

` https://huggingface.co/insha142/forensiq-checkpoints/tree/main`

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

An optional Google Gemini API key can be entered in the application sidebar. This enables the Image Agent's validated fusion approach and the narration layer. The application can still run without the API key, but the Image Agent will have lower accuracy.

## Live Demo

` https://forensiq-multi-agent-deepfake-detection-1.onrender.com ` 

## License

The code in this repository is released under the MIT License (see `LICENSE`). This only covers the author's own code. The pretrained third-party models, including MVSS-Net and UnivFD/CLIP, and the datasets used in this project have their own separate licences. Their original repositories should be checked before reuse.

## Author

**Insha Ahmed**
MSc Data Science & AI, Middlesex University Dubai

