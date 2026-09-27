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
forensiq/
  agents/       Orchestrator, per-modality agent logic, image fusion
  models/       Model architecture definitions
  data/         Dataset loaders and degradation protocol
  app.py        Streamlit application entry point
  checkpoints/  Not included, see Model Weights below
```

## Model Weights

The trained checkpoints are not included in this repository because of their file size. They are hosted separately on Hugging Face Hub:

`[your HF checkpoint repo link here]`

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

An optional Google Gemini API key can be entered in the application sidebar. This enables the Image Agent's validated fusion approach and the narration layer. The application can still run without the API key, but the Image Agent will have lower accuracy.

## Live Demo

'[]`

## License

The code in this repository is released under the MIT License (see `LICENSE`). This only covers the author's own code. The pretrained third-party models, including MVSS-Net and UnivFD/CLIP, and the datasets used in this project have their own separate licences. Their original repositories should be checked before reuse.

## Author

**Insha Ahmed**
MSc Data Science & AI, Middlesex University Dubai

