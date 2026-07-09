# Real-Time Video Prediction for Teleoperation and Autonomous Driving
[![arXiv](https://img.shields.io/badge/arXiv-2503.23185-b31b1b.svg)](https://arxiv.org/abs/2503.23185)

## Project Note

This repository is based on the original IFRVP implementation by Hirose et al. The original project and paper introduced the IFRVP model for real-time video prediction.

This fork adapts that work for **real-time video prediction for teleoperation and autonomous driving**, with additional training and evaluation scripts for BDD100K-style driving videos, demo fine-tuning workflows, and comparison/verification utilities.

## Overview

This project uses IFRVP-style future-frame prediction to reduce the visual effect of communication latency in remote driving, robot teleoperation, and autonomous-driving monitoring. Instead of showing only delayed receiver frames, the system predicts the current scene from past frames and displays a compensated view.

## Teleoperation / Autonomous Driving Adaptation

This adaptation focuses on predicting near-future driving-scene frames for latency compensation in remote operation and autonomous-driving video pipelines. The added workflow includes:

1. Preprocessing scripts for BDD100K/demo driving videos
2. Training and fine-tuning scripts for future-frame prediction
3. Evaluation utilities for comparing pretrained and fine-tuned IFRVP outputs
4. Verification scripts and demo metrics for presentation/reporting

## Our Additions

This repository extends the original IFRVP codebase with a driving-video workflow:

- **BDD100K-oriented preprocessing:** `preprocess_bdd100k_t3.py`
- **Demo-video preprocessing:** `preprocess_demo_videos.py`
- **BDD100K future-frame training:** `IFRVP/train_bdd100k_t3.py`
- **Demo fine-tuning:** `IFRVP/train_demo_finetune.py`
- **Training helpers:** `run_training.sh` and `check_training.sh`
- **Model comparison tools:** `compare_models.py` and `create_comparison_video.py`
- **Prediction validation:** `verify_prediction.py`, `debug_verification.py`, and `analyze_comparison_video.py`
- **Presentation metrics:** `DEMO_METRICS.md` and `generate_slide_metrics.py`
- **PyTorch in-place operation fix:** `IFRVP/models/IFRNet.py` was updated to avoid slice assignment inside the residual block forward pass.

## Demo Result

The teleoperation demo simulates a 200 ms network delay, equal to 3 frames at 15 FPS. IFRVP predicts the current sender frame from delayed receiver frames.

| Comparison | MSE | PSNR |
|------------|-----|------|
| Predicted vs sender | 22.80 | 34.55 dB |
| Delayed receiver vs sender | 42.80 | 31.82 dB |
| Improvement | 46.7% lower error | +2.73 dB |

The predicted frame is substantially closer to the current sender frame than the delayed receiver frame, showing the value of future-frame prediction for latency compensation.

## Repository Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Preprocess driving videos:

```bash
python preprocess_bdd100k_t3.py
python preprocess_demo_videos.py
```

Train or fine-tune the prediction model:

```bash
bash run_training.sh
python IFRVP/train_bdd100k_t3.py
python IFRVP/train_demo_finetune.py
```

Evaluate and generate comparison material:

```bash
python test_video_prediction.py
python test_finetuned.py
python compare_models.py
python create_comparison_video.py
python verify_prediction.py
```

Generated logs, checkpoints, caches, and output videos are ignored by Git so the repository stays focused on source code and reproducible scripts.

## Original IFRVP Background

The original IFRVP repository contains the implementation of "REAL-TIME VIDEO PREDICTION WITH FAST VIDEO INTERPOLATION MODEL AND PREDICTION TRAINING," a novel approach to enable zero-latency interaction in networked video applications.

## Paper Link

The full paper is available at: [IEEE Xplore](https://ieeexplore.ieee.org/document/10647865)

## Paper Abstract

Transmission latency significantly affects users' quality of experience in real-time interaction and actuation. While latency is fundamentally inevitable due to physical constraints, this work proposes IFRVP (Intermediate Feature Refinement Video Prediction) to mitigate latency through efficient video prediction. IFRVP extends a simple convolution-only frame interpolation network based on IFRNet by unifying optical flow estimation and pixel refinements into a single network. The architecture introduces ELAN-based residual blocks which significantly reduce computational complexity while maintaining high prediction accuracy. Unlike previous state models that require recursive application for multi-frame prediction, IFRVP's arbitrary and independent prediction methods can generate predictions for any future timestep in a single inference, effectively avoiding error accumulation while enabling real-time performance even on consumer hardware.

## Performance Comparison

![Performance Comparison](flopscomparison_hrs_v2.png)

## Key Contributions

1. **Three Training Methods for Video Prediction**:
   - **Recurrent Prediction**: Uses the two latest frames to recursively predict the next frame
   - **Arbitrary Prediction**: Predicts any future timestep in a single inference using timestep embedding
   - **Independent Prediction**: Utilizes specialized models for different prediction timesteps

![Prediction Methods](predictionmethods.png)

2. **ELAN-based Residual Blocks**: Lightweight architecture that improves both inference speed and prediction accuracy

3. **State-of-the-Art Performance**: Achieves the best trade-off between prediction accuracy and computational speed compared to existing methods

## Results

- IFRVP-Fast achieves comparable or better prediction quality than state-of-the-art methods while requiring only 9.9 GFLOPs (20% less computation than DMVFN)
- Models can run at 70-130 FPS on consumer GPUs depending on resolution
- Independent prediction training achieves the highest accuracy by avoiding error accumulation

## Demo

A demonstration video showing real-time prediction capabilities is available at http://bit.ly/IFRVPDemo

## Applications

- Remote control/telepresence systems
- Autonomous driving
- Cloud gaming
- Video conferencing
- Mission-critical systems requiring near-zero latency

## Citation

```
@inproceedings{hirose2024realtime,
  title={Real-Time Video Prediction with Fast Video Interpolation Model and Prediction Training},
  author={Hirose, Shota and Kotoyori, Kazuki and Arunruangsirilert, Kasidis and Lin, Fangzheng and Sun, Heming and Katto, Jiro},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2024}
}
```
