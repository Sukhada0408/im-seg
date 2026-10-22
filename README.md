# Image Segmentation Inference Pipeline

A lightweight semantic segmentation inference pipeline using **Fast-SCNN, ONNX Runtime, C++, and OpenCV**.

This project explores deploying a deep learning segmentation model beyond Python by exporting a PyTorch model to ONNX and integrating it into a native C++ inference application.

## Overview

The pipeline is designed to run semantic segmentation on input images using a Fast-SCNN model configured for 19 classes from the Cityscapes dataset.

**Key components**

* **Fast-SCNN:** Lightweight neural network architecture for semantic segmentation.
* **PyTorch:** Model definition, weight loading, and ONNX export.
* **ONNX Runtime:** Executes the exported model for inference.
* **C++ and OpenCV:** Native application for image processing and model integration.
* **CMake:** Build configuration for the C++ application.

## Pipeline

```text
Input Image
    |
    v
OpenCV Image Preprocessing
    |
    v
ONNX Runtime Inference
    |
    v
Pixel-wise Class Predictions
    |
    v
Segmentation Output
```
## Output
For image size 384x768

https://github.com/user-attachments/assets/7bebd91e-8b2c-448d-b958-4f23160a45f2

**Input resolution vs. accuracy.** Fast-SCNN is trained on Cityscapes at 1024x2048.
This pipeline runs at a reduced input size to fit CPU-only inference (8 GB RAM, no GPU).
Qualitatively, the output improves as resolution increases (see the comparison below), however,
the quantitative difference (mIoU score) has not been measured here.

| Input size | Avg. Latency (mean ± std, ms) | FPS |
| 384x768   | 143.451 ms  | 6.971 FPS |
| 768x1536  | 481.636 ms  | 2.07626 FPS |

**Domain gap.** The weights are the pretrained Cityscapes checkpoint from the original
repository. Cityscapes is windshield-height, forward-facing footage, so results degrade
on elevated or non-dashcam views (e.g. a portrait phone video filmed from a window).

## Requirements

* Python 3.12
* PyTorch
* ONNX
* ONNX Runtime
* C++17-compatible compiler
* CMake 3.15+
* OpenCV


