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

## Requirements

* Python 3.12
* PyTorch
* ONNX
* ONNX Runtime
* C++17-compatible compiler
* CMake 3.15+
* OpenCV


