// SegmentationInference.hpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

class SegmentationInference {
public:
    explicit SegmentationInference(const std::string& modelPath);

    // returns colorized segmentation mask, same size as input frame
    cv::Mat infer(const cv::Mat& frame);
    // Add this:
    double getLastLatencyMs() const { return lastLatencyMs_; }

private:
    std::vector<float> preprocess(const cv::Mat& frame);
    Ort::Value runInference(std::vector<float>& inputTensorValues);
    cv::Mat postprocess(const Ort::Value& outputTensor);
    cv::Mat colorize(const cv::Mat& classMap);

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memoryInfo_;
    Ort::SessionOptions sessionOption_;

    // fixed model I/O shape — Fast-SCNN is NOT dynamic on H/W, remember
    static constexpr int64_t kInputH = 384;
    static constexpr int64_t kInputW = 768;
    static constexpr int64_t kNumClasses = 19;

    std::string inputName_;
    std::string outputName_;

    double lastLatencyMs_ = 0.0; // set inside infer(), like your LaneDetector timing
};