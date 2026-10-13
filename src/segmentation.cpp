#include "segmentation.hpp"
#include <iostream>

SegmentationInference::SegmentationInference(const std::string& modelPath)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SegInference"),
      session_(nullptr),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU))
{
    sessionOption_.SetIntraOpNumThreads(4); // tune based on your CPU core count
    sessionOption_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Windows requires a wide-string path
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    try {
    session_ = Ort::Session(env_, wModelPath.c_str(), sessionOption_);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime load failed. Code: " << e.GetOrtErrorCode()
                << "  Message: " << e.what() << std::endl;
        throw; // rethrow so it's still a hard failure, but now you've printed the detail first
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Query input name + shape
    auto inputNameAlloc = session_.GetInputNameAllocated(0, allocator);
    inputName_ = inputNameAlloc.get();

    auto inputTypeInfo = session_.GetInputTypeInfo(0);
    auto inputShape_ = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    // Query output name
    auto outputNameAlloc = session_.GetOutputNameAllocated(0, allocator);
    outputName_ = outputNameAlloc.get();

    std::cout << "Model loaded. Input shape: ";
    for (auto d : inputShape_) std::cout << d << " ";
    std::cout << std::endl;
}

std::vector<float> SegmentationInference::preprocess(const cv::Mat& frame) {
    // TODO 1: resize frame to cv::Size(kInputW, kInputH) using cv::resize
    //         — INTER_LINEAR is fine, this is downsampling in most AD camera feeds
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(kInputW, kInputH));

    // TODO 2: convert BGR -> RGB (cv::cvtColor), since the PyTorch model
    //         was trained on RGB tensors (ImageNet convention), OpenCV loads BGR
    cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

    // TODO 3: convert to float32 and scale pixel values from [0,255] to [0,1]
    //         (resized.convertTo(floatImg, CV_32FC3, 1.0/255.0))
    cv::Mat floatImg;
    resized_frame.convertTo(floatImg, CV_32FC3, 1.0/255.0);

    // TODO 4: normalize per-channel using Cityscapes/ImageNet mean-std:
    //         mean = {0.485, 0.456, 0.406}, std = {0.229, 0.224, 0.225}
    //         applied per-channel: (pixel - mean[c]) / std[c]
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar std = {0.229, 0.224, 0.225};
    floatImg -= mean;
    floatImg /= std;

    // TODO 5: convert HWC (OpenCV's native layout) -> CHW (what the ONNX
    //         graph expects, since PyTorch tensors are channel-first).
    //         This is a manual index remap, NOT a reshape — reshape would
    //         scramble the data since memory layout actually changes.
    //         out[c * H * W + h * W + w] = in.at<cv::Vec3f>(h, w)[c]

    std::vector<float> tensorData(1 * 3 * kInputH * kInputW);
    // ... fill tensorData per above ...
    const int planeSize = kInputH*kInputW;
    for (int h=0; h<kInputH; ++h)
    {
        for (int w=0; w<kInputW; ++w)
        {
            cv::Vec3f pixel = floatImg.at<cv::Vec3f>(h,w);
            for (int c=0; c<3; ++c)
            {
                tensorData[c*planeSize + h*kInputW +w] = pixel[c];
            }
        }
    }

    return tensorData;
}

Ort::Value SegmentationInference::runInference(std::vector<float>& inputTensorValues) {
    std::array<int64_t, 4> inputShape{1, 3, kInputH, kInputW};

    // CreateTensor does NOT copy inputTensorValues — it wraps the pointer.
    // inputTensorValues must outlive this Ort::Value.
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size()
    );

    const char* inputNames[]  = { inputName_.c_str() };
    const char* outputNames[] = { outputName_.c_str() };

    auto outputTensors = session_.Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 1
    );

    // outputTensors is a std::vector<Ort::Value>; we asked for 1 output
    return std::move(outputTensors[0]);
}

cv::Mat SegmentationInference::postprocess(const Ort::Value& outputTensor) {
    const float* data = outputTensor.GetTensorData<float>();
    // data layout: [class][h][w], row-major, since NCHW with N=1

    cv::Mat classMap(kInputH, kInputW, CV_8UC1);

    const int planeSize = kInputH * kInputW; // stride between class channels

    for (int h = 0; h < kInputH; ++h) {
        for (int w = 0; w < kInputW; ++w) {
            float maxVal = data[0 * planeSize + h * kInputW + w];
            int maxClass = 0;

            // manual argmax across 19 channels — no std::max_element here
            // on purpose: keeping the index math explicit while it's new
            for (int c = 1; c < kNumClasses; ++c) {
                float val = data[c * planeSize + h * kInputW + w];
                if (val > maxVal) {
                    maxVal = val;
                    maxClass = c;
                }
            }
            classMap.at<uint8_t>(h, w) = static_cast<uint8_t>(maxClass);
        }
    }
    return classMap;
}

cv::Mat SegmentationInference::colorize(const cv::Mat& classMap) {
    // TODO: build a static 19-entry BGR color table (Cityscapes palette,
    // easy to find online) as a cv::Vec3b array, then map:
    //   colored.at<cv::Vec3b>(h,w) = palette[classMap.at<uint8_t>(h,w)]
    // for every pixel. This is a manual LUT application — cv::LUT() only
    // works per-channel independently, not for a 3-channel-out-of-1-channel-in
    // mapping like this, so a nested loop (or cv::applyColorMap for a
    // built-in palette as a fallback) is the correct tool here.
    static const std::array<cv::Vec3b, 19> palette = {{
        {128u,  64u, 128u},   // road          =   0u
         {244u,  35u, 232u},   // sidewalk      =   1u
         { 70u,  70u,  70u},   // building      =   2u
         {102u, 102u, 156u},   // wall          =   3u
         {190u, 153u, 153u},   // fence         =   4u
         {153u, 153u, 153u},   // pole          =   5u
         {250u, 170u,  30u},   // traffic light =   6u
         {220u, 220u,   0u},   // traffic sign  =   7u
         {107u, 142u,  35u},   // vegetation    =   8u
         {152u, 251u, 152u},   // terrain       =  9u
         { 70u, 130u, 180u},   // sky           =  10u
         {220u,  20u,  60u},   // pedestrian    =  11u
         {255u,   0u,   0u},   // rider         =  12u
         {  0u,   0u, 142u},   // Car           =  13u
         {  0u,   0u,  70u},   // truck         =  14u
         {  0u,  60u, 100u},   // bus           =  15u
         {  0u,  80u, 100u},   // train         =  16u
         {  0u,   0u, 230u},   // motorcycle    =  17u
         {119u,  11u,  32u},   // bicycle       =  18u
    }};

    cv::Mat colored(classMap.rows, classMap.cols, CV_8UC3);
    for (int h=0; h<classMap.rows; h++)
    {
        for (int w=0; w<classMap.cols; w++)
        {
            colored.at<cv::Vec3b>(h,w) = palette[classMap.at<uint8_t>(h,w)];
        }
    }
    return colored;
}

cv::Mat SegmentationInference::infer(const cv::Mat& frame) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> input = preprocess(frame);
    Ort::Value output = runInference(input);
    cv::Mat classMap = postprocess(output);
    cv::Mat colored = colorize(classMap);

    auto t1 = std::chrono::high_resolution_clock::now();
    lastLatencyMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return colored;
}