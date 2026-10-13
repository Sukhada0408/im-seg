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
        throw;
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
    // Resize frame
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(kInputW, kInputH));

    // Convert BGR to RGB
    cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

    // Convert to float and scale pixel values
    cv::Mat floatImg;
    resized_frame.convertTo(floatImg, CV_32FC3, 1.0/255.0);

    // Normalize per-channel using Cityscapes/ImageNet mean-std
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar std = {0.229, 0.224, 0.225};
    floatImg -= mean;
    floatImg /= std;

    // TODO 5: convert HWC to CHW

    std::vector<float> tensorData(1 * 3 * kInputH * kInputW);
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

    
    return std::move(outputTensors[0]);
}

cv::Mat SegmentationInference::postprocess(const Ort::Value& outputTensor) {
    const float* data = outputTensor.GetTensorData<float>();
    
    cv::Mat classMap(kInputH, kInputW, CV_8UC1);

    const int planeSize = kInputH * kInputW; // stride between class channels

    for (int h = 0; h < kInputH; ++h) {
        for (int w = 0; w < kInputW; ++w) {
            float maxVal = data[0 * planeSize + h * kInputW + w];
            int maxClass = 0;


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
    // build a static 19-entry BGR color table (Cityscapes palette)
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
