#include "segmentation.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>

int main()
{
    const std::string videoPath = "C:\\Users\\Sukhada\\upskill\\im-seg\\input.mp4";
    const std::string modelPath = "C:\\Users\\Sukhada\\upskill\\im-seg\\fast_scnn_citys.onnx";
    
    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened()){
        std::cerr<<"Cannot open video..."<<std::endl;
        return 1;
    }
    
    SegmentationInference segInfer(modelPath);

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps<=1.0) fps=25.0;

    cv::VideoWriter writer(
        "segmentation_output_1.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(frameWidth, frameHeight)
    );

    std::vector<double> latencies;

    cv::Mat frame;
    int frameCount = 0;

    while (cap.read(frame)) {
        // cap.read(frame) returns false when the video ends. This is to read next frame or indicate that
        // inferencing is finished.

        if (frame.empty()) {
            std::cerr << "WARNING: empty frame at index " << frameCount << ", skipping\n";
            continue;
        }

        cv::Mat coloredMask = segInfer.infer(frame);
        latencies.push_back(segInfer.getLastLatencyMs()); // see note below

        cv::Mat maskResized;
        cv::resize(coloredMask, maskResized, frame.size(), 0, 0, cv::INTER_NEAREST);

        // Blend original frame + segmentation mask
        cv::Mat overlay;
        cv::addWeighted(frame, 0.6, maskResized, 0.8, 0.0, overlay);

        // Display and write the output
        cv::imshow("Fast-SCNN Segmentation", overlay);
        writer.write(overlay);

        ++frameCount;

        if (cv::waitKey(1) == 'q') {
            std::cout << "Interrupted by user at frame " << frameCount << "\n";
            break;
        }
    } 
    
    // Cleanup
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    // Latency summary
    if (!latencies.empty()) {
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double maxLat = *std::max_element(latencies.begin(), latencies.end());
        double minLat = *std::min_element(latencies.begin(), latencies.end());

        std::cout << "\n--- Inference stats over " << frameCount << " frames ---\n";
        std::cout << "Avg latency: " << avg    << " ms  (" << (1000.0 / avg) << " FPS)\n";
        std::cout << "Min latency: " << minLat << " ms\n";
        std::cout << "Max latency: " << maxLat << " ms\n";
    }

    return 0;
}
