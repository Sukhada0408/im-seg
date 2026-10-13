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
        // cap.read(frame) returns false when the video ends — this
        // doubles as both "read next frame" and "are we done" in one call,
        // similar to Python's `while ret: ret, frame = cap.read()` pattern
        // but folded into the loop condition itself.

        if (frame.empty()) {
            std::cerr << "WARNING: empty frame at index " << frameCount << ", skipping\n";
            continue;
        }

        // --- Run the full pipeline: preprocess -> infer -> postprocess -> colorize ---
        cv::Mat coloredMask = segInfer.infer(frame);
        latencies.push_back(segInfer.getLastLatencyMs()); // see note below

        // --- Resize mask back up to original frame resolution for overlay ---
        // Your model runs at fixed 384x768, but the source video is
        // almost certainly a different resolution. Resize the COLORED
        // mask (not the raw class map!) back up using nearest-neighbor,
        // so class boundaries stay crisp instead of blurring across
        // class colors during interpolation.
        cv::Mat maskResized;
        cv::resize(coloredMask, maskResized, frame.size(), 0, 0, cv::INTER_NEAREST);

        // --- Blend original frame + segmentation mask ---
        cv::Mat overlay;
        cv::addWeighted(frame, 0.6, maskResized, 0.8, 0.0, overlay);

        // --- Display + write ---
        cv::imshow("Fast-SCNN Segmentation", overlay);
        writer.write(overlay);

        ++frameCount;

        // waitKey(1) both drives the GUI event loop AND lets you break
        // early on 'q' — without SOME waitKey call, imshow() won't
        // actually render anything to the window.
        if (cv::waitKey(1) == 'q') {
            std::cout << "Interrupted by user at frame " << frameCount << "\n";
            break;
        }
    } 
    
    // --- Cleanup ---
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    // --- Latency summary, same spirit as your LaneDetector timing prints ---
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