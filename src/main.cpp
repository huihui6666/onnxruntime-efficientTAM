#include <iostream>
#include <filesystem>
#include "SAM2.h"

void sam2(){
    auto sam2 = std::make_unique<SAM2>();
    std::vector<std::string> onnx_paths{
        "../models/etam/image_encoder.onnx",
        "../models/etam/memory_attention.onnx",
        "../models/etam/image_decoder.onnx",
        "../models/etam/memory_encoder.onnx"
    };
    auto r = sam2->initialize(onnx_paths,true);
    if(r.index() != 0){
        std::string error = std::get<std::string>(r);
        std::println("错误：{}",error);
        return;
    }

    sam2->setparms({.type=1,
                    .prompt_box = {745,695,145,230},
                    .prompt_point = {500,420}}); // 在原始图像上的box,point
    
    std::string video_path = "../assets/01_dog.mp4";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) return;
    //************************************************************
    std::cout << "视频中图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << capture.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    //************************************************************
    cv::Mat frame;
    size_t idx = 0;
    while (true) {
        if (!capture.read(frame) || frame.empty()) break;
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sam2->inference(frame);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("frame = {},duration = {}ms",idx++,duration);
        if(result.index() == 0){
            std::string text = std::format("frame = {},fps={:.1f}",idx,1000.0f/duration);
            cv::putText(frame,text,cv::Point{30,40},1,2,cv::Scalar(0,0,255),2);
            cv::imshow("frame", frame);
            int key = cv::waitKey(5);
            if (key == 'q' || key == 27) break;
        }else{
            std::string error = std::get<std::string>(result);
            std::println("错误：{}",error);
            break;
        }
    }
    capture.release();
}
int main(int argc, char const *argv[]){
    // yolo();
    // yolosam();
    // yolotrace();
    sam2();
    return 0;   
}

