cmake_minimum_required(VERSION 3.15)
project(
    main
    LANGUAGES CXX C
    DESCRIPTION "模型推理集合"
)

# 编译设置
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/include/bytetrack/include/
    /opt/onnxruntime/include
)

# 源文件
file(GLOB SRC_FILES
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# 查找依赖
find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)

# ONNX Runtime 配置（手动指定）
set(ONNXRUNTIME_INCLUDE_DIR /opt/onnxruntime/include)
set(ONNXRUNTIME_LIBRARY /opt/onnxruntime/lib/libonnxruntime.so)

# 验证 ONNX Runtime 文件是否存在
if(NOT EXISTS ${ONNXRUNTIME_LIBRARY})
    message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIBRARY}")
endif()

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SRC_FILES})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    TBB::tbb
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

# 添加库搜索路径
target_link_directories(${PROJECT_NAME} PRIVATE /opt/onnxruntime/lib)