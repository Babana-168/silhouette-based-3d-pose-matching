cmake_minimum_required(VERSION 3.15)
project(PoseMatch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pose_match
    main.cpp
)

target_link_libraries(pose_match ${OpenCV_LIBS})

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# MSVC optimizations
if(MSVC)
    target_compile_options(pose_match PRIVATE /O2 /fp:fast /openmp /arch:AVX2)
endif()
