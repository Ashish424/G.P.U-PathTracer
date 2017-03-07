cmake_minimum_required(VERSION 3.6)
project(GpuProject)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets REQUIRED )
#find_package(Qt5Core REQUIRED )
#find_package(Qt5Gui REQUIRED )
find_package(CUDA REQUIRED)
MESSAGE( STATUS "cuda toolkit location" ${CUDA_TOOLKIT_ROOT_DIR} )
MESSAGE( STATUS "QT5 Core location " ${Qt5Core_DIR})
MESSAGE( STATUS "QT5 widgets version " ${Qt5Widgets_VERSION})



if(CUDA_FOUND)
    # compared to class settings, we let NVidia's FindCUDA CMake detect
    # whether to build x64.  We tell it to support most devices, though,
    # to make sure more people can easily run class code without knowing
    # about this compiler argument
    set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/gcc-4.9;
#  -gencode;arch=compute_30,code=sm_30;
#  -gencode;arch=compute_35,code=sm_35;
#  -gencode;arch=compute_35,code=compute_35;
#  -gencode;arch=compute_20,code=sm_20;
#  -gencode;arch=compute_11,code=sm_11;
#  -gencode;arch=compute_12,code=sm_12;
#  -gencode;arch=compute_13,code=sm_13;
")


    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")


else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()


#find_package(OpenCV REQUIRED)

cuda_add_executable(
        hellocuda
        gpu.cpp
)
target_link_libraries(hellocuda Qt5::Widgets)