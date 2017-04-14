cmake_minimum_required(VERSION 3.6)
project(GpuProject)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/bin)


#setup qt
#TODO see if you require this or use just imgui

#set(CMAKE_AUTOMOC ON)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)

#find_package(Qt5Widgets REQUIRED )
#find_package(Qt5OpenGL REQUIRED )

#find_package(Qt5Core REQUIRED )
#find_package(Qt5Gui REQUIRED )
#use cuda on linux only

find_package(CUDA REQUIRED)
MESSAGE( STATUS "cuda toolkit location" ${CUDA_TOOLKIT_ROOT_DIR} )
#MESSAGE( STATUS "QT5 widgets version " ${Qt5Widgets_VERSION})
#MESSAGE( STATUS "QT5 Core location " ${Qt5Core_DIR})



#TODO delete assimp if not used fr
#add assimp library
#option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
#option(ASSIMP_BUILD_SAMPLES "" OFF)
#option(ASSIMP_BUILD_TESTS "" OFF)
#add_subdirectory(deps/assimp)


#add glraw library
#option(BUILD_SHARED_LIBS  "" OFF)
#option(OPTION_SELF_CONTAINED "" OFF)
#option(OPTION_BUILD_TESTS "" OFF)
#option(OPTION_BUILD_DOCS "" OFF)
#TODO see if you need this or use just stb image lib
#add_subdirectory(deps/glraw)


add_subdirectory(deps/glfw)


include_directories(
        "deps/glad/include"
        "deps/glfw/include"
        "deps/glm"
        "deps/tinyobjloader"
#)
        #"deps/assimp/include"
        "deps/glraw/source/glraw")

file(GLOB DEPS_SOURCES deps/glad/src/glad.c)
file(GLOB TINY_OBJ_SOURCE deps/tinyobjloader/*)
#file(GLOB IMGUI_SOURCE
#        deps/imgui/*.h
#        deps/imgui/*.cpp
#        )


set(DEPS_SOURCES ${DEPS_SOURCES})

if(CUDA_FOUND)
    #TODO see these flags

    set(CUDA_NVCC_FLAGS "-std=c++11")
    if(UNIX)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin /usr/bin/gcc-4.9;")
    endif(UNIX)
    set(CUDA_NVCC_FLAGS "
    ${CUDA_NVCC_FLAGS}
#  -gencode;arch=compute_30,code=sm_30;
#  -gencode;arch=compute_35,code=sm_35;
#  -gencode;arch=compute_35,code=compute_35;
#  -gencode;arch=compute_20,code=sm_20;
#  -gencode;arch=compute_11,code=sm_11;
#  -gencode;arch=compute_12,code=sm_12;
#  -gencode;arch=compute_13,code=sm_13;
")
#TODO off them in release build
# add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} " -G")


else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()



add_subdirectory(CpuRayTracer)
add_subdirectory(GpuPathTracer)

#C.P.U executable
#add_executable(cpuTracer src/main.cpp src/Scene.cpp
#        src/Scene.hpp src/Camera.cpp src/Camera.hpp
#        ${DEPS_SOURCES})

#cuda_add_executable(
#        hellocuda
#        gpu.cpp
#)

#set(PLATFORM_LINK_GLAD )
#if (UNIX)
#    set(PLATFORM_LINK_GLAD -ldl)
#endif (UNIX)

#target_link_libraries(cpuTracer  Qt5::Widgets ${PLATFORM_LINK_GLAD})
