cmake_minimum_required(VERSION 3.6)
#TODO decide to keep project here or not
project(CpuRayTracer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-narrowing -march=native -m64 -O3  -funroll-loops")
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCE_FILES
        src/main.cpp
        src/material.cpp
        src/objects.cpp
        src/kdtree.cpp
        src/texture.cpp
        src/camera.cpp
        src/scene.cpp
        src/renderer.cpp
        lib/rand48/rand48.c
        lib/lodepng/lodepng.cpp
        lib/tiny_obj_loader/tiny_obj_loader.cc)
file(GLOB SOURCE_HEADERS "src/*.hpp")

add_executable(CpuPathTracer ${SOURCE_FILES} ${SOURCE_HEADERS}  ${DEPS_SOURCES})
target_link_libraries(CpuPathTracer glfw ${GLFW_LIBRARIES} ${PLATFORM_LINK_GLAD})
set_target_properties(CpuPathTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CpuPathTracer)
# copy shader files to build directory
file(GLOB OBJECTS
        "obj/*"
        )
file(COPY ${OBJECTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CpuPathTracer/obj)
