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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/bin)
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
        lib/tiny_obj_loader/tiny_obj_loader.cc )
file(GLOB SOURCE_HEADERS "src/*.hpp")

add_executable(pathtracer ${SOURCE_FILES} ${SOURCE_HEADERS}  ${DEPS_SOURCES})
target_link_libraries(pathtracer glfw ${GLFW_LIBRARIES} -ldl )
# copy shader files to build directory
file(GLOB OBJECTS
        "obj/*"
        )
file(COPY ${OBJECTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/obj)
