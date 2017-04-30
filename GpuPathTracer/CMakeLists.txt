cmake_minimum_required(VERSION 3.6)




set(GpuBasicSrc App/main.cpp BasicScene.cpp  utilfun.cpp BVH.cpp BVHNode.cpp bvh_util.cpp SplitBVHBuilder.cpp Sort.cpp CudaBVH.cpp)
set(GpuBasicHdrs BasicScene.hpp  utilfun.hpp cudaUtils.h BVH.hpp BVHNode.hpp bvh_util.hpp SplitBVHBuilder.hpp SceneMesh.hpp Sort.hpp CudaBVH.hpp CpuStructs.hpp)
set(GpuBasicCudaSrc tracer.cu)

#MESSAGE(STATUS "deps " ${PLATFORM_LINK_GLAD})
cuda_add_executable(GpuBasicTracer ${GpuBasicCudaSrc}  ${GpuBasicSrc} ${GpuBasicHdrs} ${IMGUI_SOURCE} ${TINY_OBJ_SOURCE}  ${DEPS_SOURCES})
target_link_libraries(GpuBasicTracer glfw ${PLATFORM_LINK_GLAD})
set_target_properties(GpuBasicTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GpuBasicTracer)

file(GLOB Shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/*.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/*.frag
        )

file(GLOB OBJECTS ${CMAKE_SOURCE_DIR}/Assets/*.obj)

file(COPY ${Shaders} ${OBJECTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GpuBasicTracer)







