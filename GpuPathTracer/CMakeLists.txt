cmake_minimum_required(VERSION 3.6)




set(GpuBasicSrc main.cpp BasicScene.cpp Scene.cpp utilfun.cpp BVH.cpp BVHNode.cpp bvh_util.cpp SplitBVHBuilder.cpp Sort.cpp)
set(GpuBasicHdrs BasicScene.hpp Scene.hpp utilfun.hpp BVH.hpp BVHNode.hpp bvh_util.hpp SplitBVHBuilder.hpp SceneMesh.hpp Sort.hpp)
set(GpuBasicCudaSrc tracer.cu )

#MESSAGE(STATUS "deps " ${DEPS_SOURCES})
cuda_add_executable(GpuBasicTracer ${GpuBasicCudaSrc}  ${GpuBasicSrc} ${GpuBasicHdrs} ${DEPS_SOURCES})
#TODO make stuff windows compilable remove -ldl from here
target_link_libraries(GpuBasicTracer assimp glfw -ldl)
set_target_properties(GpuBasicTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GpuBasicTracer)
#TODO see if to make shaders as target so that if required
file(GLOB Shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/*.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/*.frag
        )
file(GLOB OBJECTS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.obj
        )

file(COPY ${Shaders} ${OBJECTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GpuBasicTracer)







