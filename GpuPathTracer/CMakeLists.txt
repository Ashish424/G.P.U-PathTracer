cmake_minimum_required(VERSION 3.6)




set(GpuBasicSrc main.cpp BasicScene.cpp Scene.cpp utilfun.cpp)
set(GpuBasicHdrs  BasicScene.hpp Scene.hpp utilfun.hpp)
set(GpuBasicCudaSrc tracer.cu)
cuda_add_executable(GpuBasicTracer ${GpuBasicCudaSrc} ${GpuBasicSrc} ${GpuBasicHdrs} ${DEPS_SOURCES})
target_link_libraries(GpuBasicTracer glfw ${GLFW_LIBRARIES} -ldl)
set_target_properties(GpuBasicTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GpuBasicTracer)
file(GLOB Shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/*.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/*.frag
        )


file(COPY ${Shaders} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GpuBasicTracer)






