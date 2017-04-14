cmake_minimum_required(VERSION 3.6)




set(GpuBasicSrc main.cpp BasicScene.cpp Scene.cpp utilfun.cpp)
set(GpuBasicHdrs BasicScene.hpp Scene.hpp utilfun.hpp)
set(GpuBasicCudaSrc tracer.cu )

#MESSAGE(STATUS "deps " ${DEPS_SOURCES})
cuda_add_executable(GpuBasicTracer ${GpuBasicCudaSrc}  ${GpuBasicSrc} ${GpuBasicHdrs} ${DEPS_SOURCES}  ${TINY_OBJ_SOURCE})
target_link_libraries(GpuBasicTracer glfw ${GLFW_LIBRARIES} -ldl)
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







