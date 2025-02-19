cmake_minimum_required(VERSION 3.20.0)
set (CMAKE_CXX_STANDARD 20)

project(obj_inspect_inator)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE DEBUG)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

set(CMAKE_SOURCE_DIR "${MY_ROOT}/src")
list(APPEND sourcesList "src/main.cpp" "src/renderer.cpp" "src/renderer.hpp")
add_executable(${PROJECT_NAME} ${sourcesList})

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan glfw)
