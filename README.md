# obj_inspect_inator
Obj_inspect_inator is a simple tool build for inspecting .obj files.

## Usage
Program should be passed one argument that is path to a file e.g.  
./obj_inspect_inator.exe example.obj

## Building

### Prerequisites
To build following project it's necessary to install/build VulkanSDK (https://vulkan.lunarg.com) and glfw (https://www.glfw.org/download.html)

### Setting up CMakeLists
In CMakeLists.txt file set paths to vulkan and GLFW library directories in CONFIG INFORMATION section  

### Shaders
You may need to rebuild shaders, in order to do so, provide path to glslc.exe (it should be shipped with VulkanSDK) to "compile_shaders.bat" script

## Dependencies
* vulkan - https://www.vulkan.org
* GLFW - https://www.glfw.org
* tinyobjloader - https://github.com/tinyobjloader/tinyobjloader
* stb - https://github.com/nothings/stb