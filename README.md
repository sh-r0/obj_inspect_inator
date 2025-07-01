# obj_inspect_inator
Obj_inspect_inator is a simple tool built for inspecting .obj files.

## Usage
Program should be passed one argument that is path to a file e.g.  
./obj_inspect_inator.exe example.obj

## Shaders
You may need to rebuild shaders, in order to do so, 
provide path to glslc (it should be shipped with VulkanSDK) 
to a proper "compile_shaders" script and run it.

## Dependencies
* vulkan - https://www.vulkan.org
* GLFW - https://www.glfw.org
* tinyobjloader - https://github.com/tinyobjloader/tinyobjloader
* stb - https://github.com/nothings/stb
