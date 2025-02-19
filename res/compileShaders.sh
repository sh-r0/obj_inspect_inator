#!/bin/bash

glslc -fshader-stage=vertex shaders_code/vertex.glsl -o shaders/vertex.spv 
glslc -fshader-stage=fragment shaders_code/fragment.glsl -o shaders/fragment.spv
