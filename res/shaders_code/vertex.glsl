#version 450

layout(binding = 0) uniform uniformBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in uint inTexId;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out flat uint fragTexId;
layout(location = 3) out vec3 fragPos;

void main(void) {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0f);
    fragTexCoord = inTexCoord;
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragTexId = inTexId;
    fragPos = (ubo.model * vec4(inPosition, 1.0f)).xyz;

    return;
}