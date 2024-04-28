#version 450

layout(binding = 1) uniform sampler2D texSampler[16];

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in flat uint fragTexId;
layout(location = 3) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main(void) {
    const vec3 lightPos = vec3(0, 5, 5);
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = 0.75f*max(dot(norm, lightDir), 0.0) + 0.25f;
    vec3 diffuse = diff * vec3(1,1,1);

    outColor = vec4(texture(texSampler[fragTexId], fragTexCoord).xyz * diffuse, 1.0f);

    return;
}