#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_tex_coord;
layout (location = 2) in float in_shading;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

out vec3 uv;
out float shading;
out vec3 frag_world_pos;

void main() {
    // Calculate world position for fog calculation
    frag_world_pos = (m_model * vec4(in_position, 1.0)).xyz;
    
    // Pass texture coordinates and shading
    uv = in_tex_coord;
    shading = in_shading;
    
    // Calculate final position
    gl_Position = m_proj * m_view * vec4(frag_world_pos, 1.0);
}
