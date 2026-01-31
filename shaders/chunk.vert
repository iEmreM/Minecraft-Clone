#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_tex_coord;
layout (location = 2) in float in_shading;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;  // Keep for Python compatibility (always identity)

out vec3 uv;
out float shading;
out vec3 frag_world_pos;

void main() {
    // OPTIMIZATION: Direct position assignment (m_model is always identity)
    // We keep the uniform declared to avoid Python errors
    frag_world_pos = in_position;
    
    // Pass texture coordinates and shading
    uv = in_tex_coord;
    shading = in_shading;
    
    // Calculate final position (also optimized - skip m_model)
    gl_Position = m_proj * m_view * vec4(in_position, 1.0);
}
