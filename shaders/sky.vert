#version 330 core

layout (location = 0) in vec2 in_texcoord;
layout (location = 1) in vec3 in_position;

out vec3 frag_pos;

uniform mat4 m_proj;
uniform mat4 m_view;

void main() {
    // We want the sky to be always at the background, so we set z = w = 1.0 (after perspective division)
    // But standard way for skybox is:
    // Pass vertex position as texture direction logic
    
    // For fullscreen quad approach (simpler procedural sky):
    gl_Position = vec4(in_position, 1.0);
    // Passing position for calculating view direction in fragment shader if needed
    // But actually, for a generic sky, we usually want ray direction.
    
    // Let's use a simpler approach: Fullscreen Quad (-1 to 1)
    // And calculating Ray Direction from inverse ViewProjection?
    // OR simpler: Render a big Cube following the camera.
    
    // Simplest: Fullscreen Quad.
    frag_pos = in_position;
}
