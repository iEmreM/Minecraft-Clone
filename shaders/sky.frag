#version 330 core
#include "sky_common.glsl"

out vec4 fragColor;
in vec3 frag_pos;

uniform mat4 m_inv_pv; // Inverse Projection * View matrix to rebuild the ray

void main() {
    // Ray direction through this pixel, from its NDC on the fullscreen quad.
    vec4 eye = m_inv_pv * vec4(frag_pos.xy, 1.0, 1.0); // z = 1: the far plane
    fragColor = vec4(sky_color(normalize(eye.xyz / eye.w)), 1.0);
}
