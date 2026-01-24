#version 330 core

in vec3 uv;
in float shading;
in vec3 frag_world_pos;

uniform sampler2DArray u_texture_0;
uniform vec3 bg_color;
uniform float water_line; // Water level for underwater effects

out vec4 fragColor;

// Gamma correction constants like ornek2
const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = 1.0 / gamma;

void main() {
    // Sample texture
    vec3 tex_color = texture(u_texture_0, uv).rgb;
    
    // Apply gamma correction (convert from sRGB to linear)
    tex_color = pow(tex_color, gamma);
    
    // Apply shading (ambient occlusion + face lighting)
    tex_color *= shading;
    
    // Underwater effect (from ornek2)
    if (frag_world_pos.y < water_line) {
        tex_color *= vec3(0.0, 0.3, 1.0);
    }
    
    // Calculate distance fog like ornek2
    float fog_dist = gl_FragCoord.z / gl_FragCoord.w;
    float fog_factor = 1.0 - exp2(-0.00001 * fog_dist * fog_dist);
    
    // Mix with background color for atmospheric depth
    tex_color = mix(tex_color, bg_color, fog_factor);
    
    // Convert back to sRGB for display (gamma correction)
    tex_color = pow(tex_color, inv_gamma);
    
    fragColor = vec4(tex_color, 1.0);
}
