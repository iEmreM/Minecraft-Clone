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
    // Cinematic Tint: Mix Warm Sun and Cool Shadow (Subtle Natural)
    vec3 sunlight = vec3(1.02, 1.0, 0.98); // Very subtle warm white
    vec3 shadow = vec3(0.65, 0.7, 0.75);   // Natural cool grey shadow
    
    // shading varies from 0.4 (darkest) to 1.0 (brightest)
    // Remap shading to 0.0 - 1.0 for mixing
    float light_factor = smoothstep(0.4, 1.0, shading);
    vec3 light_tint = mix(shadow, sunlight, light_factor);
    
    tex_color *= shading * light_tint;
    
    // Underwater effect (from ornek2)
    if (frag_world_pos.y < water_line) {
        tex_color *= vec3(0.0, 0.3, 1.0);
    }
    
    // Calculate distance fog (Standard Exponential)
    // Minimally visible fog for depth only
    float fog_dist = gl_FragCoord.z / gl_FragCoord.w;
    float fog_density = 0.0008; // Ultra low density
    float fog_factor = 1.0 - exp(-fog_density * fog_dist);
    
    // Mix with background color for atmospheric depth
    tex_color = mix(tex_color, bg_color, clamp(fog_factor, 0.0, 1.0));
    
    // Vibrancy Boost (Subtle Correction)
    // 1. Saturation (Reduced from 1.2 to 1.05)
    float gray = dot(tex_color, vec3(0.299, 0.587, 0.114));
    tex_color = mix(vec3(gray), tex_color, 1.05); 
    
    // Removed Contrast Boost to keep original block colors authentic
    
    // Convert back to sRGB for display (gamma correction)
    tex_color = pow(tex_color, inv_gamma);
    
    fragColor = vec4(tex_color, 1.0);
}
