#version 330 core

in vec3 uv;
in float shading;
in vec3 frag_world_pos;

uniform sampler2DArray u_texture_0;
uniform vec3 bg_color;
uniform float water_line; // Water level for underwater effects

out vec4 fragColor;

void main() {
    // Sample texture
    vec3 tex_color = texture(u_texture_0, uv).rgb;
    
    // OPTIMIZATION: Removed gamma correction (2× pow operations)
    // Textures are already in sRGB, modern displays handle this
    
    // Apply shading (ambient occlusion + face lighting)
    // Enhanced Cinematic Tint: More vibrant warm sun and cooler shadows
    vec3 sunlight = vec3(1.08, 1.02, 0.96); // Warmer, more golden sunlight
    vec3 shadow = vec3(0.62, 0.68, 0.76);   // Slightly brighter shadows for more visibility
    
    // OPTIMIZATION: Replaced smoothstep with simple remap
    // shading varies from 0.4 (darkest) to 1.0 (brightest)
    float light_factor = (shading - 0.4) / 0.6; // Remap [0.4, 1.0] → [0.0, 1.0]
    light_factor = clamp(light_factor, 0.0, 1.0);
    vec3 light_tint = mix(shadow, sunlight, light_factor);
    
    tex_color *= shading * light_tint;
    
    // Underwater effect - lighter, less intense blue tint
    if (frag_world_pos.y < water_line) {
        tex_color *= vec3(0.35, 0.55, 0.85); // Lighter underwater (was 0.0, 0.3, 1.0)
    }
    
    // OPTIMIZATION: Simplified fog calculation (linear instead of exponential)
    // Increased fog density for better atmospheric depth
    float fog_dist = gl_FragCoord.z / gl_FragCoord.w;
    float fog_factor = fog_dist * 0.0012; // Increased from 0.0008 for more visible fog
    fog_factor = clamp(fog_factor, 0.0, 1.0);
    
    // Mix with background color for atmospheric depth
    tex_color = mix(tex_color, bg_color, fog_factor);
    
    // Minimal saturation boost for color vibrancy (very cheap: 1 dot + 1 mix)
    float gray = dot(tex_color, vec3(0.299, 0.587, 0.114));
    tex_color = mix(vec3(gray), tex_color, 1.08); // Slight saturation boost
    
    fragColor = vec4(tex_color, 1.0);
}
