#version 330 core

out vec4 fragColor;
in vec3 frag_pos;

uniform vec3 u_view_pos; // Not stricly needed if just using gl_FragCoord or UV
uniform float u_time;
uniform vec2 u_resolution;
uniform mat4 m_inv_pv; // Inverse Projection * View matrix to rebuild ray

// Noise functions (Simplex 2D)
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626, // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

void main() {
    // 1. Calculate Ray Direction from Normalized Device Coordinates
    vec2 ndc = frag_pos.xy; // -1 to 1
    vec4 clip = vec4(ndc, 1.0, 1.0); // Optimized for Far Plane
    vec4 eye = m_inv_pv * clip;
    vec3 ray_dir = normalize(eye.xyz / eye.w);
    
    // 2. Sky Gradient
    // Map ray_dir.y (-1 to 1) to colors
    // 0.0 = Horizon, 1.0 = Zenith
    
    vec3 color_zenith = vec3(0.0, 0.4, 0.8); // Deep Blue
    vec3 color_horizon = vec3(0.6, 0.8, 0.95); // Misty/White Blue
    
    // Background mix
    float gradient = smoothstep(-0.2, 0.5, max(0.0, ray_dir.y));
    vec3 sky_color = mix(color_horizon, color_zenith, gradient);
    
    // 3. Clouds (Only above horizon)
    if (ray_dir.y > 0.05) {
        // Project ray to a plane at height H
        // y = H implies scaling logic
        // World Pos on cloud plane: P = O + t * D
        // Since O is camera (moving), we want clouds to move WITH camera?
        // No, clouds should be static relative to world OR parallax.
        // For simplicity: Project onto a sphere or plane effectively infinite.
        
        float cloud_scale = 1.0 / ray_dir.y; // Perspective projection
        vec2 cloud_uv = ray_dir.xz * cloud_scale * 0.5;
        
        // Blocky Clouds Logic
        // Scale UVs to create "Voxels"
        float cloud_scale_base = 1.0;
        
        // Snap UVs to grid for blocky look
        float pixel_size = 30.0; // Size of cloud "pixels"
        vec2 blocky_uv = floor(cloud_uv * pixel_size) / pixel_size;
        
        // Add time movement (Wind) - separate from UV to animate blocks moving
        // Or animate UV before snapping for "moving grid" effect?
        // Let's animate BEFORE snapping so the blocks "crawl" like in Minecraft shaders
        
        blocky_uv.x += u_time * 0.005; 
        
        // Sample noise with blocky UVs
        float noise = snoise(blocky_uv * 1.0);
        noise += snoise(blocky_uv * 2.0) * 0.5;
        noise = max(0.0, noise);
        
        // Hard step for consistent blocky edges
        // smoothstep makes it fuzzy, we want hard edges or very tight smoothstep
        float cloud_density = smoothstep(0.6, 0.65, noise); 
        
        // Mix white clouds
        sky_color = mix(sky_color, vec3(1.0), cloud_density * 0.9 * (1.0 - exp(-ray_dir.y * 5.0)));
    }
    
    // 4. Sun
    vec3 sun_dir = normalize(vec3(0.2, 0.8, 0.5)); // Arbitrary sun position
    float sun_intensity = max(0.0, dot(ray_dir, sun_dir));
    float sun_glare = pow(sun_intensity, 200.0); // Small sharp sun
    float sun_halo = pow(sun_intensity, 50.0) * 0.5; // Wider halo
    
    sky_color += vec3(1.0, 0.9, 0.7) * (sun_glare + sun_halo);

    fragColor = vec4(sky_color, 1.0);
}
