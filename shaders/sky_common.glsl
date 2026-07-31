// The sky, as one function, because three shaders have to agree on it:
// sky.frag draws it, and chunk.frag / water.frag fog to it along the view ray.
// Fully fogged terrain is therefore the sky pixel for pixel — clouds and sun
// included, not just the gradient. When only the gradient was shared, a
// half-fogged mountain standing in front of a cloud turned sky-blue and cut the
// cloud in half.
//
// Everything here is a function of the view direction alone — no camera
// position — so terrain 80 blocks out and the sky at infinity land on the same
// cloud, and the cloud reads as continuous across the mountain.
//
// Resolved by ShaderManager, not by the driver: GLSL 330 has no #include.

uniform vec3 sky_horizon;
uniform vec3 sky_zenith;
uniform float u_time;

// --- Simplex 2D noise ---
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

// What the sky looks like along `dir` (must be normalized).
vec3 sky_color(vec3 dir) {
    // Background gradient: 0 at the horizon, 1 at the zenith.
    float gradient = smoothstep(-0.2, 0.5, max(0.0, dir.y));
    vec3 color = mix(sky_horizon, sky_zenith, gradient);

    // Blocky clouds, above the horizon only — the projection onto the cloud
    // plane blows up at dir.y = 0, and this branch is also what keeps the two
    // noise samples off every downward-looking fogged fragment.
    if (dir.y > 0.05) {
        vec2 cloud_uv = dir.xz * (0.5 / dir.y);        // project onto the plane
        vec2 blocky_uv = floor(cloud_uv * 30.0) / 30.0; // snap: cloud "pixels"
        blocky_uv.x += u_time * 0.005;                  // wind

        float noise = snoise(blocky_uv) + snoise(blocky_uv * 2.0) * 0.5;
        float density = smoothstep(0.6, 0.65, noise);   // hard blocky edges
        color = mix(color, vec3(1.0),
                    density * 0.9 * (1.0 - exp(-dir.y * 5.0)));
    }

    // Sun: sharp disc plus a wider halo.
    float sun = max(0.0, dot(dir, normalize(vec3(0.2, 0.8, 0.5))));
    color += vec3(1.0, 0.9, 0.7) * (pow(sun, 200.0) + pow(sun, 50.0) * 0.5);

    return color;
}
