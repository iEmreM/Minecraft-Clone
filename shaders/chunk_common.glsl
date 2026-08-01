// The terrain fragment shader's body, shared by the two passes that draw
// chunks: `chunk.frag` (opaque) and `chunk_alpha.frag` (see-through blocks).
// They differ in four lines at the end of main(); everything that decides what
// a block looks like — face lighting, the underwater tint, the saturation
// boost, the fog — lives here, once, so the two passes cannot drift apart and a
// pane of glass cannot fog differently from the wall it is set into.
//
// Requires sky_common.glsl to be included first (for sky_color).
// Resolved by ShaderManager, not by the driver: GLSL 330 has no #include.

in vec3 uv;
in float shading;
in vec3 frag_world_pos;

uniform sampler2DArray u_texture_0;
uniform float water_line; // Water level for underwater effects
uniform vec2 fog_range;   // (start, fully opaque) in world units, render distance driven
uniform vec3 cam_pos;     // Eye position, for radial (spherical) fog

out vec4 fragColor;

// Colour and coverage for one terrain texel. Alpha comes back as 1.0 for
// anything the atlas baked opaque, so the opaque pass can ignore it entirely.
vec4 shade_terrain(vec4 texel) {
    vec3 tex_color = texel.rgb;

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

    // Saturation boost happens BEFORE the fog mix: after it, it also pushed the
    // fog colour away from the sky's, which is what made distant terrain read as
    // a brighter silhouette than the sky behind it.
    // (very cheap: 1 dot + 1 mix)
    float gray = dot(tex_color, vec3(0.299, 0.587, 0.114));
    tex_color = mix(vec3(gray), tex_color, 1.08); // Slight saturation boost

    // Fog is tied to the render distance (fog_range), so chunks fade into the sky
    // instead of popping in at the edge. The squared ramp keeps everything up to
    // fog_range.x clear and piles the haze up over the last stretch.
    //
    // Distance is radial (from the eye), like Minecraft's spherical fog — NOT
    // gl_FragCoord.z/w, which is depth along the view axis: that made the fog on
    // a fixed piece of terrain thin out as you turned and it slid to the edge of
    // the screen, because its z-depth shrinks by cos(angle off centre).
    vec3 to_frag = frag_world_pos - cam_pos;
    float fog_dist = length(to_frag);
    float fog_factor = clamp((fog_dist - fog_range.x) / (fog_range.y - fog_range.x),
                             0.0, 1.0);
    fog_factor *= fog_factor;

    // Fog colour is the sky colour along this same ray, from the same function
    // sky.frag draws with — gradient, clouds and sun. A flat colour could only
    // match the sky at one height; the gradient alone could only match it where
    // there is no cloud, so a half-fogged mountain used to cut a cloud in half.
    // Skipped entirely below fog_range.x, which is most of the screen.
    float alpha = texel.a;
    if (fog_factor > 0.0) {
        tex_color = mix(tex_color, sky_color(to_frag / fog_dist), fog_factor);
        // Glass goes opaque as it fogs, the same way distant water does. Left
        // translucent, a fogged window blends the sky over the sky and stays a
        // visibly paler rectangle against it.
        alpha = mix(alpha, 1.0, fog_factor);
    }

    return vec4(tex_color, alpha);
}
