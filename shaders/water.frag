#version 330 core

layout (location = 0) out vec4 fragColor;

const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = 1 / gamma;

in vec2 uv;
in vec3 frag_world_pos;

uniform sampler2D u_texture_0;
uniform vec2 fog_range;   // matches chunk.frag, so water and land fade together
uniform vec3 cam_pos;
uniform vec3 sky_horizon;
uniform vec3 sky_zenith;

void main() {
    vec3 tex_col = texture(u_texture_0, uv).rgb;
    tex_col = pow(tex_col, gamma);

    // Same radial fog curve as the terrain — distance from the eye, ramp squared.
    vec3 to_frag = frag_world_pos - cam_pos;
    float fog_dist = length(to_frag);
    float fog_factor = clamp((fog_dist - fog_range.x) / (fog_range.y - fog_range.x),
                             0.0, 1.0);
    fog_factor *= fog_factor;

    // gamma correction
    tex_col = pow(tex_col, inv_gamma);

    // Distant water goes to the sky colour along this ray and turns opaque, so it
    // hides the plane's edge instead of fading out and showing the void through it.
    float gradient = smoothstep(-0.2, 0.5, max(0.0, to_frag.y / fog_dist));
    fragColor = vec4(mix(tex_col, mix(sky_horizon, sky_zenith, gradient), fog_factor),
                     mix(0.5, 1.0, fog_factor));
}
