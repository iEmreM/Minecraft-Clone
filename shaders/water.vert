#version 330 core

layout (location = 1) in vec3 in_position;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform float water_area;
uniform float water_line;
uniform vec2 water_center; // Player XZ: the plane follows the camera

out vec2 uv;
out vec3 frag_world_pos;

void main() {
    // The quad is only as big as the fog can see (water_area, from render
    // distance) and is centred on the player, so its rim is always past the
    // point where fog is opaque. It used to be a fixed 4000-unit plane anchored
    // at the origin: most of it was fragments nobody could see.
    vec3 pos = in_position;
    pos.xz = (pos.xz - 0.5) * water_area + water_center;
    pos.y += water_line;

    // UVs come from world position, not from the quad, so the texture stays put
    // while the plane slides with the player. Same 1-tile-per-unit scale as before.
    uv = pos.xz;

    frag_world_pos = pos;
    gl_Position = m_proj * m_view * vec4(pos, 1.0);
}
