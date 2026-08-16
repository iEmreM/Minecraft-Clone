#version 330 core
#include "sky_common.glsl"
#include "chunk_common.glsl"

// The opaque terrain pass. Deliberately has no `discard` and no blending: the
// whole near-to-far chunk sort is worth ~20% of frame time only because early-Z
// works here, and a `discard` anywhere in this shader turns that off for every
// fragment of terrain. See-through blocks are drawn by chunk_alpha.frag after
// this pass has finished.

void main() {
    // Alpha is 1.0 for every layer this pass can reach — build_atlas.py
    // flattens it — so it is dropped here rather than written and ignored.
    fragColor = vec4(shade_terrain(sample_nearest(uv)).rgb, 1.0);
}
