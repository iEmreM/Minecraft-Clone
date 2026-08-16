#version 330 core
#include "sky_common.glsl"
#include "chunk_common.glsl"

// The see-through terrain pass: glass, stained glass, ice, copper grates.
// Same vertex shader, same vertex format and the same shading as chunk.frag —
// only the last two lines differ.
//
// The `discard` is why this is a separate program rather than a branch in
// chunk.frag. Glass is 75% holes with alpha 0, and blending alone would leave
// those fragments writing depth, so a window would silently occlude whatever is
// behind it. Discarding them costs this pass its early-Z, which is affordable
// because it covers a few windows rather than the whole world.

void main() {
    vec4 texel = sample_nearest(uv);
    if (texel.a < 0.02) {
        discard;                 // a hole in the glass, not a faint pane
    }
    fragColor = shade_terrain(texel);
}
