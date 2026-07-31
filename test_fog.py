"""Fog self-check: renders the real chunk/sky/water shaders offscreen.

Guards three things that fail as "looks wrong" rather than as an exception:

  * fog is radial — turning the camera must not change how fogged a fixed piece
    of terrain is. With the old gl_FragCoord.z/w depth fog it thinned out by
    cos(angle off centre) as terrain slid to the edge of the screen.
  * fog is tied to render distance — clear up close, opaque by fog_end.
  * fully fogged terrain is *exactly* the sky colour along the same ray, so the
    horizon has no seam and distant land is not a brighter silhouette.

Run: python test_fog.py
"""

import math
import numpy as np
import moderngl
import glm

WIDTH = HEIGHT = 256
FOV = 65.0
FOG_END = 6 * 16 * 0.9          # renderer's values for render distance 6
FOG_START = FOG_END * 0.65
SKY_HORIZON = (0.6, 0.8, 0.95)
SKY_ZENITH = (0.0, 0.4, 0.8)


def _load(ctx, name):
    with open(f'shaders/{name}.vert') as f:
        vert = f.read()
    with open(f'shaders/{name}.frag') as f:
        frag = f.read()
    return ctx.program(vertex_shader=vert, fragment_shader=frag)


def _camera(yaw_deg):
    eye = glm.vec3(0.0, 0.0, 0.0)
    yaw = math.radians(yaw_deg)
    view = glm.lookAt(eye, eye + glm.vec3(math.sin(yaw), 0.0, -math.cos(yaw)),
                      glm.vec3(0, 1, 0))
    proj = glm.perspective(glm.radians(FOV), 1.0, 0.1, 1000.0)
    return eye, view, proj


def _screen_pos(proj, view, world):
    clip = proj * view * glm.vec4(*world, 1.0)
    return (int((clip.x / clip.w * 0.5 + 0.5) * WIDTH),
            int((clip.y / clip.w * 0.5 + 0.5) * HEIGHT))


def _read(fbo, px, py):
    data = np.frombuffer(fbo.read(components=3, dtype='f4'), dtype='f4')
    return data.reshape(HEIGHT, WIDTH, 3)[py, px]


def _render_probe(ctx, prog, fbo, world, yaw_deg):
    """Draw one quad at `world` and read the pixel at its centre.

    The camera sits at the origin looking down -Z, rotated by yaw_deg, so the
    same quad can be put dead centre or far off to the side of the screen.
    """
    x, y, z = world
    half = 2.0
    verts = []
    # Upright quad in the XY plane so it faces the camera at every yaw tested.
    for dx, dy in ((-half, -half), (half, -half), (half, half),
                   (-half, -half), (half, half), (-half, half)):
        # position, uv (layer 0), shading 1.0 (unshaded, so only fog moves it)
        verts += [x + dx, y + dy, z, 0.5, 0.5, 0.0, 1.0]
    vbo = ctx.buffer(np.array(verts, dtype='f4').tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f 3f 1f',
                                   'in_position', 'in_tex_coord', 'in_shading')])

    eye, view, proj = _camera(yaw_deg)
    prog['m_proj'].write(proj.to_bytes())
    prog['m_view'].write(view.to_bytes())
    prog['cam_pos'].write(eye)
    prog['sky_horizon'].write(glm.vec3(*SKY_HORIZON))
    prog['sky_zenith'].write(glm.vec3(*SKY_ZENITH))
    prog['water_line'] = -1000.0        # keep the underwater tint out of the way
    prog['fog_range'].write(glm.vec2(FOG_START, FOG_END))

    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render()

    px, py = _screen_pos(proj, view, world)
    pixel = _read(fbo, px, py)

    vbo.release()
    vao.release()
    return pixel, (px, py)


def _render_sky(ctx, prog, fbo, yaw_deg):
    """Full-screen sky, same camera, so a pixel can be compared against fog."""
    verts = np.array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0], dtype='f4')
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_position')])

    _, view, proj = _camera(yaw_deg)
    prog['m_inv_pv'].write(glm.inverse(proj * glm.mat4(glm.mat3(view))).to_bytes())
    prog['u_time'] = 0.0
    prog['sky_horizon'].write(glm.vec3(*SKY_HORIZON))
    prog['sky_zenith'].write(glm.vec3(*SKY_ZENITH))

    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLE_STRIP)

    vbo.release()
    vao.release()


def _fog_of(pixel, unfogged, fogged):
    """Recover the fog factor from a rendered pixel (colour lerps unfogged→fogged)."""
    span = np.array(fogged, dtype='f4') - unfogged
    axis = int(np.argmax(np.abs(span)))     # most-separated channel = best signal
    return float((pixel[axis] - unfogged[axis]) / span[axis])


def main():
    ctx = moderngl.create_standalone_context()
    chunk_prog = _load(ctx, 'chunk')
    water_prog = _load(ctx, 'water')
    sky_prog = _load(ctx, 'sky')

    tex = ctx.texture_array((1, 1, 1), 3, np.array([220, 40, 40], dtype='u1').tobytes())
    tex.use(0)
    fbo = ctx.simple_framebuffer((WIDTH, HEIGHT), components=3, dtype='f4')
    fbo.use()
    ctx.disable(moderngl.CULL_FACE)

    # Reference colours, both measured rather than assumed: the shader tints and
    # saturates the texture on the way through.
    near, _ = _render_probe(ctx, chunk_prog, fbo, (0.0, 0.0, -10.0), 0.0)
    far, far_at = _render_probe(ctx, chunk_prog, fbo, (0.0, 0.0, -FOG_END), 0.0)

    # --- fully fogged terrain must be the sky, pixel for pixel ---
    _render_sky(ctx, sky_prog, fbo, 0.0)
    sky_pixel = _read(fbo, *far_at)
    delta = float(np.max(np.abs(far - sky_pixel)))
    print(f'  fogged terrain {tuple(round(float(c), 3) for c in far)} vs '
          f'sky {tuple(round(float(c), 3) for c in sky_pixel)}  (max delta {delta:.4f})')
    assert delta < 0.01, 'fully fogged terrain does not match the sky — horizon seam'

    # --- rotation invariance: same terrain, centre of view vs edge of view ---
    dist = FOG_END * 0.85
    centre, at_centre = _render_probe(ctx, chunk_prog, fbo, (0.0, 0.0, -dist), 0.0)
    fog_centre = _fog_of(centre, near, far)

    for yaw in (-20.0, -28.0):
        # Rotating the camera slides the same world point toward the screen edge.
        edge, at_edge = _render_probe(ctx, chunk_prog, fbo, (0.0, 0.0, -dist), yaw)
        fog_edge = _fog_of(edge, near, far)
        assert abs(at_edge[0] - at_centre[0]) > WIDTH * 0.2, \
            f'probe did not move off centre at yaw {yaw}: {at_edge} vs {at_centre}'
        assert abs(fog_edge - fog_centre) < 0.02, (
            f'fog changed with view angle: {fog_centre:.3f} at centre vs '
            f'{fog_edge:.3f} at yaw {yaw} — fog is not radial')
        # What the old depth fog would have done, for the record.
        depth = dist * math.cos(math.radians(yaw))
        depth_fog = max(0.0, min(1.0, (depth - FOG_START) / (FOG_END - FOG_START))) ** 2
        print(f'  yaw {yaw:6.1f}°  radial {fog_edge:.3f}  '
              f'(view-axis depth fog would give {depth_fog:.3f})')

    # --- render distance ramp ---
    ramp = []
    for d in (0.25, 0.5, 0.65, 0.8, 0.95, 1.0):
        pixel, _ = _render_probe(ctx, chunk_prog, fbo, (0.0, 0.0, -FOG_END * d), 0.0)
        ramp.append((d, _fog_of(pixel, near, far)))
        print(f'  {d * FOG_END:6.1f} units ({d:.0%} of fog_end) -> fog {ramp[-1][1]:.3f}')

    assert ramp[0][1] < 0.01, 'near terrain should be clear'
    assert ramp[2][1] < 0.01, 'fog must not start before fog_range.x'
    assert ramp[3][1] < 0.25, 'the ramp should still be gentle at 80%'
    assert ramp[-1][1] > 0.99, 'fog must be opaque at fog_end, or chunks pop in'
    assert all(b[1] >= a[1] for a, b in zip(ramp, ramp[1:])), 'ramp must be monotonic'

    # --- water shares the curve and the sky colours ---
    for name in ('fog_range', 'sky_horizon', 'sky_zenith'):
        assert name in water_prog, f'water lost {name}: it will fade differently'
    print('OK: fog is radial, ramps with render distance, and resolves to the sky')


if __name__ == '__main__':
    main()
