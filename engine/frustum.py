import math

import numpy as np

from world.modern_chunk import CHUNK_HEIGHT, CHUNK_SIZE


class Frustum:
    """Camera frustum for culling chunks.

    Gribb-Hartmann plane extraction from the view-projection matrix, then an
    exact positive-vertex test of each chunk's AABB against all six planes.

    The test runs over **every chunk at once** with numpy. It used to run one
    chunk at a time in Python, building eight glm vectors per chunk per frame,
    and at 2.5 us each that was the single largest item in the frame at any
    render distance worth having: 4.4 ms out of an 11 ms frame at render
    distance 24, and growing straight-line with the chunk count. The arithmetic
    here is the same arithmetic, so the set of chunks that survives is the same
    set — this is a rewrite of how the loop is spelled, not of what it decides.
    """

    def __init__(self):
        # left, right, bottom, top, near, far — one (a, b, c, d) tuple each
        self.planes = [(0.0, 0.0, 0.0, 0.0)] * 6

    def extract_planes(self, view_projection_matrix):
        """Pull the six frustum planes out of a view-projection matrix.

        Plain Python floats, not numpy: this is 24 numbers once a frame, and
        numpy's per-call overhead on arrays that small costs more than the
        arithmetic saves — measured 0.24 ms a frame against 0.05 ms for this.
        The per-chunk test below is the opposite case and does want numpy.
        """
        # glm hands over its columns; the extraction is written in rows.
        c0, c1, c2, c3 = view_projection_matrix.to_tuple()
        r0 = (c0[0], c1[0], c2[0], c3[0])
        r1 = (c0[1], c1[1], c2[1], c3[1])
        r2 = (c0[2], c1[2], c2[2], c3[2])
        r3 = (c0[3], c1[3], c2[3], c3[3])

        planes = self.planes
        for i, (row, sign) in enumerate(((r0, 1), (r0, -1), (r1, 1),
                                         (r1, -1), (r2, 1), (r2, -1))):
            a = r3[0] + sign * row[0]
            b = r3[1] + sign * row[1]
            c = r3[2] + sign * row[2]
            d = r3[3] + sign * row[3]
            length = math.sqrt(a * a + b * b + c * c)
            if length > 0:
                a /= length
                b /= length
                c /= length
                d /= length
            planes[i] = (a, b, c, d)

    def visible_mask(self, min_x, max_x, min_z, max_z, out=None):
        """Boolean mask over chunks whose full-height AABB survives the frustum.

        *min_x* .. *max_z* are the pre-computed world-space extents of every
        chunk, one entry each. The chunks are columns of fixed height, so only
        the X and Z extents vary and the Y term of each plane is a scalar.
        """
        count = len(min_x)
        if out is None or len(out) != count:
            out = np.ones(count, dtype=bool)
        else:
            out.fill(True)

        if count == 0:
            return out

        distance = np.empty(count)
        for a, b, c, d in self.planes:
            # The positive vertex is the corner furthest along the plane
            # normal. Each plane's normal has a fixed sign, so picking the
            # corner is choosing between two arrays rather than a per-chunk
            # branch.
            np.multiply(max_x if a >= 0 else min_x, a, out=distance)
            distance += (max_z if c >= 0 else min_z) * c
            distance += d + b * (CHUNK_HEIGHT if b >= 0 else 0.0)
            out &= distance >= 0

        return out

    def is_chunk_visible(self, chunk_x, chunk_z):
        """Single-chunk test. Only the debug/telemetry paths want this — the
        renderer goes through visible_mask."""
        one = np.array([float(chunk_x * CHUNK_SIZE)])
        two = np.array([float(chunk_z * CHUNK_SIZE)])
        return bool(self.visible_mask(one, one + CHUNK_SIZE, two, two + CHUNK_SIZE)[0])
