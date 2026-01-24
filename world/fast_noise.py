import numpy as np
from numba import njit
import math

# Fast Simplex Noise implementation compatible with Numba
# Based on public domain implementations of Simplex Noise

@njit
def fast_floor(x):
    return int(x) if x >= 0 else int(x) - 1

@njit
def dot2(g, x, y):
    return g[0]*x + g[1]*y

@njit
def dot3(g, x, y, z):
    return g[0]*x + g[1]*y + g[2]*z

# Permutation table
_perm = np.array([
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
], dtype=np.int32)

# Gradient table (reshuffled)
GRAD3 = np.array([
    1,1,0, -1,1,0, 1,-1,0, -1,-1,0,
    1,0,1, -1,0,1, 1,0,-1, -1,0,-1,
    0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1
], dtype=np.int32)

@njit
def seed_noise(seed):
    """Simple seeding by permuting the table based on seed"""
    np.random.seed(seed)
    # We can't modify global array inside njit easily if it's constant
    # So we return a new perm table or just use different offsets?
    # For now, let's keep it static but allow offset input
    pass

@njit
def fast_noise2(x, y):
    """2D Simplex Noise"""
    # Skewing
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    s = (x + y) * F2
    i = fast_floor(x + s)
    j = fast_floor(y + s)
    
    G2 = (3.0 - math.sqrt(3.0)) / 6.0
    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0
    
    # Determine which simplex we are in
    if x0 > y0:
        i1 = 1; j1 = 0
    else:
        i1 = 0; j1 = 1
        
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2
    
    # Wrap index
    ii = i & 255
    jj = j & 255
    
    # Calculate contribution from 3 corners
    t0 = 0.5 - x0*x0 - y0*y0
    if t0 < 0:
        n0 = 0.0
    else:
        t0 *= t0
        idx = (_perm[ii + _perm[jj]] % 12) * 3
        # dot product with gradient
        n0 = t0 * t0 * (GRAD3[idx]*x0 + GRAD3[idx+1]*y0)
        
    t1 = 0.5 - x1*x1 - y1*y1
    if t1 < 0:
        n1 = 0.0
    else:
        t1 *= t1
        idx = (_perm[ii + i1 + _perm[jj + j1]] % 12) * 3
        n1 = t1 * t1 * (GRAD3[idx]*x1 + GRAD3[idx+1]*y1)
        
    t2 = 0.5 - x2*x2 - y2*y2
    if t2 < 0:
        n2 = 0.0
    else:
        t2 *= t2
        idx = (_perm[ii + 1 + _perm[jj + 1]] % 12) * 3
        n2 = t2 * t2 * (GRAD3[idx]*x2 + GRAD3[idx+1]*y2)
        
    # Scale result to [-1, 1] (Simplex noise usually in [-1, 1] but might need scaling)
    return 70.0 * (n0 + n1 + n2)

@njit
def fast_noise3(x, y, z):
    """3D Simplex Noise"""
    F3 = 1.0 / 3.0
    s = (x + y + z) * F3
    i = fast_floor(x + s)
    j = fast_floor(y + s)
    k = fast_floor(z + s)
    
    G3 = 1.0 / 6.0
    t = (i + j + k) * G3
    X0 = i - t
    Y0 = j - t
    Z0 = k - t
    x0 = x - X0
    y0 = y - Y0
    z0 = z - Z0
    
    # Simplex order
    if x0 >= y0:
        if y0 >= z0:
            i1=1; j1=0; k1=0; i2=1; j2=1; k2=0
        elif x0 >= z0:
            i1=1; j1=0; k1=0; i2=1; j2=0; k2=1
        else:
            i1=0; j1=0; k1=1; i2=1; j2=0; k2=1
    else:
        if y0 < z0:
            i1=0; j1=0; k1=1; i2=0; j2=1; k2=1
        elif x0 < z0:
            i1=0; j1=1; k1=0; i2=0; j2=1; k2=1
        else:
            i1=0; j1=1; k1=0; i2=1; j2=1; k2=0
            
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    
    x2 = x0 - i2 + 2.0*G3
    y2 = y0 - j2 + 2.0*G3
    z2 = z0 - k2 + 2.0*G3
    
    x3 = x0 - 1.0 + 3.0*G3
    y3 = y0 - 1.0 + 3.0*G3
    z3 = z0 - 1.0 + 3.0*G3
    
    ii = i & 255
    jj = j & 255
    kk = k & 255
    
    n0 = 0.0
    t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
    if t0 > 0:
        t0 *= t0
        idx = (_perm[ii + _perm[jj + _perm[kk]]] % 12) * 3
        n0 = t0 * t0 * (GRAD3[idx]*x0 + GRAD3[idx+1]*y0 + GRAD3[idx+2]*z0)
        
    n1 = 0.0
    t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
    if t1 > 0:
        t1 *= t1
        idx = (_perm[ii + i1 + _perm[jj + j1 + _perm[kk + k1]]] % 12) * 3
        n1 = t1 * t1 * (GRAD3[idx]*x1 + GRAD3[idx+1]*y1 + GRAD3[idx+2]*z1)
        
    n2 = 0.0
    t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
    if t2 > 0:
        t2 *= t2
        idx = (_perm[ii + i2 + _perm[jj + j2 + _perm[kk + k2]]] % 12) * 3
        n2 = t2 * t2 * (GRAD3[idx]*x2 + GRAD3[idx+1]*y2 + GRAD3[idx+2]*z2)
        
    n3 = 0.0
    t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
    if t3 > 0:
        t3 *= t3
        idx = (_perm[ii + 1 + _perm[jj + 1 + _perm[kk + 1]]] % 12) * 3
        n3 = t3 * t3 * (GRAD3[idx]*x3 + GRAD3[idx+1]*y3 + GRAD3[idx+2]*z3)
        
    return 32.0 * (n0 + n1 + n2 + n3)
