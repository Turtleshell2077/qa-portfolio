"""Minimal vectorised signed-distance-field toolkit (numpy, float32).

Every primitive takes P with shape (N, 3) and returns (N,) distances in mm.
Negative = inside.  Fields are (approximately) 1-Lipschitz so they can be
sphere-traced for previews and polygonised with marching cubes.
"""
import numpy as np

F = np.float32


def v3(*a):
    return np.array(a, dtype=F)


def ln(v):
    """Euclidean length along the last axis."""
    return np.sqrt(np.einsum('...i,...i->...', v, v))


# ---------------------------------------------------------------- booleans
def smin(a, b, k):
    """Polynomial smooth minimum (soft union)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(*ds):
    out = ds[0]
    for d in ds[1:]:
        out = np.minimum(out, d)
    return out


def subtract(a, b):
    """a minus b."""
    return np.maximum(a, -b)


def ssubtract(a, b, k):
    return smax(a, -b, k)


# ------------------------------------------------------------- transforms
def rotmat(rx=0.0, ry=0.0, rz=0.0):
    """Local->world rotation matrix, angles in degrees, order Rz*Ry*Rx."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                              np.sin(ry), np.cos(rz), np.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(F)


def local(P, c, R):
    """World points -> local frame centred on c with orientation R."""
    return (P - np.asarray(c, F)) @ R


# ------------------------------------------------------------- primitives
def sphere(P, c, r):
    return ln(P - np.asarray(c, F)) - F(r)


def ellipsoid(P, c, r, R=None):
    p = P - np.asarray(c, F)
    if R is not None:
        p = p @ R
    r = np.asarray(r, F)
    k0 = ln(p / r)
    k1 = ln(p / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, F(1e-9))


def capsule(P, a, b, rad):
    a = np.asarray(a, F)
    ba = np.asarray(b, F) - a
    pa = P - a
    h = np.clip((pa @ ba) / float(ba @ ba), 0.0, 1.0)
    return ln(pa - h[:, None] * ba) - F(rad)


def round_cone(P, a, b, r1, r2):
    """Tapered capsule with exact distance (Inigo Quilez)."""
    a = np.asarray(a, F)
    b = np.asarray(b, F)
    ba = b - a
    l2 = float(ba @ ba)
    rr = float(r1 - r2)
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    t = pa * l2 - np.outer(y, ba)
    x2 = np.einsum('ij,ij->i', t, t)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    far = np.sqrt(np.maximum(x2 + z2, 0.0)) * il2 - r2
    near = np.sqrt(np.maximum(x2 + y2, 0.0)) * il2 - r1
    side = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, far,
                    np.where(np.sign(y) * a2 * y2 < k, near, side)).astype(F)


def box(P, c, b, rad=0.0, R=None):
    p = P - np.asarray(c, F)
    if R is not None:
        p = p @ R
    q = np.abs(p) - np.asarray(b, F)
    return (ln(np.maximum(q, 0.0)) + np.minimum(q.max(axis=-1), 0.0) - F(rad))


def torus(P, c, big, small, R=None):
    """Torus around the local Z axis."""
    p = P - np.asarray(c, F)
    if R is not None:
        p = p @ R
    q = np.hypot(p[:, 0], p[:, 1]) - F(big)
    return np.hypot(q, p[:, 2]) - F(small)


def cylinder(P, c, r, h, rad=0.0, R=None):
    """Rounded-edge cylinder, total height 2h, radius r, edge fillet rad."""
    p = P - np.asarray(c, F)
    if R is not None:
        p = p @ R
    qx = np.hypot(p[:, 0], p[:, 1]) - (F(r) - F(rad))
    qy = np.abs(p[:, 2]) - (F(h) - F(rad))
    return (np.minimum(np.maximum(qx, qy), 0.0)
            + np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) - F(rad))


def plane(P, axis, val, sign=1.0):
    """Half space: sign*(coord - val) <= 0 is solid."""
    return F(sign) * (P[:, axis] - F(val))


def slab(P, axis, centre, half):
    return np.abs(P[:, axis] - F(centre)) - F(half)


# ------------------------------------------------------------ 2D profiles
def seg2(p, a, b, r1, r2=None):
    """2D tapered capsule; p is (N,2)."""
    a = np.asarray(a, F)
    ba = np.asarray(b, F) - a
    pa = p - a
    h = np.clip((pa @ ba) / float(ba @ ba), 0.0, 1.0)
    r = F(r1) if r2 is None else F(r1) + (F(r2) - F(r1)) * h
    return ln(pa - h[:, None] * ba) - r


def circle2(p, c, r):
    return ln(p - np.asarray(c, F)) - F(r)


def star5(p, r, rf=0.42):
    """Five pointed star, outer radius r, rf = inner/outer ratio (I. Quilez)."""
    k1 = np.array([0.809016994375, -0.587785252292], dtype=F)
    k2 = np.array([-k1[0], k1[1]], dtype=F)
    p = p.copy()
    p[:, 0] = np.abs(p[:, 0])
    p -= 2.0 * np.maximum(p @ k1, 0.0)[:, None] * k1
    p -= 2.0 * np.maximum(p @ k2, 0.0)[:, None] * k2
    p[:, 0] = np.abs(p[:, 0])
    p[:, 1] -= F(r)
    ba = F(rf) * np.array([-k1[1], k1[0]], dtype=F) - np.array([0.0, 1.0], dtype=F)
    h = np.clip((p @ ba) / float(ba @ ba), 0.0, float(r))
    d = ln(p - h[:, None] * ba)
    return d * np.sign(p[:, 1] * ba[0] - p[:, 0] * ba[1])


def sphere_proj(P, centre, radius):
    """Map points onto arc-length coordinates of a sphere around `centre`.

    A 2D profile evaluated in these coordinates sweeps a radial cone rather
    than a prism, so decals wrap around curved surfaces (a face, a skull)
    instead of sticking out as flat tabs where the surface turns away."""
    q = P - np.asarray(centre, F)
    r = np.maximum(ln(q), F(1e-6))
    u = np.arctan2(q[:, 0], q[:, 1]) * F(radius)
    v = np.arcsin(np.clip(q[:, 2] / r, -1.0, 1.0)) * F(radius)
    return np.stack([u, v], axis=-1)


def cyl_proj(P, axis_y, radius):
    """Decal coordinates for a mostly-forward-facing surface: horizontal arc
    length around a vertical axis, and z kept as-is so vertical placement of a
    feature is exact (a spherical mapping shifts it down by the difference
    between the assumed and the real radius)."""
    u = np.arctan2(P[:, 0], P[:, 1] - F(axis_y)) * F(radius)
    return np.stack([u, P[:, 2]], axis=-1)


def cyl_uv(x, z, radius):
    return (float(np.arcsin(np.clip(float(x) / radius, -1.0, 1.0)) * radius), float(z))


def sphere_uv(x, z, centre, radius):
    """Arc-length coordinates of the surface point above (x, z)."""
    dz = float(z) - centre[2]
    v = np.arcsin(np.clip(dz / radius, -1.0, 1.0)) * radius
    rho = np.sqrt(max(radius * radius - dz * dz, 1e-6))
    u = np.arcsin(np.clip(float(x) / rho, -1.0, 1.0)) * radius
    return (float(u), float(v))


def proj(P, axes, centre=(0.0, 0.0)):
    """Project 3D points onto a 2D plane given by two axis indices."""
    c = np.asarray(centre, F)
    return np.stack([P[:, axes[0]] - c[0], P[:, axes[1]] - c[1]], axis=-1)


def chain(P, pts, radii, k=0.0):
    """Smoothly blended chain of tapered capsules through `pts`."""
    d = None
    for i in range(len(pts) - 1):
        seg = round_cone(P, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = seg if d is None else (np.minimum(d, seg) if k <= 0 else smin(d, seg, k))
    return d


def shell(P, c, r_out, r_in):
    """Hollow sphere wall."""
    return np.maximum(sphere(P, c, r_out), -sphere(P, c, r_in))


def lens(P, c, half_h, half_w, R=None, axis=1):
    """Almond aperture: a 2D lens (two overlapping circles) extruded along
    `axis`.  Extruding matters - a lens built from two *spheres* is a flat
    disc that never cuts through the full thickness of an eyelid."""
    p = P - np.asarray(c, F)
    if R is not None:
        p = p @ R
    u = p[:, 0] if axis != 0 else p[:, 1]
    v = p[:, 2] if axis != 2 else p[:, 1]
    a = (half_w * half_w - half_h * half_h) / (2.0 * half_h)
    ra = F(a + half_h)
    return np.maximum(np.hypot(u, v - F(a)) - ra, np.hypot(u, v + F(a)) - ra)


# --------------------------------------------------------------- surface detail
def emboss(body, profile, height, clip=None):
    """Raised detail that hugs the surface of `body` (a prism intersected
    with the body inflated by `height`).  `clip` is an optional extra SDF
    used to keep the detail on one side only."""
    d = np.maximum(profile, body - F(height))
    if clip is not None:
        d = np.maximum(d, clip)
    return d


def engrave(body, profile, depth, clip=None):
    """Cutter solid for a groove of `depth` sunk into the surface."""
    d = np.maximum(profile, -(body + F(depth)))
    if clip is not None:
        d = np.maximum(d, clip)
    return d
